"""
Big-context RLM example: comprehensive architectural review of a codebase.

The combined source code of the `rlm/` package (~40 files / ~300 KB of Python)
is passed as a dict of {relative_path: source_code} entries — comfortably larger
than a single LLM's usable context window once you account for reasoning room.
RLM handles this automatically: the model can slice, inspect individual files,
spawn sub-LM calls per module, and synthesise the results iteratively without
any manual chunking on your end.

Tracing is provided by Langfuse 4 (OTel-based) using the @observe decorator.
RLM iteration and sub-call callbacks emit nested events/spans inside the root
trace, giving full visibility into every step of the recursive reasoning process.

Prerequisites:
    uv pip install -e ".[langfuse]"   # installs langfuse

    All credentials are read from .env (see .env.example):
        PROXY_BASE_URL      - OpenAI-compatible proxy base URL
        PROXY_API_KEY       - API key for the proxy
        PROXY_MODEL         - model name
        LANGFUSE_PUBLIC_KEY - Langfuse project public key
        LANGFUSE_SECRET_KEY - Langfuse project secret key
        LANGFUSE_BASE_URL   - Langfuse host (default: https://cloud.langfuse.com)

Run with:
    make langfuse-example
    # or directly:
    uv run python -m examples.langfuse_example
"""

import logging
import os
import pathlib

from dotenv import load_dotenv
from langfuse import Langfuse, observe

from rlm import RLM
from rlm.logger import RLMLogger

load_dotenv()

# Show INFO-level progress from the RLM core; suppress noisier third-party loggers.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Langfuse 4 — uses its own OTel-based SDK; no callback integration needed.
# Reads LANGFUSE_BASE_URL (project convention) or LANGFUSE_HOST.
# ---------------------------------------------------------------------------
langfuse_public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
langfuse_secret_key = os.getenv("LANGFUSE_SECRET_KEY")
langfuse_host = os.getenv("LANGFUSE_BASE_URL") or os.getenv(
    "LANGFUSE_HOST", "https://cloud.langfuse.com"
)

langfuse_client: Langfuse | None = None
if langfuse_public_key and langfuse_secret_key:
    langfuse_client = Langfuse(
        public_key=langfuse_public_key,
        secret_key=langfuse_secret_key,
        host=langfuse_host,
    )
    print(f"Langfuse tracing enabled  →  {langfuse_host}")
else:
    print(
        "Warning: LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY not set — "
        "continuing without traces.  Add them to .env to enable."
    )

# ---------------------------------------------------------------------------
# Build the large context: load every Python source file in the project and
# assemble a dict  { "rlm/core/rlm.py": "<source>" , ... }.
#
# On this repo's rlm/ package: ~40 files / ~300 KB / ~75 K tokens.  That fills
# a typical 128 K-token window leaving almost no room for reasoning.  RLM lets
# the model tackle it iteratively, delegating individual file analysis to
# sub-LM calls.
# ---------------------------------------------------------------------------
REPO_ROOT = pathlib.Path(__file__).parent.parent

# Only index the core library (rlm/) — still ~40 files / ~300 KB, comfortably
# larger than any single LLM context window but tractable for RLM recursion.
source_files: dict[str, str] = {}
for py_file in sorted((REPO_ROOT / "rlm").rglob("*.py")):
    if "__pycache__" in py_file.parts:
        continue
    try:
        source_files[str(py_file.relative_to(REPO_ROOT))] = py_file.read_text(
            encoding="utf-8"
        )
    except OSError:
        continue

total_chars = sum(len(v) for v in source_files.values())
total_lines = sum(v.count("\n") for v in source_files.values())
print(
    f"Loaded {len(source_files)} Python files — "
    f"{total_chars:,} chars / {total_lines:,} lines from rlm/."
)
print("The model will work through this recursively rather than reading it all at once.\n")

# ---------------------------------------------------------------------------
# RLM — OpenAI-compatible backend pointing at the LiteLLM proxy.
# Reads PROXY_BASE_URL / PROXY_API_KEY / PROXY_MODEL from .env, with
# fallback to the legacy LITELLM_* names for backward compatibility.
# ---------------------------------------------------------------------------
api_base = os.getenv("PROXY_BASE_URL") or os.getenv("LITELLM_PROXY_URL", "")
api_key = os.getenv("PROXY_API_KEY") or os.getenv("LITELLM_API_KEY")
model_name = os.getenv("PROXY_MODEL") or os.getenv("LITELLM_MODEL", "openai/bedrock/global.anthropic.claude-sonnet-4-6")

if not api_key:
    raise ValueError(
        "PROXY_API_KEY is not set.  Copy .env.example to .env and fill in your credentials."
    )

logger = RLMLogger(log_dir="./logs")

# ---------------------------------------------------------------------------
# RLM callbacks — called synchronously inside rlm.completion(), which runs
# inside the @observe-decorated function below.  Langfuse's OTel context
# propagation therefore automatically nests these spans under the root trace.
#
# Each RLM iteration opens a Langfuse span (on_iteration_start) that is closed
# with timing metadata (on_iteration_complete).  Sub-calls each get their own
# nested "agent" span.  All spans are nested under the root @observe trace.
# ---------------------------------------------------------------------------
_iteration_spans: dict[tuple[int, int], object] = {}  # (depth, iter) → open span
_subcall_spans: dict[int, object] = {}  # depth → open span


def _on_iteration_start(depth: int, iteration_num: int) -> None:
    if langfuse_client is None:
        return
    span = langfuse_client.start_observation(
        name=f"iteration-{iteration_num}",
        as_type="span",
        metadata={"depth": depth, "iteration": iteration_num},
    )
    _iteration_spans[(depth, iteration_num)] = span


def _on_iteration_complete(depth: int, iteration_num: int, duration: float) -> None:
    span = _iteration_spans.pop((depth, iteration_num), None)
    if span is not None:
        span.update(  # type: ignore[union-attr]
            metadata={"depth": depth, "iteration": iteration_num, "duration_s": round(duration, 3)},
        )
        span.end()  # type: ignore[union-attr]


def _on_subcall_start(depth: int, model: str, prompt_preview: str) -> None:
    if langfuse_client is None:
        return
    span = langfuse_client.start_observation(
        name="rlm-subcall",
        as_type="agent",
        model=model,
        input={"depth": depth, "prompt_preview": prompt_preview[:500]},
    )
    _subcall_spans[depth] = span


def _on_subcall_complete(depth: int, model: str, duration: float, error: str | None) -> None:
    span = _subcall_spans.pop(depth, None)
    if span is not None:
        span.update(  # type: ignore[union-attr]
            metadata={"duration_s": round(duration, 3), "error": error},
            level="ERROR" if error else "DEFAULT",
        )
        span.end()  # type: ignore[union-attr]


rlm = RLM(
    backend="openai",
    backend_kwargs={
        "model_name": model_name,
        "api_key": api_key,
        "base_url": api_base,
    },
    environment="local",
    max_depth=2,       # allow one level of recursive sub-calls
    max_iterations=25,
    logger=logger,
    verbose=True,
    on_iteration_start=_on_iteration_start,
    on_iteration_complete=_on_iteration_complete,
    on_subcall_start=_on_subcall_start,
    on_subcall_complete=_on_subcall_complete,
)

# ---------------------------------------------------------------------------
# Task — the root_prompt is the question; the large dict is the context.
# The RLM will have it available as `context` inside the REPL and can slice,
# search, and delegate pieces to sub-LM calls as it sees fit.
# ---------------------------------------------------------------------------
root_prompt = (
    "You are a senior software architect reviewing the RLM (Recursive Language Models) "
    "library.  The `context` variable in the REPL is a **dict[str, str]** where each key "
    "is a relative file path (e.g. 'rlm/core/rlm.py') and the value is the full source code "
    "of that file.  Iterate over it with `for path, src in context.items():`.\n\n"
    "Produce a **comprehensive architectural review** covering:\n\n"
    "1. **Module overview** – list every module/package and its one-line purpose.\n"
    "2. **Core abstractions** – key base classes and their design contracts "
    "(BaseLM, NonIsolatedEnv, IsolatedEnv, etc.).\n"
    "3. **Data-flow walkthrough** – trace a single `RLM.completion()` call end-to-end: "
    "from user code through the LM handler socket protocol, environment execution, and back.\n"
    "4. **Design patterns** – concrete examples (Template Method, Strategy, Context Manager, "
    "Protocol…) with file/class references.\n"
    "5. **Top-5 actionable improvements** – concrete suggestions with file references.\n\n"
    "IMPORTANT REPL RULES:\n"
    "- Never assign long text as a triple-quoted string literal in REPL code — it will be "
    "truncated and cause a SyntaxError.  Instead, build reports by calling `llm_query()` "
    "and storing the returned string: `report = llm_query('...')`.\n"
    "- Build each section separately with `llm_query`, store in variables, then join them: "
    "`report = '\\n\\n'.join([sec1, sec2, sec3, sec4, sec5])`.\n\n"
    "Store the finished report in a variable called `report` and call FINAL_VAR(\"report\")."
)

# ---------------------------------------------------------------------------
# Run — wrapped with @observe so langfuse records a root trace for the entire
# completion: timing, input metadata, and the final output are all captured.
# Iteration events and sub-call spans nest under this trace automatically.
# ---------------------------------------------------------------------------
@observe(name="rlm-codebase-review", as_type="agent", capture_input=False)
def run_rlm_completion() -> str:
    result = rlm.completion(source_files, root_prompt=root_prompt)
    if langfuse_client is not None:
        langfuse_client.update_current_span(
            output=result.response[:2000],
            metadata={
                "execution_time_s": round(result.execution_time, 2),
                "files_analysed": len(source_files),
                "total_chars": total_chars,
                "model": model_name,
            },
        )
    return result.response


print("=" * 70)
print("Starting RLM completion — the model will iterate over the codebase…")
print("=" * 70)

response = run_rlm_completion()

print("\n" + "=" * 70)
print("ARCHITECTURAL REVIEW")
print("=" * 70)
print(response)

if langfuse_client is not None:
    langfuse_client.flush()
    print(f"\nTraces → {langfuse_host}")
