# Detective Agent

Investigation engine using Recursive Language Models (RLMs). A single LM self-programs its investigation strategy through code execution in a REPL environment.

## Use Cases

### 1. Behavioral Deviation Detection (default)

Given the README/documentation of a multi-agent application and its actual OpenTelemetry traces, the agent detects whether the app is behaving as documented — flagging derailments, failures, protocol violations, and anomalies.

**Inputs:**
- `APP_README_PATH` — README or documentation of the app under test
- `TRACES_PATH` — OpenTelemetry JSONL traces from a live run

**Output:** A `BEHAVIORAL COMPLIANCE REPORT` with per-component findings classified as `DERAILED / DEGRADED / ANOMALOUS / HEALTHY`.

**Example dataset:** `examples/banking_app/` — SentinelBank AI, a multi-agent banking system.

### 2. Root Cause Analysis

Given a tarball of infrastructure logs (network, Kubernetes, syslog, etc.) and a problem statement, the agent identifies the root cause of an incident.

**Inputs:**
- `TEST_DATASET` — path to a `.tar` / `.tar.gz` log archive
- `PROBLEM_STATEMENT` — description of the observed failure

**Output:** A `ROOT CAUSE ANALYSIS REPORT` with evidence, causal chain, and confidence rating.

## Running

```bash
# From the detective_agent directory
uv run python -m examples.detective_agent.main

# With environment variable overrides
APP_README_PATH=examples/banking_app/README.md \
TRACES_PATH=examples/banking_app/otel-traces.jsonl \
uv run python -m examples.detective_agent.main
```

## Configuration

Settings are read from `examples/detective_agent/.env` via Pydantic Settings (`config.py`).

### Required

| Variable | Description |
|----------|-------------|
| `AZURE_OPENAI_API_KEY` | Azure OpenAI API key |
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI endpoint URL |

### Optional (with defaults)

| Variable | Default | Description |
|----------|---------|-------------|
| `AZURE_OPENAI_API_VERSION` | `2024-02-15-preview` | API version |
| `AZURE_OPENAI_MODEL` | `gpt-4o` | Model deployment name |
| `MAX_ITERATIONS` | `20` | Top-level LLM iteration limit |
| `MAX_DEPTH` | `1` | Recursive sub-call depth (`1` = single model) |
| `CONTEXT_TOKEN_LIMIT` | `100000` | Token budget ceiling; truncation tiers at 50/70/85% |
| `APP_README_PATH` | `examples/banking_app/README.md` | Path to app documentation |
| `TRACES_PATH` | `examples/banking_app/otel-traces.jsonl` | Path to OTEL JSONL traces |
| `INVESTIGATION_FOCUS` | *(empty)* | Optional hint (e.g. `"focus on the security agent"`) |
| `LOG_DIR` | `results` | Output directory for logs and result files |
| `VERBOSE` | `true` | Enable rich console output |

## Project Structure

```
detective_agent/
├── main.py              # Entry point (behavioral deviation detection)
├── config.py            # Pydantic Settings configuration
├── preprocessor.py      # Tarball extraction and file manifest (RCA use case)
├── log_utils.py         # Log setup
├── clients/             # LLM clients (Azure OpenAI, Anthropic, LiteLLM, OpenAI)
├── core/                # RLM loop, LM handler, parsing, types
├── context/             # Context management
│   ├── budget.py        # ContextBudget — 4-tier token truncation
│   └── history_manager.py
├── environment/         # REPL environments
│   ├── local_repl.py    # Base local REPL
│   ├── detector_repl.py # DetectorREPL — OTEL trace helpers injected
│   ├── detector_helpers.py  # load_traces, extract_log_records, get_errors, …
│   ├── rca_repl.py      # RcaREPL — filesystem helpers injected
│   └── helpers.py       # Filesystem helpers (read_file_safe, grep, …)
├── logger/              # Iteration logger (JSONL + rich console)
├── prompts/
│   ├── detector_system_prompt.py  # Behavioral deviation detection prompt
│   └── system_prompt.py           # RCA prompt
└── examples/
    └── banking_app/     # SentinelBank AI example dataset
        ├── README.md
        └── otel-traces.jsonl
```

## How It Works

1. **Input loading** — The app README and OTEL traces file are resolved from config and read into memory. No extraction step needed.
2. **RLM initialization** — A `DetectorREPL` environment is created with OTEL helpers (`load_traces`, `extract_log_records`, `get_errors`, `get_component_timeline`, `summarize_component`, …) and `app_readme` injected into the REPL namespace. `ContextBudget` tracks token usage with 4-tier truncation.
3. **Investigation loop** — The LM extracts the expected spec from the README, loads and triages traces, cross-references actual vs. documented behaviour per component, and builds a compliance verdict. Phase-injection inserts budget-pressure reminders at key milestones.
4. **Output** — The compliance report and a cost/iteration summary are written to `results/`.

## Troubleshooting

| Error | Fix |
|-------|-----|
| `README not found` | Check `APP_README_PATH` in `.env` |
| `Traces file not found` | Check `TRACES_PATH` in `.env` |
| `AZURE_OPENAI_API_KEY not set` | Set `AZURE_OPENAI_API_KEY` in `.env` |
| Investigation terminates early | Increase `MAX_ITERATIONS` in `.env` |
| `ModuleNotFoundError` | Run `uv sync` in the `detective_agent` directory |

## References

- [RLM Paper](https://arxiv.org/abs/2512.24601)
- [RLM Framework](https://github.com/alexzhang13/rlm)
