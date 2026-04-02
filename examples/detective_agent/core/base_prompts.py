"""Base prompt utilities for the RLM loop. Forked from rlms library."""

import textwrap

from examples.detective_agent.core.types import QueryMetadata

# Generic base system prompt — overridden by our RCA-specific system_prompt.py
RLM_BASE_SYSTEM_PROMPT = textwrap.dedent(
    """You are tasked with answering a query with associated context. You can access, transform,
and analyze this context interactively in a REPL environment. You will be queried iteratively
until you provide a final answer.

Available REPL globals:
- `context` / `context_0`: the loaded context payload
- `llm_query(prompt: str) -> str`: query the LM from REPL code
- `llm_query_batched(prompts: list[str]) -> list[str]`: concurrent LM queries
- `SHOW_VARS()`: see all REPL variables
- `context_budget()`: check remaining context budget

Execute Python code with ```repl blocks. Provide final answer via FINAL_VAR(variable_name).
""",
)

USER_PROMPT = (
    "Review what you have learned so far (in plain text), then write ```repl code to gather "
    "more data or delegate analysis to llm_query(). Your next action:"
)
USER_PROMPT_WITH_ROOT = (
    'Continue your investigation. Recall the original task: "{root_prompt}"'
    "\n\nState your reasoning in plain text, then write ```repl code for your next step:"
)


def build_rlm_system_prompt(
    system_prompt: str,
    query_metadata: QueryMetadata,
) -> list[dict[str, str]]:
    """
    Build the initial message list for the RLM loop.

    Returns [system message, assistant acknowledgement with context metadata].
    """
    context_lengths = query_metadata.context_lengths
    context_total_length = query_metadata.context_total_length
    context_type = query_metadata.context_type

    if len(context_lengths) > 100:
        others = len(context_lengths) - 100
        context_lengths_str = str(context_lengths[:100]) + f"... [{others} others]"
    else:
        context_lengths_str = str(context_lengths)

    metadata_prompt = (
        f"Your context is a {context_type} with {context_total_length} total characters, "
        f"broken into chunks of char lengths: {context_lengths_str}."
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "assistant", "content": metadata_prompt},
    ]


def build_user_prompt(
    root_prompt: str | None = None,
    iteration: int = 0,
    context_count: int = 1,
    history_count: int = 0,
) -> dict[str, str]:
    """Build the per-iteration user message."""
    if iteration == 0:
        safeguard = (
            " You haven't accessed the REPL yet. Start by inspecting file_manifest "
            "to see what files are available, then build your investigation plan."
        )
    else:
        safeguard = " The history above is your previous interactions."

    if history_count > 0:
        history_note = (
            f" You have {history_count} prior conversation histories available "
            f"as history_0 ... history_{history_count - 1}."
        )
    else:
        history_note = ""

    if context_count > 1:
        context_note = (
            f" You have {context_count} contexts loaded: "
            f"context_0 ... context_{context_count - 1} (context aliases context_0)."
        )
    else:
        context_note = ""

    if root_prompt:
        content = USER_PROMPT_WITH_ROOT.format(root_prompt=root_prompt) + safeguard + context_note + history_note
    else:
        content = USER_PROMPT + safeguard + context_note + history_note

    return {"role": "user", "content": content}
