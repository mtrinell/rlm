"""Parsing utilities for RLM trajectories. Forked from rlms library."""

import re
from typing import TYPE_CHECKING

from examples.detective_agent.core.types import REPLResult, RLMIteration

if TYPE_CHECKING:
    from examples.detective_agent.environment.base_env import BaseEnv


def find_code_blocks(text: str) -> list[str]:
    """Find REPL code blocks in text wrapped in triple backticks."""
    pattern = r"```repl\s*\n(.*?)\n```"
    results = []
    for match in re.finditer(pattern, text, re.DOTALL):
        code_content = match.group(1).strip()
        results.append(code_content)
    return results


def find_final_answer(text: str, environment: "BaseEnv | None" = None) -> str | None:
    """
    Detect when the model has provided a final answer and return it.

    Detection order:
    1. REPL namespace: FINAL_VAR was assigned as a string (model wrote `FINAL_VAR = "..."`)
    2. Text regex: FINAL_VAR("varname") call pattern in response text
    3. Text regex: FINAL("...") inline pattern

    Returns None if no final answer is found.
    """
    # 1. Check REPL namespace first — catches the natural assignment form
    #    `FINAL_VAR = """..."""` that the model writes inside a ```repl block.
    #    execute_code() persists the string value to environment.locals when detected.
    if environment is not None and hasattr(environment, "locals"):
        fv = environment.locals.get("FINAL_VAR")
        if isinstance(fv, str) and fv.strip():
            return fv.strip()

    # 2. FINAL_VAR("varname") call pattern — older rlm-library style
    final_var_pattern = r"^\s*FINAL_VAR\((.*?)\)"
    match = re.search(final_var_pattern, text, re.MULTILINE | re.DOTALL)
    if match:
        variable_name = match.group(1).strip().strip('"').strip("'")
        if environment is not None:
            result = environment.execute_code(f"print(FINAL_VAR({variable_name!r}))")
            final_answer = result.stdout.strip()
            if final_answer == "":
                final_answer = result.stderr.strip() or ""
            return final_answer
        return None

    # 3. FINAL("inline text") pattern
    final_pattern = r"^\s*FINAL\((.*)\)\s*$"
    match = re.search(final_pattern, text, re.MULTILINE | re.DOTALL)
    if match:
        return match.group(1).strip()

    return None


def format_iteration(
    iteration: RLMIteration,
    max_character_length: int = 8000,
) -> list[dict[str, str]]:
    """
    Format an RLM iteration to append to the message history for the next iteration.

    Truncates code execution results that exceed max_character_length.
    """
    messages = [{"role": "assistant", "content": iteration.response}]

    for code_block in iteration.code_blocks:
        code = code_block.code
        result = code_block.result
        result_str = format_execution_result(result)
        if len(result_str) > max_character_length:
            result_str = (
                result_str[:max_character_length] + f"... + [{len(result_str) - max_character_length} chars truncated]"
            )

        execution_message = {
            "role": "user",
            "content": f"Code executed:\n```python\n{code}\n```\n\nREPL output:\n{result_str}",
        }
        messages.append(execution_message)
    return messages


def format_execution_result(result: REPLResult) -> str:
    """Format the execution result as a string for display."""
    result_parts = []

    if result.stdout:
        result_parts.append(f"\n{result.stdout}")

    if result.stderr:
        result_parts.append(f"\n{result.stderr}")

    # Show key variable names (for introspection, not full values)
    important_vars = {}
    for key, value in result.locals.items():
        if (
            not key.startswith("_")
            and key not in ("__builtins__", "__name__", "__doc__")
            and isinstance(value, (str, int, float, bool, list, dict, tuple))
        ):
            important_vars[key] = ""

    if important_vars:
        result_parts.append(f"REPL variables: {list(important_vars.keys())}\n")

    return "\n\n".join(result_parts) if result_parts else "No output"
