"""Rich console printer for RLM verbose output. Forked from rlm/logger/verbose.py."""

from typing import Any

from rich.console import Console, Group
from rich.panel import Panel
from rich.rule import Rule
from rich.style import Style
from rich.table import Table
from rich.text import Text

from examples.detective_agent.core.types import CodeBlock, RLMIteration, RLMMetadata

COLORS = {
    "primary": "#7AA2F7",
    "secondary": "#BB9AF7",
    "success": "#9ECE6A",
    "warning": "#E0AF68",
    "error": "#F7768E",
    "text": "#A9B1D6",
    "muted": "#565F89",
    "accent": "#7DCFFF",
    "bg_subtle": "#1A1B26",
    "border": "#3B4261",
    "code_bg": "#24283B",
}

STYLE_PRIMARY = Style(color=COLORS["primary"], bold=True)
STYLE_SECONDARY = Style(color=COLORS["secondary"])
STYLE_SUCCESS = Style(color=COLORS["success"])
STYLE_WARNING = Style(color=COLORS["warning"])
STYLE_ERROR = Style(color=COLORS["error"])
STYLE_TEXT = Style(color=COLORS["text"])
STYLE_MUTED = Style(color=COLORS["muted"])
STYLE_ACCENT = Style(color=COLORS["accent"], bold=True)


def _to_str(value: Any) -> str:
    if isinstance(value, str):
        return value
    return str(value)


class VerbosePrinter:
    """Rich console printer for RLM verbose output."""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self.console = Console() if enabled else None
        self._iteration_count = 0

    def print_metadata(self, metadata: RLMMetadata) -> None:
        if not self.enabled:
            return
        model = metadata.backend_kwargs.get("model_name", "unknown")
        other = list(metadata.other_backends) if metadata.other_backends else None
        self._print_header(
            backend=metadata.backend,
            model=model,
            environment=metadata.environment_type,
            max_iterations=metadata.max_iterations,
            max_depth=metadata.max_depth,
            other_backends=other,
        )

    def _print_header(
        self,
        backend: str,
        model: str,
        environment: str,
        max_iterations: int,
        max_depth: int,
        other_backends: list[str] | None = None,
    ) -> None:
        if not self.enabled:
            return
        title = Text()
        title.append("◆ ", style=STYLE_ACCENT)
        title.append("Detective Agent", style=STYLE_PRIMARY)
        title.append(" ━ RLM implementation", style=STYLE_MUTED)

        config_table = Table(show_header=False, show_edge=False, box=None, padding=(0, 2), expand=True)
        config_table.add_column("key", style=STYLE_MUTED, width=16)
        config_table.add_column("value", style=STYLE_TEXT)
        config_table.add_column("key2", style=STYLE_MUTED, width=16)
        config_table.add_column("value2", style=STYLE_TEXT)
        config_table.add_row(
            "Backend",
            Text(backend, style=STYLE_SECONDARY),
            "Environment",
            Text(environment, style=STYLE_SECONDARY),
        )
        config_table.add_row(
            "Model",
            Text(model, style=STYLE_ACCENT),
            "Max Iterations",
            Text(str(max_iterations), style=STYLE_WARNING),
        )
        if other_backends:
            config_table.add_row(
                "Sub-models",
                Text(", ".join(other_backends), style=STYLE_SECONDARY),
                "Max Depth",
                Text(str(max_depth), style=STYLE_WARNING),
            )
        else:
            config_table.add_row("Max Depth", Text(str(max_depth), style=STYLE_WARNING), "", "")

        panel = Panel(config_table, title=title, title_align="left", border_style=COLORS["border"], padding=(1, 2))
        self.console.print()
        self.console.print(panel)
        self.console.print()

    def print_iteration(self, iteration: RLMIteration, iteration_num: int) -> None:
        if not self.enabled:
            return
        self._iteration_count = iteration_num
        rule = Rule(Text(f" Iteration {iteration_num} ", style=STYLE_PRIMARY), style=COLORS["border"], characters="─")
        self.console.print(rule)

        # Response
        header = Text()
        header.append("◇ ", style=STYLE_ACCENT)
        header.append("LLM Response", style=STYLE_PRIMARY)
        if iteration.iteration_time:
            header.append(f"  ({iteration.iteration_time:.2f}s)", style=STYLE_MUTED)
        response_str = _to_str(iteration.response)
        panel = Panel(
            Text(response_str[:3000], style=STYLE_TEXT),
            title=header,
            title_align="left",
            border_style=COLORS["muted"],
            padding=(0, 1),
        )
        self.console.print(panel)

        for code_block in iteration.code_blocks:
            self._print_code(code_block)

            # Print any sub-calls made during this code block
            for call in code_block.result.rlm_calls:
                self._print_subcall(call)

    def _print_subcall(self, call: Any) -> None:
        if not self.enabled:
            return
        header = Text()
        header.append("  ↳ ", style=STYLE_SECONDARY)
        header.append("Sub-call: ", style=STYLE_SECONDARY)
        header.append(str(call.root_model), style=STYLE_ACCENT)
        if call.execution_time:
            header.append(f"  ({call.execution_time:.2f}s)", style=STYLE_MUTED)

        content = Text()
        prompt_str = str(call.prompt) if call.prompt else ""
        response_str = str(call.response) if call.response else ""
        content.append("Prompt: ", style=STYLE_MUTED)
        content.append(prompt_str[:300] + ("..." if len(prompt_str) > 300 else ""), style=STYLE_TEXT)
        content.append("\nResponse: ", style=STYLE_MUTED)
        content.append(response_str[:500] + ("..." if len(response_str) > 500 else ""), style=STYLE_TEXT)

        panel = Panel(content, title=header, title_align="left", border_style=COLORS["secondary"], padding=(0, 1))
        self.console.print(panel)

    def _print_code(self, code_block: CodeBlock) -> None:
        if not self.enabled:
            return
        result = code_block.result
        header = Text()
        header.append("▸ ", style=STYLE_SUCCESS)
        header.append("Code Execution", style=Style(color=COLORS["success"], bold=True))
        if result.execution_time:
            header.append(f"  ({result.execution_time:.3f}s)", style=STYLE_MUTED)

        parts = []
        code_text = Text()
        code_text.append("Code:\n", style=STYLE_MUTED)
        code_text.append(_to_str(code_block.code), style=STYLE_TEXT)
        parts.append(code_text)

        stdout_str = _to_str(result.stdout) if result.stdout else ""
        if stdout_str.strip():
            out = Text()
            out.append("\nOutput:\n", style=STYLE_MUTED)
            out.append(stdout_str[:2000], style=STYLE_SUCCESS)
            parts.append(out)

        stderr_str = _to_str(result.stderr) if result.stderr else ""
        if stderr_str.strip():
            err = Text()
            err.append("\nError:\n", style=STYLE_MUTED)
            err.append(stderr_str[:500], style=STYLE_ERROR)
            parts.append(err)

        panel = Panel(Group(*parts), title=header, title_align="left", border_style=COLORS["success"], padding=(0, 1))
        self.console.print(panel)

    def print_final_answer(self, answer: Any) -> None:
        if not self.enabled:
            return
        title = Text()
        title.append("★ ", style=STYLE_WARNING)
        title.append("Final Answer", style=Style(color=COLORS["warning"], bold=True))
        panel = Panel(
            Text(_to_str(answer), style=STYLE_TEXT),
            title=title,
            title_align="left",
            border_style=COLORS["warning"],
            padding=(1, 2),
        )
        self.console.print()
        self.console.print(panel)
        self.console.print()

    def print_summary(
        self,
        total_iterations: int,
        total_time: float,
        usage_summary: dict[str, Any] | None = None,
    ) -> None:
        if not self.enabled:
            return
        summary_table = Table(show_header=False, show_edge=False, box=None, padding=(0, 2))
        summary_table.add_column("metric", style=STYLE_MUTED)
        summary_table.add_column("value", style=STYLE_ACCENT)
        summary_table.add_row("Iterations", str(total_iterations))
        summary_table.add_row("Total Time", f"{total_time:.2f}s")
        if usage_summary:
            total_input = sum(
                m.get("total_input_tokens", 0) for m in usage_summary.get("model_usage_summaries", {}).values()
            )
            total_output = sum(
                m.get("total_output_tokens", 0) for m in usage_summary.get("model_usage_summaries", {}).values()
            )
            if total_input or total_output:
                summary_table.add_row("Input Tokens", f"{total_input:,}")
                summary_table.add_row("Output Tokens", f"{total_output:,}")
        self.console.print()
        self.console.print(Rule(style=COLORS["border"], characters="═"))
        self.console.print(summary_table, justify="center")
        self.console.print(Rule(style=COLORS["border"], characters="═"))
        self.console.print()
