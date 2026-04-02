"""
Progressive history truncation for the RCA investigation loop.

Applies 4-tier truncation to the message_history list when context budget
pressure increases. Preserves critical messages (system prompt, initial task,
recent iterations) while compressing old tool outputs.

Truncation policy mirrors the prep agent's ContextManager, adapted for
the RLM message format where REPL outputs appear as "user" messages.

Tier | Trigger  | Strategy
-----|----------|------------------------------------------
none | < 50%    | No changes
light| 50-70%   | Deduplicate consecutive identical lines
medium| 70-85%  | Drop low-signal INFO/DEBUG from old user REPL messages
     |          | (preserving lines matching state-change keywords)
aggressive|>85% | Keep only ERROR/WARN lines + first/last 10
                 | from old REPL messages; cap each at 3000 chars
"""

from __future__ import annotations

import re
from typing import Any

from examples.detective_agent.context.budget import ContextBudget, TruncationLevel

# Lines matching these patterns are ALWAYS kept regardless of truncation level.
# Reasoning: root causes in networking infrastructure are frequently logged at
# INFO level — "interface state change", "BGP peer deletion", "commit by user",
# "load patch" — and early truncation would destroy exactly this evidence.
_KEEP_PATTERNS = re.compile(
    r"\b(ERROR|WARN|WARNING|CRITICAL|FATAL|EXCEPTION|PANIC"
    r"|down|up|change|changed|shutdown|disabled|reset|restart|crash"
    r"|delete|deleted|remove|removed|commit|load patch"
    r"|PeerConfig|bgp|bfd|ospf|isis|mpls|lldp"
    r"|interface|link|port|cable|sfp|fiber|optic"
    r"|memory|cpu|disk|resource|exhaustion|leak"
    r"|cert|certificate|expir|tls|ssl"
    r"|authentication|auth|login|denied"
    r"|config|configuration|rollback"
    r"|traceback|stack trace|assert)\b",
    re.IGNORECASE,
)

# Lines that are low-signal when truncating
_LOW_SIGNAL_PATTERNS = re.compile(
    r"^(?:"
    r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}[^:]*:\s*(?:DEBUG|INFO|TRACE|VERBOSE)"
    r"|\s*#.*"  # comments
    r"|REPL variables: \[.*\]"  # variable listings (verbose)
    r")\b",
    re.IGNORECASE,
)

# Markers that indicate a message contains REPL execution output
_REPL_OUTPUT_MARKER = "REPL output:"

# Protected message indices:
# - index 0: system prompt
# - index 1: initial assistant metadata acknowledgement
_PROTECTED_LEADING = 2

# Also protect the last N messages (most recent iterations)
_PROTECTED_TRAILING = 4

# Per-message character cap under aggressive truncation
_AGGRESSIVE_CHAR_CAP = 3_000


class HistoryManager:
    """
    Applies progressive truncation to the RLM message_history list.

    Usage:
        manager = HistoryManager()
        message_history = manager.apply(message_history, budget)
    """

    def apply(
        self,
        message_history: list[dict[str, Any]],
        budget: ContextBudget,
    ) -> list[dict[str, Any]]:
        """
        Apply truncation at the appropriate level and return the modified list.

        Args:
            message_history: Current full message history list (modified in place).
            budget: Current context budget to determine truncation level.

        Returns:
            The (possibly modified) message_history list.

        """
        level = budget.level
        if level == "none":
            return message_history

        # Identify the range of messages we're allowed to modify
        n = len(message_history)
        start = _PROTECTED_LEADING
        end = max(start, n - _PROTECTED_TRAILING)

        if start >= end:
            return message_history  # not enough messages to truncate safely

        for i in range(start, end):
            msg = message_history[i]
            if msg.get("role") != "user":
                continue
            content = msg.get("content", "")
            if _REPL_OUTPUT_MARKER not in content:
                continue

            # Split at REPL output boundary — preamble (code) is preserved as-is
            parts = content.split(_REPL_OUTPUT_MARKER, 1)
            preamble = parts[0] + _REPL_OUTPUT_MARKER
            repl_output = parts[1] if len(parts) > 1 else ""

            compressed = self._compress(repl_output, level)
            message_history[i] = {
                **msg,
                "content": preamble + compressed,
            }

        return message_history

    def _compress(self, text: str, level: TruncationLevel) -> str:
        """Apply text compression at the given level."""
        if level == "light":
            return self._deduplicate_lines(text)

        lines = text.splitlines()

        if level == "medium":
            filtered = self._drop_low_signal(lines)
            return "\n".join(filtered)

        if level == "aggressive":
            filtered = self._keep_critical_only(lines)
            result = "\n".join(filtered)
            if len(result) > _AGGRESSIVE_CHAR_CAP:
                result = (
                    result[:_AGGRESSIVE_CHAR_CAP]
                    + f"\n[{len(result) - _AGGRESSIVE_CHAR_CAP} chars truncated by context manager]"
                )
            return result

        return text

    def _deduplicate_lines(self, text: str) -> str:
        """Remove consecutive duplicate lines (keeps first occurrence)."""
        lines = text.splitlines()
        if not lines:
            return text

        deduped = [lines[0]]
        run_count = 1
        for line in lines[1:]:
            if line.strip() == deduped[-1].strip() and line.strip():
                run_count += 1
                if run_count == 2:
                    deduped.append("  [line repeated — omitting duplicates]")
            else:
                run_count = 1
                deduped.append(line)
        return "\n".join(deduped)

    def _drop_low_signal(self, lines: list[str]) -> list[str]:
        """
        Drop low-signal INFO/DEBUG lines that don't match keep patterns.

        Always keeps:
        - First 20 lines (preamble context)
        - Last 20 lines (most recent state)
        - Any line matching _KEEP_PATTERNS
        """
        n = len(lines)
        if n <= 40:
            return lines

        head = lines[:20]
        tail = lines[-20:]
        middle = lines[20 : n - 20]

        filtered_middle = []
        for line in middle:
            if _KEEP_PATTERNS.search(line):
                filtered_middle.append(line)
            elif _LOW_SIGNAL_PATTERNS.search(line):
                pass  # drop it
            else:
                filtered_middle.append(line)

        if len(filtered_middle) < len(middle):
            dropped = len(middle) - len(filtered_middle)
            filtered_middle.insert(0, f"  [{dropped} low-signal DEBUG/INFO lines dropped]")

        return head + filtered_middle + tail

    def _keep_critical_only(self, lines: list[str]) -> list[str]:
        """
        Aggressive: keep first 10 + last 10 + all lines matching critical patterns.

        Everything else is dropped.
        """
        n = len(lines)
        if n <= 20:
            return lines

        head = lines[:10]
        tail = lines[-10:]
        critical = [line for line in lines[10 : n - 10] if _KEEP_PATTERNS.search(line)]

        parts = head[:]
        if critical:
            parts.append(f"  [kept {len(critical)} critical lines from {n - 20} middle lines]")
            parts.extend(critical)
        else:
            dropped = n - 20
            parts.append(f"  [{dropped} middle lines dropped — no critical patterns found]")
        parts.extend(tail)
        return parts
