"""
Context budget tracker for the RCA investigation loop.

Tracks cumulative token usage across the conversation and REPL outputs.
Provides a 4-level budget classification used to trigger progressive
history truncation:

  none       < 50% used   no action
  light      50-70%       deduplicate repetitive lines
  medium     70-85%       drop low-signal INFO/DEBUG from old messages
  aggressive > 85%        keep only errors/warnings + first/last lines
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

TruncationLevel = Literal["none", "light", "medium", "aggressive"]

# Thresholds match the prep agent's ContextManager (validated on real RCA data)
_THRESHOLDS: dict[TruncationLevel, float] = {
    "none": 0.50,
    "light": 0.70,
    "medium": 0.85,
    "aggressive": 1.00,
}

# Heuristic: 4 bytes ≈ 1 token (works for ASCII-heavy log content)
_BYTES_PER_TOKEN: int = 4


@dataclass
class ContextBudget:
    """
    Tracks token usage across the full conversation.

    Usage:
        budget = ContextBudget(total_token_limit=100_000)
        budget.record_message("system", system_prompt)
        budget.record_message("user", user_message)
        budget.record_repl_output(stdout)

        if budget.level != "none":
            history_manager.apply(message_history, budget)
    """

    total_token_limit: int = 100_000
    _used_tokens: int = field(default=0, init=False, repr=False)
    _message_count: int = field(default=0, init=False, repr=False)

    def record_message(self, role: str, content: str) -> None:  # noqa: ARG002
        """Record a new message added to history (system, user, or assistant)."""
        if content:
            self._used_tokens += len(content) // _BYTES_PER_TOKEN
            self._message_count += 1

    def record_messages(self, messages: list[dict[str, Any]]) -> None:
        """Record a batch of messages."""
        for msg in messages:
            self.record_message(msg.get("role", ""), msg.get("content", "") or "")

    def record_repl_output(self, stdout: str) -> None:
        """Record REPL stdout as token usage (same estimator)."""
        if stdout:
            self._used_tokens += len(stdout) // _BYTES_PER_TOKEN

    def reset(self) -> None:
        self._used_tokens = 0
        self._message_count = 0

    @property
    def used_tokens(self) -> int:
        return self._used_tokens

    @property
    def used_fraction(self) -> float:
        """Fraction of total budget used (0.0-1.0+)."""
        if self.total_token_limit == 0:
            return 0.0
        return min(self._used_tokens / self.total_token_limit, 1.0)

    @property
    def level(self) -> TruncationLevel:
        """Current truncation level based on fraction used."""
        frac = self.used_fraction
        if frac < _THRESHOLDS["none"]:
            return "none"
        if frac < _THRESHOLDS["light"]:
            return "light"
        if frac < _THRESHOLDS["medium"]:
            return "medium"
        return "aggressive"

    @property
    def tokens_remaining(self) -> int:
        return max(0, self.total_token_limit - self._used_tokens)

    def snapshot(self) -> dict[str, Any]:
        """Return a serializable snapshot for the context_budget() REPL helper."""
        return {
            "used_tokens": self._used_tokens,
            "total_tokens": self.total_token_limit,
            "tokens_remaining": self.tokens_remaining,
            "percent_used": round(self.used_fraction * 100, 1),
            "level": self.level,
            "message_count": self._message_count,
        }

    def __repr__(self) -> str:
        return (
            f"ContextBudget(used={self._used_tokens}/{self.total_token_limit} tokens, "
            f"{self.used_fraction * 100:.1f}%, level={self.level})"
        )
