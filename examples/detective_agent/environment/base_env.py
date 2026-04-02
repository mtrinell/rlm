"""Base environment interface. Forked from rlms library."""

from abc import ABC, abstractmethod
from typing import Any, Never, Protocol, runtime_checkable

from examples.detective_agent.core.types import REPLResult


class BaseEnv(ABC):
    """Abstract base REPL-like environment."""

    def __init__(self, persistent: bool = False, depth: int = 1, **kwargs: Any) -> None:
        self.persistent = persistent
        self.depth = depth
        self.kwargs = kwargs

    @abstractmethod
    def setup(self) -> Never:
        raise NotImplementedError

    @abstractmethod
    def load_context(self, context_payload: dict | list | str) -> Never:
        raise NotImplementedError

    @abstractmethod
    def execute_code(self, code: str) -> REPLResult:
        raise NotImplementedError


class IsolatedEnv(BaseEnv, ABC):
    """Environments that sit on a separate machine from the LM."""

    def __init__(self, persistent: bool = False, **kwargs: Any) -> None:
        super().__init__(persistent=persistent, **kwargs)

    @abstractmethod
    def setup(self) -> None: ...

    @abstractmethod
    def load_context(self, context_payload: dict | list | str) -> None: ...

    @abstractmethod
    def execute_code(self, code: str) -> REPLResult: ...


class NonIsolatedEnv(BaseEnv, ABC):
    """Environments running on the same machine as the LM."""

    def __init__(self, persistent: bool = False, **kwargs: Any) -> None:
        super().__init__(persistent=persistent, **kwargs)

    @abstractmethod
    def setup(self) -> None: ...

    @abstractmethod
    def load_context(self, context_payload: dict | list | str) -> None: ...

    @abstractmethod
    def execute_code(self, code: str) -> REPLResult: ...


@runtime_checkable
class SupportsPersistence(Protocol):
    """Protocol for environments that support persistent multi-turn sessions."""

    def update_handler_address(self, address: tuple[str, int]) -> None: ...

    def add_context(
        self,
        context_payload: dict | list | str,
        context_index: int | None = None,
    ) -> int: ...

    def get_context_count(self) -> int: ...

    def add_history(
        self,
        message_history: list[dict[str, Any]],
        history_index: int | None = None,
    ) -> int: ...

    def get_history_count(self) -> int: ...
