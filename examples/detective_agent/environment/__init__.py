"""Environment classes for the RCA and detector engines."""

from typing import Any

from examples.detective_agent.environment.base_env import BaseEnv, SupportsPersistence
from examples.detective_agent.environment.detector_repl import DetectorREPL
from examples.detective_agent.environment.local_repl import LocalREPL
from examples.detective_agent.environment.rca_repl import RcaREPL


def get_environment(environment_type: str, kwargs: dict[str, Any]) -> BaseEnv:
    """
    Factory: create environment instance by type name.

    Supported types:
    - "local"    → LocalREPL
    - "rca"      → RcaREPL (LocalREPL subclass with injected helpers + file manifest)
    - "detector" → DetectorREPL (LocalREPL subclass with OTEL trace helpers)
    """
    if environment_type == "rca":
        return RcaREPL(**kwargs)
    if environment_type == "detector":
        return DetectorREPL(**kwargs)
    if environment_type == "local":
        return LocalREPL(**kwargs)
    raise ValueError(
        f"Unknown environment type: '{environment_type}'. Supported: 'local', 'rca', 'detector'",
    )


__all__ = [
    "BaseEnv",
    "DetectorREPL",
    "LocalREPL",
    "RcaREPL",
    "SupportsPersistence",
    "get_environment",
]
