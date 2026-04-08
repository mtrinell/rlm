"""Environment classes for the detector and RCA engines."""

from typing import Any

from examples.detective_agent.environment.detector_repl import DetectorREPL
from examples.detective_agent.environment.base_env import BaseEnv, SupportsPersistence
from examples.detective_agent.environment.local_repl import LocalREPL
from examples.detective_agent.environment.rca_repl import RcaREPL


def get_environment(environment_type: str, kwargs: dict[str, Any]) -> BaseEnv:
    """
    Factory: create environment instance by type name.

    Supported types:
    - "local"    → LocalREPL
    - "detector" → DetectorREPL (generic file-access helpers; any dataset format)
    - "rca"      → RcaREPL (LocalREPL subclass with injected helpers + file manifest)
    """
    if environment_type == "detector":
        return DetectorREPL(**kwargs)
    if environment_type == "rca":
        return RcaREPL(**kwargs)
    if environment_type == "local":
        return LocalREPL(**kwargs)
    raise ValueError(
        f"Unknown environment type: '{environment_type}'. Supported: 'local', 'detector', 'rca'",
    )


__all__ = [
    "DetectorREPL",
    "BaseEnv",
    "LocalREPL",
    "RcaREPL",
    "SupportsPersistence",
    "get_environment",
]
