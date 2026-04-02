"""Prompts package — RCA and behavioral deviation detection."""

from examples.detective_agent.prompts.detector_system_prompt import (
    DETECTOR_SYSTEM_PROMPT,
    build_detector_task_prompt,
    get_detector_system_prompt,
)
from examples.detective_agent.prompts.system_prompt import (
    RCA_SYSTEM_PROMPT,
    build_rca_task_prompt,
    get_rca_system_prompt,
)

__all__ = [
    "DETECTOR_SYSTEM_PROMPT",
    "RCA_SYSTEM_PROMPT",
    "build_detector_task_prompt",
    "build_rca_task_prompt",
    "get_detector_system_prompt",
    "get_rca_system_prompt",
]
