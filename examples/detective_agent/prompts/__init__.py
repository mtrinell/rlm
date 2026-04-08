"""Prompts package — behavioral deviation detection."""

from examples.detective_agent.prompts.detector_system_prompt import (
    DETECTOR_SYSTEM_PROMPT,
    build_detector_task_prompt,
    get_detector_system_prompt,
)

__all__ = [
    "DETECTOR_SYSTEM_PROMPT",
    "build_detector_task_prompt",
    "get_detector_system_prompt",
]
