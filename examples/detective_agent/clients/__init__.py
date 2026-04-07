"""Client adapters for language model backends. Forked from rlms library."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Literal

from examples.detective_agent.clients.base_lm import BaseLM

if TYPE_CHECKING:
    from examples.detective_agent.clients.rlm_call_capture import RLMCallCapture

# Define ClientBackend locally to avoid circular import with src.rlm.core.types
ClientBackend = Literal["azure_openai", "openai", "vllm", "openrouter", "vercel", "anthropic"]

__all__ = ["BaseLM", "get_client"]


def get_client(
    backend: ClientBackend,
    backend_kwargs: dict[str, Any] | None = None,
    replayer: Replayer | None = None,
    logger: logging.Logger | None = None,
) -> BaseLM:
    """
    Route a backend identifier + kwargs to the appropriate client implementation.

    The client is always wrapped with RLMCallCapture for retry logic.
    If replayer is provided, calls will also be recorded/replayed.

    Args:
        backend: Backend identifier (azure_openai, openai, vllm, etc.).
        backend_kwargs: Keyword arguments for the backend client.
        replayer: Optional Replayer for recording/replay mode.
        logger: Optional logger for retry and debug messages.

    Returns:
        BaseLM: The client wrapped with RLMCallCapture for retry support.

    Raises:
        ValueError: If backend is not recognized.

    """
    kwargs = backend_kwargs or {}
    client: BaseLM

    if backend == "azure_openai":
        from examples.detective_agent.clients.azure_openai import AzureOpenAIClient  # noqa: PLC0415

        client = AzureOpenAIClient(**kwargs)

    elif backend in ("openai", "vllm", "openrouter", "vercel"):
        from examples.detective_agent.clients.openai_client import OpenAIClient  # noqa: PLC0415

        if backend == "openrouter":
            kwargs.setdefault("base_url", "https://openrouter.ai/api/v1")
        elif backend == "vercel":
            kwargs.setdefault("base_url", "https://ai-gateway.vercel.sh/v1")
        client = OpenAIClient(**kwargs)

    elif backend == "anthropic":
        from examples.detective_agent.clients.anthropic_client import (
            AnthropicClient,  # noqa: PLC0415
        )

        client = AnthropicClient(**kwargs)

    else:
        raise ValueError(
            f"Unknown backend: {backend}. "
            "Supported: ['azure_openai', 'openai', 'vllm', 'openrouter', 'vercel', 'anthropic']",
        )

    return client
