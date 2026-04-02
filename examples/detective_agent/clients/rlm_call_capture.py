"""
RLM Call Capture wrapper for recording and replaying LLM calls.

This module provides a wrapper around RLM's BaseLM clients that:
1. Adds retry logic with exponential backoff for transient errors
2. Records LLM calls for replay (deterministic testing)
3. Replays stored responses instead of making actual API calls

The wrapper integrates with the existing Replayer infrastructure used by
LangChain agents, allowing both systems to share replay files.

"""

import json
import logging
import time
from typing import Any

try:
    from src.modules.llm_wrapper import deserialize_response_object, serialize_response_object
    from src.modules.replayer import Replayer
    from src.modules.retry_utils import retry_with_backoff
except ImportError:
    deserialize_response_object = None  # type: ignore[assignment]
    serialize_response_object = None  # type: ignore[assignment]
    Replayer = None  # type: ignore[assignment,misc]
    retry_with_backoff = None  # type: ignore[assignment]
from examples.detective_agent.clients.base_lm import BaseLM
from examples.detective_agent.core.types import ModelUsageSummary, UsageSummary


class RLMCallCapture(BaseLM):
    """
    Wrapper for RLM LLM clients with retry, recording, and replay capabilities.

    This wrapper intercepts all LLM calls and:
    - Applies retry logic with exponential backoff for transient errors
    - Records calls to Replayer for deterministic replay
    - Returns stored responses during replay mode

    Attributes:
        wrapped_client: The underlying BaseLM client to wrap.
        replayer: Replayer instance for recording/replay (optional).
        logger: Logger for debug and retry messages.
        operation_name: Name used in retry log messages.

    """

    def __init__(
        self,
        wrapped_client: BaseLM,
        replayer: Replayer | None = None,
        logger: logging.Logger | None = None,
        operation_name: str = "RLM completion",
    ) -> None:
        """
        Initialize RLM call capture wrapper.

        Args:
            wrapped_client: The BaseLM client to wrap.
            replayer: Replayer for recording/replay. If None, no recording/replay.
            logger: Logger instance. If None, uses module logger.
            operation_name: Name for retry log messages.

        """
        # Initialize base class with wrapped client's model_name
        super().__init__(model_name=wrapped_client.model_name or "unknown")
        self.wrapped_client = wrapped_client
        self.replayer = replayer
        self.logger = logger or logging.getLogger(__name__)
        self.operation_name = operation_name
        self._call_counter = 0

    def completion(
        self,
        prompt: str | list[dict[str, Any]],
        model: str | None = None,
    ) -> str:
        """
        Execute LLM completion with retry logic and optional recording/replay.

        Args:
            prompt: The prompt to send to the LLM.
            model: Optional model override.

        Returns:
            str: The LLM response text.

        """
        # Convert prompt to string for lookup key
        input_key = self._prompt_to_key(prompt)

        # Check for replay mode
        if self.replayer and not self.replayer.record:
            return self._get_replay_response(input_key)

        # Execute with retry logic
        response = self._completion_with_retry(prompt, model)

        # Record if in record mode
        if self.replayer and self.replayer.record:
            self._record_call(input_key, response, model)

        return response

    @retry_with_backoff(operation_name="RLM completion")
    def _completion_with_retry(
        self,
        prompt: str | list[dict[str, Any]],
        model: str | None = None,
    ) -> str:
        """
        Execute completion with retry decorator applied.

        Args:
            prompt: The prompt to send.
            model: Optional model override.

        Returns:
            str: The LLM response.

        """
        return self.wrapped_client.completion(prompt, model)

    async def acompletion(
        self,
        prompt: str | list[dict[str, Any]],
        model: str | None = None,
    ) -> str:
        """
        Async completion (delegates to wrapped client without replay support).

        Note: Async replay is not currently supported. This method will always
        make actual API calls even in replay mode.

        Args:
            prompt: The prompt to send.
            model: Optional model override.

        Returns:
            str: The LLM response.

        """
        # Async replay not supported - just delegate
        self.logger.warning("Async completion called - replay not supported for async")
        return await self.wrapped_client.acompletion(prompt, model)

    def get_usage_summary(self) -> UsageSummary:
        """
        Get usage summary from wrapped client.

        Returns:
            UsageSummary: Aggregated usage statistics.

        """
        return self.wrapped_client.get_usage_summary()

    def get_last_usage(self) -> ModelUsageSummary:
        """
        Get last call usage from wrapped client.

        Returns:
            ModelUsageSummary: Usage statistics for the last call.

        """
        return self.wrapped_client.get_last_usage()

    def _prompt_to_key(
        self,
        prompt: str | list[dict[str, Any]],
    ) -> str:
        """
        Convert prompt to a string key for replay lookup.

        Args:
            prompt: The prompt (string or message list).

        Returns:
            str: JSON string key for lookup.

        """
        if isinstance(prompt, str):
            return json.dumps({"text": prompt})
        return json.dumps(prompt)

    def _get_replay_response(
        self,
        input_key: str,
    ) -> str:
        """
        Get stored response from replayer.

        Args:
            input_key: The input key to look up.

        Returns:
            str: The stored response.

        Raises:
            ValueError: If no stored response is found.

        """
        if not self.replayer:
            raise ValueError("Replayer not configured")

        call_record = self.replayer.get_stored_llm_call_record(input_key)
        if call_record is None:
            raise ValueError(f"No stored response found for RLM input: {input_key[:100]}...")

        if "output_data" not in call_record:
            raise ValueError("Stored record missing output_data field")

        # Deserialize the response (handles raw_string type)
        response = deserialize_response_object(call_record["output_data"])

        if not isinstance(response, str):
            raise TypeError(f"Expected string response from replay, got {type(response).__name__}")

        self.logger.debug(f"Replayed RLM response: {response[:100]}...")
        return response

    def _record_call(
        self,
        input_key: str,
        response: str,
        model: str | None,
    ) -> None:
        """
        Record an LLM call to the replayer.

        Args:
            input_key: The serialized input key.
            response: The LLM response string.
            model: The model used (optional).

        """
        if not self.replayer:
            return

        self._call_counter += 1

        # Serialize the response using the extended serialization
        output_serialized = serialize_response_object(response)

        # Build capture entry in the same format as LLMCallCapture
        capture_entry = {
            "input_data": {
                "input": input_key,
                "kwargs": json.dumps({"model": model}),
            },
            "output_data": output_serialized,
            "call_info": {
                "call_id": self._call_counter,
                "call_type": "rlm_completion",
                "timestamp": time.time(),
                "model": model or self.wrapped_client.model_name or "unknown",
                "client_type": "RLMCallCapture",
            },
        }

        self.replayer._add_capture_entry(capture_entry)  # noqa: SLF001
        self.logger.debug(f"Recorded RLM call #{self._call_counter}")


def create_rlm_client_with_capture(
    client_class: type[BaseLM],
    replayer: Replayer | None = None,
    logger: logging.Logger | None = None,
    **client_kwargs: Any,
) -> RLMCallCapture:
    """
    Factory function to create an RLM client wrapped with capture.

    This is a convenience function that creates the underlying client
    and wraps it with RLMCallCapture in one step.

    Args:
        client_class: The BaseLM client class to instantiate.
        replayer: Replayer for recording/replay (optional).
        logger: Logger instance (optional).
        **client_kwargs: Arguments to pass to the client constructor.

    Returns:
        RLMCallCapture: Wrapped client with retry and recording capabilities.

    Example:
        client = create_rlm_client_with_capture(
            AzureOpenAIClient,
            replayer=replayer,
            logger=logger,
            api_key="...",
            azure_endpoint="...",
            model_name="gpt-4o",
        )

    """
    # Create the underlying client
    underlying_client = client_class(**client_kwargs)

    # Wrap with capture
    return RLMCallCapture(
        wrapped_client=underlying_client,
        replayer=replayer,
        logger=logger,
    )
