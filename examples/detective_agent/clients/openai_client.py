"""OpenAI client. Forked from rlms library."""

import os
from collections import defaultdict
from typing import Any

import openai

from examples.detective_agent.clients.base_lm import BaseLM
from examples.detective_agent.core.types import ModelUsageSummary, UsageSummary


class OpenAIClient(BaseLM):
    """LM Client for OpenAI API (also works with vLLM and OpenRouter)."""

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str | None = None,
        base_url: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_name=model_name, **kwargs)

        if api_key is None:
            api_key = os.getenv("LLM_API_KEY")
        if base_url is None:
            base_url = os.getenv("LLM_BASE_URL")
        if model_name is None:
            model_name = os.getenv("LLM_MODEL")

        self.client = openai.OpenAI(api_key=api_key, base_url=base_url)
        self.async_client = openai.AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name

        self.model_call_counts: dict[str, int] = defaultdict(int)
        self.model_input_tokens: dict[str, int] = defaultdict(int)
        self.model_output_tokens: dict[str, int] = defaultdict(int)
        self.model_total_tokens: dict[str, int] = defaultdict(int)
        self.last_prompt_tokens: int = 0
        self.last_completion_tokens: int = 0

    def completion(self, prompt: str | list[dict[str, Any]], model: str | None = None) -> str:
        messages = self._to_messages(prompt)
        model = model or self.model_name
        if not model:
            raise ValueError("Model name is required.")
        response = self.client.chat.completions.create(model=model, messages=messages)
        self._track_cost(response, model)
        return response.choices[0].message.content

    async def acompletion(
        self,
        prompt: str | list[dict[str, Any]],
        model: str | None = None,
    ) -> str:
        messages = self._to_messages(prompt)
        model = model or self.model_name
        if not model:
            raise ValueError("Model name is required.")
        response = await self.async_client.chat.completions.create(model=model, messages=messages)
        self._track_cost(response, model)
        return response.choices[0].message.content

    def _to_messages(self, prompt: str | list[dict[str, Any]]) -> list[dict[str, Any]]:
        if isinstance(prompt, str):
            return [{"role": "user", "content": prompt}]
        return prompt

    def _track_cost(self, response: Any, model: str) -> None:
        self.model_call_counts[model] += 1
        usage = getattr(response, "usage", None)
        if usage is None:
            raise ValueError("No usage data received.")
        self.model_input_tokens[model] += usage.prompt_tokens
        self.model_output_tokens[model] += usage.completion_tokens
        self.model_total_tokens[model] += usage.total_tokens
        self.last_prompt_tokens = usage.prompt_tokens
        self.last_completion_tokens = usage.completion_tokens

    def get_usage_summary(self) -> UsageSummary:
        return UsageSummary(
            model_usage_summaries={
                model: ModelUsageSummary(
                    total_calls=self.model_call_counts[model],
                    total_input_tokens=self.model_input_tokens[model],
                    total_output_tokens=self.model_output_tokens[model],
                )
                for model in self.model_call_counts
            },
        )

    def get_last_usage(self) -> ModelUsageSummary:
        return ModelUsageSummary(
            total_calls=1,
            total_input_tokens=self.last_prompt_tokens,
            total_output_tokens=self.last_completion_tokens,
        )
