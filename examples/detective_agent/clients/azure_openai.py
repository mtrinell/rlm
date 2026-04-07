"""Azure OpenAI client. Forked from rlms library."""

import os
from collections import defaultdict
from typing import Any

import openai

from examples.detective_agent.clients.base_lm import BaseLM
from examples.detective_agent.core.types import ModelUsageSummary, UsageSummary


class AzureOpenAIClient(BaseLM):
    """LM Client for Azure OpenAI API."""

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str | None = None,
        azure_endpoint: str | None = None,
        api_version: str | None = None,
        azure_deployment: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_name=model_name, **kwargs)

        if api_key is None:
            api_key = os.getenv("AZURE_OPENAI_API_KEY")
        if azure_endpoint is None:
            azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        if api_version is None:
            api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
        if azure_deployment is None:
            azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")

        if azure_endpoint is None:
            raise ValueError(
                "azure_endpoint is required. Set via argument or AZURE_OPENAI_ENDPOINT env var.",
            )

        self.client = openai.AzureOpenAI(
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            api_version=api_version,
            azure_deployment=azure_deployment,
        )
        self.async_client = openai.AsyncAzureOpenAI(
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            api_version=api_version,
            azure_deployment=azure_deployment,
        )
        self.model_name = model_name
        self.azure_deployment = azure_deployment

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
