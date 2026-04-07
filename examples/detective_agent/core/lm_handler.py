"""
LMHandler - Routes LLM requests from the RLM process and environment subprocesses.

Uses a multi-threaded socket server. Protocol: 4-byte length prefix + JSON payload.
Forked from rlms library.
"""

import asyncio
import time
import types
from socketserver import StreamRequestHandler, ThreadingTCPServer
from threading import Thread

from examples.detective_agent.clients.base_lm import BaseLM
from examples.detective_agent.core.comms_utils import (
    LMRequest,
    LMResponse,
    socket_recv,
    socket_send,
)
from examples.detective_agent.core.types import RLMChatCompletion, UsageSummary


class LMRequestHandler(StreamRequestHandler):
    """Socket handler for LLM completion requests."""

    def handle(self) -> None:
        try:
            request_data = socket_recv(self.connection)
            if not isinstance(request_data, dict):
                self._safe_send(LMResponse.error_response("Request must be a JSON object"))
                return

            request = LMRequest.from_dict(request_data)
            handler: LMHandler = self.server.lm_handler  # type: ignore[attr-defined]

            if request.is_batched:
                response = self._handle_batched(request, handler)
            elif request.prompt:
                response = self._handle_single(request, handler)
            else:
                response = LMResponse.error_response("Missing 'prompt' or 'prompts' in request.")

            self._safe_send(response)

        except (BrokenPipeError, ConnectionError, ConnectionResetError, OSError):
            pass
        except Exception as e:
            self._safe_send(LMResponse.error_response(str(e)))

    def _safe_send(self, response: LMResponse) -> bool:
        try:
            socket_send(self.connection, response.to_dict())
            return True
        except (BrokenPipeError, ConnectionError, ConnectionResetError, OSError):
            return False

    def _handle_single(self, request: LMRequest, handler: "LMHandler") -> LMResponse:
        client = handler.get_client(request.model, request.depth)
        start_time = time.perf_counter()
        content = client.completion(request.prompt)
        end_time = time.perf_counter()

        model_usage = client.get_last_usage()
        root_model = request.model or client.model_name
        usage_summary = UsageSummary(model_usage_summaries={root_model: model_usage})
        return LMResponse.success_response(
            chat_completion=RLMChatCompletion(
                root_model=root_model,
                prompt=request.prompt,
                response=content,
                usage_summary=usage_summary,
                execution_time=end_time - start_time,
            ),
        )

    def _handle_batched(self, request: LMRequest, handler: "LMHandler") -> LMResponse:
        client = handler.get_client(request.model, request.depth)
        start_time = time.perf_counter()

        async def run_all() -> list[str]:
            tasks = [client.acompletion(prompt) for prompt in request.prompts]
            return await asyncio.gather(*tasks)

        results = asyncio.run(run_all())
        end_time = time.perf_counter()
        total_time = end_time - start_time

        model_usage = client.get_last_usage()
        root_model = request.model or client.model_name
        usage_summary = UsageSummary(model_usage_summaries={root_model: model_usage})

        chat_completions = [
            RLMChatCompletion(
                root_model=root_model,
                prompt=prompt,
                response=content,
                usage_summary=usage_summary,
                execution_time=total_time / len(request.prompts),
            )
            for prompt, content in zip(request.prompts, results, strict=True)
        ]
        return LMResponse.batched_success_response(chat_completions=chat_completions)


class ThreadingLMServer(ThreadingTCPServer):
    """Multi-threaded TCP server for LM requests."""

    daemon_threads = True
    allow_reuse_address = True


class LMHandler:
    """
    Handles all LM calls from the main process and environment subprocesses.

    Uses a multi-threaded socket server for concurrent requests.
    """

    def __init__(
        self,
        client: BaseLM,
        host: str = "127.0.0.1",
        port: int = 0,
        other_backend_client: BaseLM | None = None,
    ) -> None:
        self.default_client = client
        self.other_backend_client = other_backend_client
        self.clients: dict[str, BaseLM] = {}
        self.host = host
        self._server: ThreadingLMServer | None = None
        self._thread: Thread | None = None
        self._port = port

        self.register_client(client.model_name, client)

    def register_client(self, model_name: str, client: BaseLM) -> None:
        self.clients[model_name] = client

    def get_client(self, model: str | None = None, depth: int = 0) -> BaseLM:
        """
        Get client by model name or depth routing.

        Routing:
        - depth=0: default_client (orchestrator model)
        - depth=1: other_backend_client if set, else default_client
        - model specified: use named client (overrides depth)
        """
        if model and model in self.clients:
            return self.clients[model]
        if depth == 1 and self.other_backend_client is not None:
            return self.other_backend_client
        return self.default_client

    @property
    def port(self) -> int:
        if self._server:
            return self._server.server_address[1]
        return self._port

    @property
    def address(self) -> tuple[str, int]:
        return (self.host, self.port)

    def start(self) -> tuple[str, int]:
        if self._server is not None:
            return self.address
        self._server = ThreadingLMServer((self.host, self._port), LMRequestHandler)
        self._server.lm_handler = self  # type: ignore[attr-defined]
        self._thread = Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self.address

    def stop(self) -> None:
        if self._server:
            self._server.shutdown()
            self._server = None
            self._thread = None

    def completion(self, prompt: str, model: str | None = None) -> str:
        return self.get_client(model).completion(prompt)

    def __enter__(self) -> "LMHandler":
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> bool:
        self.stop()
        return False

    def get_usage_summary(self) -> UsageSummary:
        merged = {}
        merged.update(self.default_client.get_usage_summary().model_usage_summaries)
        if self.other_backend_client is not None:
            merged.update(self.other_backend_client.get_usage_summary().model_usage_summaries)
        for client in self.clients.values():
            merged.update(client.get_usage_summary().model_usage_summaries)
        return UsageSummary(model_usage_summaries=merged)
