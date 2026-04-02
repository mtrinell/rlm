"""
Modified fork of rlm/core/rlm.py.

Key additions over the upstream version:
1. Accepts optional ContextBudget + HistoryManager for progressive context management.
2. Records token usage into the budget after each LLM response.
3. Calls history_manager.apply() after each format_iteration() to compress old REPL outputs.
4. Phase injection: inserts a concise budget-pressure reminder at medium/aggressive levels
   and a plan reminder once the REPL local variable "plan" is set.
5. Single-model focus: other_backends is kept for API compatibility but not required.

All imports use relative paths — no dependency on the rlm package.
"""

import logging
import time
import types
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from examples.detective_agent.clients import BaseLM, get_client
from examples.detective_agent.context.budget import ContextBudget
from examples.detective_agent.context.history_manager import HistoryManager
from examples.detective_agent.environment.base_env import BaseEnv, SupportsPersistence
from examples.detective_agent.logger import RLMLogger, VerbosePrinter

from examples.detective_agent.core.base_prompts import (
    RLM_BASE_SYSTEM_PROMPT,
    QueryMetadata,
    build_rlm_system_prompt,
    build_user_prompt,
)
from examples.detective_agent.core.lm_handler import LMHandler
from examples.detective_agent.core.parsing import (
    find_code_blocks,
    find_final_answer,
    format_iteration,
)
from examples.detective_agent.core.rlm_utils import filter_sensitive_keys
from examples.detective_agent.core.types import (
    ClientBackend,
    CodeBlock,
    EnvironmentType,
    REPLResult,
    RLMChatCompletion,
    RLMIteration,
    RLMMetadata,
)

# ─── Phase injection messages ──────────────────────────────────────────────────
# Injected into message_history when budget pressure reaches medium/aggressive
# to remind the model to converge to a conclusion.

_PHASE_MEDIUM_REMINDER = (
    "Context limit reminder (medium pressure): "
    "You have used approximately 70-85% of the available context. "
    "Prioritise completing your current investigation phase and moving toward a "
    "root-cause conclusion. Avoid reading large new files unless strictly necessary. "
    "Use summarize_file() for any new files. Consolidate findings into the 'findings' "
    "variable so they are preserved if earlier messages are truncated."
)

_PHASE_AGGRESSIVE_REMINDER = (
    "Context budget note: you have used over 85% of the available context. "
    "Please wrap up your investigation and write your root-cause analysis "
    "using FINAL_VAR. If you need a one-line summary of anything, use llm_query(). "
    "Avoid reading additional files at this stage."
)

_PLAN_REMINDER_TEMPLATE = "Reminder: your investigation plan is:\n{plan}\nFocus on completing the remaining phases."


def _get_repl_local(environment: BaseEnv, name: str) -> Any:
    """Safely extract a local variable from REPL environment."""
    try:
        # LocalREPL stores user-defined variables in self.locals
        locals_dict = getattr(environment, "locals", None)
        if isinstance(locals_dict, dict):
            return locals_dict.get(name)
        # Fallback: try globals dict
        globals_dict = getattr(environment, "globals", None)
        if isinstance(globals_dict, dict):
            return globals_dict.get(name)
    except Exception:  # noqa: S110
        pass
    return None


class RLM:
    """
    Recursive Language Model for root-cause analysis.

    Forked from rlm/core/rlm.py with context management extensions.
    """

    def __init__(
        self,
        backend: ClientBackend = "azure_openai",
        backend_kwargs: dict[str, Any] | None = None,
        environment: EnvironmentType = "local",
        environment_kwargs: dict[str, Any] | None = None,
        depth: int = 0,
        max_depth: int = 1,
        max_iterations: int = 30,
        custom_system_prompt: str | None = None,
        other_backends: list[ClientBackend] | None = None,
        other_backend_kwargs: list[dict[str, Any]] | None = None,
        logger: RLMLogger | None = None,
        verbose: bool = False,
        persistent: bool = False,
        # ── Context management (new) ───────────────────────────────────────
        budget: ContextBudget | None = None,
        history_manager: HistoryManager | None = None,
        phase_injection_enabled: bool = True,
        # ── Replay support (new) ───────────────────────────────────────────
        replayer: "Replayer | None" = None,
        replay_logger: logging.Logger | None = None,
    ) -> None:
        """
        Initialize the RLM.

        Args:
            backend: LLM backend identifier.
            backend_kwargs: Keyword arguments for the backend client.
            environment: Environment type ("local" or "docker").
            environment_kwargs: Kwargs for the environment.
            depth: Current depth index (0 = root).
            max_depth: Maximum recursion depth (1 = single model).
            max_iterations: Maximum iterations before forced conclusion.
            custom_system_prompt: Override the system prompt.
            other_backends: Optional additional backends for sub-calls (single-model: leave None).
            other_backend_kwargs: Kwargs for other_backends.
            logger: RLMLogger instance for JSONL logging.
            verbose: Enable rich console output.
            persistent: Reuse environment across completion() calls.
            budget: ContextBudget for tracking token usage and truncation level.
            history_manager: HistoryManager for progressive history compression.
            phase_injection_enabled: Whether to inject phase reminders into history.
            replayer: Optional Replayer for recording/replaying LLM calls.
            replay_logger: Optional logger for replay/retry messages.

        """
        self.backend = backend
        self.backend_kwargs = backend_kwargs
        self.replayer = replayer
        self.replay_logger = replay_logger
        self.environment_type = environment
        self.environment_kwargs = environment_kwargs.copy() if environment_kwargs is not None else {}

        if other_backends is not None and len(other_backends) != 1:
            raise ValueError(
                "We currently only support one additional backend for recursive sub-calls.",
            )

        self.other_backends = other_backends
        self.other_backend_kwargs = other_backend_kwargs

        self.depth = depth
        self.max_depth = max_depth
        self.max_iterations = max_iterations
        self.system_prompt = custom_system_prompt if custom_system_prompt else RLM_BASE_SYSTEM_PROMPT
        self.logger = logger
        self.verbose = VerbosePrinter(enabled=verbose)

        # Context management
        self.budget = budget
        self.history_manager = history_manager
        self.phase_injection_enabled = phase_injection_enabled

        # Track which phase reminders have already been injected (avoid repeats)
        self._injected_medium_reminder = False
        self._injected_aggressive_reminder = False
        self._injected_plan_reminder = False

        # Persistence support
        self.persistent = persistent
        self._persistent_env: SupportsPersistence | None = None

        if self.persistent:
            self._validate_persistent_environment_support()

        if self.logger or verbose:
            metadata = RLMMetadata(
                root_model=(backend_kwargs.get("model_name", "unknown") if backend_kwargs else "unknown"),
                max_depth=max_depth,
                max_iterations=max_iterations,
                backend=backend,
                backend_kwargs=filter_sensitive_keys(backend_kwargs) if backend_kwargs else {},
                environment_type=environment,
                environment_kwargs=filter_sensitive_keys(environment_kwargs) if environment_kwargs else {},
                other_backends=other_backends,
            )
            if self.logger and hasattr(self.logger, "log_metadata"):
                self.logger.log_metadata(metadata)
            self.verbose.print_metadata(metadata)

    @contextmanager
    def _spawn_completion_context(self, prompt: str | dict[str, Any]) -> Any:
        """Spawn an LM handler and environment for a single completion call."""
        # Create main client with optional replay/retry support
        client: BaseLM = get_client(
            self.backend,
            self.backend_kwargs,
            replayer=self.replayer,
            logger=self.replay_logger,
        )

        other_backend_client: BaseLM | None = None
        if self.other_backends and self.other_backend_kwargs:
            other_backend_client = get_client(
                self.other_backends[0],
                self.other_backend_kwargs[0],
                replayer=self.replayer,
                logger=self.replay_logger,
            )

        lm_handler = LMHandler(client, other_backend_client=other_backend_client)

        if self.other_backends and self.other_backend_kwargs:
            for backend, kwargs in zip(self.other_backends, self.other_backend_kwargs, strict=True):
                other_client: BaseLM = get_client(
                    backend,
                    kwargs,
                    replayer=self.replayer,
                    logger=self.replay_logger,
                )
                lm_handler.register_client(other_client.model_name, other_client)

        lm_handler.start()

        if self.persistent and self._persistent_env is not None:
            environment = self._persistent_env
            if not self._env_supports_persistence(environment):
                raise RuntimeError(
                    f"Persistent environment of type '{type(environment).__name__}' does not "
                    f"implement required methods.",
                )
            environment.update_handler_address((lm_handler.host, lm_handler.port))
            environment.add_context(prompt)
        else:
            env_kwargs = self.environment_kwargs.copy()
            env_kwargs["lm_handler_address"] = (lm_handler.host, lm_handler.port)
            env_kwargs["context_payload"] = prompt
            env_kwargs["depth"] = self.depth + 1

            from examples.detective_agent.environment import get_environment  # noqa: PLC0415

            environment: BaseEnv = get_environment(self.environment_type, env_kwargs)

            if self.persistent:
                self._persistent_env = environment

        try:
            yield lm_handler, environment
        finally:
            lm_handler.stop()
            if not self.persistent and hasattr(environment, "cleanup"):
                environment.cleanup()

    def _setup_prompt(self, prompt: str | dict[str, Any]) -> list[dict[str, Any]]:
        """Initialise message_history with system prompt + task framing."""
        metadata = QueryMetadata(prompt)
        message_history = build_rlm_system_prompt(
            system_prompt=self.system_prompt,
            query_metadata=metadata,
        )
        # Seed the budget with the initial messages
        if self.budget is not None:
            self.budget.record_messages(message_history)
        return message_history

    # ─── Phase injection helpers ───────────────────────────────────────────────

    def _maybe_inject_phase_messages(
        self,
        message_history: list[dict[str, Any]],
        iteration: int,
        environment: BaseEnv,
    ) -> None:
        """Inject phase-boundary reminder messages into history when appropriate."""
        if not self.phase_injection_enabled or self.budget is None:
            return

        level = self.budget.level

        # Aggressive reminder — highest priority
        if level == "aggressive" and not self._injected_aggressive_reminder:
            msg = {"role": "user", "content": _PHASE_AGGRESSIVE_REMINDER}
            message_history.append(msg)
            self._injected_aggressive_reminder = True
            if self.budget:
                self.budget.record_message("user", _PHASE_AGGRESSIVE_REMINDER)
            return  # one injection per iteration

        # Medium reminder
        if level in ("medium", "aggressive") and not self._injected_medium_reminder:
            msg = {"role": "user", "content": _PHASE_MEDIUM_REMINDER}
            message_history.append(msg)
            self._injected_medium_reminder = True
            if self.budget:
                self.budget.record_message("user", _PHASE_MEDIUM_REMINDER)

        # Plan reminder (injected after iteration 3 if plan variable is set)
        if iteration >= 3 and not self._injected_plan_reminder:
            plan_val = _get_repl_local(environment, "plan")
            if plan_val:
                plan_text = str(plan_val)[:1000]  # cap at 1000 chars
                content = _PLAN_REMINDER_TEMPLATE.format(plan=plan_text)
                msg = {"role": "user", "content": content}
                message_history.append(msg)
                self._injected_plan_reminder = True
                if self.budget:
                    self.budget.record_message("user", content)

    def completion(
        self,
        prompt: str | dict[str, Any],
        root_prompt: str | None = None,
    ) -> RLMChatCompletion:
        """
        Main RCA completion entry point.

        Runs the RLM loop with context management hooks:
        - After each LLM response: records tokens in budget, applies history compression
        - At key iterations: injects phase-boundary reminder messages
        - On budget exhaustion: forces conclusion via _default_answer

        Args:
            prompt: Task description — the RCA investigation request.
            root_prompt: Optional short root prompt shown to the LLM each iteration.

        Returns:
            RLMChatCompletion containing the root-cause analysis.

        """
        time_start = time.perf_counter()

        if self.depth >= self.max_depth:
            return self._fallback_answer(prompt)

        # Reset per-run injection flags
        self._injected_medium_reminder = False
        self._injected_aggressive_reminder = False
        self._injected_plan_reminder = False

        with self._spawn_completion_context(prompt) as (lm_handler, environment):
            message_history = self._setup_prompt(prompt)

            for i in range(self.max_iterations):
                # Emergency exit: aggressive pressure AND more than 10 iterations done
                if self.budget is not None and self.budget.level == "aggressive" and i > 10:
                    break

                context_count = environment.get_context_count() if isinstance(environment, SupportsPersistence) else 1
                history_count = environment.get_history_count() if isinstance(environment, SupportsPersistence) else 0
                current_prompt = [*message_history, build_user_prompt(root_prompt, i, context_count, history_count)]

                iteration_result: RLMIteration = self._completion_turn(
                    prompt=current_prompt,
                    lm_handler=lm_handler,
                    environment=environment,
                )

                # Record the LLM response tokens into budget
                if self.budget is not None:
                    self.budget.record_message("assistant", iteration_result.response)

                # Check for final answer
                final_answer = find_final_answer(
                    iteration_result.response,
                    environment=environment,
                )
                iteration_result.final_answer = final_answer

                # Only call custom log() if logger supports single-argument iteration logging
                # Standard LoggerAdapter.log() requires (level, msg) so we skip it
                if self.logger and hasattr(self.logger, "log_iteration"):
                    self.logger.log_iteration(iteration_result)

                self.verbose.print_iteration(iteration_result, i + 1)

                if final_answer is not None:
                    time_end = time.perf_counter()
                    usage = lm_handler.get_usage_summary()
                    self.verbose.print_final_answer(final_answer)
                    self.verbose.print_summary(i + 1, time_end - time_start, usage.to_dict())

                    if self.persistent and isinstance(environment, SupportsPersistence):
                        environment.add_history(message_history)

                    return RLMChatCompletion(
                        root_model=(
                            self.backend_kwargs.get("model_name", "unknown") if self.backend_kwargs else "unknown"
                        ),
                        prompt=prompt,
                        response=final_answer,
                        usage_summary=usage,
                        execution_time=time_end - time_start,
                    )

                # Format iteration into new messages
                new_messages = format_iteration(iteration_result)

                # Record new user-facing messages in budget
                if self.budget is not None:
                    self.budget.record_messages(new_messages)

                # Extend history
                message_history.extend(new_messages)

                # Apply history compression based on current budget level
                if (
                    self.history_manager is not None
                    and self.budget is not None
                    and self.budget.level != "none"
                ):
                    message_history = self.history_manager.apply(message_history, self.budget)

                # Maybe inject phase reminder messages
                self._maybe_inject_phase_messages(message_history, i, environment)

            # ── Iterations exhausted — force a final answer ────────────────────
            time_end = time.perf_counter()
            final_answer = self._default_answer(message_history, lm_handler)
            usage = lm_handler.get_usage_summary()
            self.verbose.print_final_answer(final_answer)
            self.verbose.print_summary(
                self.max_iterations,
                time_end - time_start,
                usage.to_dict(),
            )

            if self.persistent and isinstance(environment, SupportsPersistence):
                environment.add_history(message_history)

            return RLMChatCompletion(
                root_model=(self.backend_kwargs.get("model_name", "unknown") if self.backend_kwargs else "unknown"),
                prompt=prompt,
                response=final_answer,
                usage_summary=usage,
                execution_time=time_end - time_start,
            )

    def _completion_turn(
        self,
        prompt: list[dict[str, Any]],
        lm_handler: LMHandler,
        environment: BaseEnv,
    ) -> RLMIteration:
        """Single iteration: LLM call → code extraction → REPL execution."""
        iter_start = time.perf_counter()
        response = lm_handler.completion(prompt)
        code_block_strs = find_code_blocks(response)
        code_blocks = []

        for code_block_str in code_block_strs:
            code_result: REPLResult = environment.execute_code(code_block_str)
            # Record REPL output size in budget
            if self.budget is not None and code_result.stdout:
                self.budget.record_repl_output(code_result.stdout)
            code_blocks.append(CodeBlock(code=code_block_str, result=code_result))

        iteration_time = time.perf_counter() - iter_start
        return RLMIteration(
            prompt=prompt,
            response=response,
            code_blocks=code_blocks,
            iteration_time=iteration_time,
        )

    def _default_answer(
        self,
        message_history: list[dict[str, Any]],
        lm_handler: LMHandler,
    ) -> str:
        """Force a final answer when iterations are exhausted."""
        current_prompt = [
            *message_history,
            {
                "role": "user",
                "content": (
                    "You have used all available iterations. Based on the "
                    "evidence gathered so far, please provide your root-cause "
                    "analysis. Assign it to FINAL_VAR."
                ),
            },
        ]
        try:
            response = lm_handler.completion(current_prompt)
        except Exception as exc:
            # Content filter or other API error — return a graceful summary
            # using only the last few messages to minimise prompt injection risk
            short_prompt = [
                *message_history[-4:],
                {
                    "role": "user",
                    "content": (
                        "Summarise the root-cause findings identified in this "
                        "investigation so far, based on the evidence already gathered."
                    ),
                },
            ]
            try:
                response = lm_handler.completion(short_prompt)
            except Exception:
                response = (
                    f"[Investigation incomplete — final summary generation failed: {exc}]\n\n"
                    "All evidence gathered in the iterations above is available in the "
                    "conversation history."
                )
        # Only call custom log_iteration() if logger supports it
        # Standard LoggerAdapter.log() requires (level, msg) so we skip it
        if self.logger and hasattr(self.logger, "log_iteration"):
            self.logger.log_iteration(
                RLMIteration(
                    prompt=current_prompt,
                    response=response,
                    final_answer=response,
                    code_blocks=[],
                ),
            )

        return response

    def _fallback_answer(self, message: str | dict[str, Any]) -> str:
        """Fallback: act as plain LM when at max depth."""
        client: BaseLM = get_client(
            self.backend,
            self.backend_kwargs,
            replayer=self.replayer,
            logger=self.replay_logger,
        )
        return client.completion(message)

    def _validate_persistent_environment_support(self) -> None:
        """Validate that the configured environment supports persistent mode."""
        persistent_supported_environments = {"local", "rca"}

        if self.environment_type not in persistent_supported_environments:
            raise ValueError(
                f"persistent=True is not supported for environment type "
                f"'{self.environment_type}'. Supported: {sorted(persistent_supported_environments)}",
            )

    @staticmethod
    def _env_supports_persistence(env: BaseEnv) -> bool:
        """Check if an environment instance supports persistent mode."""
        return isinstance(env, SupportsPersistence)

    def close(self) -> None:
        """Clean up persistent environment."""
        if self._persistent_env is not None:
            if hasattr(self._persistent_env, "cleanup"):
                self._persistent_env.cleanup()
            self._persistent_env = None

    def __enter__(self) -> "RLM":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> bool:
        self.close()
        return False
