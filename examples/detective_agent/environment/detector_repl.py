"""
DetectorREPL — LocalREPL subclass for behavioral deviation detection.

Extends the base REPL with:
- OTEL trace parsing helpers (load_traces, extract_log_records, extract_spans, …)
- app_readme and traces_path injected as REPL variables
- ContextBudget awareness via context_budget() helper
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from examples.detective_agent.environment.detector_helpers import build_detector_helpers
from examples.detective_agent.environment.local_repl import LocalREPL

if TYPE_CHECKING:
    from examples.detective_agent.context.budget import ContextBudget


class DetectorREPL(LocalREPL):
    """
    REPL environment pre-loaded with OTEL trace analysis helpers.

    The LLM gets reliable, structured access to traces without needing
    to write trace-parsing boilerplate from scratch each run.

    Additional REPL globals injected at setup:
        app_readme          — str: full README / documentation of the app under test
        traces_path         — str: path to OTEL JSONL trace file
        load_traces         — load and parse OTEL JSONL into list of dicts
        extract_log_records — flatten all log records from loaded traces
        extract_spans       — flatten all span records from loaded traces
        get_errors          — filter records to severity >= 17 (ERROR/FATAL)
        get_component_timeline — group records by scope, sorted by time
        search_traces       — regex search over record bodies
        summarize_component — text summary of a component's records for llm_query()
        context_budget      — check remaining token budget
    """

    def __init__(
        self,
        traces_path: str,
        app_readme: str = "",
        budget: ContextBudget | None = None,
        inject_helpers: bool = True,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the DetectorREPL.

        Args:
            traces_path: Path to the OTEL JSONL traces file.
            app_readme: Full README / documentation content for the app under test.
            budget: ContextBudget instance for context_budget() REPL helper.
            inject_helpers: Whether to inject trace helper functions (default True).
            **kwargs: Forwarded to LocalREPL (lm_handler_address, context_payload, etc.)

        """
        self._traces_path = traces_path
        self._app_readme = app_readme
        self._budget = budget
        self._inject_helpers = inject_helpers

        super().__init__(**kwargs)

    def setup(self) -> None:
        """Setup namespace with OTEL helpers and stdlib pre-imports."""
        super().setup()

        if not self._inject_helpers:
            return

        # Inject OTEL trace helpers
        helpers = build_detector_helpers(self._traces_path, budget=self._budget)
        self.globals.update(helpers)

        # Pre-import commonly needed stdlib modules
        stdlib_setup = """
import os, re, json
from pathlib import Path
from datetime import datetime, timedelta, timezone
from collections import Counter, defaultdict, OrderedDict
import itertools
import fnmatch
"""
        self.execute_code(stdlib_setup.strip())

        # Inject traces_path and app_readme as REPL variables
        self.locals["traces_path"] = self._traces_path
        self.globals["traces_path"] = self._traces_path

        self.locals["app_readme"] = self._app_readme
        self.globals["app_readme"] = self._app_readme
