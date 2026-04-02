"""
RcaREPL — LocalREPL subclass for RCA investigations.

Extends the base REPL with:
- Injected file system helpers (read_file_safe, list_files, grep, etc.)
- Pre-imported stdlib modules for log analysis
- files_path + file_manifest as REPL variables
- ContextBudget awareness via context_budget() helper
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from examples.detective_agent.environment.helpers import build_helpers
from examples.detective_agent.environment.local_repl import LocalREPL

if TYPE_CHECKING:
    from examples.detective_agent.context.budget import ContextBudget


class RcaREPL(LocalREPL):
    """
    REPL environment pre-loaded with RCA investigation helpers.

    The LLM gets reliable, fast access to the filesystem without needing
    to generate error-prone file-read code from first principles each run.

    Additional REPL globals injected at setup:
        files_path       — str path to extracted investigation directory
        file_manifest    — list of {path, size_bytes, modified, type} dicts
        read_file_safe   — read a file safely (binary-safe, size-limited)
        list_files       — list files with metadata (rg-backed)
        grep             — regex search across files (rg-backed)
        filter_by_timerange — filter grep results by timestamp window
        summarize_file   — triage view (first+last+errors)
        context_budget   — check remaining token budget
    """

    def __init__(
        self,
        files_path: str,
        file_manifest: list[dict[str, Any]] | None = None,
        budget: ContextBudget | None = None,
        inject_helpers: bool = True,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the RcaREPL.

        Args:
            files_path: Path to the extracted log/config directory from Preprocessor.
            file_manifest: Pre-built file listing from Preprocessor.list_files().
            budget: ContextBudget instance for context_budget() REPL helper.
            inject_helpers: Whether to inject rca helper functions (default True).
            **kwargs: Forwarded to LocalREPL (lm_handler_address, context_payload, etc.)

        """
        self._files_path = files_path
        self._file_manifest = file_manifest or []
        self._budget = budget
        self._inject_helpers = inject_helpers

        super().__init__(**kwargs)

    def setup(self) -> None:
        """Setup namespace with helpers and stdlib pre-imports."""
        # Run base setup first (registers FINAL_VAR, SHOW_VARS, llm_query etc.)
        super().setup()

        if not self._inject_helpers:
            return

        # Inject helpers as globals so they're available without importing
        helpers = build_helpers(self._files_path, budget=self._budget)
        self.globals.update(helpers)

        # Pre-import commonly needed stdlib modules into locals
        # (avoids "import not allowed" issues and saves LLM tokens)
        stdlib_setup = """
import os, re, json
from pathlib import Path
from datetime import datetime, timedelta
from collections import Counter, defaultdict, OrderedDict
import itertools
import fnmatch
"""
        self.execute_code(stdlib_setup.strip())

        # Set files_path as a REPL variable (both locals and direct)
        self.locals["files_path"] = self._files_path
        self.globals["files_path"] = self._files_path

        # Inject file manifest
        if self._file_manifest:
            manifest_json = json.dumps(self._file_manifest)
            self.execute_code(
                f"import json as _json\nfile_manifest = _json.loads({manifest_json!r})\ndel _json",
            )
        else:
            # Bootstrap manifest from list_files helper
            self.execute_code("file_manifest = list_files()")

    def update_budget(self, budget: ContextBudget) -> None:
        """Update the budget reference (called by RLMLoop after each iteration)."""
        self._budget = budget
        # Rebuild helpers with updated budget reference
        helpers = build_helpers(self._files_path, budget=self._budget)
        self.globals["context_budget"] = helpers["context_budget"]
