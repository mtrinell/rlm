"""
DetectorREPL — generic LocalREPL subclass for dataset analysis.

Extends the base REPL with:
- Generic file-access helpers (list_files, read_file, read_json, read_jsonl,
  read_lines, read_csv, detect_format, extract_archive, search_lines, context_budget)
- dataset_path, spec, and user_prompt injected as REPL variables
- ContextBudget awareness via the context_budget() helper
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from examples.detective_agent.environment.detector_helpers import build_detector_helpers
from examples.detective_agent.environment.local_repl import LocalREPL

if TYPE_CHECKING:
    from examples.detective_agent.context.budget import ContextBudget


class DetectorREPL(LocalREPL):
    """
    REPL environment pre-loaded with generic file-access helpers.

    The LLM can freely explore any dataset — a tarball, a folder, a single
    JSON/YAML/CSV/log file, or any other format — without needing to write
    low-level I/O boilerplate from scratch.

    Additional REPL globals injected at setup:

        dataset_path    — str: path to the dataset (file, folder, or archive)
        spec            — str: specification / documentation (empty if not provided)
        user_prompt     — str: the end-user's question or directive (empty if not provided)

        list_files      — list all files in a path (directory, archive, or single file)
        read_file       — read a file as UTF-8 text (with optional max_bytes cap)
        read_lines      — read a file as a list of lines
        read_json       — parse a JSON file
        read_jsonl      — parse a JSONL file (one JSON object per line)
        read_csv        — parse a CSV/TSV file into a list of row dicts
        detect_format   — detect a file's format ("json", "yaml", "csv", "log", "archive", …)
        extract_archive — extract a tar or zip archive to a directory
        search_lines    — grep-style line search across a file
        context_budget  — check remaining token budget
    """

    def __init__(
        self,
        dataset_path: str,
        spec: str = "",
        user_prompt: str = "",
        budget: ContextBudget | None = None,
        inject_helpers: bool = True,
        **kwargs: Any,
    ) -> None:
        """
        Initialise the DetectorREPL.

        Args:
            dataset_path: Path to the dataset (file, folder, or archive).
            spec: Specification / documentation content (empty string if not provided).
            user_prompt: The end-user's original question or investigation directive.
            budget: ContextBudget instance for context_budget() REPL helper.
            inject_helpers: Whether to inject file-access helper functions (default True).
            **kwargs: Forwarded to LocalREPL (lm_handler_address, context_payload, etc.)

        """
        self._dataset_path = dataset_path
        self._spec = spec
        self._user_prompt = user_prompt
        self._budget = budget
        self._inject_helpers = inject_helpers

        super().__init__(**kwargs)

    def setup(self) -> None:
        """Set up namespace with generic file-access helpers and stdlib pre-imports."""
        super().setup()

        if not self._inject_helpers:
            return

        # Inject generic file-access helpers
        helpers = build_detector_helpers(self._dataset_path, budget=self._budget)
        self.globals.update(helpers)

        # Pre-import commonly needed stdlib modules
        stdlib_setup = """
import os, re, json, csv, tarfile, zipfile
from pathlib import Path
from datetime import datetime, timedelta, timezone
from collections import Counter, defaultdict, OrderedDict
import itertools
import fnmatch
"""
        self.execute_code(stdlib_setup.strip())

        # Inject dataset_path, spec, and user_prompt as REPL variables
        for name, value in (
            ("dataset_path", self._dataset_path),
            ("spec", self._spec),
            ("user_prompt", self._user_prompt),
        ):
            self.locals[name] = value
            self.globals[name] = value
