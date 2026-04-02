"""
REPL helper functions injected into the RCA investigation namespace.

These helpers give the LLM reliable, performant access to the filesystem
without needing to generate file-access code from scratch each run.
They mirror the prep agent's tools but live as REPL-callable Python,
preserving the LLM's code-generation flexibility.
"""

from __future__ import annotations

import contextlib
import os
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Timezone abbreviation → UTC offset map (common networking log timezones)
_TZ_OFFSETS: dict[str, int] = {
    "UTC": 0,
    "GMT": 0,
    "Z": 0,
    "EST": -5,
    "EDT": -4,
    "CST": -6,
    "CDT": -5,
    "MST": -7,
    "MDT": -6,
    "PST": -8,
    "PDT": -7,
    "CET": 1,
    "CEST": 2,
    "EET": 2,
    "EEST": 3,
    "IST": 5,
    "JST": 9,
    "AEST": 10,
    "AEDT": 11,
}

# File type classification patterns (same as prep agent)
_FILE_TYPE_PATTERNS: dict[str, list[str]] = {
    "log": [r"\.log$", r"\.log\.\d+$", r"_log$", r"\.out$", r"\.stderr$", r"\.stdout$"],
    "config": [
        r"\.conf$",
        r"\.cfg$",
        r"\.yaml$",
        r"\.yml$",
        r"\.json$",
        r"\.ini$",
        r"\.toml$",
        r"\.xml$",
        r"\.properties$",
    ],
    "topology": [r"topology", r"topo", r"diagram", r"network_map"],
}


def _classify_file(path: str) -> str:
    name = os.path.basename(path).lower()
    for file_type, patterns in _FILE_TYPE_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, name):
                return file_type
    return "unknown"


def _run_ripgrep(*args: str, timeout: int = 30) -> tuple[str, bool]:
    """Run ripgrep; return (output, success). Falls back gracefully if rg not found."""
    try:
        result = subprocess.run(
            ["rg", *list(args)],  # noqa: S607
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.stdout, True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return "", False


def build_helpers(files_path: str, budget: Any = None) -> dict[str, Any]:
    """
    Build the dict of helper functions to inject into the REPL namespace.

    Args:
        files_path: Path to the extracted log/config directory.
        budget: Optional ContextBudget instance for context_budget() calls.

    Returns:
        Dict mapping function names to callables, ready for REPL injection.

    """

    def read_file_safe(path: str, max_bytes: int = 500_000) -> str:
        """
        Read a file safely, handling binary files and large files.

        Args:
            path: Absolute or relative path (relative paths resolved against files_path).
            max_bytes: Maximum bytes to read (default 500KB to stay within context budget).

        Returns:
            File content as string, truncated if > max_bytes, with [TRUNCATED] marker.

        """
        if not os.path.isabs(path):
            path = os.path.join(files_path, path)
        try:
            stat = os.stat(path)
            raw = Path(path).read_bytes()[:max_bytes]
            content = raw.decode("utf-8", errors="replace")
            if stat.st_size > max_bytes:
                truncated_kb = (stat.st_size - max_bytes) // 1024
                content += (
                    f"\n\n[TRUNCATED: {truncated_kb}KB omitted"
                    " -- use grep() or filter_by_timerange() for targeted reads]"
                )
            return content
        except FileNotFoundError:
            return f"Error: File not found: {path}"
        except PermissionError:
            return f"Error: Permission denied: {path}"
        except Exception as e:
            return f"Error reading {path}: {type(e).__name__}: {e}"

    def list_files(
        path: str | None = None,
        extensions: list[str] | None = None,
        recursive: bool = True,
    ) -> list[dict[str, Any]]:
        """
        List all files in the investigation directory.

        Uses ripgrep (rg) for fast traversal; falls back to os.walk.

        Args:
            path: Directory to scan (defaults to files_path).
            extensions: Optional filter, e.g. ['.log', '.conf'].
            recursive: Include subdirectories (default True).

        Returns:
            List of dicts with keys: path, size_bytes, modified (ISO string), type.
            Sorted by size_bytes descending.

        """
        target = path or files_path

        # Try ripgrep first (much faster)
        rg_output, rg_ok = _run_ripgrep("--files", "--no-ignore", target)
        if rg_ok and rg_output.strip():
            file_paths = [p for p in rg_output.splitlines() if p.strip()]
        else:
            # Fallback to os.walk
            file_paths = []
            if recursive:
                for root, _, names in os.walk(target):
                    for name in names:
                        file_paths.append(os.path.join(root, name))
            else:
                with contextlib.suppress(OSError):
                    file_paths = [
                        os.path.join(target, f) for f in os.listdir(target) if os.path.isfile(os.path.join(target, f))
                    ]

        results = []
        for fp in file_paths:
            if extensions and not any(fp.lower().endswith(ext.lower()) for ext in extensions):
                continue
            try:
                stat = os.stat(fp)
                results.append(
                    {
                        "path": fp,
                        "size_bytes": stat.st_size,
                        "modified": datetime.fromtimestamp(
                            stat.st_mtime, tz=UTC,
                        ).isoformat(),
                        "type": _classify_file(fp),
                    },
                )
            except OSError:
                pass

        return sorted(results, key=lambda x: x["size_bytes"], reverse=True)

    def grep(
        pattern: str,
        path: str | None = None,
        flags: str = "i",
        max_results: int = 500,
        context_lines: int = 0,
    ) -> list[dict[str, Any]]:
        """
        Search for a regex pattern across files.

        Uses ripgrep for performance. Falls back to Python re if rg not available.

        Args:
            pattern: Regex or literal string to search for.
            path: File or directory to search (defaults to files_path).
            flags: Ripgrep flags — 'i' = case-insensitive (default), 'n' = case-sensitive.
            max_results: Maximum number of matching lines to return.
            context_lines: Number of context lines before/after each match (like grep -C).

        Returns:
            List of dicts with keys: file, line_number, text.

        """
        # Resolve relative paths against files_path (same as read_file_safe)
        if path and not os.path.isabs(path):
            target = os.path.join(files_path, path)
        else:
            target = path or files_path
        rg_args = ["--line-number", "--no-heading"]
        if "i" in flags:
            rg_args.append("--ignore-case")
        if context_lines > 0:
            rg_args.extend(["--context", str(context_lines)])

        rg_output, rg_ok = _run_ripgrep(*rg_args, pattern, target)

        results = []
        if rg_ok and rg_output.strip():
            for line in rg_output.splitlines():
                if len(results) >= max_results:
                    break
                # ripgrep format: filename:line_num:text  (or filename-line_num-text for context)
                separator = ":" if ":" in line else "-"
                parts = line.split(separator, 2)
                if len(parts) == 3:
                    with contextlib.suppress(ValueError, IndexError):
                        results.append(
                            {
                                "file": parts[0],
                                "line_number": int(parts[1]),
                                "text": parts[2],
                            },
                        )
        else:
            # Python fallback
            target_path = Path(target)
            file_iter = (
                [target_path] if target_path.is_file() else list(target_path.rglob("*")) if target_path.is_dir() else []
            )
            re_flags = re.IGNORECASE if "i" in flags else 0
            try:
                compiled = re.compile(pattern, re_flags)
            except re.error:
                return [{"file": "error", "line_number": 0, "text": f"Invalid regex: {pattern}"}]

            for fpath in file_iter:
                if not fpath.is_file():
                    continue
                try:
                    content = fpath.read_bytes().decode("utf-8", errors="replace")
                    for lineno, line in enumerate(content.splitlines(), 1):
                        if compiled.search(line):
                            results.append(
                                {
                                    "file": str(fpath),
                                    "line_number": lineno,
                                    "text": line,
                                },
                            )
                            if len(results) >= max_results:
                                break
                except OSError:
                    pass
                if len(results) >= max_results:
                    break

        return results

    def filter_by_timerange(
        events: list[dict[str, Any]],
        start_dt: datetime | str,
        end_dt: datetime | str,
        time_key: str = "text",
    ) -> list[dict[str, Any]]:
        """
        Filter a list of event dicts (e.g. from grep()) by timestamp range.

        Handles multiple common timestamp formats found in network logs.

        Args:
            events: List of dicts, each containing a string field with a timestamp.
            start_dt: Start of window (datetime or ISO string).
            end_dt: End of window (datetime or ISO string).
            time_key: Dict key containing the timestamp text (default 'text').

        Returns:
            Filtered list, events where parsed timestamp falls within [start_dt, end_dt].

        """
        # Normalize start/end to timezone-naive datetimes
        if isinstance(start_dt, str):
            start_dt = _parse_timestamp_from_text(start_dt) or datetime.min.replace(tzinfo=UTC)
        if isinstance(end_dt, str):
            end_dt = _parse_timestamp_from_text(end_dt) or datetime.max.replace(tzinfo=UTC)

        # Strip tzinfo for naive comparison
        if hasattr(start_dt, "tzinfo") and start_dt.tzinfo:
            start_dt = start_dt.replace(tzinfo=None)
        if hasattr(end_dt, "tzinfo") and end_dt.tzinfo:
            end_dt = end_dt.replace(tzinfo=None)

        filtered = []
        for event in events:
            text = event.get(time_key, "")
            ts = _parse_timestamp_from_text(str(text))
            if ts and start_dt <= ts <= end_dt:
                filtered.append(event)
        return filtered

    def summarize_file(path: str, max_lines: int = 200) -> str:
        """
        Get a quick triage view of a file.

        Returns first 50 lines + all ERROR/WARN lines + last 50 lines,
        capped at max_lines. Useful for getting a sense of a file before
        deciding whether to read it fully.

        Args:
            path: File path (absolute or relative to files_path).
            max_lines: Maximum total lines to return.

        Returns:
            String with triage summary.

        """
        content = read_file_safe(path)
        if content.startswith("Error:"):
            return content

        lines = content.splitlines()
        total = len(lines)
        if total <= max_lines:
            return content

        head = lines[:50]
        tail = lines[-50:]
        error_warn_lines = [
            (i, line)
            for i, line in enumerate(lines)
            if re.search(r"\b(ERROR|WARN|CRITICAL|FATAL|EXCEPTION|TRACEBACK)\b", line, re.IGNORECASE)
            and i >= 50
            and i < total - 50
        ]

        # Cap error lines to avoid flooding context
        max_error_lines = max_lines - 100
        if len(error_warn_lines) > max_error_lines:
            error_warn_lines = error_warn_lines[:max_error_lines]

        parts = [
            f"=== {path} | {total} total lines ===",
            "--- FIRST 50 LINES ---",
            "\n".join(head),
        ]
        if error_warn_lines:
            parts.append(f"\n--- ERROR/WARN LINES ({len(error_warn_lines)} found) ---")
            for lineno, line in error_warn_lines:
                parts.append(f"  {lineno + 1}: {line}")
        parts.extend(
            [
                "\n--- LAST 50 LINES ---",
                "\n".join(tail),
            ],
        )
        return "\n".join(parts)

    def context_budget() -> dict[str, Any]:
        """
        Get the current context budget status.

        Returns:
            Dict with: used_tokens, total_tokens, percent_used (0-100),
                       level ('none'|'light'|'medium'|'aggressive'),
                       tokens_remaining, iterations_remaining (estimate).

        """
        if budget is None:
            return {
                "used_tokens": 0,
                "total_tokens": 100_000,
                "percent_used": 0,
                "level": "none",
                "tokens_remaining": 100_000,
                "iterations_remaining": "unknown",
            }
        return budget.snapshot()

    return {
        "read_file_safe": read_file_safe,
        "list_files": list_files,
        "grep": grep,
        "filter_by_timerange": filter_by_timerange,
        "summarize_file": summarize_file,
        "context_budget": context_budget,
    }


# =============================================================================
# Timestamp parsing helpers
# =============================================================================

_TIMESTAMP_FORMATS = [
    # ISO 8601 variants
    r"(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?)",
    # Cisco/IOS: Sep 20 14:23:45 or Sep 20 2023 14:23:45
    r"([A-Za-z]{3}\s+\d{1,2}\s+\d{4}\s+\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:\s+[A-Z]{2,4})?)",
    r"([A-Za-z]{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:\s+[A-Z]{2,4})?)",
    # Unix syslog: 2024-01-15T14:23:45+00:00
    r"(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})",
    # Juniper: 2024-09-20 14:23:45
    r"(\d{2}/\d{2}/\d{4}\s+\d{2}:\d{2}:\d{2})",
]

_PARSED_FORMATS = [
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%SZ",
    "%b %d %Y %H:%M:%S",
    "%b %d %H:%M:%S",
    "%m/%d/%Y %H:%M:%S",
]


def _parse_timestamp_from_text(text: str) -> datetime | None:
    """Extract and parse the first timestamp from a text string."""
    for pattern in _TIMESTAMP_FORMATS:
        m = re.search(pattern, text)
        if not m:
            continue
        ts_str = m.group(1).strip()

        # Strip known timezone abbreviations before parsing
        stripped = ts_str
        for tz_abbr in sorted(_TZ_OFFSETS.keys(), key=len, reverse=True):
            stripped = re.sub(r"\s+" + re.escape(tz_abbr) + r"$", "", stripped).strip()

        for fmt in _PARSED_FORMATS:
            try:
                return datetime.strptime(stripped, fmt).replace(tzinfo=UTC)
            except ValueError:
                continue

        # Try ISO with fromisoformat
        try:
            clean = re.sub(r"[+-]\d{2}:\d{2}$", "", ts_str).rstrip("Z").strip()
            return datetime.fromisoformat(clean)
        except ValueError:
            continue

    return None
