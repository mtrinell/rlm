"""
Generic file-access REPL helpers for the detector environment.

These helpers give the LLM structured access to any dataset — a single file,
a directory, or an archive — without needing format-specific boilerplate.
The LLM decides how to read and parse the data based on its content.
"""

from __future__ import annotations

import csv as _csv
import json
import re
import tarfile
import tempfile
import zipfile
from pathlib import Path
from typing import Any


def build_detector_helpers(dataset_path: str, budget: Any = None) -> dict[str, Any]:
    """
    Build the dict of generic file-access helpers for REPL injection.

    Args:
        dataset_path: Path to the dataset (file, folder, or archive).
        budget: Optional ContextBudget instance for context_budget() calls.

    Returns:
        Dict mapping function names to callables, ready for REPL injection.

    """

    def list_files(path: str | None = None) -> list[dict[str, Any]]:
        """
        List files at path. Works on directories, archives (tar/zip), and single files.

        Args:
            path: Path to explore. Defaults to dataset_path if not provided.

        Returns:
            List of dicts: {path, name, size, ext, is_dir}

        """
        target = Path(path) if path is not None else Path(dataset_path)
        results: list[dict[str, Any]] = []

        if target.is_dir():
            for f in sorted(target.rglob("*")):
                results.append(
                    {
                        "path": str(f),
                        "name": f.name,
                        "size": f.stat().st_size if f.is_file() else 0,
                        "ext": f.suffix.lower(),
                        "is_dir": f.is_dir(),
                    }
                )
            return results

        if not target.is_file():
            return [{"error": f"Path not found: {target}"}]

        # Try tar first (before zip — .tar.gz passes zipfile check on some platforms)
        try:
            if tarfile.is_tarfile(str(target)):
                with tarfile.open(str(target), "r:*") as tf:
                    for member in tf.getmembers():
                        results.append(
                            {
                                "path": member.name,
                                "name": Path(member.name).name,
                                "size": member.size,
                                "ext": Path(member.name).suffix.lower(),
                                "is_dir": member.isdir(),
                            }
                        )
                return results
        except Exception:
            pass

        try:
            if zipfile.is_zipfile(str(target)):
                with zipfile.ZipFile(str(target), "r") as zf:
                    for info in zf.infolist():
                        results.append(
                            {
                                "path": info.filename,
                                "name": Path(info.filename).name,
                                "size": info.file_size,
                                "ext": Path(info.filename).suffix.lower(),
                                "is_dir": info.is_dir(),
                            }
                        )
                return results
        except Exception:
            pass

        # Single plain file
        results.append(
            {
                "path": str(target),
                "name": target.name,
                "size": target.stat().st_size,
                "ext": target.suffix.lower(),
                "is_dir": False,
            }
        )
        return results

    def read_file(path: str, max_bytes: int | None = None) -> str:
        """
        Read a file as UTF-8 text (non-UTF-8 bytes are replaced with the replacement char).

        Args:
            path: Path to the file.
            max_bytes: If set, read only the first N bytes (useful for sampling large files).

        Returns:
            File contents as a string.

        """
        p = Path(path)
        try:
            with open(p, "rb") as fh:
                raw = fh.read(max_bytes) if max_bytes is not None else fh.read()
            return raw.decode("utf-8", errors="replace")
        except FileNotFoundError:
            return f"[ERROR] File not found: {path}"
        except OSError as e:
            return f"[ERROR] {e}"

    def read_lines(path: str, max_lines: int | None = None) -> list[str]:
        """
        Read a file as a list of lines (newlines stripped).

        Args:
            path: Path to the file.
            max_lines: If set, return only the first N lines.

        Returns:
            List of line strings.

        """
        p = Path(path)
        try:
            lines: list[str] = []
            with open(p, encoding="utf-8", errors="replace") as fh:
                for i, line in enumerate(fh):
                    if max_lines is not None and i >= max_lines:
                        break
                    lines.append(line.rstrip("\n"))
            return lines
        except FileNotFoundError:
            return [f"[ERROR] File not found: {path}"]
        except OSError as e:
            return [f"[ERROR] {e}"]

    def read_json(path: str) -> Any:
        """
        Parse a JSON file and return the parsed object.

        Args:
            path: Path to the JSON file.

        Returns:
            Parsed JSON object (dict or list).

        """
        p = Path(path)
        try:
            with open(p, encoding="utf-8", errors="replace") as fh:
                return json.load(fh)
        except FileNotFoundError:
            return {"error": f"File not found: {path}"}
        except json.JSONDecodeError as e:
            return {"error": f"JSON parse error: {e}"}
        except OSError as e:
            return {"error": str(e)}

    def read_jsonl(path: str, max_lines: int | None = None) -> list[Any]:
        """
        Parse a JSONL file (one JSON object per line). Blank and malformed lines are skipped.

        Args:
            path: Path to the JSONL file.
            max_lines: If set, parse only the first N non-blank lines.

        Returns:
            List of parsed JSON objects.

        """
        p = Path(path)
        results: list[Any] = []
        try:
            with open(p, encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    if max_lines is not None and len(results) >= max_lines:
                        break
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        results.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        except FileNotFoundError:
            return [{"error": f"File not found: {path}"}]
        except OSError as e:
            return [{"error": str(e)}]
        return results

    def detect_format(path: str) -> str:
        """
        Detect the format of a file by extension, then by content inspection.

        Returns one of: "json", "jsonl", "yaml", "csv", "xml", "html", "toml",
        "ini", "log", "text", "archive", "binary", or "unknown".

        Args:
            path: Path to the file.

        Returns:
            Format string.

        """
        p = Path(path)
        ext = p.suffix.lower()

        ext_map: dict[str, str] = {
            ".json": "json",
            ".jsonl": "jsonl",
            ".ndjson": "jsonl",
            ".yaml": "yaml",
            ".yml": "yaml",
            ".csv": "csv",
            ".tsv": "csv",
            ".xml": "xml",
            ".html": "html",
            ".htm": "html",
            ".toml": "toml",
            ".ini": "ini",
            ".cfg": "ini",
            ".conf": "ini",
            ".log": "log",
            ".txt": "text",
            ".md": "text",
            ".rst": "text",
            ".tar": "archive",
            ".gz": "archive",
            ".tgz": "archive",
            ".zip": "archive",
            ".bz2": "archive",
            ".xz": "archive",
        }
        if ext in ext_map:
            return ext_map[ext]

        # Content-based detection
        try:
            with open(p, "rb") as fh:
                header = fh.read(512)
        except OSError:
            return "unknown"

        # Magic bytes
        if header[:2] == b"\x1f\x8b":
            return "archive"  # gzip
        if header[:4] == b"PK\x03\x04":
            return "archive"  # zip
        if header[:5] == b"BZh91":
            return "archive"  # bzip2

        try:
            text = header.decode("utf-8").strip()
        except UnicodeDecodeError:
            return "binary"

        first = text[:1]
        if first in ("{", "["):
            # Could be JSON or first line of JSONL
            first_line = text.splitlines()[0].strip() if text.splitlines() else text
            if first_line.startswith("{") or first_line.startswith("["):
                # Check second line for JSONL detection
                lines = text.splitlines()
                if len(lines) > 1 and lines[1].strip().startswith("{"):
                    return "jsonl"
                return "json"
        if first == "<":
            return "xml"
        if "," in text and "\n" in text:
            return "csv"

        return "text"

    def extract_archive(path: str, dest: str | None = None) -> str:
        """
        Extract a tar (.tar, .tar.gz, .tgz, .tar.bz2) or zip archive.

        Args:
            path: Path to the archive file.
            dest: Destination directory. If None, extracts to a temp directory.

        Returns:
            Absolute path to the extraction directory as a string.

        Raises:
            ValueError: If the file is not a recognised archive.

        """
        src = Path(path)
        if dest is None:
            dest_path = Path(tempfile.mkdtemp(prefix="rlm_extract_"))
        else:
            dest_path = Path(dest)
            dest_path.mkdir(parents=True, exist_ok=True)

        # Try tar first
        try:
            if tarfile.is_tarfile(str(src)):
                with tarfile.open(str(src), "r:*") as tf:
                    for member in tf.getmembers():
                        # Security: reject absolute paths and path traversal
                        member_path = Path(member.name)
                        if member_path.is_absolute() or ".." in member_path.parts:
                            continue
                        tf.extract(member, dest_path, set_attrs=False)
                return str(dest_path)
        except Exception:
            pass

        # Try zip
        try:
            if zipfile.is_zipfile(str(src)):
                with zipfile.ZipFile(str(src), "r") as zf:
                    for name in zf.namelist():
                        parts = Path(name).parts
                        # Security: reject absolute paths and path traversal
                        if ".." in parts or (parts and Path(parts[0]).is_absolute()):
                            continue
                        zf.extract(name, dest_path)
                return str(dest_path)
        except Exception:
            pass

        raise ValueError(f"Could not extract archive (unsupported or corrupt format): {path}")

    def search_lines(
        path: str,
        pattern: str,
        flags: str = "i",
        max_results: int = 200,
    ) -> list[dict[str, Any]]:
        """
        Search a file's lines for a regex pattern (grep-style).

        Args:
            path: Path to the file.
            pattern: Python regex pattern.
            flags: Regex flags string — "i" = case-insensitive (default), "" = none.
            max_results: Maximum number of matching lines to return.

        Returns:
            List of dicts: {path, line_no (1-based), line}

        """
        re_flags = re.IGNORECASE if "i" in flags else 0
        try:
            compiled = re.compile(pattern, re_flags)
        except re.error as e:
            return [{"error": f"Invalid regex: {e}"}]

        results: list[dict[str, Any]] = []
        p = Path(path)
        try:
            with open(p, encoding="utf-8", errors="replace") as fh:
                for i, line in enumerate(fh, start=1):
                    if compiled.search(line):
                        results.append({"path": str(p), "line_no": i, "line": line.rstrip("\n")})
                        if len(results) >= max_results:
                            break
        except FileNotFoundError:
            return [{"error": f"File not found: {path}"}]
        except OSError as e:
            return [{"error": str(e)}]
        return results

    def read_csv(path: str, max_rows: int | None = None) -> list[dict[str, str]]:
        """
        Parse a CSV (or TSV) file into a list of row dicts keyed by header.

        Args:
            path: Path to the CSV file.
            max_rows: If set, return only the first N data rows.

        Returns:
            List of row dicts.

        """
        p = Path(path)
        delimiter = "\t" if p.suffix.lower() == ".tsv" else ","
        rows: list[dict[str, str]] = []
        try:
            with open(p, encoding="utf-8", errors="replace", newline="") as fh:
                reader = _csv.DictReader(fh, delimiter=delimiter)
                for i, row in enumerate(reader):
                    if max_rows is not None and i >= max_rows:
                        break
                    rows.append(dict(row))
        except FileNotFoundError:
            return [{"error": f"File not found: {path}"}]
        except OSError as e:
            return [{"error": str(e)}]
        return rows

    def context_budget() -> dict[str, Any]:
        """
        Return the current context token budget snapshot.

        Returns:
            Dict with keys: used_tokens, total_tokens, percent_used, level, tokens_remaining.
            level is one of: "none", "light", "medium", "aggressive".

        """
        if budget is None:
            return {
                "used_tokens": 0,
                "total_tokens": 0,
                "percent_used": 0.0,
                "level": "none",
                "tokens_remaining": 0,
            }
        return budget.snapshot()

    return {
        "list_files": list_files,
        "read_file": read_file,
        "read_lines": read_lines,
        "read_json": read_json,
        "read_jsonl": read_jsonl,
        "read_csv": read_csv,
        "detect_format": detect_format,
        "extract_archive": extract_archive,
        "search_lines": search_lines,
        "context_budget": context_budget,
    }
