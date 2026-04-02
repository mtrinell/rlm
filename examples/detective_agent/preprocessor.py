"""
Preprocessor: extracts tarball and builds file manifest before handing off to RLM.

Solves the "wasted iteration 1" problem in the old prototype: the LLM was given
the tarball path and spent its first iteration just extracting it. Now extraction
happens before the loop starts, and the file manifest is injected into the REPL
so the LLM can immediately begin analysis.

Usage:
    result = Preprocessor(tarball_path).run()
    # result.extract_dir   -> Path to extracted files
    # result.file_manifest -> dict[relative_path, FileInfo]
    # result.cleanup()     -> remove temp dir when done
"""

from __future__ import annotations

import logging
import shutil
import tarfile
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class FileInfo:
    """Metadata for a single file in the extracted archive."""

    size: int
    modified: str  # ISO-8601 mtime string or empty
    lines: int  # estimated line count (0 if binary)
    path: str  # path relative to extract root


@dataclass
class PreprocessResult:
    """Result from Preprocessor.run()."""

    tarball_path: Path
    extract_dir: Path
    file_manifest: dict[str, dict[str, Any]]  # relative_path → FileInfo-dict
    total_files: int
    total_size_bytes: int
    _owned_tempdir: bool = field(default=False, repr=False)

    @property
    def summary(self) -> str:
        mb = self.total_size_bytes / 1024 / 1024
        return f"{self.tarball_path.name}: {self.total_files} files, {mb:.1f} MB extracted at {self.extract_dir}"

    def cleanup(self) -> None:
        """Remove the temporary extraction directory (only if we created it)."""
        if self._owned_tempdir and self.extract_dir.exists():
            shutil.rmtree(self.extract_dir, ignore_errors=True)
            logger.debug(f"Removed temp dir: {self.extract_dir}")


_MAX_LINE_ESTIMATE_BYTES = 64 * 1024  # read first 64 KB to estimate line count


def _estimate_lines(path: Path) -> int:
    """Count newlines in the first 64 KB of a file to estimate line count."""
    try:
        with open(path, "rb") as fh:
            chunk = fh.read(_MAX_LINE_ESTIMATE_BYTES)
        newlines = chunk.count(b"\n")
        if not newlines:
            return 0
        # Extrapolate from sample if file is larger
        size = path.stat().st_size
        if size > _MAX_LINE_ESTIMATE_BYTES:
            return int(newlines * size / _MAX_LINE_ESTIMATE_BYTES)
        return newlines
    except OSError:
        return 0


class Preprocessor:
    """
    Extract a tarball and build a file manifest for RcaREPL injection.

    Args:
        tarball_path: Path to the .tar, .tar.gz, .tgz, or .tar.bz2 file.
        extract_dir:  Optional explicit extraction directory.
                      If None, a temporary directory is created and owned by
                      PreprocessResult (will be cleaned up after use).

    """

    def __init__(
        self,
        tarball_path: str | Path,
        extract_dir: str | Path | None = None,
    ) -> None:
        self.tarball_path = Path(tarball_path).resolve()
        self._explicit_extract_dir = Path(extract_dir).resolve() if extract_dir else None

    def run(self) -> PreprocessResult:
        """Extract the archive and return a PreprocessResult."""
        if not self.tarball_path.exists():
            raise FileNotFoundError(f"Tarball not found: {self.tarball_path}")

        # Determine / create extraction directory
        owned = False
        if self._explicit_extract_dir:
            extract_dir = self._explicit_extract_dir
            extract_dir.mkdir(parents=True, exist_ok=True)
        else:
            extract_dir = Path(tempfile.mkdtemp(prefix="a3po_rca_")).resolve()
            owned = True

        logger.info(f"Extracting {self.tarball_path} → {extract_dir}")

        try:
            with tarfile.open(self.tarball_path, "r:*") as tar:
                # Security: prevent path traversal
                for member in tar.getmembers():
                    member_path = Path(member.name)
                    if member_path.is_absolute():
                        raise ValueError(f"Archive contains absolute path: {member.name}")
                    resolved = (extract_dir / member_path).resolve()
                    if not str(resolved).startswith(str(extract_dir)):
                        raise ValueError(f"Archive path traversal detected: {member.name}")

                tar.extractall(path=extract_dir)  # noqa: S202 — validated above
        except Exception as e:
            if owned:
                shutil.rmtree(extract_dir, ignore_errors=True)
            raise RuntimeError(f"Extraction failed: {e}") from e

        # Build file manifest
        manifest: dict[str, dict[str, Any]] = {}
        total_size = 0

        for path in extract_dir.rglob("*"):
            if not path.is_file():
                continue
            try:
                stat = path.stat()
            except OSError:
                continue

            size = stat.st_size
            total_size += size

            try:
                import datetime  # noqa: PLC0415

                mtime = datetime.datetime.fromtimestamp(
                    stat.st_mtime, tz=datetime.UTC,
                ).isoformat()
            except (OSError, ValueError):
                mtime = ""

            rel = str(path.relative_to(extract_dir))
            manifest[rel] = {
                "size": size,
                "modified": mtime,
                "lines": _estimate_lines(path),
                "path": rel,
            }

        logger.info(f"Manifest built: {len(manifest)} files, {total_size / 1024:.0f} KB")

        return PreprocessResult(
            tarball_path=self.tarball_path,
            extract_dir=extract_dir,
            file_manifest=manifest,
            total_files=len(manifest),
            total_size_bytes=total_size,
            _owned_tempdir=owned,
        )

    def validate(self) -> bool:
        """Check that the tarball exists and is a valid archive (without extracting)."""
        if not self.tarball_path.exists():
            return False
        try:
            with tarfile.open(self.tarball_path, "r:*") as tar:
                _ = tar.getmembers()
            return True
        except tarfile.TarError:
            return False
