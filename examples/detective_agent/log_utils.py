"""Minimal logging configuration."""

import logging
from pathlib import Path


def setup_logging(log_file: Path, level: int = logging.INFO) -> logging.Logger:
    """
    Configure logging to file and console.

    Args:
        log_file: Path to write logs
        level: Logging level

    Returns:
        Root logger

    """
    # Create log file directory if needed
    log_file.parent.mkdir(parents=True, exist_ok=True)

    # Clear any existing handlers
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(level)

    # Console handler - INFO and above
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    root_logger.addHandler(console)

    # File handler - everything
    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"),
    )
    root_logger.addHandler(file_handler)

    return root_logger
