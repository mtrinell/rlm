"""JSONL logger for RLM iterations. Forked from rlms library."""

import json
import os
import uuid
from datetime import UTC, datetime

from examples.detective_agent.core.types import RLMIteration, RLMMetadata


class RLMLogger:
    """Writes RLMIteration data to a JSON-lines file for analysis and replay."""

    def __init__(self, log_dir: str, file_name: str = "rlm") -> None:
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        timestamp = datetime.now(tz=UTC).strftime("%Y-%m-%d_%H-%M-%S")
        run_id = str(uuid.uuid4())[:8]
        self.log_file_path = os.path.join(log_dir, f"{file_name}_{timestamp}_{run_id}.jsonl")

        self._iteration_count = 0
        self._metadata_logged = False

    def log_metadata(self, metadata: RLMMetadata) -> None:
        if self._metadata_logged:
            return
        entry = {
            "type": "metadata",
            "timestamp": datetime.now(tz=UTC).isoformat(),
            **metadata.to_dict(),
        }
        with open(self.log_file_path, "a") as f:
            json.dump(entry, f)
            f.write("\n")
        self._metadata_logged = True

    def log(self, iteration: RLMIteration) -> None:
        self._iteration_count += 1
        entry = {
            "type": "iteration",
            "iteration": self._iteration_count,
            "timestamp": datetime.now(tz=UTC).isoformat(),
            **iteration.to_dict(),
        }
        with open(self.log_file_path, "a") as f:
            json.dump(entry, f)
            f.write("\n")

    def log_iteration(self, iteration: RLMIteration) -> None:
        """Alias for log() to match rlm_loop.py's expected interface."""
        self.log(iteration)

    @property
    def iteration_count(self) -> int:
        return self._iteration_count
