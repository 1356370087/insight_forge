"""Checkpoint-based recovery for async SubAgent research tasks.

When a background researcher task fails or is cancelled, the last saved
checkpoint allows a retry to resume from the most recent stable state
rather than starting from scratch.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ResearcherCheckpoint:
    """A snapshot of researcher state that can be persisted and restored.

    Captures enough information to skip already-completed queries and
    sources when resuming, avoiding duplicate API calls and wasted tokens.
    """

    task_id: str
    phase: str  # "researching" | "compressing"
    messages_snapshot: list[dict[str, Any]] = field(default_factory=list)
    tool_call_iterations: int = 0
    completed_queries: list[str] = field(default_factory=list)
    fetched_sources: list[str] = field(default_factory=list)
    compressed_research: Optional[str] = None
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "task_id": self.task_id,
            "phase": self.phase,
            "messages_snapshot": self.messages_snapshot,
            "tool_call_iterations": self.tool_call_iterations,
            "completed_queries": self.completed_queries,
            "fetched_sources": self.fetched_sources,
            "compressed_research": self.compressed_research,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ResearcherCheckpoint":
        """Deserialize from a dictionary."""
        return cls(
            task_id=data.get("task_id", ""),
            phase=data.get("phase", "researching"),
            messages_snapshot=data.get("messages_snapshot", []),
            tool_call_iterations=data.get("tool_call_iterations", 0),
            completed_queries=data.get("completed_queries", []),
            fetched_sources=data.get("fetched_sources", []),
            compressed_research=data.get("compressed_research"),
            timestamp=data.get("timestamp", time.time()),
        )


class CheckpointManager:
    """Persists and loads :class:`ResearcherCheckpoint` instances to/from disk.

    Checkpoints are stored as JSON files under
    ``{runs_dir}/{run_id}/checkpoints/{task_id}.json``.
    """

    def __init__(self, runs_dir: str, run_id: str) -> None:
        self._dir = os.path.join(runs_dir, run_id, "checkpoints")
        os.makedirs(self._dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    def _path(self, task_id: str) -> str:
        return os.path.join(self._dir, f"{task_id}.json")

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def save(self, checkpoint: ResearcherCheckpoint) -> None:
        """Write (or overwrite) a checkpoint for the given task."""
        checkpoint.timestamp = time.time()
        with open(self._path(checkpoint.task_id), "w", encoding="utf-8") as fh:
            json.dump(checkpoint.to_dict(), fh, ensure_ascii=False, indent=2)

    def load(self, task_id: str) -> Optional[ResearcherCheckpoint]:
        """Load the checkpoint for *task_id*, or ``None`` if it doesn't exist."""
        path = self._path(task_id)
        if not os.path.isfile(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as fh:
                return ResearcherCheckpoint.from_dict(json.load(fh))
        except (json.JSONDecodeError, KeyError):
            return None

    def delete(self, task_id: str) -> None:
        """Remove the checkpoint file for *task_id* (e.g. after successful completion)."""
        path = self._path(task_id)
        if os.path.isfile(path):
            os.remove(path)
