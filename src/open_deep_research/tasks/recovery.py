"""Checkpoint-based recovery for async SubAgent research tasks.

When a background researcher task fails or is cancelled, the last saved
checkpoint allows a retry to resume from the most recent stable state
rather than starting from scratch.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from open_deep_research.tasks.mailbox import (
    atomic_write_json,
    read_json_file,
    validate_component,
)


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
    research_topic: str = ""
    run_id: str = ""
    user_id: Optional[str] = None
    memory_context: Optional[str] = None
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
            "research_topic": self.research_topic,
            "run_id": self.run_id,
            "user_id": self.user_id,
            "memory_context": self.memory_context,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ResearcherCheckpoint:
        """Deserialize from a dictionary."""
        return cls(
            task_id=data.get("task_id", ""),
            phase=data.get("phase", "researching"),
            messages_snapshot=data.get("messages_snapshot", []),
            tool_call_iterations=data.get("tool_call_iterations", 0),
            completed_queries=data.get("completed_queries", []),
            fetched_sources=data.get("fetched_sources", []),
            compressed_research=data.get("compressed_research"),
            research_topic=data.get("research_topic", ""),
            run_id=data.get("run_id", ""),
            user_id=data.get("user_id"),
            memory_context=data.get("memory_context"),
            timestamp=data.get("timestamp", time.time()),
        )


class CheckpointManager:
    """Persists and loads :class:`ResearcherCheckpoint` instances to/from disk.

    Checkpoints are stored as JSON files under
    ``{runs_dir}/{run_id}/checkpoints/{task_id}.json``.
    """

    def __init__(self, runs_dir: str, run_id: str) -> None:
        """Initialize the run-scoped checkpoint directory."""
        root = Path(runs_dir).resolve()
        safe_run_id = validate_component(run_id, "run_id")
        self._dir = root / safe_run_id / "checkpoints"
        self._dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    def _path(self, task_id: str) -> Path:
        safe_task_id = validate_component(task_id, "task_id")
        return self._dir / f"{safe_task_id}.json"

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def save(self, checkpoint: ResearcherCheckpoint) -> None:
        """Write (or overwrite) a checkpoint for the given task."""
        checkpoint.timestamp = time.time()
        atomic_write_json(self._path(checkpoint.task_id), checkpoint.to_dict())

    def load(self, task_id: str) -> Optional[ResearcherCheckpoint]:
        """Load the checkpoint for *task_id*, or ``None`` if it doesn't exist."""
        path = self._path(task_id)
        if not path.is_file():
            return None
        return ResearcherCheckpoint.from_dict(read_json_file(path))

    def delete(self, task_id: str) -> None:
        """Remove the checkpoint file for *task_id* (e.g. after successful completion)."""
        path = self._path(task_id)
        if path.is_file():
            path.unlink()

    def list_checkpoints(self) -> list[ResearcherCheckpoint]:
        """Load every valid checkpoint for this run."""
        checkpoints: list[ResearcherCheckpoint] = []
        if not self._dir.is_dir():
            return checkpoints
        for path in self._dir.iterdir():
            if path.suffix != ".json":
                continue
            checkpoint = self.load(path.stem)
            if checkpoint is not None:
                checkpoints.append(checkpoint)
        return checkpoints
