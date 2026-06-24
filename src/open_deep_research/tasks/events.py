"""Structured JSONL event logging system for async SubAgent research.

Provides event type enumerations, a Pydantic event model, and an append-only
JSONL writer that persists one file per research run under the configured runs_dir.
"""

import os
import time
import uuid
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class EventType(str, Enum):
    """Well-known event types emitted throughout the research lifecycle."""

    TASK_CREATED = "task.created"
    TASK_STARTED = "task.started"
    TASK_PHASE_CHANGE = "task.phase_change"
    TASK_TOOL_CALL = "task.tool_call"
    TASK_TOOL_RESULT = "task.tool_result"
    TASK_COMPLETED = "task.completed"
    TASK_FAILED = "task.failed"
    TASK_CANCELLED = "task.cancelled"
    TASK_TIMED_OUT = "task.timed_out"
    TASK_UPDATED = "task.updated"
    TASK_DOMAIN_CONFIRMATION_REQUESTED = "task.domain_confirmation_requested"
    TASK_DOMAIN_DECISION = "task.domain_decision"
    TASK_NOTIFICATION_FAILED = "task.notification_failed"
    CHECKPOINT_SAVED = "checkpoint.saved"
    CHECKPOINT_LOADED = "checkpoint.loaded"
    MESSAGES_SUMMARIZED = "messages.summarized"

    # Mem0 long-term memory events (lead-agent only, privacy-preserving)
    MEMORY_RECALLED = "memory.recalled"
    MEMORY_CANDIDATE_EXTRACTED = "memory.candidate_extracted"
    MEMORY_WRITTEN = "memory.written"
    MEMORY_SKIPPED = "memory.skipped"
    MEMORY_FAILED = "memory.failed"

    # Docker sandbox lifecycle events
    SANDBOX_WORKSPACE_CREATED = "sandbox.workspace_created"
    SANDBOX_CONTAINER_CREATED = "sandbox.container_created"
    SANDBOX_CONTAINER_STARTED = "sandbox.container_started"
    SANDBOX_NETWORK_POLICY_APPLIED = "sandbox.network_policy_applied"
    SANDBOX_OUTPUT_COLLECTED = "sandbox.output_collected"
    SANDBOX_TMP_CLEANED = "sandbox.tmp_cleaned"
    SANDBOX_CONTAINER_REMOVED = "sandbox.container_removed"
    SANDBOX_FAILED = "sandbox.failed"


class ResearchEvent(BaseModel):
    """A single structured event emitted during a research task.

    Every event carries enough context to reconstruct the task timeline
    without reading LangGraph internal state.
    """

    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    event_type: EventType
    task_id: str
    run_id: str
    timestamp: float = Field(default_factory=time.time)
    phase: Optional[str] = None
    data: dict[str, Any] = Field(default_factory=dict)


class JSONLEventWriter:
    """Append-only JSONL writer that creates one file per research run.

    Usage::

        writer = JSONLEventWriter(run_id="abc123", runs_dir=".runs")
        writer.write(ResearchEvent(
            event_type=EventType.TASK_STARTED,
            task_id="task-1",
            run_id="abc123",
            phase="researching",
        ))
        writer.close()
    """

    def __init__(self, run_id: str, runs_dir: str = ".runs") -> None:
        """Open the run-scoped JSONL file for appending."""
        self.run_id = run_id
        self.run_dir = os.path.join(runs_dir, run_id)
        os.makedirs(self.run_dir, exist_ok=True)

        self._file_path = os.path.join(self.run_dir, "events.jsonl")
        self._file = open(self._file_path, "a", buffering=1)

    def write(self, event: ResearchEvent) -> None:
        """Write a single event as a JSON line (appended immediately)."""
        self._file.write(event.model_dump_json() + "\n")

    def close(self) -> None:
        """Flush and close the underlying file handle."""
        if not self._file.closed:
            self._file.close()

    def __enter__(self) -> "JSONLEventWriter":
        """Return this writer for context-manager usage."""
        return self

    def __exit__(self, *args: Any) -> None:
        """Close the writer when leaving a context-manager block."""
        self.close()

    @property
    def file_path(self) -> str:
        """Return the JSONL file path."""
        return self._file_path
