"""In-memory task registry for async SubAgent lifecycle management.

The registry lives outside LangGraph state because it holds live ``asyncio.Task``
handles and ``asyncio.Event`` / ``asyncio.Queue`` objects that cannot be
JSON-serialised. A module-level singleton (``_registry``) is used so that all
nodes in the same process share the same task records.

Every :class:`TaskRecord` carries a ``run_id`` to prevent cross-run leakage.
Query methods accept an optional ``run_id`` filter; callers should always pass
the current run's ID to avoid mixing results from concurrent or prior runs.
"""

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class TaskStatus(str, Enum):
    """Lifecycle status of a single research task."""

    PENDING = "pending"
    RUNNING = "running"
    WAITING_FOR_CONFIRMATION = "waiting_for_confirmation"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"


class TaskPhase(str, Enum):
    """Coarse-grained execution phase within a research task."""

    RESEARCHING = "researching"
    COMPRESSING = "compressing"
    COMPLETED = "completed"


@dataclass
class TaskRecord:
    """All metadata for a single async research task.

    This record is mutated in-place by the background executor and read by the
    supervisor tool handlers.  It is intentionally **not** stored in LangGraph
    state — only the final ``result`` is written back when the supervisor
    collects completed outputs.
    """

    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    research_topic: str = ""

    # Isolation — prevents cross-run / cross-user contamination
    run_id: str = ""
    user_id: Optional[str] = None

    # Lifecycle
    status: TaskStatus = TaskStatus.PENDING
    phase: TaskPhase = TaskPhase.RESEARCHING

    # Timestamps (float = time.time())
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None

    # Output
    result: Optional[dict[str, Any]] = None  # {"compressed_research": ..., "raw_notes": [...]}
    assigned_teammate_id: Optional[str] = None
    admission_status: str = "pending"
    result_artifact_path: Optional[str] = None
    result_artifact_sha256: Optional[str] = None
    trace_parent_span_id: Optional[str] = None
    langfuse_parent_span_id: Optional[str] = None
    memory_context: Optional[str] = None
    assignment_attempt: int = 0
    pending_update_instructions: list[str] = field(default_factory=list)
    error_message: Optional[str] = None

    # Pending domain-confirmation (set while WAITING_FOR_CONFIRMATION).
    # The live asyncio.Future lives in DomainApprovalRegistry, not here — these
    # scalars are for snapshot display only.
    pending_domain: Optional[str] = None
    pending_domain_tool: Optional[str] = None

    # Docker sandbox metadata
    sandbox_enabled: bool = False
    workspace_path: Optional[str] = None
    container_id: Optional[str] = None
    sandbox_network_mode: Optional[str] = None
    output_archive_path: Optional[str] = None
    last_sandbox_event: Optional[str] = None

    # Runtime handles (NOT serialisable — live only in the registry)
    background_task: Optional[asyncio.Task[None]] = field(default=None, repr=False)
    control_queue: asyncio.Queue[dict[str, Any]] = field(
        default_factory=asyncio.Queue, repr=False
    )
    cancelled: asyncio.Event = field(default_factory=asyncio.Event, repr=False)

    # Quality metrics
    query_count: int = 0
    source_count: int = 0
    citation_count: int = 0
    retry_count: int = 0
    source_urls: set[str] = field(default_factory=set, repr=False)

    @property
    def elapsed_seconds(self) -> float:
        """Wall-clock time since creation (or until completion)."""
        if self.completed_at is not None:
            return self.completed_at - self.created_at
        return time.time() - self.created_at


class TaskRegistry:
    """Thread-unsafe, in-memory registry for async research tasks.

    Intended to be used as a **module-level singleton**.  All methods are
    synchronous because the registry is only accessed from the same event loop
    that owns the background tasks.

    Query methods accept an optional *run_id* to scope results to a single
    research run.  When omitted, all runs are included (useful for debugging).
    """

    def __init__(self) -> None:
        """Initialize an empty task registry."""
        self._tasks: dict[str, TaskRecord] = {}

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def create(
        self,
        research_topic: str,
        *,
        run_id: str = "",
        user_id: Optional[str] = None,
    ) -> TaskRecord:
        """Create a new PENDING task record and return it."""
        record = TaskRecord(
            research_topic=research_topic,
            run_id=run_id,
            user_id=user_id,
        )
        self._tasks[record.task_id] = record
        return record

    def get(self, task_id: str) -> Optional[TaskRecord]:
        """Return the record for *task_id*, or ``None``."""
        return self._tasks.get(task_id)

    def restore(self, record: TaskRecord) -> TaskRecord:
        """Register a record reconstructed from durable run state."""
        self._tasks[record.task_id] = record
        return record

    def list(
        self,
        status_filter: Optional[TaskStatus] = None,
        run_id: Optional[str] = None,
    ) -> list[TaskRecord]:
        """Return records, optionally filtered by status and/or run."""
        records = self._tasks.values()
        if run_id is not None:
            records = [t for t in records if t.run_id == run_id]
        if status_filter is not None:
            records = [t for t in records if t.status == status_filter]
        return list(records)

    def update_status(
        self, task_id: str, status: TaskStatus, **kwargs: Any
    ) -> None:
        """Atomically update the status (and optional extra fields) of a task."""
        record = self._tasks.get(task_id)
        if record is None:
            return
        record.status = status
        for key, value in kwargs.items():
            if hasattr(record, key):
                setattr(record, key, value)

    # ------------------------------------------------------------------
    # Convenience queries
    # ------------------------------------------------------------------

    def all_completed(self, run_id: Optional[str] = None) -> bool:
        """Return ``True`` when every task in *run_id* has reached a terminal status."""
        terminal = {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED, TaskStatus.TIMED_OUT}
        records = self.list(run_id=run_id) if run_id is not None else self._tasks.values()
        return all(t.status in terminal for t in records)

    def count_running(self, run_id: Optional[str] = None) -> int:
        """Count tasks currently RUNNING, optionally scoped to *run_id*."""
        records = self.list(run_id=run_id) if run_id is not None else self._tasks.values()
        return sum(1 for t in records if t.status == TaskStatus.RUNNING)

    def count_active(self, run_id: Optional[str] = None) -> int:
        """Count admitted tasks that hold a run concurrency slot.

        Paused-for-confirmation tasks still occupy resources, so this is the
        concurrency check used against ``max_in_flight_tasks``.
        """
        records = self.list(run_id=run_id) if run_id is not None else self._tasks.values()
        active = {
            TaskStatus.PENDING,
            TaskStatus.RUNNING,
            TaskStatus.WAITING_FOR_CONFIRMATION,
        }
        return sum(1 for t in records if t.status in active)

    def __len__(self) -> int:
        """Return the number of registered tasks."""
        return len(self._tasks)


# ------------------------------------------------------------------
# Module-level singleton — the single source of truth for the process.
# ------------------------------------------------------------------
_registry: Optional[TaskRegistry] = None


def get_task_registry() -> TaskRegistry:
    """Return the module-level :class:`TaskRegistry` singleton."""
    global _registry
    if _registry is None:
        _registry = TaskRegistry()
    return _registry
