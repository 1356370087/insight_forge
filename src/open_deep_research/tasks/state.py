"""File-backed latest-state snapshots for asynchronous research tasks."""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, List, Optional

import portalocker
from pydantic import BaseModel, Field

from open_deep_research.tasks.mailbox import (
    atomic_write_json,
    read_json_file,
    validate_component,
)
from open_deep_research.tasks.registry import TaskPhase, TaskRecord, TaskStatus


class TaskSnapshot(BaseModel):
    """Serializable, versioned latest-state snapshot for a research task."""

    task_id: str
    run_id: str = ""
    user_id: Optional[str] = None
    memory_context: Optional[str] = None
    status: TaskStatus = TaskStatus.PENDING
    phase: TaskPhase = TaskPhase.RESEARCHING
    research_topic: str = ""
    display_title: str = ""
    wave_id: str = ""
    plan_task_id: str = ""
    created_at: float = Field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    updated_at: float = Field(default_factory=time.time)
    version: int = 1
    result: Optional[dict[str, Any]] = None
    error_message: Optional[str] = None
    pending_domain: Optional[str] = None
    pending_domain_tool: Optional[str] = None
    assigned_teammate_id: Optional[str] = None
    assignment_attempt: int = 0
    pending_update_instructions: list[str] = Field(default_factory=list)
    admission_status: str = "pending"
    result_artifact_path: Optional[str] = None
    result_artifact_sha256: Optional[str] = None
    trace_parent_span_id: Optional[str] = None
    langfuse_parent_span_id: Optional[str] = None
    metrics: dict[str, int] = Field(default_factory=dict)
    sandbox: dict[str, Any] = Field(default_factory=dict)

    @property
    def elapsed_seconds(self) -> float:
        """Wall-clock seconds covered by this snapshot."""
        return max(0.0, (self.completed_at or time.time()) - self.created_at)

    @classmethod
    def from_record(
        cls,
        record: TaskRecord,
        *,
        version: int = 1,
        updated_at: Optional[float] = None,
    ) -> TaskSnapshot:
        """Build a serializable snapshot from the live runtime record."""
        return cls(
            task_id=record.task_id,
            run_id=record.run_id or "default",
            user_id=record.user_id,
            memory_context=record.memory_context,
            status=record.status,
            phase=record.phase,
            research_topic=record.research_topic,
            display_title=record.display_title,
            wave_id=record.wave_id,
            plan_task_id=record.plan_task_id,
            created_at=record.created_at,
            started_at=record.started_at,
            completed_at=record.completed_at,
            updated_at=updated_at or time.time(),
            version=version,
            result=record.result,
            error_message=record.error_message,
            pending_domain=record.pending_domain,
            pending_domain_tool=record.pending_domain_tool,
            assigned_teammate_id=record.assigned_teammate_id,
            assignment_attempt=record.assignment_attempt,
            pending_update_instructions=list(record.pending_update_instructions),
            admission_status=record.admission_status,
            result_artifact_path=record.result_artifact_path,
            result_artifact_sha256=record.result_artifact_sha256,
            trace_parent_span_id=record.trace_parent_span_id,
            langfuse_parent_span_id=record.langfuse_parent_span_id,
            metrics={
                "query_count": record.query_count,
                "source_count": record.source_count,
                "citation_count": record.citation_count,
                "retry_count": record.retry_count,
            },
            sandbox={
                "enabled": record.sandbox_enabled,
                "workspace_path": record.workspace_path,
                "container_id": record.container_id,
                "network_mode": record.sandbox_network_mode,
                "output_archive_path": record.output_archive_path,
                "last_event": record.last_sandbox_event,
            },
        )


class TaskStateStore(ABC):
    """Async interface for authoritative latest task state."""

    @abstractmethod
    async def upsert(self, snapshot: TaskSnapshot) -> TaskSnapshot:
        """Create or replace a task snapshot."""

    async def update_from_record(self, record: TaskRecord) -> TaskSnapshot:
        """Persist a runtime record with a monotonic version."""
        current = await self.get(record.task_id, run_id=record.run_id or "default")
        snapshot = TaskSnapshot.from_record(record, version=(current.version + 1) if current else 1)
        return await self.upsert(snapshot)

    @abstractmethod
    async def get(
        self, task_id: str, *, run_id: Optional[str] = None
    ) -> Optional[TaskSnapshot]:
        """Return one task snapshot."""

    @abstractmethod
    async def list(
        self,
        *,
        status_filter: Optional[TaskStatus] = None,
        run_id: Optional[str] = None,
    ) -> List[TaskSnapshot]:
        """List snapshots filtered by run and status."""

    async def count_running(self, *, run_id: Optional[str] = None) -> int:
        """Count running tasks."""
        return len(await self.list(status_filter=TaskStatus.RUNNING, run_id=run_id))

    async def count_active(self, *, run_id: Optional[str] = None) -> int:
        """Count admitted tasks that hold a run concurrency slot."""
        pending = await self.list(status_filter=TaskStatus.PENDING, run_id=run_id)
        running = await self.list(status_filter=TaskStatus.RUNNING, run_id=run_id)
        waiting = await self.list(status_filter=TaskStatus.WAITING_FOR_CONFIRMATION, run_id=run_id)
        return len(pending) + len(running) + len(waiting)

    async def collect_completed(self, *, run_id: Optional[str] = None) -> List[TaskSnapshot]:
        """Return completed tasks."""
        return await self.list(status_filter=TaskStatus.COMPLETED, run_id=run_id)


class MemoryTaskStateStore(TaskStateStore):
    """Process-local state store reserved for unit tests."""

    def __init__(self) -> None:
        """Initialize an empty test store."""
        self._snapshots: dict[str, TaskSnapshot] = {}

    async def upsert(self, snapshot: TaskSnapshot) -> TaskSnapshot:
        """Store one snapshot in process memory."""
        self._snapshots[snapshot.task_id] = snapshot
        return snapshot

    async def get(
        self, task_id: str, *, run_id: Optional[str] = None
    ) -> Optional[TaskSnapshot]:
        """Return one process-local snapshot."""
        snapshot = self._snapshots.get(task_id)
        if snapshot is None or (run_id and snapshot.run_id != run_id):
            return None
        return snapshot

    async def list(
        self,
        *,
        status_filter: Optional[TaskStatus] = None,
        run_id: Optional[str] = None,
    ) -> List[TaskSnapshot]:
        """List process-local snapshots with optional filters."""
        snapshots = list(self._snapshots.values())
        if run_id:
            snapshots = [snapshot for snapshot in snapshots if snapshot.run_id == run_id]
        if status_filter is not None:
            snapshots = [snapshot for snapshot in snapshots if snapshot.status == status_filter]
        return snapshots

    def clear(self) -> None:
        """Remove every process-local snapshot."""
        self._snapshots.clear()


class FileTaskStateStore(TaskStateStore):
    """Cross-process snapshots stored as one locked JSON file per task."""

    def __init__(self, runs_dir: str, *, lock_timeout_seconds: float = 5) -> None:
        """Initialize the shared runs directory and lock timeout."""
        self.runs_dir = Path(runs_dir).resolve()
        self.lock_timeout_seconds = lock_timeout_seconds

    def _paths(self, run_id: str, task_id: str) -> tuple[Path, Path]:
        run = validate_component(run_id, "run_id")
        task = validate_component(task_id, "task_id")
        directory = self.runs_dir / run / "coordination" / "tasks"
        return directory / f"{task}.json", directory / f"{task}.lock"

    def _upsert_sync(self, snapshot: TaskSnapshot) -> TaskSnapshot:
        path, lock_path = self._paths(snapshot.run_id, snapshot.task_id)
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with portalocker.Lock(str(lock_path), mode="a+b", timeout=self.lock_timeout_seconds):
            if path.exists():
                current = TaskSnapshot.model_validate(read_json_file(path))
                terminal_statuses = {
                    TaskStatus.COMPLETED,
                    TaskStatus.FAILED,
                    TaskStatus.CANCELLED,
                    TaskStatus.TIMED_OUT,
                }
                if current.status in terminal_statuses and snapshot.status != current.status:
                    return current
                if snapshot.version <= current.version:
                    snapshot.version = current.version + 1
            atomic_write_json(path, snapshot.model_dump(mode="json"))
        return snapshot

    async def upsert(self, snapshot: TaskSnapshot) -> TaskSnapshot:
        """Atomically write one versioned task snapshot."""
        return await asyncio.to_thread(self._upsert_sync, snapshot)

    async def get(
        self, task_id: str, *, run_id: Optional[str] = None
    ) -> Optional[TaskSnapshot]:
        """Find a task snapshot, scoped to a run when supplied."""
        task = validate_component(task_id, "task_id")
        if run_id:
            path, _ = self._paths(run_id, task)
            if not path.exists():
                return None
            return TaskSnapshot.model_validate(await asyncio.to_thread(read_json_file, path))
        for path in self.runs_dir.glob(f"*/coordination/tasks/{task}.json"):
            return TaskSnapshot.model_validate(await asyncio.to_thread(read_json_file, path))
        return None

    async def list(
        self,
        *,
        status_filter: Optional[TaskStatus] = None,
        run_id: Optional[str] = None,
    ) -> List[TaskSnapshot]:
        """List file snapshots with optional run and status filters."""
        if run_id:
            run = validate_component(run_id, "run_id")
            paths = list((self.runs_dir / run / "coordination" / "tasks").glob("*.json"))
        else:
            paths = list(self.runs_dir.glob("*/coordination/tasks/*.json"))
        snapshots = [
            TaskSnapshot.model_validate(await asyncio.to_thread(read_json_file, path))
            for path in paths
        ]
        if status_filter is not None:
            snapshots = [snapshot for snapshot in snapshots if snapshot.status == status_filter]
        return sorted(snapshots, key=lambda snapshot: snapshot.created_at)


_memory_store = MemoryTaskStateStore()
_file_stores: dict[tuple[str, float], FileTaskStateStore] = {}


def get_task_state_store(configurable: Any) -> TaskStateStore:
    """Return the file store, or the explicit test-only memory store."""
    backend = getattr(configurable, "task_state_backend", "file")
    if backend == "memory":
        return _memory_store
    if backend != "file":
        raise ValueError(f"Unsupported task_state_backend: {backend}")
    runs_dir = str(getattr(configurable, "runs_dir", ".runs"))
    timeout = float(getattr(configurable, "mailbox_lock_timeout_seconds", 5))
    key = (str(Path(runs_dir).resolve()), timeout)
    if key not in _file_stores:
        _file_stores[key] = FileTaskStateStore(runs_dir, lock_timeout_seconds=timeout)
    return _file_stores[key]


def reset_memory_task_state_store() -> None:
    """Clear the test-only process-local store."""
    _memory_store.clear()
