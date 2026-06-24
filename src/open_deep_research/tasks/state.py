"""Shared task-state stores for async SubAgent orchestration."""

from __future__ import annotations

import json
import os
import time
from abc import ABC, abstractmethod
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

from open_deep_research.tasks.registry import TaskPhase, TaskRecord, TaskStatus


class TaskSnapshot(BaseModel):
    """Serializable latest-state snapshot for a research task."""

    task_id: str
    run_id: str = ""
    user_id: Optional[str] = None
    status: TaskStatus = TaskStatus.PENDING
    phase: TaskPhase = TaskPhase.RESEARCHING
    research_topic: str = ""
    created_at: float = Field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    updated_at: float = Field(default_factory=time.time)
    version: int = 1
    result: Optional[dict[str, Any]] = None
    error_message: Optional[str] = None
    pending_domain: Optional[str] = None
    pending_domain_tool: Optional[str] = None
    metrics: dict[str, int] = Field(default_factory=dict)
    sandbox: dict[str, Any] = Field(default_factory=dict)

    @property
    def elapsed_seconds(self) -> float:
        """Wall-clock seconds covered by this snapshot."""
        end = self.completed_at or time.time()
        return max(0.0, end - self.created_at)

    @classmethod
    def from_record(
        cls, record: TaskRecord, *, version: int = 1, updated_at: Optional[float] = None
    ) -> TaskSnapshot:
        """Build a serializable snapshot from an in-process runtime record."""
        return cls(
            task_id=record.task_id,
            run_id=record.run_id,
            user_id=record.user_id,
            status=record.status,
            phase=record.phase,
            research_topic=record.research_topic,
            created_at=record.created_at,
            started_at=record.started_at,
            completed_at=record.completed_at,
            updated_at=updated_at or time.time(),
            version=version,
            result=record.result,
            error_message=record.error_message,
            pending_domain=record.pending_domain,
            pending_domain_tool=record.pending_domain_tool,
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
    """Async interface for latest task-state persistence."""

    @abstractmethod
    async def upsert(self, snapshot: TaskSnapshot) -> TaskSnapshot:
        """Create or replace a task snapshot."""

    async def update_from_record(self, record: TaskRecord) -> TaskSnapshot:
        """Persist a runtime record, incrementing the stored snapshot version."""
        current = await self.get(record.task_id)
        version = (current.version + 1) if current else 1
        snapshot = TaskSnapshot.from_record(record, version=version)
        return await self.upsert(snapshot)

    @abstractmethod
    async def get(self, task_id: str) -> Optional[TaskSnapshot]:
        """Return a task snapshot by id."""

    @abstractmethod
    async def list(
        self,
        *,
        status_filter: Optional[TaskStatus] = None,
        run_id: Optional[str] = None,
    ) -> list[TaskSnapshot]:
        """Return snapshots, optionally filtered by run and status."""

    async def count_running(self, *, run_id: Optional[str] = None) -> int:
        """Count running tasks."""
        return len(await self.list(status_filter=TaskStatus.RUNNING, run_id=run_id))

    async def count_active(self, *, run_id: Optional[str] = None) -> int:
        """Count tasks that are RUNNING or WAITING_FOR_CONFIRMATION (holding a slot)."""
        running = await self.list(status_filter=TaskStatus.RUNNING, run_id=run_id)
        waiting = await self.list(
            status_filter=TaskStatus.WAITING_FOR_CONFIRMATION, run_id=run_id
        )
        return len(running) + len(waiting)

    async def collect_completed(self, *, run_id: Optional[str] = None) -> list[TaskSnapshot]:
        """Return completed tasks."""
        return await self.list(status_filter=TaskStatus.COMPLETED, run_id=run_id)


class MemoryTaskStateStore(TaskStateStore):
    """In-process task-state store used for tests and local development."""

    def __init__(self) -> None:
        """Initialize an empty in-process snapshot dictionary."""
        self._snapshots: dict[str, TaskSnapshot] = {}

    async def upsert(self, snapshot: TaskSnapshot) -> TaskSnapshot:
        """Create or replace a snapshot."""
        self._snapshots[snapshot.task_id] = snapshot
        return snapshot

    async def get(self, task_id: str) -> Optional[TaskSnapshot]:
        """Return a snapshot by task id."""
        return self._snapshots.get(task_id)

    async def list(
        self,
        *,
        status_filter: Optional[TaskStatus] = None,
        run_id: Optional[str] = None,
    ) -> list[TaskSnapshot]:
        """Return snapshots filtered by run and status."""
        snapshots = list(self._snapshots.values())
        if run_id is not None:
            snapshots = [s for s in snapshots if s.run_id == run_id]
        if status_filter is not None:
            snapshots = [s for s in snapshots if s.status == status_filter]
        return snapshots

    def clear(self) -> None:
        """Remove all snapshots."""
        self._snapshots.clear()


class RedisTaskStateStore(TaskStateStore):
    """Redis-backed latest-state store."""

    def __init__(self, redis_url: str, *, ttl_seconds: Optional[int] = None) -> None:
        """Initialize the Redis connection settings."""
        self.redis_url = redis_url
        self.ttl_seconds = ttl_seconds
        self._client = None

    def _redis(self):
        if self._client is None:
            try:
                from redis import asyncio as redis_async
            except ImportError as exc:
                raise RuntimeError("Install redis>=5 to use Redis task state.") from exc
            self._client = redis_async.from_url(self.redis_url, decode_responses=True)
        return self._client

    @staticmethod
    def _task_key(task_id: str) -> str:
        return f"odr:task:{task_id}"

    @staticmethod
    def _run_key(run_id: str) -> str:
        return f"odr:run:{run_id}:tasks"

    async def upsert(self, snapshot: TaskSnapshot) -> TaskSnapshot:
        """Create or replace a Redis snapshot."""
        client = self._redis()
        payload = snapshot.model_dump_json()
        key = self._task_key(snapshot.task_id)
        await client.set(key, payload, ex=self.ttl_seconds)
        await client.sadd(self._run_key(snapshot.run_id), snapshot.task_id)
        if self.ttl_seconds:
            await client.expire(self._run_key(snapshot.run_id), self.ttl_seconds)
        return snapshot

    async def get(self, task_id: str) -> Optional[TaskSnapshot]:
        """Return a snapshot by task id."""
        payload = await self._redis().get(self._task_key(task_id))
        if not payload:
            return None
        return TaskSnapshot.model_validate_json(payload)

    async def list(
        self,
        *,
        status_filter: Optional[TaskStatus] = None,
        run_id: Optional[str] = None,
    ) -> list[TaskSnapshot]:
        """Return Redis snapshots filtered by run and status."""
        client = self._redis()
        if run_id is not None:
            task_ids = await client.smembers(self._run_key(run_id))
            keys = [self._task_key(task_id) for task_id in task_ids]
        else:
            keys = [key async for key in client.scan_iter("odr:task:*")]
        if not keys:
            return []
        payloads = await client.mget(keys)
        snapshots = [
            TaskSnapshot.model_validate_json(payload)
            for payload in payloads
            if payload
        ]
        if status_filter is not None:
            snapshots = [s for s in snapshots if s.status == status_filter]
        return snapshots


class PostgresTaskStateStore(TaskStateStore):
    """Postgres-backed latest-state store."""

    def __init__(self, conninfo: str) -> None:
        """Initialize the Postgres connection settings."""
        self.conninfo = conninfo
        self._pool = None
        self._setup_done = False

    async def _ensure_pool(self):
        if self._pool is None:
            try:
                from psycopg_pool import AsyncConnectionPool
            except ImportError as exc:
                raise RuntimeError(
                    "Postgres task state requires psycopg_pool. "
                    "Run `uv sync` after installing dependencies."
                ) from exc
            self._pool = AsyncConnectionPool(
                conninfo=self.conninfo,
                open=False,
                kwargs={"autocommit": True, "prepare_threshold": 0},
            )
            await self._pool.open()
        if not self._setup_done:
            await self.setup()
            self._setup_done = True
        return self._pool

    async def setup(self) -> None:
        """Create the task-state table and indexes if needed."""
        pool = self._pool
        if pool is None:
            return
        async with pool.connection() as conn:
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS research_task_state (
                    task_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    user_id TEXT,
                    status TEXT NOT NULL,
                    phase TEXT NOT NULL,
                    research_topic TEXT NOT NULL,
                    snapshot_json JSONB NOT NULL,
                    result_json JSONB,
                    updated_at DOUBLE PRECISION NOT NULL,
                    created_at DOUBLE PRECISION NOT NULL,
                    completed_at DOUBLE PRECISION
                )
                """
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_research_task_state_run_id "
                "ON research_task_state(run_id)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_research_task_state_run_status "
                "ON research_task_state(run_id, status)"
            )

    async def upsert(self, snapshot: TaskSnapshot) -> TaskSnapshot:
        """Create or replace a Postgres snapshot row."""
        pool = await self._ensure_pool()
        snapshot_json = snapshot.model_dump_json()
        result_json = json.dumps(snapshot.result) if snapshot.result is not None else None
        async with pool.connection() as conn:
            await conn.execute(
                """
                INSERT INTO research_task_state (
                    task_id, run_id, user_id, status, phase, research_topic,
                    snapshot_json, result_json, updated_at, created_at, completed_at
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb, %s, %s, %s
                )
                ON CONFLICT (task_id) DO UPDATE SET
                    run_id = EXCLUDED.run_id,
                    user_id = EXCLUDED.user_id,
                    status = EXCLUDED.status,
                    phase = EXCLUDED.phase,
                    research_topic = EXCLUDED.research_topic,
                    snapshot_json = EXCLUDED.snapshot_json,
                    result_json = EXCLUDED.result_json,
                    updated_at = EXCLUDED.updated_at,
                    created_at = EXCLUDED.created_at,
                    completed_at = EXCLUDED.completed_at
                """,
                (
                    snapshot.task_id,
                    snapshot.run_id,
                    snapshot.user_id,
                    snapshot.status.value,
                    snapshot.phase.value,
                    snapshot.research_topic,
                    snapshot_json,
                    result_json,
                    snapshot.updated_at,
                    snapshot.created_at,
                    snapshot.completed_at,
                ),
            )
        return snapshot

    async def get(self, task_id: str) -> Optional[TaskSnapshot]:
        """Return a snapshot by task id."""
        pool = await self._ensure_pool()
        async with pool.connection() as conn:
            row = await conn.execute(
                "SELECT snapshot_json FROM research_task_state WHERE task_id = %s",
                (task_id,),
            )
            result = await row.fetchone()
        if result is None:
            return None
        return TaskSnapshot.model_validate(result[0])

    async def list(
        self,
        *,
        status_filter: Optional[TaskStatus] = None,
        run_id: Optional[str] = None,
    ) -> list[TaskSnapshot]:
        """Return Postgres snapshots filtered by run and status."""
        pool = await self._ensure_pool()
        clauses = []
        params: list[Any] = []
        if run_id is not None:
            clauses.append("run_id = %s")
            params.append(run_id)
        if status_filter is not None:
            clauses.append("status = %s")
            params.append(status_filter.value)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        async with pool.connection() as conn:
            rows = await conn.execute(
                f"SELECT snapshot_json FROM research_task_state {where} ORDER BY created_at",
                params,
            )
            results = await rows.fetchall()
        return [TaskSnapshot.model_validate(row[0]) for row in results]


_memory_store = MemoryTaskStateStore()
_store_cache: dict[tuple[Any, ...], TaskStateStore] = {}


def resolve_task_state_postgres_uri(configurable: Any) -> Optional[str]:
    """Resolve the Postgres URI for task state using project env conventions."""
    return (
        getattr(configurable, "task_state_postgres_uri", None)
        or os.getenv("TASK_STATE_POSTGRES_URI")
        or os.getenv("LANGGRAPH_POSTGRES_URI")
        or os.getenv("POSTGRES_URI")
        or os.getenv("DATABASE_URL")
    )


def get_task_state_store(configurable: Any) -> TaskStateStore:
    """Return the configured latest-state store."""
    backend: Literal["memory", "redis", "postgres"] = getattr(
        configurable, "task_state_backend", "memory"
    )
    if backend == "memory":
        return _memory_store

    if backend == "redis":
        redis_url = getattr(configurable, "redis_url", None) or os.getenv("REDIS_URL")
        if not redis_url:
            raise RuntimeError("task_state_backend='redis' requires REDIS_URL or redis_url.")
        ttl = getattr(configurable, "task_state_ttl_seconds", None)
        key = ("redis", redis_url, ttl)
        if key not in _store_cache:
            _store_cache[key] = RedisTaskStateStore(redis_url, ttl_seconds=ttl)
        return _store_cache[key]

    if backend == "postgres":
        conninfo = resolve_task_state_postgres_uri(configurable)
        if not conninfo:
            raise RuntimeError(
                "task_state_backend='postgres' requires TASK_STATE_POSTGRES_URI, "
                "LANGGRAPH_POSTGRES_URI, POSTGRES_URI, or DATABASE_URL."
            )
        key = ("postgres", conninfo)
        if key not in _store_cache:
            _store_cache[key] = PostgresTaskStateStore(conninfo)
        return _store_cache[key]

    raise ValueError(f"Unsupported task_state_backend: {backend}")


def reset_memory_task_state_store() -> None:
    """Clear the process-local memory store, for tests."""
    _memory_store.clear()
