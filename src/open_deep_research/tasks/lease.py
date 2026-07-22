"""Single-owner file lease for a run's Lead orchestrator."""

from __future__ import annotations

import asyncio
import os
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

import portalocker
from pydantic import BaseModel

from open_deep_research.tasks.mailbox import (
    atomic_write_json,
    read_json_file,
    validate_component,
)

PROCESS_INSTANCE_ID = f"{os.getpid()}-{uuid.uuid4()}"
_INSTANCE_PID = os.getpid()
_T = TypeVar("_T")


class LeaseConflictError(RuntimeError):
    """Raised when another live owner holds the run lease."""


class FenceLostError(RuntimeError):
    """Raised when an operation belongs to an expired or superseded epoch."""


def current_owner_id() -> str:
    """Return an owner identity regenerated after fork/process change."""
    global PROCESS_INSTANCE_ID, _INSTANCE_PID
    current_pid = os.getpid()
    if current_pid != _INSTANCE_PID:
        PROCESS_INSTANCE_ID = f"{current_pid}-{uuid.uuid4()}"
        _INSTANCE_PID = current_pid
    return PROCESS_INSTANCE_ID


class LeaderLease(BaseModel):
    """Persistent owner lease for one research run."""

    run_id: str
    owner_instance_id: str
    pid: int
    acquired_at: float
    heartbeat_at: float
    lease_expires_at: float
    fence_token: int = 0


class LeaderLeaseManager:
    """Acquire and renew a cross-process Lead lease."""

    def __init__(
        self,
        *,
        runs_dir: str,
        run_id: str,
        lease_seconds: float = 15,
        lock_timeout: float = 5,
        owner_id: str | None = None,
    ) -> None:
        """Initialize the run-scoped lease paths and timing policy."""
        self.run_id = validate_component(run_id, "run_id")
        self.root = Path(runs_dir).resolve() / self.run_id / "coordination"
        self.path = self.root / "leader_lease.json"
        self.lock_path = self.root / "leader_lease.lock"
        self.lease_seconds = lease_seconds
        self.lock_timeout = lock_timeout
        self.owner_id = owner_id or current_owner_id()
        self.fence_token: int | None = None

    def _locked_update(self, operation):
        self.root.mkdir(parents=True, exist_ok=True)
        with portalocker.Lock(str(self.lock_path), mode="a+b", timeout=self.lock_timeout):
            current = None
            if self.path.exists():
                current = LeaderLease.model_validate(read_json_file(self.path))
            result = operation(current)
            if result is not None:
                atomic_write_json(self.path, result.model_dump(mode="json"))
            return result

    async def acquire(self) -> LeaderLease:
        """Acquire an absent/expired lease or refresh the current live epoch."""
        def operation(current: LeaderLease | None) -> LeaderLease:
            now = time.time()
            current_is_live = current is not None and current.lease_expires_at > now
            if current is not None and current_is_live and current.owner_instance_id != self.owner_id:
                raise LeaseConflictError(
                    f"Run {self.run_id} is owned by {current.owner_instance_id}"
                )
            same_live_epoch = (
                current is not None
                and current_is_live
                and current.owner_instance_id == self.owner_id
            )
            acquired_at = current.acquired_at if same_live_epoch and current else now
            fence_token = (
                current.fence_token
                if same_live_epoch and current
                else (current.fence_token if current else 0) + 1
            )
            return LeaderLease(
                run_id=self.run_id,
                owner_instance_id=self.owner_id,
                pid=os.getpid(),
                acquired_at=acquired_at,
                heartbeat_at=now,
                lease_expires_at=now + self.lease_seconds,
                fence_token=fence_token,
            )

        lease = await asyncio.to_thread(self._locked_update, operation)
        self.fence_token = lease.fence_token
        return lease

    async def renew(
        self,
        *,
        expected_fence_token: int | None = None,
    ) -> LeaderLease:
        """Renew only the current unexpired ownership epoch."""
        token = expected_fence_token if expected_fence_token is not None else self.fence_token

        def operation(current: LeaderLease | None) -> LeaderLease:
            now = time.time()
            if current is None or current.owner_instance_id != self.owner_id:
                raise FenceLostError(f"Lost Lead lease for run {self.run_id}")
            if token is None or current.fence_token != token:
                raise FenceLostError(f"Lost fence token for run {self.run_id}")
            if current.lease_expires_at <= now:
                raise FenceLostError(f"Lead lease expired for run {self.run_id}")
            current.heartbeat_at = now
            current.lease_expires_at = now + self.lease_seconds
            return current

        lease = await asyncio.to_thread(self._locked_update, operation)
        self.fence_token = lease.fence_token
        return lease

    async def is_owner(
        self,
        *,
        expected_fence_token: int | None = None,
    ) -> bool:
        """Return whether this owner holds the expected unexpired epoch."""
        def operation(current: LeaderLease | None) -> LeaderLease | None:
            return current

        lease = await asyncio.to_thread(self._locked_update, operation)
        token = expected_fence_token if expected_fence_token is not None else self.fence_token
        return (
            lease is not None
            and lease.owner_instance_id == self.owner_id
            and token is not None
            and lease.fence_token == token
            and lease.lease_expires_at > time.time()
        )

    async def run_fenced(self, fence_token: int, operation: Callable[[], _T]) -> _T:
        """Run a short commit while the exact live lease epoch is locked."""
        def fenced(current: LeaderLease | None) -> LeaderLease | None:
            if (
                current is None
                or current.owner_instance_id != self.owner_id
                or current.fence_token != fence_token
                or current.lease_expires_at <= time.time()
            ):
                raise FenceLostError(f"Lost Lead lease for run {self.run_id}")
            operation_result.append(operation())
            return current

        operation_result: list[_T] = []
        await asyncio.to_thread(self._locked_update, fenced)
        return operation_result[0]

    async def release(self, *, expected_fence_token: int | None = None) -> None:
        """Expire only the exact ownership epoch held by this manager."""
        token = expected_fence_token if expected_fence_token is not None else self.fence_token

        def operation(current: LeaderLease | None) -> LeaderLease | None:
            if (
                current is None
                or current.owner_instance_id != self.owner_id
                or token is None
                or current.fence_token != token
            ):
                raise FenceLostError(f"Lost Lead lease for run {self.run_id}")
            current.lease_expires_at = 0
            current.heartbeat_at = time.time()
            return current

        await asyncio.to_thread(self._locked_update, operation)
        self.fence_token = None
