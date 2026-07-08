"""Single-owner file lease for a run's Lead orchestrator."""

from __future__ import annotations

import asyncio
import os
import time
import uuid
from pathlib import Path

import portalocker
from pydantic import BaseModel

from open_deep_research.tasks.mailbox import (
    atomic_write_json,
    read_json_file,
    validate_component,
)

PROCESS_INSTANCE_ID = f"{os.getpid()}-{uuid.uuid4()}"


class LeaderLease(BaseModel):
    """Persistent owner lease for one research run."""

    run_id: str
    owner_instance_id: str
    pid: int
    acquired_at: float
    heartbeat_at: float
    lease_expires_at: float


class LeaderLeaseManager:
    """Acquire and renew a cross-process Lead lease."""

    def __init__(self, *, runs_dir: str, run_id: str, lease_seconds: float = 15, lock_timeout: float = 5) -> None:
        """Initialize the run-scoped lease paths and timing policy."""
        self.run_id = validate_component(run_id, "run_id")
        self.root = Path(runs_dir).resolve() / self.run_id / "coordination"
        self.path = self.root / "leader_lease.json"
        self.lock_path = self.root / "leader_lease.lock"
        self.lease_seconds = lease_seconds
        self.lock_timeout = lock_timeout
        self.owner_id = PROCESS_INSTANCE_ID

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
        """Acquire an absent/expired lease or refresh this process's lease."""
        def operation(current: LeaderLease | None) -> LeaderLease:
            now = time.time()
            if current and current.owner_instance_id != self.owner_id and current.lease_expires_at > now:
                raise RuntimeError(f"Run {self.run_id} is owned by {current.owner_instance_id}")
            acquired_at = current.acquired_at if current and current.owner_instance_id == self.owner_id else now
            return LeaderLease(
                run_id=self.run_id,
                owner_instance_id=self.owner_id,
                pid=os.getpid(),
                acquired_at=acquired_at,
                heartbeat_at=now,
                lease_expires_at=now + self.lease_seconds,
            )

        return await asyncio.to_thread(self._locked_update, operation)

    async def renew(self) -> LeaderLease:
        """Renew only when this process still owns the run."""
        def operation(current: LeaderLease | None) -> LeaderLease:
            if current is None or current.owner_instance_id != self.owner_id:
                raise RuntimeError(f"Lost Lead lease for run {self.run_id}")
            now = time.time()
            current.heartbeat_at = now
            current.lease_expires_at = now + self.lease_seconds
            return current

        return await asyncio.to_thread(self._locked_update, operation)

    async def is_owner(self) -> bool:
        """Return whether this process owns an unexpired lease."""
        def operation(current: LeaderLease | None) -> LeaderLease | None:
            return current

        lease = await asyncio.to_thread(self._locked_update, operation)
        return (
            lease is not None
            and lease.owner_instance_id == self.owner_id
            and lease.lease_expires_at > time.time()
        )

    async def release(self) -> None:
        """Expire this process's lease without deleting audit state."""
        def operation(current: LeaderLease | None) -> LeaderLease | None:
            if current is not None and current.owner_instance_id == self.owner_id:
                current.lease_expires_at = 0
                current.heartbeat_at = time.time()
            return current

        await asyncio.to_thread(self._locked_update, operation)
