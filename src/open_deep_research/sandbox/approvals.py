"""Durable, concurrent security approvals owned by the API."""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
import uuid
from pathlib import Path
from typing import Any, Literal

import portalocker
from pydantic import BaseModel, ConfigDict, Field

from open_deep_research.tasks.mailbox import (
    atomic_write_json,
    read_json_file,
    validate_component,
)

ApprovalKind = Literal["network", "tool_effect", "filesystem", "command", "mcp_oauth"]
ApprovalDecision = Literal["allow_once", "allow_run", "deny"]


class SecurityApproval(BaseModel):
    """One redacted, auditable approval request."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    approval_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    run_id: str
    task_id: str
    fence_token: int = Field(ge=1)
    kind: ApprovalKind
    capability: str
    target: dict[str, Any]
    target_fingerprint: str
    status: Literal["pending", "resolved", "expired", "consumed"] = "pending"
    decision: ApprovalDecision | None = None
    reason: str = ""
    requested_at: float = Field(default_factory=time.time)
    expires_at: float
    resolved_at: float | None = None
    resolved_by: str | None = None
    operation_id: str
    version: int = 1


class SecurityApprovalStore:
    """Run-scoped file store supporting multiple concurrent pending requests."""

    def __init__(self, run_id: str, *, runs_dir: str = ".runs") -> None:
        """Bind the approval index to one validated run directory."""
        self.run_id = validate_component(run_id, "run_id")
        self.root = Path(runs_dir).resolve() / self.run_id / "sandbox" / "approvals"
        self.index_path = self.root / "index.json"
        self.lock_path = self.root / "index.lock"

    @staticmethod
    def fingerprint(kind: str, capability: str, target: dict[str, Any]) -> str:
        """Hash one normalized capability target for allow-run reuse."""
        encoded = json.dumps(
            {"kind": kind, "capability": capability, "target": target},
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode()
        return hashlib.sha256(encoded).hexdigest()

    def _load(self) -> dict[str, Any]:
        if not self.index_path.exists():
            return {"schema_version": 1, "version": 0, "approvals": {}}
        return read_json_file(self.index_path)

    def _locked(self, operation):
        self.root.mkdir(parents=True, exist_ok=True)
        with portalocker.Lock(str(self.lock_path), mode="a+b", timeout=30):
            data = self._load()
            result = operation(data)
            atomic_write_json(self.index_path, data)
            return result

    def request(
        self,
        *,
        task_id: str,
        fence_token: int,
        kind: ApprovalKind,
        capability: str,
        target: dict[str, Any],
        operation_id: str,
        expires_at: float,
    ) -> SecurityApproval:
        """Create or return an idempotent pending request for an operation."""
        fingerprint = self.fingerprint(kind, capability, target)
        validated_task_id = validate_component(task_id, "task_id")

        def create(data: dict[str, Any]) -> SecurityApproval:
            for raw in data["approvals"].values():
                existing = SecurityApproval.model_validate(raw)
                if existing.operation_id == operation_id:
                    if (
                        existing.fence_token != fence_token
                        or existing.task_id != validated_task_id
                        or existing.target_fingerprint != fingerprint
                        or existing.kind != kind
                        or existing.capability != capability
                    ):
                        raise ValueError("security_approval_operation_mismatch")
                    return existing
            if len(data["approvals"]) >= 1000:
                raise ValueError("security_approval_run_limit_exceeded")
            pending_count = sum(
                1
                for raw in data["approvals"].values()
                if raw.get("status") == "pending"
            )
            if pending_count >= 100:
                raise ValueError("security_approval_pending_limit_exceeded")
            approval = SecurityApproval(
                run_id=self.run_id,
                task_id=validated_task_id,
                fence_token=fence_token,
                kind=kind,
                capability=capability,
                target=target,
                target_fingerprint=fingerprint,
                operation_id=operation_id,
                expires_at=expires_at,
            )
            data["approvals"][approval.approval_id] = approval.model_dump(mode="json")
            data["version"] = int(data.get("version", 0)) + 1
            return approval

        return self._locked(create)

    def list(self, *, status: str | None = None) -> tuple[int, list[SecurityApproval]]:
        """List approvals and expire stale pending entries atomically."""
        def read(data: dict[str, Any]) -> tuple[int, list[SecurityApproval]]:
            now = time.time()
            changed = False
            values = []
            for approval_id, raw in list(data["approvals"].items()):
                item = SecurityApproval.model_validate(raw)
                if item.status == "pending" and item.expires_at <= now:
                    item.status = "expired"
                    item.decision = "deny"
                    item.resolved_at = now
                    item.version += 1
                    data["approvals"][approval_id] = item.model_dump(mode="json")
                    changed = True
                if status is None or item.status == status:
                    values.append(item)
            if changed:
                data["version"] = int(data.get("version", 0)) + 1
            values.sort(key=lambda item: (item.requested_at, item.approval_id))
            return int(data.get("version", 0)), values

        return self._locked(read)

    def resolve(
        self,
        approval_id: str,
        *,
        decision: ApprovalDecision,
        actor: str,
        reason: str,
        expected_fence_token: int,
    ) -> SecurityApproval:
        """Resolve one pending approval for the expected ownership epoch."""
        def apply(data: dict[str, Any]) -> SecurityApproval:
            raw = data["approvals"].get(approval_id)
            if raw is None:
                raise KeyError(approval_id)
            item = SecurityApproval.model_validate(raw)
            if item.fence_token != expected_fence_token:
                raise ValueError("stale_fence")
            if item.status != "pending":
                if item.decision == decision:
                    return item
                raise ValueError("security_approval_already_resolved")
            if item.expires_at <= time.time():
                raise ValueError("security_approval_timeout")
            item.status = "resolved"
            item.decision = decision
            item.reason = reason[:1000]
            item.resolved_by = actor
            item.resolved_at = time.time()
            item.version += 1
            data["approvals"][approval_id] = item.model_dump(mode="json")
            data["version"] = int(data.get("version", 0)) + 1
            return item

        return self._locked(apply)

    def consume(
        self,
        approval_id: str,
        *,
        operation_id: str,
        expected_fence_token: int,
    ) -> SecurityApproval:
        """Atomically consume allow-once; allow-run remains reusable."""

        def apply(data: dict[str, Any]) -> SecurityApproval:
            raw = data["approvals"].get(approval_id)
            if raw is None:
                raise KeyError(approval_id)
            item = SecurityApproval.model_validate(raw)
            if item.fence_token != expected_fence_token:
                raise ValueError("stale_fence")
            if item.decision == "deny" or item.status in {"expired", "consumed"}:
                raise ValueError("security_approval_not_usable")
            if item.decision == "allow_once":
                if item.operation_id != operation_id:
                    raise ValueError("security_approval_operation_mismatch")
                item.status = "consumed"
                item.version += 1
                data["approvals"][approval_id] = item.model_dump(mode="json")
                data["version"] = int(data.get("version", 0)) + 1
            elif item.decision != "allow_run":
                raise ValueError("security_approval_not_resolved")
            return item

        return self._locked(apply)

    def deny_pending(self, *, actor: str, reason: str) -> int:
        """Resolve every pending approval when its run becomes terminal."""

        def apply(data: dict[str, Any]) -> int:
            now = time.time()
            changed = 0
            for approval_id, raw in list(data["approvals"].items()):
                item = SecurityApproval.model_validate(raw)
                if item.status != "pending":
                    continue
                item.status = "resolved"
                item.decision = "deny"
                item.reason = reason[:1000]
                item.resolved_by = actor
                item.resolved_at = now
                item.version += 1
                data["approvals"][approval_id] = item.model_dump(mode="json")
                changed += 1
            if changed:
                data["version"] = int(data.get("version", 0)) + 1
            return changed

        return self._locked(apply)

    async def wait_for_change(
        self,
        after_version: int,
        *,
        timeout_seconds: float = 25,
    ) -> tuple[int, list[SecurityApproval]]:
        """Long-poll until the durable index version advances or times out."""
        deadline = time.monotonic() + min(25.0, max(0.1, timeout_seconds))
        while True:
            version, approvals = await asyncio.to_thread(self.list)
            if version > after_version or time.monotonic() >= deadline:
                return version, approvals
            await asyncio.sleep(0.2)
