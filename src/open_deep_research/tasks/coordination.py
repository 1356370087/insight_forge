"""Mailbox adapters for task events, Lead updates, and domain decisions."""

from __future__ import annotations

import asyncio
import hashlib
import time
from pathlib import Path
from typing import Any

import portalocker
from langchain_core.runnables import RunnableConfig

from open_deep_research.configuration import Configuration
from open_deep_research.tasks.events import EventType
from open_deep_research.tasks.mailbox import (
    CoordinationError,
    FileMailbox,
    MailboxMessage,
    atomic_write_json,
    read_json_file,
    validate_component,
)
from open_deep_research.tasks.state import TaskSnapshot, get_task_state_store

LEAD_AGENT_ID = "lead"


def _verify_result_artifact(
    configurable: Configuration,
    run_id: str,
    snapshot: TaskSnapshot,
) -> None:
    """Verify that a completed result exists inside the run and matches its digest."""
    if not snapshot.result_artifact_path or not snapshot.result_artifact_sha256:
        raise CoordinationError(f"Task {snapshot.task_id} completion is missing its result artifact")
    run_root = (Path(configurable.runs_dir).resolve() / validate_component(run_id, "run_id")).resolve()
    artifact = (run_root / snapshot.result_artifact_path).resolve()
    if run_root not in artifact.parents:
        raise CoordinationError(f"Task {snapshot.task_id} artifact escapes the run directory")
    try:
        digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    except OSError as exc:
        raise CoordinationError(f"Task {snapshot.task_id} result artifact is unavailable") from exc
    if digest != snapshot.result_artifact_sha256:
        raise CoordinationError(f"Task {snapshot.task_id} result artifact hash mismatch")


def get_run_id(config: RunnableConfig) -> str:
    """Return the run identifier used to isolate coordination state."""
    return str(config.get("metadata", {}).get("run_id", "default"))


def get_mailbox(configurable: Configuration, run_id: str) -> FileMailbox:
    """Construct a mailbox using the run's coordination settings."""
    return FileMailbox(
        runs_dir=configurable.runs_dir,
        run_id=run_id,
        lock_timeout_seconds=configurable.mailbox_lock_timeout_seconds,
        claim_lease_seconds=configurable.mailbox_claim_lease_seconds,
        max_delivery_attempts=configurable.mailbox_max_delivery_attempts,
        acked_retention_seconds=configurable.mailbox_acked_retention_seconds,
        compaction_threshold=configurable.mailbox_compaction_threshold,
    )


def _event_message_type(event_type: EventType | str) -> str:
    value = event_type.value if isinstance(event_type, EventType) else str(event_type)
    suffix = value.rsplit(".", 1)[-1]
    return {
        "started": "task_started",
        "completed": "task_completed",
        "failed": "task_failed",
        "cancelled": "task_cancelled",
        "timed_out": "task_timed_out",
        "domain_confirmation_requested": "domain_approval_request",
    }.get(suffix, "task_progress")


async def publish_task_update(
    configurable: Configuration,
    snapshot: TaskSnapshot,
    event_type: EventType | str,
) -> MailboxMessage:
    """Notify the Lead after the authoritative snapshot has been committed."""
    mailbox = get_mailbox(configurable, snapshot.run_id)
    message_type = _event_message_type(event_type)
    priority = 0 if message_type in {"task_failed", "task_cancelled", "domain_approval_request"} else 40
    return await mailbox.send(
        recipient=LEAD_AGENT_ID,
        sender=snapshot.assigned_teammate_id or "orchestrator",
        message_type=message_type,
        priority=priority,
        dedupe_key=f"{snapshot.task_id}:{message_type}:{snapshot.version}",
        payload={
            "task_id": snapshot.task_id,
            "snapshot_version": snapshot.version,
            "status": snapshot.status.value,
            "phase": snapshot.phase.value,
            "artifact_path": snapshot.result_artifact_path,
            "artifact_sha256": snapshot.result_artifact_sha256,
            "pending_domain": snapshot.pending_domain,
        },
    )


async def claim_lead_updates(
    configurable: Configuration,
    *,
    run_id: str,
    consumer_id: str,
    timeout_seconds: float = 0,
    processed_message_ids: set[str] | None = None,
) -> tuple[list[MailboxMessage], str]:
    """Claim Lead messages and render their authoritative task snapshots."""
    mailbox = get_mailbox(configurable, run_id)
    if timeout_seconds > 0:
        messages = await mailbox.wait_and_claim(
            agent_id=LEAD_AGENT_ID,
            consumer_id=consumer_id,
            timeout_seconds=timeout_seconds,
            poll_interval_seconds=configurable.mailbox_poll_interval_ms / 1000,
        )
    else:
        messages = await mailbox.claim(agent_id=LEAD_AGENT_ID, consumer_id=consumer_id)
    if not messages:
        return [], ""
    store = get_task_state_store(configurable)
    parts: list[str] = []
    processed = processed_message_ids or set()
    for message in messages:
        if message.message_id in processed:
            continue
        task_id = str(message.payload.get("task_id", ""))
        snapshot = await store.get(task_id) if task_id else None
        if snapshot is None:
            parts.append(f"{message.type}: task={task_id or '(none)'}")
            continue
        if message.type == "task_completed":
            _verify_result_artifact(configurable, run_id, snapshot)
        result_text = ""
        if snapshot.result and snapshot.status.value == "completed":
            result_text = str(snapshot.result.get("compressed_research", ""))
        parts.append(
            f"### {snapshot.task_id} - {snapshot.status.value.upper()}\n"
            f"Teammate: {snapshot.assigned_teammate_id or '(unassigned)'}\n"
            f"Topic: {snapshot.research_topic}\n"
            f"Version: {snapshot.version}\n"
            f"{result_text}"
        )
    context = "Mailbox task updates:\n\n" + "\n---\n".join(parts) if parts else ""
    return messages, context


async def ack_lead_updates(
    configurable: Configuration,
    *,
    run_id: str,
    consumer_id: str,
    message_ids: list[str],
) -> None:
    """ACK Lead updates after their Supervisor state delta is durable."""
    if message_ids:
        await get_mailbox(configurable, run_id).ack(
            agent_id=LEAD_AGENT_ID,
            consumer_id=consumer_id,
            message_ids=message_ids,
        )


class FileDomainDecisionStore:
    """Locked persistent cache for per-run domain approvals."""

    def __init__(self, configurable: Configuration, run_id: str) -> None:
        """Initialize the run-scoped decision document."""
        safe_run_id = validate_component(run_id, "run_id")
        self.root = Path(configurable.runs_dir).resolve() / safe_run_id / "coordination"
        self.path = self.root / "domain_decisions.json"
        self.lock_path = self.root / "domain_decisions.lock"
        self.timeout = configurable.mailbox_lock_timeout_seconds

    def _update(self, domain: str, allowed: bool) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        with portalocker.Lock(str(self.lock_path), mode="a+b", timeout=self.timeout):
            payload: dict[str, Any] = {"schema_version": 1, "decisions": {}}
            if self.path.exists():
                payload = read_json_file(self.path)
            payload.setdefault("decisions", {})[domain.lower()] = {
                "allowed": allowed,
                "updated_at": time.time(),
            }
            atomic_write_json(self.path, payload)

    async def record(self, domain: str, allowed: bool) -> None:
        """Persist one allow/deny decision."""
        await asyncio.to_thread(self._update, domain, allowed)

    async def get(self, domain: str) -> bool | None:
        """Return a persisted decision when available."""
        def read_locked() -> bool | None:
            self.root.mkdir(parents=True, exist_ok=True)
            with portalocker.Lock(str(self.lock_path), mode="a+b", timeout=self.timeout):
                if not self.path.exists():
                    return None
                payload = read_json_file(self.path)
                decision = payload.get("decisions", {}).get(domain.lower())
                return None if decision is None else bool(decision["allowed"])

        return await asyncio.to_thread(read_locked)
