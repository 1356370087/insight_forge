"""Durable JSON mailboxes used by the Lead and persistent research teammates."""

from __future__ import annotations

import asyncio
import json
import os
import re
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Literal

import portalocker
from pydantic import BaseModel, Field

SCHEMA_VERSION = 1
_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
MessageStatus = Literal["pending", "processing", "acked", "dead_letter"]


class CoordinationError(RuntimeError):
    """Base error for file-backed task coordination."""


class CoordinationCorruptedError(CoordinationError):
    """Raised when a coordination JSON file is unreadable or invalid."""


class MailboxMessage(BaseModel):
    """One typed, lease-aware mailbox message."""

    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    sequence: int = 0
    run_id: str
    sender: str
    recipient: str
    type: str
    priority: int = 50
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: float = Field(default_factory=time.time)
    status: MessageStatus = "pending"
    attempt_count: int = 0
    lease_owner: str | None = None
    lease_expires_at: float | None = None
    acked_at: float | None = None
    dedupe_key: str | None = None


class MailboxFile(BaseModel):
    """On-disk representation of one agent inbox."""

    schema_version: int = SCHEMA_VERSION
    agent_id: str
    next_sequence: int = 1
    messages: list[MailboxMessage] = Field(default_factory=list)


def validate_component(value: str, kind: str = "path component") -> str:
    """Reject path traversal and ambiguous coordination identifiers."""
    if not _SAFE_COMPONENT.fullmatch(value) or ".." in value:
        raise ValueError(f"Invalid {kind}: {value!r}")
    return value


def atomic_write_json(path: Path, payload: Any) -> None:
    """Atomically replace *path* with an fsync'd JSON document."""
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def read_json_file(path: Path) -> Any:
    """Read JSON without treating corruption as an empty document."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except Exception as exc:  # noqa: BLE001 - normalized corruption boundary
        raise CoordinationCorruptedError(f"Corrupt coordination file: {path}") from exc


class FileMailbox:
    """Cross-process JSON mailbox with claim/ACK leases and dead-lettering."""

    def __init__(
        self,
        *,
        runs_dir: str,
        run_id: str,
        lock_timeout_seconds: float = 5,
        claim_lease_seconds: float = 30,
        max_delivery_attempts: int = 5,
        acked_retention_seconds: float = 86400,
        compaction_threshold: int = 1000,
    ) -> None:
        """Initialize one run's mailbox transport and reliability limits."""
        self.run_id = validate_component(run_id, "run_id")
        self.root = Path(runs_dir).resolve() / self.run_id / "coordination"
        self.inboxes_dir = self.root / "inboxes"
        self.dead_letters_dir = self.root / "dead_letters"
        self.lock_timeout_seconds = lock_timeout_seconds
        self.claim_lease_seconds = claim_lease_seconds
        self.max_delivery_attempts = max_delivery_attempts
        self.acked_retention_seconds = acked_retention_seconds
        self.compaction_threshold = compaction_threshold
        self.inboxes_dir.mkdir(parents=True, exist_ok=True)
        self.dead_letters_dir.mkdir(parents=True, exist_ok=True)

    def _paths(self, agent_id: str) -> tuple[Path, Path]:
        safe = validate_component(agent_id, "agent_id")
        return self.inboxes_dir / f"{safe}.json", self.inboxes_dir / f"{safe}.lock"

    def _mark_corrupted(self, source: Path, error: BaseException) -> None:
        marker = self.root / "coordination_corrupted.json"
        atomic_write_json(marker, {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "source": str(source),
            "error": str(error),
            "timestamp": time.time(),
        })

    def _load_locked(self, agent_id: str, path: Path) -> MailboxFile:
        if not path.exists():
            return MailboxFile(agent_id=agent_id)
        try:
            mailbox = MailboxFile.model_validate(read_json_file(path))
        except Exception as exc:
            self._mark_corrupted(path, exc)
            raise
        if mailbox.agent_id != agent_id:
            mismatch_error = CoordinationCorruptedError(f"Mailbox owner mismatch: {path}")
            self._mark_corrupted(path, mismatch_error)
            raise mismatch_error
        return mailbox

    def _with_lock(self, agent_id: str, operation):
        path, lock_path = self._paths(agent_id)
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with portalocker.Lock(str(lock_path), mode="a+b", timeout=self.lock_timeout_seconds):
            mailbox = self._load_locked(agent_id, path)
            result = operation(mailbox)
            atomic_write_json(path, mailbox.model_dump(mode="json"))
            return result

    def _compact(self, mailbox: MailboxFile, now: float) -> None:
        if len(mailbox.messages) < self.compaction_threshold:
            return
        cutoff = now - self.acked_retention_seconds
        mailbox.messages = [
            message
            for message in mailbox.messages
            if message.status != "acked" or (message.acked_at or now) >= cutoff
        ]

    def _append_dead_letters(self, agent_id: str, messages: list[MailboxMessage]) -> None:
        if not messages:
            return
        path = self.dead_letters_dir / f"{validate_component(agent_id, 'agent_id')}.json"
        existing: list[dict[str, Any]] = []
        if path.exists():
            payload = read_json_file(path)
            if not isinstance(payload, list):
                raise CoordinationCorruptedError(f"Invalid dead-letter file: {path}")
            existing = payload
        existing.extend(message.model_dump(mode="json") for message in messages)
        atomic_write_json(path, existing)

    async def send(
        self,
        *,
        recipient: str,
        sender: str,
        message_type: str,
        payload: dict[str, Any] | None = None,
        priority: int = 50,
        dedupe_key: str | None = None,
    ) -> MailboxMessage:
        """Append a message, returning an existing message for duplicate keys."""
        validate_component(sender, "sender")

        def operation(mailbox: MailboxFile) -> MailboxMessage:
            if dedupe_key:
                existing = next((m for m in mailbox.messages if m.dedupe_key == dedupe_key), None)
                if existing is not None:
                    return existing
            message = MailboxMessage(
                sequence=mailbox.next_sequence,
                run_id=self.run_id,
                sender=sender,
                recipient=recipient,
                type=message_type,
                priority=priority,
                payload=payload or {},
                dedupe_key=dedupe_key,
            )
            mailbox.next_sequence += 1
            mailbox.messages.append(message)
            self._compact(mailbox, time.time())
            return message

        return await asyncio.to_thread(self._with_lock, recipient, operation)

    async def claim(
        self,
        *,
        agent_id: str,
        consumer_id: str,
        limit: int = 20,
    ) -> list[MailboxMessage]:
        """Claim pending messages by priority, recovering expired leases."""
        validate_component(consumer_id, "consumer_id")

        def operation(mailbox: MailboxFile) -> list[MailboxMessage]:
            now = time.time()
            dead: list[MailboxMessage] = []
            for message in mailbox.messages:
                if message.status == "processing" and (message.lease_expires_at or 0) <= now:
                    message.status = "pending"
                    message.lease_owner = None
                    message.lease_expires_at = None
                if message.status == "pending" and message.attempt_count >= self.max_delivery_attempts:
                    message.status = "dead_letter"
                    dead.append(message.model_copy(deep=True))
            candidates = sorted(
                (message for message in mailbox.messages if message.status == "pending"),
                key=lambda message: (message.priority, message.sequence),
            )[:limit]
            for message in candidates:
                message.status = "processing"
                message.attempt_count += 1
                message.lease_owner = consumer_id
                message.lease_expires_at = now + self.claim_lease_seconds
            self._append_dead_letters(agent_id, dead)
            self._compact(mailbox, now)
            return [message.model_copy(deep=True) for message in candidates]

        return await asyncio.to_thread(self._with_lock, agent_id, operation)

    async def ack(self, *, agent_id: str, consumer_id: str, message_ids: list[str]) -> None:
        """Acknowledge messages currently leased by *consumer_id*."""
        ids = set(message_ids)

        def operation(mailbox: MailboxFile) -> None:
            now = time.time()
            for message in mailbox.messages:
                if message.message_id in ids:
                    if message.status != "processing" or message.lease_owner != consumer_id:
                        raise CoordinationError(f"Message {message.message_id} is not owned by {consumer_id}")
                    message.status = "acked"
                    message.acked_at = now
                    message.lease_owner = None
                    message.lease_expires_at = None
            self._compact(mailbox, now)

        await asyncio.to_thread(self._with_lock, agent_id, operation)

    async def nack(self, *, agent_id: str, consumer_id: str, message_ids: list[str]) -> None:
        """Release claimed messages for retry or dead-lettering."""
        ids = set(message_ids)

        def operation(mailbox: MailboxFile) -> None:
            dead: list[MailboxMessage] = []
            for message in mailbox.messages:
                if message.message_id not in ids:
                    continue
                if message.status != "processing" or message.lease_owner != consumer_id:
                    raise CoordinationError(f"Message {message.message_id} is not owned by {consumer_id}")
                message.lease_owner = None
                message.lease_expires_at = None
                if message.attempt_count >= self.max_delivery_attempts:
                    message.status = "dead_letter"
                    dead.append(message.model_copy(deep=True))
                else:
                    message.status = "pending"
            self._append_dead_letters(agent_id, dead)

        await asyncio.to_thread(self._with_lock, agent_id, operation)

    async def wait_and_claim(
        self,
        *,
        agent_id: str,
        consumer_id: str,
        timeout_seconds: float,
        poll_interval_seconds: float = 0.5,
        limit: int = 20,
    ) -> list[MailboxMessage]:
        """Wait without model calls until messages arrive or the timeout expires."""
        deadline = time.monotonic() + max(0, timeout_seconds)
        while True:
            messages = await self.claim(agent_id=agent_id, consumer_id=consumer_id, limit=limit)
            if messages or time.monotonic() >= deadline:
                return messages
            await asyncio.sleep(min(poll_interval_seconds, max(0, deadline - time.monotonic())))

    async def stats(self, agent_id: str) -> dict[str, int]:
        """Return status counts for observability and tests."""
        def operation(mailbox: MailboxFile) -> dict[str, int]:
            now = time.time()
            result = {
                status: sum(message.status == status for message in mailbox.messages)
                for status in ("pending", "processing", "acked", "dead_letter")
            }
            result["available"] = result["pending"] + sum(
                message.status == "processing" and (message.lease_expires_at or 0) <= now
                for message in mailbox.messages
            )
            return result

        return await asyncio.to_thread(self._with_lock, agent_id, operation)
