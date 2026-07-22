"""File-backed Query session journal and authoritative run artifacts."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import time
import uuid
from contextlib import contextmanager, nullcontext
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Optional, TypeAlias

import portalocker
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    ToolMessage,
    message_to_dict,
    messages_from_dict,
)
from langchain_core.messages.utils import count_tokens_approximately
from pydantic import BaseModel, Field

from open_deep_research.tasks.lease import FenceLostError, LeaderLease
from open_deep_research.tasks.mailbox import read_json_file

SCHEMA_VERSION = 1
RecordType: TypeAlias = Literal[
    "message_delta",
    "state_delta",
    "stage_checkpoint",
    "context_compacted",
    "artifact_committed",
    "run_status",
]
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SECRET_KEY_RE = re.compile(
    r"(?:api[_-]?key|authorization|oauth[_-]?token|access[_-]?token|refresh[_-]?token|cookie|password|secret)",
    re.IGNORECASE,
)
_SECRET_TEXT_RE = re.compile(
    r"(?i)\b(api[_-]?key|authorization|oauth[_-]?token|access[_-]?token|refresh[_-]?token|cookie|password|secret)"
    r"\s*[:=]\s*([^\s,;]+)"
)


class RunContextError(RuntimeError):
    """Base error for run-context persistence."""


class JournalCorruptedError(RunContextError):
    """Raised when a journal cannot be replayed safely."""


class ResearchBriefPersistenceError(RunContextError):
    """Raised when the authoritative research brief cannot be persisted."""


class SessionJournalRecord(BaseModel):
    """One append-only Query session journal record."""

    schema_version: int = SCHEMA_VERSION
    seq: int
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    run_id: str
    timestamp: float = Field(default_factory=time.time)
    channel: Literal["lead", "supervisor"] = "lead"
    record_type: RecordType
    stage: str
    payload: dict[str, Any] = Field(default_factory=dict)
    artifact_refs: list[str] = Field(default_factory=list)


class RunManifest(BaseModel):
    """Fast run index; the journal remains the recovery source of truth."""

    schema_version: int = SCHEMA_VERSION
    run_id: str
    owner_id: Optional[str] = None
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)
    status: str = "pending"
    last_stable_stage: str = "received"
    next_stage: str = "summarize_messages"
    last_journal_seq: int = 0
    last_public_event_seq: int = 0
    research_brief_sha256: Optional[str] = None
    persistence_degraded: bool = False
    persistence_error: Optional[str] = None
    recovered_from_degraded_persistence: bool = False
    config: dict[str, Any] = Field(default_factory=dict)
    result: Optional[dict[str, Any]] = None
    final_artifacts: dict[str, Any] = Field(default_factory=dict)
    coordination_schema_version: int = 0
    coordination_backend: str = "legacy"
    fence_token: int = 0
    fence_owner_id: Optional[str] = None


class ReplayResult(BaseModel):
    """Reconstructed Query and Supervisor state."""

    state: dict[str, Any] = Field(default_factory=dict)
    supervisor_state: dict[str, Any] = Field(default_factory=dict)
    manifest: RunManifest
    records: list[SessionJournalRecord] = Field(default_factory=list)


def _sha256_text(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _redact_text(value: str) -> str:
    return _SECRET_TEXT_RE.sub(lambda match: f"{match.group(1)}=[REDACTED]", value)


def _sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        sanitized: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            sanitized[key_text] = "[REDACTED]" if _SECRET_KEY_RE.search(key_text) else _sanitize(item)
        return sanitized
    if isinstance(value, list | tuple):
        return [_sanitize(item) for item in value]
    if isinstance(value, str):
        return _redact_text(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, bool | int | float):
        return value
    return str(value)


def _sanitize_config(value: Any) -> Any:
    """Remove credential-bearing config keys instead of persisting placeholders."""
    if isinstance(value, dict):
        return {
            str(key): _sanitize_config(item)
            for key, item in value.items()
            if not _SECRET_KEY_RE.search(str(key))
        }
    if isinstance(value, list | tuple):
        return [_sanitize_config(item) for item in value]
    return _sanitize(value)


class RunContextStore:
    """Persist one run's recoverable state under ``runs_dir/run_id/context``."""

    _locks: dict[str, asyncio.Lock] = {}

    def __init__(
        self,
        run_id: str,
        *,
        runs_dir: str = ".runs",
        inline_content_max_chars: int = 32768,
    ) -> None:
        """Initialize paths and the process-local append lock for one run."""
        if not _RUN_ID_RE.fullmatch(run_id) or ".." in run_id:
            raise ValueError("Invalid run_id")
        self.run_id = run_id
        self.run_dir = Path(runs_dir).resolve() / run_id
        self.context_dir = self.run_dir / "context"
        self.journal_path = self.context_dir / "session_memory.jsonl"
        self.manifest_path = self.context_dir / "manifest.json"
        self.journal_lock_path = self.context_dir / "session_memory.lock"
        self.manifest_lock_path = self.context_dir / "manifest.lock"
        self.lease_path = self.run_dir / "coordination" / "leader_lease.json"
        self.lease_lock_path = self.run_dir / "coordination" / "leader_lease.lock"
        self.brief_path = self.context_dir / "research_brief.md"
        self.inline_content_max_chars = inline_content_max_chars
        self._fence_token: int | None = None
        self._fence_owner_id: str | None = None
        lock_key = str(self.context_dir)
        self._lock = self._locks.setdefault(lock_key, asyncio.Lock())

    @staticmethod
    def _safe_config(config: dict[str, Any]) -> dict[str, Any]:
        return _sanitize_config(config)

    @contextmanager
    def _fence_guard(self):
        """Hold the lease lock while validating this store's ownership epoch."""
        if self._fence_token is None or self._fence_owner_id is None:
            if self.manifest_path.exists():
                manifest = self.load_manifest()
                if manifest.fence_token > 0:
                    raise FenceLostError(f"Unfenced write rejected for run {self.run_id}")
            yield
            return

        self.lease_lock_path.parent.mkdir(parents=True, exist_ok=True)
        with portalocker.Lock(str(self.lease_lock_path), mode="a+b", timeout=30):
            try:
                lease = LeaderLease.model_validate(read_json_file(self.lease_path))
            except Exception as exc:
                raise FenceLostError(f"Missing Lead lease for run {self.run_id}") from exc
            if (
                lease.owner_instance_id != self._fence_owner_id
                or lease.fence_token != self._fence_token
                or lease.lease_expires_at <= time.time()
            ):
                raise FenceLostError(f"Lost Lead lease for run {self.run_id}")
            yield

    def bind_fence_token(
        self,
        fence_token: int,
        owner_id: str,
        *,
        advance_manifest: bool = True,
    ) -> None:
        """Bind writes to the exact live Lead ownership epoch."""
        previous = (self._fence_token, self._fence_owner_id)
        self._fence_token = int(fence_token)
        self._fence_owner_id = str(owner_id)
        try:
            if advance_manifest and self.manifest_path.exists():
                self._update_manifest(
                    allow_fence_advance=True,
                    fence_token=self._fence_token,
                    fence_owner_id=self._fence_owner_id,
                )
        except Exception:
            self._fence_token, self._fence_owner_id = previous
            raise

    def initialize(self, owner_id: Optional[str], config: dict[str, Any]) -> RunManifest:
        """Create the run context and initial manifest if absent."""
        self.context_dir.mkdir(parents=True, exist_ok=True)
        (self.context_dir / "artifacts" / "messages").mkdir(parents=True, exist_ok=True)
        (self.context_dir / "artifacts" / "research_tasks").mkdir(parents=True, exist_ok=True)
        if self.manifest_path.exists():
            return self.load_manifest()
        manifest = RunManifest(
            run_id=self.run_id,
            owner_id=owner_id,
            status="running",
            config=self._safe_config(config),
            coordination_schema_version=1,
            coordination_backend="file_mailbox",
            fence_token=self._fence_token or 0,
            fence_owner_id=self._fence_owner_id,
        )
        with self._fence_guard():
            self._write_json_atomic_path(self.manifest_path, manifest.model_dump(mode="json"))
        return manifest

    def _resolve_artifact(self, relative_path: str) -> Path:
        relative = Path(relative_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Artifact path must stay inside the run context")
        target = (self.context_dir / relative).resolve()
        if self.context_dir != target and self.context_dir not in target.parents:
            raise ValueError("Artifact path escapes the run context")
        return target

    @staticmethod
    def _write_bytes_atomic_path(path: Path, content: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            with temp.open("wb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp, path)
        finally:
            if temp.exists():
                temp.unlink()

    @classmethod
    def _write_json_atomic_path(cls, path: Path, payload: Any) -> None:
        cls._write_bytes_atomic_path(
            path,
            json.dumps(payload, ensure_ascii=False, indent=2, default=str).encode("utf-8"),
        )

    def write_text_atomic(self, relative_path: str, content: str) -> str:
        """Atomically write UTF-8 text and return its SHA-256."""
        target = self._resolve_artifact(relative_path)
        with self._fence_guard():
            self._write_bytes_atomic_path(target, content.encode("utf-8"))
        return _sha256_text(content)

    def write_json_atomic(self, relative_path: str, payload: Any) -> str:
        """Atomically write sanitized JSON and return its SHA-256."""
        content = json.dumps(_sanitize(payload), ensure_ascii=False, indent=2, default=str)
        return self.write_text_atomic(relative_path, content)

    def persist_research_brief(self, content: str) -> str:
        """Strictly persist the complete authoritative research brief."""
        try:
            digest = self.write_text_atomic("research_brief.md", content)
            self._update_manifest(research_brief_sha256=digest)
            return digest
        except Exception as exc:
            raise ResearchBriefPersistenceError("research_brief_persistence_failed") from exc

    def load_research_brief(self, *, verify: bool = True) -> str:
        """Load the authoritative brief and optionally verify its manifest hash."""
        if not self.brief_path.exists():
            raise ResearchBriefPersistenceError("research_brief_missing")
        content = self.brief_path.read_text(encoding="utf-8")
        if verify:
            expected = self.load_manifest().research_brief_sha256
            if expected and _sha256_text(content) != expected:
                raise ResearchBriefPersistenceError("research_brief_hash_mismatch")
        return content

    def load_manifest(self) -> RunManifest:
        """Load the run manifest."""
        try:
            return RunManifest.model_validate_json(self.manifest_path.read_text(encoding="utf-8"))
        except Exception:
            return self._rebuild_manifest_from_journal()

    def _rebuild_manifest_from_journal(self) -> RunManifest:
        """Rebuild a missing/corrupt manifest cache from durable journal records."""
        records = self._read_records()
        if not records:
            raise JournalCorruptedError("manifest_corrupted")
        manifest = RunManifest(run_id=self.run_id)
        for record in records:
            payload = record.payload
            if record.seq == 1:
                manifest.owner_id = payload.get("owner_id")
                manifest.config = payload.get("config", {})
                manifest.coordination_schema_version = int(
                    payload.get("coordination_schema_version", 0)
                )
                manifest.coordination_backend = str(
                    payload.get("coordination_backend", "legacy")
                )
            if record.record_type == "stage_checkpoint":
                manifest.last_stable_stage = record.stage
                manifest.next_stage = str(payload.get("next_stage", manifest.next_stage))
                if record.stage in {"completed", "cancelled"}:
                    manifest.status = record.stage
            elif record.record_type == "run_status":
                manifest.status = str(payload.get("status", manifest.status))
                if manifest.status == "failed":
                    manifest.result = {"status": "error", "error": payload.get("error")}
        manifest.last_journal_seq = records[-1].seq
        if self.brief_path.exists():
            manifest.research_brief_sha256 = _sha256_text(self.brief_path.read_text(encoding="utf-8"))
        self.context_dir.mkdir(parents=True, exist_ok=True)
        with portalocker.Lock(str(self.manifest_lock_path), mode="a+b", timeout=30):
            self._write_json_atomic_path(self.manifest_path, manifest.model_dump(mode="json"))
        return manifest

    def _update_manifest(
        self,
        *,
        allow_failed_resume: bool = False,
        allow_fence_advance: bool = False,
        fence_already_held: bool = False,
        **updates: Any,
    ) -> RunManifest:
        fallback = self.load_manifest()
        self.context_dir.mkdir(parents=True, exist_ok=True)
        guard = nullcontext() if fence_already_held else self._fence_guard()
        with guard:
            with portalocker.Lock(str(self.manifest_lock_path), mode="a+b", timeout=30):
                try:
                    manifest = RunManifest.model_validate_json(
                        self.manifest_path.read_text(encoding="utf-8")
                    )
                except Exception:
                    manifest = fallback
                if manifest.fence_token > 0:
                    if self._fence_token is None or self._fence_owner_id is None:
                        raise FenceLostError(f"Unfenced manifest write rejected for run {self.run_id}")
                    if allow_fence_advance:
                        if self._fence_token < manifest.fence_token:
                            raise FenceLostError(f"Stale fence token for run {self.run_id}")
                    elif (
                        self._fence_token != manifest.fence_token
                        or self._fence_owner_id != manifest.fence_owner_id
                    ):
                        raise FenceLostError(f"Stale fence token for run {self.run_id}")
                for key, value in updates.items():
                    if not hasattr(manifest, key):
                        continue
                    if key in {"last_journal_seq", "last_public_event_seq"}:
                        value = max(int(getattr(manifest, key)), int(value))
                    if key == "status" and manifest.status in {"completed", "cancelled"}:
                        if value != manifest.status:
                            continue
                    if (
                        key == "status"
                        and manifest.status == "failed"
                        and value != "failed"
                        and not (allow_failed_resume and value == "running")
                    ):
                        continue
                    setattr(manifest, key, value)
                manifest.updated_at = time.time()
                self._write_json_atomic_path(
                    self.manifest_path,
                    manifest.model_dump(mode="json"),
                )
                return manifest

    async def _encode(self, value: Any, artifact_refs: list[str]) -> Any:
        if isinstance(value, BaseMessage):
            serialized = message_to_dict(value)
            content = serialized.get("data", {}).get("content", "")
            content_text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False, default=str)
            if len(content_text) > self.inline_content_max_chars:
                artifact_id = uuid.uuid4().hex
                relative = f"artifacts/messages/{artifact_id}.json"
                self.write_json_atomic(relative, {"kind": "message", "message": serialized})
                artifact_refs.append(relative)
                return {
                    "__message_artifact__": relative,
                    "sha256": _sha256_text(json.dumps(serialized, ensure_ascii=False, default=str)),
                    "length": len(content_text),
                    "preview": _redact_text(content_text[:512]),
                }
            return {"__message__": _sanitize(serialized)}
        if isinstance(value, dict):
            encoded: dict[str, Any] = {}
            for key, item in value.items():
                key_text = str(key)
                encoded[key_text] = "[REDACTED]" if _SECRET_KEY_RE.search(key_text) else await self._encode(item, artifact_refs)
            return encoded
        if isinstance(value, list | tuple):
            return [await self._encode(item, artifact_refs) for item in value]
        return _sanitize(value)

    def _decode(self, value: Any) -> Any:
        if isinstance(value, dict):
            if "__message__" in value:
                return messages_from_dict([value["__message__"]])[0]
            if "__message_artifact__" in value:
                artifact = self._resolve_artifact(value["__message_artifact__"])
                try:
                    payload = json.loads(artifact.read_text(encoding="utf-8"))
                    return messages_from_dict([payload["message"]])[0]
                except Exception as exc:
                    raise JournalCorruptedError("message_artifact_corrupted") from exc
            return {key: self._decode(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._decode(item) for item in value]
        return value

    def _read_records_unlocked(self, *, repair_tail: bool = True) -> list[SessionJournalRecord]:
        if not self.journal_path.exists():
            return []
        content = self.journal_path.read_bytes()
        has_complete_tail = content.endswith(b"\n")
        raw_lines = content.splitlines()
        records: list[SessionJournalRecord] = []
        expected_seq = 1
        for index, raw in enumerate(raw_lines):
            if not raw.strip():
                continue
            try:
                record = SessionJournalRecord.model_validate_json(raw)
            except Exception as exc:
                is_partial_tail = index == len(raw_lines) - 1 and not has_complete_tail
                if repair_tail and is_partial_tail:
                    valid_prefix = b"\n".join(raw_lines[:index])
                    if valid_prefix:
                        valid_prefix += b"\n"
                    with self.journal_path.open("wb") as handle:
                        handle.write(valid_prefix)
                        handle.flush()
                        os.fsync(handle.fileno())
                    break
                raise JournalCorruptedError("journal_corrupted") from exc
            if record.run_id != self.run_id or record.seq != expected_seq:
                raise JournalCorruptedError("journal_corrupted")
            records.append(record)
            expected_seq += 1
        return records

    def _read_records(self) -> list[SessionJournalRecord]:
        if not self.journal_path.exists():
            return []
        self.context_dir.mkdir(parents=True, exist_ok=True)
        with portalocker.Lock(str(self.journal_lock_path), mode="a+b", timeout=30):
            return self._read_records_unlocked()

    async def append(
        self,
        *,
        channel: Literal["lead", "supervisor"],
        record_type: RecordType,
        stage: str,
        payload: dict[str, Any],
    ) -> SessionJournalRecord:
        """Append one durable journal line under process and file locks."""
        artifact_refs: list[str] = []
        encoded = await self._encode(payload, artifact_refs)

        def append_locked() -> SessionJournalRecord:
            self.context_dir.mkdir(parents=True, exist_ok=True)
            with self._fence_guard():
                with portalocker.Lock(
                    str(self.journal_lock_path),
                    mode="a+b",
                    timeout=30,
                ):
                    manifest = self.load_manifest()
                    seq = max(
                        manifest.last_journal_seq,
                        len(self._read_records_unlocked()),
                    ) + 1
                    record = SessionJournalRecord(
                        seq=seq,
                        run_id=self.run_id,
                        channel=channel,
                        record_type=record_type,
                        stage=stage,
                        payload=encoded,
                        artifact_refs=artifact_refs,
                    )
                    self.journal_path.parent.mkdir(parents=True, exist_ok=True)
                    data = (record.model_dump_json() + "\n").encode("utf-8")
                    fd = os.open(
                        self.journal_path,
                        os.O_APPEND | os.O_CREAT | os.O_WRONLY,
                        0o600,
                    )
                    try:
                        os.write(fd, data)
                        os.fsync(fd)
                    finally:
                        os.close(fd)
                    self._update_manifest(
                        fence_already_held=True,
                        last_journal_seq=seq,
                    )
                    return record

        async with self._lock:
            return await asyncio.to_thread(append_locked)

    async def checkpoint(
        self,
        stage: str,
        next_stage: str,
        *,
        status: str = "running",
        channel: Literal["lead", "supervisor"] = "lead",
        payload: Optional[dict[str, Any]] = None,
    ) -> SessionJournalRecord:
        """Append a stable-stage checkpoint and update the manifest cache."""
        record = await self.append(
            channel=channel,
            record_type="stage_checkpoint",
            stage=stage,
            payload={"next_stage": next_stage, **(payload or {})},
        )
        self._update_manifest(
            status=status,
            last_stable_stage=stage,
            next_stage=next_stage,
            last_journal_seq=record.seq,
        )
        return record

    def mark_persistence_degraded(self, error: Exception | str) -> None:
        """Best-effort mark that recovery is only guaranteed to the last fsync."""
        try:
            self._update_manifest(
                persistence_degraded=True,
                persistence_error=str(error)[:1000],
            )
        except Exception:
            return

    def mark_event_persistence_failed(self, error: Exception | str) -> None:
        """Mark a public-event contract failure as a terminal run failure."""
        try:
            self._update_manifest(
                status="failed",
                persistence_degraded=True,
                persistence_error=f"event_persistence_failed: {error}"[:1000],
            )
        except Exception:
            return

    def load_evidence_artifact(
        self,
        relative_path: str,
        *,
        expected_sha256: str,
    ) -> dict[str, Any]:
        """Load one evidence artifact after path and content-hash validation."""
        if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
            raise ValueError("Invalid artifact SHA-256")
        target = self._resolve_artifact(relative_path)
        content = target.read_text(encoding="utf-8")
        if _sha256_text(content) != expected_sha256:
            raise ValueError("Evidence artifact hash mismatch")
        payload = json.loads(content)
        if not isinstance(payload, dict):
            raise ValueError("Evidence artifact must contain an object")
        return payload

    def persist_task_result(self, task_id: str, result: dict[str, Any]) -> str:
        """Persist a completed research task before its in-memory context is released."""
        if not _RUN_ID_RE.fullmatch(task_id) or ".." in task_id:
            raise ValueError(f"Invalid task_id: {task_id!r}")
        return self.write_json_atomic(f"artifacts/research_tasks/{task_id}.json", result)

    def load_task_result(
        self,
        task_id: str,
        *,
        expected_sha256: str,
    ) -> dict[str, Any]:
        """Load a task artifact after validating its task id and content digest."""
        if not _RUN_ID_RE.fullmatch(task_id) or ".." in task_id:
            raise ValueError(f"Invalid task_id: {task_id!r}")
        if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
            raise ValueError("Invalid artifact SHA-256")
        target = self._resolve_artifact(f"artifacts/research_tasks/{task_id}.json")
        try:
            content = target.read_text(encoding="utf-8")
        except OSError as exc:
            raise FileNotFoundError(f"Research artifact is unavailable: {task_id}") from exc
        if _sha256_text(content) != expected_sha256:
            raise ValueError(f"Research artifact hash mismatch: {task_id}")
        payload = json.loads(content)
        if not isinstance(payload, dict):
            raise ValueError(f"Research artifact must contain an object: {task_id}")
        return payload

    def replay(self) -> ReplayResult:
        """Replay state deltas and stable checkpoints from the journal."""
        from open_deep_research.runtime import apply_update_to_state

        manifest = self.load_manifest()
        records = self._read_records()
        state: dict[str, Any] = {}
        supervisor_state: dict[str, Any] = {}
        last_stage = manifest.last_stable_stage
        next_stage = manifest.next_stage
        for record in records:
            payload = self._decode(record.payload)
            if record.record_type in {"state_delta", "message_delta", "context_compacted"}:
                update = payload.get("update", {})
                scope = payload.get("scope", "main")
                target = supervisor_state if scope == "supervisor" else state
                apply_update_to_state(target, update)
            elif record.record_type == "stage_checkpoint":
                last_stage = record.stage
                next_stage = str(payload.get("next_stage", next_stage))
        manifest.last_stable_stage = last_stage
        manifest.next_stage = next_stage
        manifest.last_journal_seq = records[-1].seq if records else 0
        if manifest.research_brief_sha256:
            state["research_brief"] = self.load_research_brief()
        return ReplayResult(
            state=state,
            supervisor_state=supervisor_state,
            manifest=manifest,
            records=records,
        )

    def build_projection(self, channel: Literal["lead", "supervisor"], token_budget: int) -> dict[str, Any]:
        """Return the latest durable summary and recent messages for a channel."""
        records = [record for record in self._read_records() if record.channel == channel]
        latest_summary: Optional[str] = None
        summary_seq = 0
        messages: list[BaseMessage] = []
        for record in records:
            payload = self._decode(record.payload)
            if record.record_type == "context_compacted":
                latest_summary = payload.get("summary")
                summary_seq = record.seq
                messages = list(payload.get("recent_messages", []))
            elif record.seq > summary_seq and record.record_type in {"state_delta", "message_delta"}:
                update = payload.get("update", {})
                key = "supervisor_messages" if channel == "supervisor" else "messages"
                value = update.get(key, [])
                if isinstance(value, dict):
                    value = value.get("value", [])
                if isinstance(value, list):
                    messages.extend(item for item in value if isinstance(item, BaseMessage))
        used = 0
        boundary = len(messages)
        for index in range(len(messages) - 1, -1, -1):
            size = count_tokens_approximately([messages[index]])
            if used and used + size > token_budget:
                break
            used += size
            boundary = index
        if boundary < len(messages) and isinstance(messages[boundary], ToolMessage):
            tool_call_id = messages[boundary].tool_call_id
            for index in range(boundary - 1, -1, -1):
                candidate = messages[index]
                if isinstance(candidate, AIMessage) and any(
                    str(call.get("id", "")) == tool_call_id for call in candidate.tool_calls
                ):
                    boundary = index
                    break
        messages = messages[boundary:]
        return {
            "summary": latest_summary,
            "recent_messages": messages,
            "token_budget": token_budget,
        }
