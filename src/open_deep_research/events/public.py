"""Durable, sanitized public events for research run progress.

This module is deliberately separate from the Query recovery journal and the
diagnostic task log.  Its JSONL file is the only source exposed through SSE.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import uuid
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Optional
from urllib.parse import urlsplit, urlunsplit

import portalocker
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from open_deep_research.configuration import Configuration

PUBLIC_EVENT_SCHEMA_VERSION = 2
PUBLIC_STAGES = (
    "preparing",
    "planning",
    "researching",
    "synthesizing",
    "writing",
    "finalizing",
)
TERMINAL_EVENT_TYPES = {"run.completed", "run.failed", "run.cancelled"}
_COMPONENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SECRET_KEY_RE = re.compile(
    r"(?:api[_-]?key|authorization|token|cookie|password|secret|prompt|reasoning|memory_context|tool_result)",
    re.IGNORECASE,
)
_SECRET_TEXT_RE = re.compile(
    r"(?i)\b(api[_-]?key|authorization|access[_-]?token|refresh[_-]?token|cookie|password|secret)"
    r"\s*[:=]\s*([^\s,;]+)"
)
_PUBLIC_URL_RE = re.compile(r"https?://[^\s<>\]\[\"']+", re.IGNORECASE)

_COMMON_KEYS = {
    "status",
    "error_code",
    "message",
    "recovered",
    "termination_reason",
    "result_status",
    "permission_denial_count",
}
_PAYLOAD_KEYS: dict[str, set[str]] = {
    "run.created": {"status"},
    "run.started": {"status", "recovered"},
    "run.resumed": {"status", "recovered"},
    "run.interrupted": {
        "status", "error_code", "message", "termination_reason", "result_status",
        "permission_denial_count",
    },
    "run.completed": {
        "status", "result_ref", "termination_reason", "result_status",
        "permission_denial_count",
    },
    "run.failed": {
        "status", "error_code", "message", "termination_reason", "result_status",
        "permission_denial_count",
    },
    "run.cancelled": {
        "status", "termination_reason", "result_status", "permission_denial_count",
    },
    "stage.started": {"stage_id", "stage_index", "stage_count"},
    "stage.completed": {"stage_id", "stage_index", "stage_count"},
    "stage.failed": {"stage_id", "stage_index", "stage_count", "error_code", "message"},
    "plan.created": {"plan_id", "revision", "objective", "stages"},
    "plan.revised": {"plan_id", "revision"},
    "plan.task.added": {"task_id", "wave_id", "title", "mode", "status"},
    "research.wave.started": {"wave_id", "mode", "task_ids", "task_count"},
    "research.wave.completed": {
        "wave_id", "mode", "task_ids", "task_count", "completed", "failed", "rejected"
    },
    "research.task.created": {
        "task_id", "wave_id", "plan_task_id", "title", "mode", "status", "phase"
    },
    "research.task.started": {
        "task_id", "wave_id", "plan_task_id", "title", "mode", "status", "phase"
    },
    "research.task.progress": {
        "task_id", "wave_id", "mode", "status", "phase", "iteration", "source_count",
        "tool_categories", "activity_phase", "activity_label", "last_activity_at",
        "activity_event_count", "model_call_count", "tool_call_count", "retry_count",
        "warning_count", "activity_available",
    },
    "research.source.discovered": {"task_id", "source_id", "title", "domain", "url"},
    "research.task.completed": {
        "task_id", "wave_id", "mode", "status", "phase", "elapsed_ms", "source_count",
        "admission_status", "reason_code", "summary_status", "message"
    },
    "research.task.failed": {
        "task_id", "wave_id", "mode", "status", "phase", "elapsed_ms", "error_code", "message"
    },
    "research.task.cancelled": {"task_id", "wave_id", "mode", "status", "phase", "elapsed_ms"},
    "research.task.timed_out": {
        "task_id", "wave_id", "mode", "status", "phase", "elapsed_ms", "error_code", "message"
    },
    "findings.updated": {"task_id", "wave_id", "summary", "sources", "source_count"},
    "report.started": {"status"},
    "report.completed": {"status", "result_ref", "sha256", "length"},
    "approval.required": {
        "action_id", "approval_type", "status", "plan_id", "revision",
        "content_markdown", "allowed_actions",
    },
    "approval.resolved": {"action_id", "approval_type", "action", "status"},
    "clarification.required": {
        "action_id", "question", "status", "allowed_actions",
    },
    "clarification.resolved": {"action_id", "action", "status"},
    "feedback.received": {"feedback_id", "feedback_type", "task_id", "status"},
    "query.model_fallback": {"turn", "from_model", "to_model", "reason"},
    "model.circuit_state": {
        "provider", "model", "from_state", "to_state", "reason",
        "failure_count", "slow_count", "sample_count", "slow_ratio",
        "cooldown_seconds", "forced_probe",
    },
    "system.warning": {"warning_code", "message"},
}


class PublicEventError(RuntimeError):
    """Base error for public-event persistence."""


class PublicEventLogCorrupted(PublicEventError):
    """Raised when a non-tail event record cannot be recovered."""


class PublicEvent(BaseModel):
    """One stable event exposed to external clients."""

    schema_version: int = PUBLIC_EVENT_SCHEMA_VERSION
    event_id: str
    sequence: int = Field(ge=1)
    run_id: str
    type: str
    timestamp: str
    stage: Optional[Literal[
        "preparing", "planning", "researching", "synthesizing", "writing", "finalizing"
    ]] = None
    payload: dict[str, Any] = Field(default_factory=dict)
    dedupe_key: str

    def public_dict(self) -> dict[str, Any]:
        """Return the wire representation without persistence-only metadata."""
        return self.model_dump(exclude={"dedupe_key"})


class PublicRunProjection(BaseModel):
    """Replay-derived current state for a public research run."""

    status: str = "pending"
    current_stage: Optional[str] = None
    stage_index: int = 0
    stage_count: int = len(PUBLIC_STAGES)
    plan: dict[str, Any] = Field(default_factory=dict)
    tasks: dict[str, int] = Field(default_factory=lambda: {
        "total": 0,
        "pending": 0,
        "running": 0,
        "completed": 0,
        "failed": 0,
        "cancelled": 0,
        "timed_out": 0,
    })
    waves: dict[str, int] = Field(default_factory=lambda: {"total": 0, "completed": 0})
    latest_findings: list[dict[str, Any]] = Field(default_factory=list)
    task_items: dict[str, dict[str, Any]] = Field(default_factory=dict)
    sources: list[dict[str, Any]] = Field(default_factory=list)
    pending_human_action: Optional[dict[str, Any]] = None
    last_event_id: int = 0


class PublicFindingsSummary(BaseModel):
    """Structured, user-visible summary produced from compressed findings."""

    findings: list[str] = Field(default_factory=list, max_length=3)


def utc_timestamp() -> str:
    """Return an RFC 3339 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def public_display_title(value: str, *, limit: int = 160) -> str:
    """Derive a short display label without exposing a complete task contract."""
    text = _SECRET_TEXT_RE.sub(lambda match: f"{match.group(1)}=[REDACTED]", str(value))
    for prefix in ("Objective:", "**Objective**:", "目标：", "目标:"):
        if prefix in text:
            text = text.split(prefix, 1)[1]
            break
    first = next((part.strip(" -*#\t") for part in text.splitlines() if part.strip()), "Research task")
    first = re.split(r"(?<=[.!?。！？])\s+", first, maxsplit=1)[0]
    return first[:limit] or "Research task"


def canonical_public_source(url: str, title: str = "") -> Optional[dict[str, str]]:
    """Return a safe public source reference, dropping credentials and URL queries."""
    try:
        parsed = urlsplit(str(url).strip())
    except ValueError:
        return None
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
        return None
    host = parsed.hostname.lower()
    try:
        port = f":{parsed.port}" if parsed.port else ""
    except ValueError:
        return None
    canonical = urlunsplit((parsed.scheme.lower(), f"{host}{port}", parsed.path or "/", "", ""))
    return {
        "source_id": hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:24],
        "title": public_display_title(title or host, limit=200),
        "domain": host,
        "url": canonical,
    }


def extract_public_sources(result: dict[str, Any], *, limit: int = 10) -> list[dict[str, str]]:
    """Extract and canonicalize source metadata without exposing result bodies."""
    found: dict[str, dict[str, str]] = {}

    def visit(value: Any) -> None:
        if len(found) >= limit:
            return
        if isinstance(value, dict):
            url = value.get("url") or value.get("source_url") or value.get("link")
            if url:
                source = canonical_public_source(
                    str(url),
                    str(value.get("title") or value.get("name") or ""),
                )
                if source is not None:
                    found.setdefault(source["source_id"], source)
            for key, item in value.items():
                if key not in {"content", "raw_content", "text", "snippet"}:
                    visit(item)
        elif isinstance(value, list | tuple):
            for item in value:
                visit(item)
        elif isinstance(value, str):
            for raw_url in _PUBLIC_URL_RE.findall(value):
                source = canonical_public_source(raw_url.rstrip(".,;:!?)}`"))
                if source is not None:
                    found.setdefault(source["source_id"], source)

    for registry_key in (
        "evidence_registry",
        "document_registry",
        "candidate_registry",
        "sources",
        "citations",
        "source_urls",
        "compressed_research",
        "raw_notes",
    ):
        visit(result.get(registry_key, []))
    return list(found.values())[:limit]


async def summarize_public_findings(
    result: dict[str, Any],
    config: RunnableConfig,
) -> Optional[str]:
    """Create a bounded public summary; failure never exposes raw findings."""
    compressed = str(result.get("compressed_research") or "").strip()
    if not compressed:
        return None
    configurable = Configuration.from_runnable_config(config)
    try:
        from langchain.chat_models import init_chat_model
        from langchain_core.messages import HumanMessage

        from open_deep_research.models.resolution import build_model_config

        model = init_chat_model(**build_model_config(
            configurable.summarization_model,
            min(configurable.summarization_model_max_tokens, 800),
            config,
            role="summarization",
        )).with_structured_output(PublicFindingsSummary, method="function_calling")
        response = await model.ainvoke([
            HumanMessage(content=(
                "Summarize the completed research into at most three concise user-visible findings. "
                "Do not mention prompts, hidden reasoning, tool internals, credentials, or implementation details. "
                "Preserve uncertainty and do not add claims.\n\nCompressed research:\n"
                + compressed[:50_000]
            ))
        ])
        findings = [str(item).strip() for item in response.findings if str(item).strip()][:3]
        if not findings:
            return None
        return "\n".join(f"- {item}" for item in findings)[: configurable.public_event_summary_max_chars]
    except Exception:
        return None


def _sanitize_scalar(value: Any) -> Any:
    if value is None or isinstance(value, bool | int | float):
        return value
    text = _SECRET_TEXT_RE.sub(lambda match: f"{match.group(1)}=[REDACTED]", str(value))
    return text


def _sanitize_nested(value: Any, *, depth: int = 0) -> Any:
    if depth > 5:
        return "[TRUNCATED]"
    if isinstance(value, dict):
        return {
            str(key): _sanitize_nested(item, depth=depth + 1)
            for key, item in value.items()
            if not _SECRET_KEY_RE.search(str(key))
        }
    if isinstance(value, list | tuple):
        return [_sanitize_nested(item, depth=depth + 1) for item in value[:100]]
    return _sanitize_scalar(value)


def sanitize_public_payload(event_type: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Apply an event-specific field allowlist and generic secret redaction."""
    allowed = _PAYLOAD_KEYS.get(event_type, _COMMON_KEYS)
    return {
        key: _sanitize_nested(value)
        for key, value in payload.items()
        if key in allowed and not _SECRET_KEY_RE.search(key)
    }


class RunEventStore:
    """Inter-process-safe append-only public event store for one run."""

    def __init__(
        self,
        run_id: str,
        *,
        runs_dir: str = ".runs",
        lock_timeout_seconds: float = 30,
        fence_token: int | None = None,
        fence_owner_id: str | None = None,
    ) -> None:
        """Initialize paths and the inter-process lock timeout."""
        if not _COMPONENT_RE.fullmatch(run_id) or ".." in run_id:
            raise ValueError("Invalid run_id")
        self.run_id = run_id
        self.run_dir = Path(runs_dir).resolve() / run_id
        self.path = self.run_dir / "public_events.jsonl"
        self.index_path = self.run_dir / "public_events_index.json"
        self.lock_path = self.run_dir / "public_events.lock"
        self.lock_timeout_seconds = lock_timeout_seconds
        self.fence_token = fence_token
        self.fence_owner_id = fence_owner_id

    def _fenced_context(self):
        """Return a bound run context and lease guard for event commits."""
        if self.fence_token is None or not self.fence_owner_id:
            return None, nullcontext()
        from open_deep_research.run_context import RunContextStore

        context = RunContextStore(self.run_id, runs_dir=str(self.run_dir.parent))
        context.bind_fence_token(
            self.fence_token,
            self.fence_owner_id,
            advance_manifest=False,
        )
        return context, context._fence_guard()  # noqa: SLF001

    @property
    def exists(self) -> bool:
        """Return whether this run has a public event stream."""
        return self.path.exists()

    def _read_records_unlocked(self, *, repair_tail: bool = True) -> list[PublicEvent]:
        if not self.path.exists():
            return []
        content = self.path.read_bytes()
        has_complete_tail = content.endswith(b"\n")
        raw_lines = content.splitlines()
        records: list[PublicEvent] = []
        expected = 1
        for index, raw in enumerate(raw_lines):
            if not raw.strip():
                continue
            try:
                event = PublicEvent.model_validate_json(raw)
            except Exception as exc:
                is_partial_tail = index == len(raw_lines) - 1 and not has_complete_tail
                if repair_tail and is_partial_tail:
                    valid = b"\n".join(raw_lines[:index])
                    if valid:
                        valid += b"\n"
                    with self.path.open("wb") as handle:
                        handle.write(valid)
                        handle.flush()
                        os.fsync(handle.fileno())
                    break
                raise PublicEventLogCorrupted("public_event_log_corrupted") from exc
            if event.run_id != self.run_id or event.sequence != expected:
                raise PublicEventLogCorrupted("public_event_log_corrupted")
            records.append(event)
            expected += 1
        return records

    def _mark_event_persistence_failed(self, error: Exception) -> None:
        try:
            from open_deep_research.run_context import RunContextStore

            context = RunContextStore(self.run_id, runs_dir=str(self.run_dir.parent))
            if context.manifest_path.exists():
                context.mark_event_persistence_failed(error)
        except Exception:
            pass

    def _read_records_locked(self) -> list[PublicEvent]:
        try:
            with portalocker.Lock(str(self.lock_path), mode="a+b", timeout=self.lock_timeout_seconds):
                return self._read_records_unlocked()
        except Exception as exc:
            self._mark_event_persistence_failed(exc)
            raise

    def read(self, after: int = 0) -> list[PublicEvent]:
        """Return durable events with sequence greater than ``after``."""
        if not self.path.exists():
            return []
        records = self._read_records_locked()
        return [event for event in records if event.sequence > after]

    def last_sequence(self) -> int:
        """Return the last durable sequence, or zero for an empty stream."""
        if not self.path.exists():
            return 0
        records = self._read_records_locked()
        return records[-1].sequence if records else 0

    def _write_index(self, last_sequence: int, keys: dict[str, int]) -> None:
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.index_path.with_name(f".{self.index_path.name}.{uuid.uuid4().hex}.tmp")
        data = {"schema_version": 1, "last_sequence": last_sequence, "keys": keys}
        try:
            with temp.open("w", encoding="utf-8") as handle:
                json.dump(data, handle, ensure_ascii=False, separators=(",", ":"))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp, self.index_path)
        finally:
            if temp.exists():
                temp.unlink()

    def _load_index(self, records: list[PublicEvent]) -> tuple[int, dict[str, int]]:
        try:
            data = json.loads(self.index_path.read_text(encoding="utf-8"))
            keys = {str(key): int(value) for key, value in data.get("keys", {}).items()}
            last = int(data.get("last_sequence", 0))
            expected_keys = {event.dedupe_key: event.sequence for event in records}
            if last == (records[-1].sequence if records else 0) and keys == expected_keys:
                return last, keys
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass
        keys = {event.dedupe_key: event.sequence for event in records}
        last = records[-1].sequence if records else 0
        self._write_index(last, keys)
        return last, keys

    def _append_sync(
        self,
        event_type: str,
        *,
        stage: Optional[str],
        payload: dict[str, Any],
        dedupe_key: str,
    ) -> PublicEvent:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        context, fence_guard = self._fenced_context()
        with fence_guard:
            with portalocker.Lock(
                str(self.lock_path),
                mode="a+b",
                timeout=self.lock_timeout_seconds,
            ):
                records = self._read_records_unlocked()
                last, keys = self._load_index(records)
                terminal = next(
                    (
                        event
                        for event in reversed(records)
                        if event.type in TERMINAL_EVENT_TYPES
                    ),
                    None,
                )
                if dedupe_key in keys:
                    sequence = keys[dedupe_key]
                    existing = next(
                        event for event in records if event.sequence == sequence
                    )
                    if existing.type != event_type and (
                        existing.type in TERMINAL_EVENT_TYPES
                        or event_type in TERMINAL_EVENT_TYPES
                    ):
                        raise RuntimeError("terminal_event_conflict")
                    return existing
                if terminal is not None:
                    if event_type in TERMINAL_EVENT_TYPES:
                        raise RuntimeError("terminal_event_conflict")
                    raise RuntimeError("run_already_terminal")
                sequence = last + 1
                event = PublicEvent(
                    event_id=str(
                        uuid.uuid5(uuid.NAMESPACE_URL, f"{self.run_id}:{dedupe_key}")
                    ),
                    sequence=sequence,
                    run_id=self.run_id,
                    type=event_type,
                    timestamp=utc_timestamp(),
                    stage=stage,
                    payload=sanitize_public_payload(event_type, payload),
                    dedupe_key=dedupe_key,
                )
                data = (event.model_dump_json() + "\n").encode("utf-8")
                fd = os.open(
                    self.path,
                    os.O_APPEND | os.O_CREAT | os.O_WRONLY,
                    0o600,
                )
                try:
                    os.write(fd, data)
                    os.fsync(fd)
                finally:
                    os.close(fd)
                keys[dedupe_key] = sequence
                self._write_index(sequence, keys)
                try:
                    if context is None:
                        from open_deep_research.run_context import RunContextStore

                        context = RunContextStore(
                            self.run_id,
                            runs_dir=str(self.run_dir.parent),
                        )
                    if context.manifest_path.exists():
                        context._update_manifest(  # noqa: SLF001
                            fence_already_held=self.fence_token is not None,
                            last_public_event_seq=sequence,
                        )
                except Exception:
                    # The event log remains authoritative; manifest is only a fast cache.
                    pass
                return event

    async def append(
        self,
        event_type: str,
        *,
        stage: Optional[str] = None,
        payload: Optional[dict[str, Any]] = None,
        dedupe_key: str,
    ) -> PublicEvent:
        """Persist one idempotent public event."""
        try:
            return await asyncio.to_thread(
                self._append_sync,
                event_type,
                stage=stage,
                payload=payload or {},
                dedupe_key=dedupe_key,
            )
        except RuntimeError as exc:
            if str(exc) in {"terminal_event_conflict", "run_already_terminal"}:
                raise
            self._mark_event_persistence_failed(exc)
            raise
        except Exception as exc:
            self._mark_event_persistence_failed(exc)
            raise

    def project(self) -> PublicRunProjection:
        """Replay this stream into its current public projection."""
        if not self.path.exists():
            return PublicRunProjection()
        records = self._read_records_locked()
        return project_public_events(records)


class RunEventPublisher:
    """Small async facade used by runtime components."""

    def __init__(self, store: RunEventStore) -> None:
        """Wrap a run-scoped durable event store."""
        self.store = store

    async def publish(
        self,
        event_type: str,
        *,
        stage: Optional[str] = None,
        payload: Optional[dict[str, Any]] = None,
        dedupe_key: str,
    ) -> PublicEvent:
        """Persist and return one sanitized public event."""
        return await self.store.append(
            event_type,
            stage=stage,
            payload=payload,
            dedupe_key=dedupe_key,
        )


def event_store_from_config(config: RunnableConfig | dict[str, Any]) -> RunEventStore:
    """Build a run-scoped store from runtime configuration."""
    configurable = Configuration.from_runnable_config(config)
    metadata = config.get("metadata") or {}
    run_id = str(metadata.get("run_id", "default"))
    fence_token = metadata.get("run_fence_token")
    return RunEventStore(
        run_id,
        runs_dir=configurable.runs_dir,
        fence_token=int(fence_token) if fence_token is not None else None,
        fence_owner_id=(
            str(metadata["run_lease_owner_id"])
            if metadata.get("run_lease_owner_id")
            else None
        ),
    )


def event_publisher_from_config(config: RunnableConfig | dict[str, Any]) -> RunEventPublisher:
    """Build a run-scoped public event publisher from runtime config."""
    return RunEventPublisher(event_store_from_config(config))


def project_public_events(events: list[PublicEvent]) -> PublicRunProjection:
    """Reduce public events into a deterministic client-facing progress snapshot."""
    projection = PublicRunProjection()
    task_statuses: dict[str, str] = {}
    waves: set[str] = set()
    completed_waves: set[str] = set()
    findings: dict[str, dict[str, Any]] = {}
    sources: dict[str, dict[str, Any]] = {}
    for event in events:
        projection.last_event_id = event.sequence
        payload = event.payload
        if event.type.startswith("run.") and payload.get("status"):
            projection.status = str(payload["status"])
            if event.type in TERMINAL_EVENT_TYPES:
                projection.pending_human_action = None
        if event.type.startswith("stage.") and payload.get("stage_id"):
            projection.current_stage = str(payload["stage_id"])
            projection.stage_index = int(payload.get("stage_index", 0))
        elif event.type in {"plan.created", "plan.revised"}:
            if event.type == "plan.created":
                projection.plan = dict(payload)
            else:
                projection.plan.update(payload)
        elif event.type == "plan.task.added":
            tasks = projection.plan.setdefault("tasks", [])
            if not any(item.get("task_id") == payload.get("task_id") for item in tasks):
                tasks.append(dict(payload))
            task_statuses[str(payload.get("task_id"))] = str(payload.get("status", "pending"))
            projection.task_items[str(payload.get("task_id"))] = dict(payload)
        elif event.type.startswith("research.task."):
            task_id = str(payload.get("task_id", ""))
            if task_id:
                status = payload.get("status")
                if status:
                    task_statuses[task_id] = str(status)
                previous = projection.task_items.get(task_id, {})
                merged = {**previous, **dict(payload)}
                for monotonic_key in ("iteration", "source_count"):
                    prior_value = previous.get(monotonic_key)
                    next_value = payload.get(monotonic_key)
                    if isinstance(prior_value, int) and isinstance(next_value, int):
                        merged[monotonic_key] = max(prior_value, next_value)
                projection.task_items[task_id] = merged
        elif event.type == "research.source.discovered":
            source_key = str(payload.get("source_id") or payload.get("url") or "")
            if source_key:
                sources[source_key] = dict(payload)
        elif event.type == "research.wave.started":
            waves.add(str(payload.get("wave_id")))
        elif event.type == "research.wave.completed":
            wave_id = str(payload.get("wave_id"))
            waves.add(wave_id)
            completed_waves.add(wave_id)
        elif event.type == "findings.updated":
            findings[str(payload.get("task_id"))] = dict(payload)
        elif event.type == "approval.required":
            approval_type = str(payload.get("approval_type") or "plan")
            projection.status = f"awaiting_{approval_type}_approval"
            projection.pending_human_action = {
                "action_id": payload.get("action_id"),
                "type": f"{approval_type}_approval",
                "payload": {
                    "content_markdown": payload.get("content_markdown", ""),
                },
                "allowed_actions": payload.get("allowed_actions", ["approve", "revise", "cancel"]),
            }
        elif event.type == "clarification.required":
            projection.status = "awaiting_clarification"
            projection.pending_human_action = {
                "action_id": payload.get("action_id"),
                "type": "clarification",
                "payload": {"question": payload.get("question", "")},
                "allowed_actions": payload.get("allowed_actions", ["answer", "cancel"]),
            }
        elif event.type in {"approval.resolved", "clarification.resolved"}:
            pending = projection.pending_human_action or {}
            if pending.get("action_id") == payload.get("action_id"):
                projection.pending_human_action = None
            projection.status = "running"

    counts = {key: 0 for key in ("pending", "running", "completed", "failed", "cancelled", "timed_out")}
    for status in task_statuses.values():
        normalized = "running" if status in {"created", "researching", "compressing"} else status
        if normalized in counts:
            counts[normalized] += 1
    projection.tasks = {"total": len(task_statuses), **counts}
    projection.waves = {"total": len(waves), "completed": len(completed_waves)}
    projection.latest_findings = list(findings.values())
    projection.sources = list(sources.values())
    return projection


def is_terminal_event(event: PublicEvent) -> bool:
    """Return whether an event closes the public stream."""
    return event.type in TERMINAL_EVENT_TYPES
