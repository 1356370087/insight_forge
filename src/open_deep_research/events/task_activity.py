"""Durable, sanitized activity streams for individual research tasks."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import portalocker
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from open_deep_research.configuration import Configuration
from open_deep_research.events.public import canonical_public_source, utc_timestamp

logger = logging.getLogger(__name__)

PUBLIC_TASK_ACTIVITY_SCHEMA_VERSION = 1
TASK_TERMINAL_TYPES = {
    "task.completed",
    "task.failed",
    "task.cancelled",
    "task.timed_out",
}

ActivityKind = Literal[
    "lifecycle",
    "model",
    "tool",
    "source",
    "quality",
    "checkpoint",
    "control",
    "security",
    "error",
]
ActivityPhase = Literal[
    "queued",
    "initializing",
    "reasoning",
    "tool_execution",
    "evidence_review",
    "quality_check",
    "gap_recovery",
    "compressing",
    "handoff",
    "terminal",
]
ActivityStatus = Literal[
    "pending",
    "running",
    "success",
    "warning",
    "error",
    "cancelled",
]

_COMPONENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SECRET_KEY_RE = re.compile(
    r"(?:api[_-]?key|authorization|(?:access|refresh|bearer)[_-]?token|cookie|password|"
    r"secret|prompt|reasoning_(?:content|text)|chain_of_thought|memory_context|"
    r"tool_result|raw_(?:content|notes)|content|messages?)",
    re.IGNORECASE,
)
_SECRET_TEXT_RE = re.compile(
    r"(?i)\b(api[_-]?key|authorization|access[_-]?token|refresh[_-]?token|cookie|"
    r"password|secret)\s*[:=]\s*([^\s,;]+)"
)
_PAYLOAD_KEYS: dict[str, set[str]] = {
    "model.started": {"provider", "model", "attempt"},
    "model.completed": {
        "provider", "model", "input_tokens", "output_tokens", "reasoning_tokens",
        "tool_call_count", "retry_count",
    },
    "model.retrying": {"provider", "model", "attempt", "error_code", "delay_ms"},
    "model.failed": {"provider", "model", "error_code", "retry_count"},
    "model.circuit_open": {
        "provider", "model", "reason", "cooldown_seconds", "failure_count",
        "slow_count", "sample_count",
    },
    "model.circuit_recovered": {
        "provider", "model", "reason", "cooldown_seconds", "failure_count",
        "slow_count", "sample_count",
    },
    "tool.started": {
        "tool_call_id", "tool_name", "tool_category", "args_summary", "args_keys",
    },
    "tool.completed": {
        "tool_call_id", "tool_name", "tool_category", "source_count", "result_chars",
        "retry_count", "urls",
    },
    "tool.failed": {
        "tool_call_id", "tool_name", "tool_category", "error_code", "retry_count",
    },
    "source.discovered": {"source_id", "title", "domain", "url"},
    "quality.started": {"evaluation_type", "attempt"},
    "quality.completed": {
        "evaluation_type", "decision", "admission_status", "scores", "gap_count",
        "recovery_attempt",
    },
    "quality.failed": {"evaluation_type", "error_code"},
    "checkpoint.loaded": {"completed_queries", "restored"},
    "checkpoint.saved": {"completed_queries"},
    "control.received": {"control_type", "instruction_summary"},
    "security.blocked": {"reason_code", "tool_name", "domain"},
    "task.phase.changed": {"activity_label"},
    "task.started": {"mode", "wave_id"},
    "task.completed": {"mode", "wave_id", "source_count", "admission_status"},
    "task.failed": {"error_code", "mode", "wave_id"},
    "task.cancelled": {"mode", "wave_id"},
    "task.timed_out": {"error_code", "timeout_seconds", "mode", "wave_id"},
    "recovery.started": {"attempt", "requirement_count"},
    "recovery.completed": {"attempt", "decision"},
}


class PublicTaskActivityEvent(BaseModel):
    """One stable, browser-safe event within a single Subagent task."""

    schema_version: int = PUBLIC_TASK_ACTIVITY_SCHEMA_VERSION
    event_id: str
    sequence: int = Field(ge=1)
    run_id: str
    task_id: str
    timestamp: str
    type: str
    kind: ActivityKind
    phase: ActivityPhase
    status: ActivityStatus
    title: str = Field(max_length=120)
    summary: str = Field(max_length=500)
    iteration: int | None = Field(default=None, ge=0)
    duration_ms: int | None = Field(default=None, ge=0)
    payload: dict[str, Any] = Field(default_factory=dict)
    dedupe_key: str

    def public_dict(self) -> dict[str, Any]:
        """Return the wire shape without persistence-only metadata."""
        return self.model_dump(exclude={"dedupe_key"})


def _clean_text(value: Any, *, limit: int) -> str:
    text = " ".join(str(value or "").split())
    text = _SECRET_TEXT_RE.sub(lambda match: f"{match.group(1)}=[REDACTED]", text)
    return text[:limit]


def _sanitize_nested(value: Any, *, depth: int = 0) -> Any:
    if depth > 3:
        return "[TRUNCATED]"
    if value is None or isinstance(value, bool | int | float):
        return value
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in list(value.items())[:30]:
            safe_key = str(key)
            if _SECRET_KEY_RE.search(safe_key):
                continue
            result[safe_key] = _sanitize_nested(item, depth=depth + 1)
        return result
    if isinstance(value, list | tuple):
        return [_sanitize_nested(item, depth=depth + 1) for item in value[:20]]
    return _clean_text(value, limit=500)


def sanitize_task_activity_payload(event_type: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Apply a strict event-specific allowlist and normalize public URLs."""
    allowed = _PAYLOAD_KEYS.get(event_type, set())
    result = {
        key: _sanitize_nested(value)
        for key, value in payload.items()
        if key in allowed and not _SECRET_KEY_RE.search(key)
    }
    if event_type == "source.discovered" and result.get("url"):
        source = canonical_public_source(str(result["url"]), str(result.get("title", "")))
        return source or {}
    urls = result.get("urls")
    if isinstance(urls, list):
        safe_urls: list[str] = []
        for url in urls:
            source = canonical_public_source(str(url))
            if source and source["url"] not in safe_urls:
                safe_urls.append(source["url"])
        result["urls"] = safe_urls[:20]
    return result


class TaskActivityStore:
    """Inter-process-safe append-only activity store for one research task."""

    def __init__(
        self,
        run_id: str,
        task_id: str,
        *,
        runs_dir: str = ".runs",
        lock_timeout_seconds: float = 30,
    ) -> None:
        """Initialize the task-local activity paths and retention limits."""
        if not _COMPONENT_RE.fullmatch(run_id) or ".." in run_id:
            raise ValueError("Invalid run_id")
        if not _COMPONENT_RE.fullmatch(task_id) or ".." in task_id:
            raise ValueError("Invalid task_id")
        self.run_id = run_id
        self.task_id = task_id
        self.directory = Path(runs_dir).resolve() / run_id / "task_activity"
        self.path = self.directory / f"{task_id}.jsonl"
        self.index_path = self.directory / f"{task_id}.index.json"
        self.lock_path = self.directory / f"{task_id}.lock"
        self.lock_timeout_seconds = lock_timeout_seconds

    @property
    def exists(self) -> bool:
        """Return whether this task has a native activity journal."""
        return self.path.exists()

    def _read_unlocked(self, *, repair_tail: bool = True) -> list[PublicTaskActivityEvent]:
        if not self.path.exists():
            return []
        content = self.path.read_bytes()
        complete_tail = content.endswith(b"\n")
        raw_lines = content.splitlines()
        records: list[PublicTaskActivityEvent] = []
        for index, raw in enumerate(raw_lines):
            if not raw.strip():
                continue
            try:
                event = PublicTaskActivityEvent.model_validate_json(raw)
            except Exception as exc:
                partial_tail = index == len(raw_lines) - 1 and not complete_tail
                if repair_tail and partial_tail:
                    valid = b"\n".join(raw_lines[:index])
                    self.path.write_bytes(valid + (b"\n" if valid else b""))
                    break
                raise RuntimeError("task_activity_log_corrupted") from exc
            expected = len(records) + 1
            if (
                event.run_id != self.run_id
                or event.task_id != self.task_id
                or event.sequence != expected
            ):
                raise RuntimeError("task_activity_log_corrupted")
            records.append(event)
        return records

    def _read_locked(self) -> list[PublicTaskActivityEvent]:
        with portalocker.Lock(
            str(self.lock_path), mode="a+b", timeout=self.lock_timeout_seconds
        ):
            return self._read_unlocked()

    def read(self, after: int = 0) -> list[PublicTaskActivityEvent]:
        """Read valid activity events strictly after a sequence cursor."""
        if not self.path.exists():
            return []
        return [event for event in self._read_locked() if event.sequence > after]

    def page(
        self,
        *,
        before: int | None = None,
        limit: int = 100,
        kind: str | None = None,
    ) -> tuple[list[PublicTaskActivityEvent], bool, int]:
        """Return a reverse-page window ordered from oldest to newest."""
        records = self._read_locked() if self.path.exists() else []
        last_sequence = records[-1].sequence if records else 0
        filtered = [
            event
            for event in records
            if (before is None or event.sequence < before)
            and (kind is None or event.kind == kind)
        ]
        has_more = len(filtered) > limit
        return filtered[-limit:], has_more, last_sequence

    def last_sequence(self) -> int:
        """Return the most recent durable sequence for this task."""
        if not self.path.exists():
            return 0
        records = self._read_locked()
        return records[-1].sequence if records else 0

    def _write_index(self, records: list[PublicTaskActivityEvent]) -> None:
        data = {
            "schema_version": 1,
            "last_sequence": records[-1].sequence if records else 0,
            "keys": {event.dedupe_key: event.sequence for event in records},
        }
        temp = self.index_path.with_name(f".{self.index_path.name}.{uuid.uuid4().hex}.tmp")
        temp.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        os.replace(temp, self.index_path)

    def _append_sync(
        self,
        event_type: str,
        *,
        kind: ActivityKind,
        phase: ActivityPhase,
        status: ActivityStatus,
        title: str,
        summary: str,
        iteration: int | None,
        duration_ms: int | None,
        payload: dict[str, Any],
        dedupe_key: str,
    ) -> PublicTaskActivityEvent:
        self.directory.mkdir(parents=True, exist_ok=True)
        with portalocker.Lock(
            str(self.lock_path), mode="a+b", timeout=self.lock_timeout_seconds
        ):
            records = self._read_unlocked()
            existing = next((item for item in records if item.dedupe_key == dedupe_key), None)
            if existing is not None:
                return existing
            event = PublicTaskActivityEvent(
                event_id=str(uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    f"{self.run_id}:{self.task_id}:{dedupe_key}",
                )),
                sequence=len(records) + 1,
                run_id=self.run_id,
                task_id=self.task_id,
                timestamp=utc_timestamp(),
                type=event_type,
                kind=kind,
                phase=phase,
                status=status,
                title=_clean_text(title, limit=120) or "任务活动",
                summary=_clean_text(summary, limit=500),
                iteration=iteration,
                duration_ms=duration_ms,
                payload=sanitize_task_activity_payload(event_type, payload),
                dedupe_key=dedupe_key,
            )
            data = (event.model_dump_json() + "\n").encode("utf-8")
            fd = os.open(self.path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
            try:
                os.write(fd, data)
                os.fsync(fd)
            finally:
                os.close(fd)
            records.append(event)
            self._write_index(records)
            return event

    async def append(self, event_type: str, **kwargs: Any) -> PublicTaskActivityEvent:
        """Append an activity event without blocking the async runtime loop."""
        return await asyncio.to_thread(self._append_sync, event_type, **kwargs)


class TaskActivityPublisher:
    """Fail-open publisher used by runtime instrumentation."""

    def __init__(self, store: TaskActivityStore) -> None:
        """Initialize a fail-open publisher around a durable store."""
        self.store = store

    async def publish(self, event_type: str, **kwargs: Any) -> PublicTaskActivityEvent | None:
        """Publish an activity event, swallowing observability-only failures."""
        try:
            return await self.store.append(event_type, **kwargs)
        except Exception as exc:  # noqa: BLE001 - observability must not fail research
            logger.warning("Task activity write failed: %s", exc)
            return None


def task_activity_store_from_config(
    config: RunnableConfig | dict[str, Any],
    task_id: str | None = None,
) -> TaskActivityStore | None:
    """Create a task-local store using run metadata in a runnable config."""
    configurable = Configuration.from_runnable_config(config)
    metadata = config.get("metadata") or {}
    resolved_task_id = str(task_id or metadata.get("task_id") or "")
    if not resolved_task_id:
        return None
    return TaskActivityStore(
        str(metadata.get("run_id", "default")),
        resolved_task_id,
        runs_dir=configurable.runs_dir,
    )


def task_activity_publisher_from_config(
    config: RunnableConfig | dict[str, Any],
    task_id: str | None = None,
) -> TaskActivityPublisher | None:
    """Create a fail-open publisher when run and task metadata are available."""
    store = task_activity_store_from_config(config, task_id)
    return TaskActivityPublisher(store) if store is not None else None


async def publish_task_activity(
    config: RunnableConfig | dict[str, Any],
    event_type: str,
    *,
    task_id: str | None = None,
    update_run_summary: bool = False,
    **kwargs: Any,
) -> PublicTaskActivityEvent | None:
    """Publish one fail-open task event and optionally refresh its run card."""
    publisher = task_activity_publisher_from_config(config, task_id)
    if publisher is None:
        return None
    event = await publisher.publish(event_type, **kwargs)
    if event is None or not update_run_summary:
        return event
    try:
        from open_deep_research.events.public import event_publisher_from_config

        summary = activity_summary(publisher.store.read())
        metadata = config.get("metadata") or {}
        is_completed = event.type == "task.completed"
        is_cancelled = event.type == "task.cancelled"
        is_failed = event.type in {"task.failed", "task.timed_out"}
        public_phase = (
            "completed" if is_completed else
            "compressing" if event.phase == "compressing" else
            "researching"
        )
        public_status = (
            "completed" if is_completed else
            "cancelled" if is_cancelled else
            "failed" if is_failed else
            "running"
        )
        await event_publisher_from_config(config).publish(
            "research.task.progress",
            stage="researching",
            payload={
                "task_id": event.task_id,
                "wave_id": metadata.get("research_wave_id") or metadata.get("wave_id") or "",
                "mode": metadata.get("research_mode") or "sync",
                "status": public_status,
                "phase": public_phase,
                **summary,
            },
            dedupe_key=f"task:{event.task_id}:activity:{event.sequence}",
        )
    except Exception as exc:  # noqa: BLE001 - card projection is also fail-open
        logger.debug("Unable to update task activity projection: %s", exc)
    return event


def activity_summary(events: list[PublicTaskActivityEvent]) -> dict[str, Any]:
    """Project bounded task-card counters from an activity stream."""
    if not events:
        return {"activity_available": False}
    latest = events[-1]
    return {
        "activity_phase": latest.phase,
        "activity_label": latest.title,
        "last_activity_at": latest.timestamp,
        "activity_event_count": len(events),
        "model_call_count": sum(event.type == "model.completed" for event in events),
        "tool_call_count": sum(event.type == "tool.completed" for event in events),
        "retry_count": sum(event.type == "model.retrying" for event in events),
        "warning_count": sum(event.status in {"warning", "error"} for event in events),
        "activity_available": True,
    }


def derive_trace_activity(
    run_id: str,
    task_id: str,
    spans: list[dict[str, Any]],
) -> list[PublicTaskActivityEvent]:
    """Build a safe best-effort timeline for legacy synchronous tasks."""
    parsed: dict[str, dict[str, Any]] = {}
    root_id: str | None = None
    for row in spans:
        item = dict(row)
        try:
            attributes = json.loads(str(item.get("attributes_json") or "{}"))
        except (TypeError, ValueError, json.JSONDecodeError):
            attributes = {}
        item["_attributes"] = attributes
        span_id = str(item.get("span_id") or "")
        parsed[span_id] = item
        if (
            item.get("name") == "tool.ConductResearch"
            and str(attributes.get("tool_call_id") or "") == task_id
        ):
            root_id = span_id
    if root_id is None:
        return []
    descendants = {root_id}
    changed = True
    while changed:
        changed = False
        for span_id, item in parsed.items():
            if span_id not in descendants and str(item.get("parent_span_id") or "") in descendants:
                descendants.add(span_id)
                changed = True
    rows = sorted(
        (parsed[span_id] for span_id in descendants),
        key=lambda item: float(item.get("started_at") or 0),
    )
    events: list[PublicTaskActivityEvent] = []
    for row in rows:
        name = str(row.get("name") or "")
        kind = str(row.get("kind") or "")
        status_value: ActivityStatus = "error" if row.get("status") == "error" else "success"
        event_type = "task.phase.changed"
        event_kind: ActivityKind = "lifecycle"
        phase: ActivityPhase = "initializing"
        title = "初始化研究任务"
        payload: dict[str, Any] = {}
        if kind == "tool" and name != "tool.ConductResearch":
            event_type = "tool.completed" if status_value == "success" else "tool.failed"
            event_kind = "tool"
            phase = "tool_execution"
            tool_name = name.removeprefix("tool.")
            title = f"执行工具 · {tool_name}"
            attributes = row["_attributes"]
            payload = {
                "tool_call_id": attributes.get("tool_call_id"),
                "tool_name": tool_name,
                "tool_category": "search" if "search" in tool_name.lower() else "tool",
                "source_count": attributes.get("source_count", 0),
                "result_chars": attributes.get("result_chars", 0),
                "retry_count": row.get("retry_count", 0),
                "error_code": row.get("error_type"),
            }
        elif kind == "llm":
            event_kind = "quality" if row.get("agent_role") == "quality_evaluator" else "model"
            if event_kind == "quality":
                event_type = "quality.completed" if status_value == "success" else "quality.failed"
                phase = "quality_check"
                title = "质量复核"
                payload = {"evaluation_type": name, "error_code": row.get("error_type")}
            else:
                event_type = "model.completed" if status_value == "success" else "model.failed"
                phase = "compressing" if "compress" in name else "reasoning"
                title = "压缩研究发现" if phase == "compressing" else "模型规划"
                payload = {
                    "provider": row.get("provider"),
                    "model": row.get("model"),
                    "input_tokens": row.get("input_tokens", 0),
                    "output_tokens": row.get("output_tokens", 0),
                    "reasoning_tokens": row.get("reasoning_tokens", 0),
                    "retry_count": row.get("retry_count", 0),
                    "error_code": row.get("error_type"),
                }
        else:
            continue
        sequence = len(events) + 1
        events.append(PublicTaskActivityEvent(
            event_id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"legacy:{run_id}:{task_id}:{row.get('span_id')}")),
            sequence=sequence,
            run_id=run_id,
            task_id=task_id,
            timestamp=(
                datetime.fromtimestamp(float(row["started_at"]), tz=timezone.utc)
                .isoformat(timespec="milliseconds")
                .replace("+00:00", "Z")
                if row.get("started_at")
                else utc_timestamp()
            ),
            type=event_type,
            kind=event_kind,
            phase=phase,
            status=status_value,
            title=title,
            summary="由历史 Trace 安全推导，部分细节可能不可用。",
            duration_ms=max(0, int(row.get("duration_ms") or 0)),
            payload=sanitize_task_activity_payload(event_type, payload),
            dedupe_key=f"legacy:{row.get('span_id')}",
        ))
    return events
