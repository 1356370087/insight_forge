"""Fail-open tracing and usage collection for Deep Research runs."""

# ruff: noqa: D102,D105,D107,UP037

from __future__ import annotations

import asyncio
import contextvars
import hashlib
import inspect
import json
import logging
import random
import re
import sqlite3
import threading
import time
import uuid
from collections.abc import Awaitable, Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import (
    AIMessageChunk,
    BaseMessage,
    get_buffer_string,
    message_chunk_to_message,
)
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel

from open_deep_research.budgets import BudgetGate
from open_deep_research.configuration import Configuration
from open_deep_research.events.public import event_publisher_from_config
from open_deep_research.events.task_activity import publish_task_activity
from open_deep_research.models.circuit import (
    CircuitFailureKind,
    CircuitOpenError,
    CircuitPermit,
    CircuitTransition,
    ModelCircuitBreaker,
    ModelCircuitState,
    get_model_circuit_registry,
    model_circuit_policy_from_configuration,
)
from open_deep_research.observability.telemetry import (
    create_langfuse_sink,
    get_prometheus_metrics,
    monotonic_time,
)
from open_deep_research.security.redaction import redact_text as _redact_text

logger = logging.getLogger(__name__)

_current_run_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "open_deep_research_run_id",
    default=None,
)
_current_span_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "open_deep_research_span_id",
    default=None,
)
# The currently-entered SpanContext (set in SpanContext.__enter__). Governance
# retry code uses TraceRecorder.active_span() to reach it and record retries on
# the span opened by observe_tool_call, without needing the span handle itself.
_current_span_ctx: contextvars.ContextVar["SpanContext | NoopSpanContext | None"] = (
    contextvars.ContextVar("open_deep_research_span_ctx", default=None)
)
_stores: dict[str, "SQLiteTraceStore"] = {}
_stores_lock = threading.Lock()

_current_langfuse_span_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "open_deep_research_langfuse_span_id",
    default=None,
)


def _now() -> float:
    return time.time()


def _json(data: Any) -> str:
    try:
        return json.dumps(data, ensure_ascii=False, default=str)
    except Exception:  # noqa: BLE001 - observability must never fail business logic
        return "{}"


def _safe_int(value: Any) -> int:
    try:
        if value is None:
            return 0
        return int(value)
    except (TypeError, ValueError):
        return 0


def _safe_http_status(exc: BaseException) -> int | None:
    """Best-effort HTTP status extraction from an SDK/LangChain exception."""
    for attr in ("status_code", "status"):
        status = getattr(exc, attr, None)
        if isinstance(status, int):
            return status
    return None


def _is_uncertain_model_failure(exc: BaseException) -> bool:
    """Return whether the provider may have executed an indeterminate request."""
    if isinstance(exc, TimeoutError | ConnectionError | asyncio.CancelledError):
        return True
    name = type(exc).__name__.lower()
    message = _exc_message(exc).lower()
    uncertain_markers = (
        "timeout",
        "connection",
        "disconnect",
        "network",
        "transport",
        "brokenpipe",
        "incomplete read",
    )
    return any(marker in name or marker in message for marker in uncertain_markers)


def _exc_message(exc: BaseException) -> str:
    """Render an exception message, tolerating exceptions whose ``__str__`` raises."""
    try:
        text = str(exc)
    except Exception:  # noqa: BLE001
        text = ""
    return text or type(exc).__name__


def _provider_model(model_name: str | None) -> tuple[str | None, str | None]:
    if not model_name:
        return None, None
    if ":" not in model_name:
        return None, model_name
    provider, model = model_name.split(":", 1)
    return provider, model


_CIRCUIT_ROLE_STAGE = {
    "supervisor": "planning",
    "researcher": "researching",
    "summarization": "researching",
    "message_summary": "researching",
    "compression": "synthesizing",
    "final_report": "writing",
    "quality_evaluator": "finalizing",
    "quality_evaluation": "finalizing",
}


async def observe_model_circuit_transition(
    transition: CircuitTransition | None,
    config: RunnableConfig | None,
    *,
    agent_role: str | None = None,
) -> None:
    """Best-effort publish one circuit transition to every configured sink."""
    if transition is None:
        return
    try:
        recorder = get_trace_recorder(config)
        if recorder.prometheus is not None:
            recorder._safe(  # noqa: SLF001 - shared fail-open recorder boundary
                recorder.prometheus.observe_model_circuit_transition,
                transition,
            )
        provider, model = _provider_model(transition.model_id)
        payload = {
            "provider": provider or "unknown",
            "model": model or transition.model_id,
            "from_state": transition.from_state.value,
            "to_state": transition.to_state.value,
            "reason": transition.reason,
            "failure_count": transition.failure_count,
            "slow_count": transition.slow_count,
            "sample_count": transition.sample_count,
            "slow_ratio": transition.slow_ratio,
            "cooldown_seconds": transition.cooldown_seconds,
            "forced_probe": transition.forced_probe,
        }
        metadata = (config or {}).get("metadata") or {}
        if metadata.get("run_id"):
            await event_publisher_from_config(config or {}).publish(
                "model.circuit_state",
                stage=_CIRCUIT_ROLE_STAGE.get(agent_role or "", "researching"),
                payload=payload,
                dedupe_key=(
                    f"model-circuit:{transition.model_id}:"
                    f"{transition.timestamp}:{transition.to_state.value}"
                ),
            )
        if metadata.get("task_id") and transition.to_state in {
            ModelCircuitState.OPEN,
            ModelCircuitState.CLOSED,
        }:
            recovered = transition.to_state is ModelCircuitState.CLOSED
            await publish_task_activity(
                config or {},
                "model.circuit_recovered" if recovered else "model.circuit_open",
                kind="model",
                phase="reasoning",
                status="success" if recovered else "warning",
                title="模型线路已恢复" if recovered else "模型线路已暂时隔离",
                summary=(
                    "半开探针成功，后续调用已恢复。"
                    if recovered
                    else "连续可恢复故障达到阈值，后续调用将优先切换候选。"
                ),
                iteration=None,
                duration_ms=None,
                payload=payload,
                dedupe_key=(
                    f"activity:model-circuit:{transition.model_id}:"
                    f"{transition.timestamp}:{transition.to_state.value}"
                ),
                update_run_summary=True,
            )
    except Exception as exc:  # noqa: BLE001 - circuit observability fails open
        logger.debug("Model circuit transition publication failed open: %s", exc)


def _message_preview(messages: Any, limit: int | None, *, redact: bool = True) -> str | None:
    if limit is not None and limit <= 0:
        return None
    try:
        if isinstance(messages, list) and all(isinstance(m, BaseMessage) for m in messages):
            text = get_buffer_string(messages)
        elif isinstance(messages, BaseMessage):
            text = str(messages.content)
        else:
            text = str(messages)
    except Exception:  # noqa: BLE001
        return None
    if redact:
        text = _redact_text(text)
    if limit is None or len(text) <= limit:
        return text
    return text[:limit] + "\n[truncated]"


def current_span_ids() -> tuple[str | None, str | None]:
    """Return the current run/span ids carried by contextvars."""
    return _current_run_id.get(), _current_span_id.get()


@contextmanager
def bind_span_context(
    run_id: str | None,
    parent_span_id: str | None,
    langfuse_parent_span_id: str | None = None,
):
    """Explicitly bind trace context for background and cross-boundary work."""
    run_token = _current_run_id.set(run_id)
    span_token = _current_span_id.set(parent_span_id)
    langfuse_token = _current_langfuse_span_id.set(langfuse_parent_span_id)
    ctx_token = _current_span_ctx.set(None)
    try:
        yield
    finally:
        _current_span_ctx.reset(ctx_token)
        _current_langfuse_span_id.reset(langfuse_token)
        _current_span_id.reset(span_token)
        _current_run_id.reset(run_token)


@dataclass
class TokenUsage:
    """Normalized token usage across LangChain/provider metadata variants."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cached_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    reasoning_tokens: int = 0
    estimated_cost_usd: float = 0.0
    raw_usage: dict[str, Any] = field(default_factory=dict)
    estimated_input_tokens: int = 0
    estimated_output_tokens: int = 0
    estimated_total_tokens: int = 0
    usage_source: str = "provider_reported"
    cost_source: str = "unavailable"
    response_status: str = "success"

    @classmethod
    def from_response(cls, response: Any) -> "TokenUsage":
        """Extract usage in priority order from a model response."""
        candidates: list[dict[str, Any]] = []
        usage_metadata = getattr(response, "usage_metadata", None)
        if isinstance(usage_metadata, dict):
            candidates.append(usage_metadata)

        response_metadata = getattr(response, "response_metadata", None)
        if isinstance(response_metadata, dict):
            token_usage = response_metadata.get("token_usage")
            if isinstance(token_usage, dict):
                candidates.append(token_usage)
            usage = response_metadata.get("usage")
            if isinstance(usage, dict):
                candidates.append(usage)
        llm_output = getattr(response, "llm_output", None)
        if isinstance(llm_output, dict):
            for key in ("token_usage", "usage"):
                usage = llm_output.get(key)
                if isinstance(usage, dict):
                    candidates.append(usage)

        for usage in candidates:
            input_tokens = _safe_int(
                usage.get("input_tokens")
                or usage.get("prompt_tokens")
                or usage.get("input_token_count")
            )
            output_tokens = _safe_int(
                usage.get("output_tokens")
                or usage.get("completion_tokens")
                or usage.get("output_token_count")
            )
            total_tokens = _safe_int(usage.get("total_tokens"))
            input_details = usage.get("input_token_details") or usage.get("prompt_tokens_details") or {}
            output_details = usage.get("output_token_details") or usage.get("completion_tokens_details") or {}
            if not isinstance(input_details, dict):
                input_details = {}
            if not isinstance(output_details, dict):
                output_details = {}
            cached_input_tokens = _safe_int(
                usage.get("cached_input_tokens")
                or usage.get("cache_read_input_tokens")
                or input_details.get("cached_tokens")
                or input_details.get("cache_read")
            )
            cache_creation_input_tokens = _safe_int(
                usage.get("cache_creation_input_tokens")
                or input_details.get("cache_creation")
            )
            reasoning_tokens = _safe_int(
                usage.get("reasoning_tokens")
                or output_details.get("reasoning_tokens")
                or output_details.get("reasoning")
            )
            if not total_tokens:
                total_tokens = input_tokens + output_tokens
            if not any((input_tokens, output_tokens, total_tokens, cached_input_tokens, cache_creation_input_tokens, reasoning_tokens)):
                continue
            estimated_cost = usage.get("estimated_cost_usd") or usage.get("cost_usd") or usage.get("cost") or 0
            try:
                estimated_cost_usd = float(estimated_cost)
            except (TypeError, ValueError):
                estimated_cost_usd = 0.0
            return cls(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                cached_input_tokens=cached_input_tokens,
                cache_creation_input_tokens=cache_creation_input_tokens,
                reasoning_tokens=reasoning_tokens,
                estimated_cost_usd=estimated_cost_usd,
                raw_usage=dict(usage),
            )
        return cls()

    def as_dict(self) -> dict[str, int | float]:
        """Return the token counters as a plain dict."""
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "cached_input_tokens": self.cached_input_tokens,
            "cache_creation_input_tokens": self.cache_creation_input_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "estimated_cost_usd": self.estimated_cost_usd,
        }

    @property
    def has_reported_tokens(self) -> bool:
        """Return whether this record contains any provider-reported token value."""
        return any((self.input_tokens, self.output_tokens, self.total_tokens))

    @property
    def has_estimated_tokens(self) -> bool:
        """Return whether this record contains a local fallback estimate."""
        return any(
            (
                self.estimated_input_tokens,
                self.estimated_output_tokens,
                self.estimated_total_tokens,
            )
        )


class UsageCaptureCallback(BaseCallbackHandler):
    """Capture raw provider usage before structured-output parsers discard it."""

    def __init__(
        self,
        *,
        recorder: "TraceRecorder | None" = None,
        config: RunnableConfig | None = None,
        messages: list[BaseMessage] | None = None,
        model: Any = None,
        model_name: str | None = None,
        span_id: str | None = None,
        attempt_index: int = 1,
        agent_role: str | None = None,
        budget_gate: BudgetGate | None = None,
    ) -> None:
        self.raise_error = True
        self._seen_run_ids: set[str] = set()
        self._terminal_run_ids: set[str] = set()
        self.records: list[TokenUsage] = []
        self._budget_keys: dict[str, str] = {}
        self._settled_budget_keys: set[str] = set()
        self._pending_physical_ids: list[str] = []
        self._physical_counter = 0
        self._model_name = model_name or "unknown"
        self._model = model
        self._messages = list(messages or [])
        self._estimation_enabled = bool(
            recorder is not None
            and recorder.configuration.token_usage_estimation_enabled
        )
        self._estimated_input = 0
        self._estimated_output = 1
        self._budget_gate: BudgetGate | None = budget_gate
        if recorder is not None:
            metadata = (config or {}).get("metadata") or {}
            run_id = str(metadata.get("run_id") or "")
            if run_id and self._budget_gate is None:
                started_at = None
                if recorder.store is not None:
                    stored = recorder._safe(recorder.store.get_run, run_id) or {}
                    started_at = stored.get("started_at")
                self._budget_gate = BudgetGate.from_config(
                    recorder.configuration,
                    run_id,
                    started_at=float(started_at) if started_at else None,
                )
            try:
                self._estimated_input = max(
                    1, int(count_tokens_approximately(messages or []))
                )
            except Exception:  # noqa: BLE001
                self._estimated_input = max(
                    1, len(get_buffer_string(messages or [])) // 4
                )
            output_fields = {
                "researcher": "research_model_max_tokens",
                "supervisor": "research_model_max_tokens",
                "summarization": "summarization_model_max_tokens",
                "message_summary": "message_summary_model_max_tokens",
                "compression": "compression_model_max_tokens",
                "final_report": "final_report_model_max_tokens",
                "quality_evaluator": "quality_evaluation_model_max_tokens",
            }
            self._estimated_output = max(
                1,
                int(
                    getattr(
                        recorder.configuration,
                        output_fields.get(agent_role or "", "research_model_max_tokens"),
                        1,
                    )
                    or 1
                ),
            )
        self._budget_key_prefix = f"usage:{span_id or 'unknown'}:{attempt_index}"

    def begin_physical_attempt(self) -> None:
        """Pre-reserve a physical request before adapter callbacks can run."""
        self._physical_counter += 1
        placeholder = f"outer-{self._physical_counter}"
        operation_key = f"{self._budget_key_prefix}:{placeholder}"
        if self._budget_gate is not None:
            self._budget_gate.reserve_model_call(
                operation_key,
                estimated_input_tokens=self._estimated_input,
                estimated_output_tokens=self._estimated_output,
                model_name=self._model_name,
            )
        self._budget_keys[placeholder] = operation_key
        self._pending_physical_ids.append(placeholder)

    def _reserve_budget(self, run_id: Any) -> None:
        callback_run_id = str(run_id)
        if callback_run_id in self._budget_keys:
            return
        if self._pending_physical_ids:
            placeholder = self._pending_physical_ids.pop(0)
            self._budget_keys[callback_run_id] = self._budget_keys.pop(placeholder)
            return
        if self._budget_gate is None:
            return
        operation_key = f"{self._budget_key_prefix}:{callback_run_id}"
        self._budget_gate.reserve_model_call(
            operation_key,
            estimated_input_tokens=self._estimated_input,
            estimated_output_tokens=self._estimated_output,
            model_name=self._model_name,
        )
        self._budget_keys[callback_run_id] = operation_key

    def on_chat_model_start(
        self, _serialized: dict[str, Any], _messages: list[list[BaseMessage]], *, run_id: Any, **_kwargs: Any
    ) -> None:
        self._reserve_budget(run_id)

    def on_llm_start(
        self, _serialized: dict[str, Any], _prompts: list[str], *, run_id: Any, **_kwargs: Any
    ) -> None:
        self._reserve_budget(run_id)

    def on_llm_end(self, response: Any, *, run_id: Any, **_kwargs: Any) -> None:
        callback_run_id = str(run_id)
        if callback_run_id in self._pending_physical_ids:
            self._pending_physical_ids.remove(callback_run_id)
        if callback_run_id in self._seen_run_ids:
            return
        self._seen_run_ids.add(callback_run_id)
        self._terminal_run_ids.add(callback_run_id)
        candidates: list[Any] = []
        for generation_group in getattr(response, "generations", None) or []:
            for generation in generation_group or []:
                candidates.append(getattr(generation, "message", generation))
        candidates.append(response)
        for candidate in candidates:
            usage = TokenUsage.from_response(candidate)
            if usage.has_reported_tokens:
                usage.usage_source = (
                    "provider_reported"
                    if usage.input_tokens > 0 and usage.output_tokens > 0
                    else "provider_partial"
                )
                self.records.append(usage)
                operation_key = self._budget_keys.get(callback_run_id)
                if operation_key and self._budget_gate is not None:
                    self._budget_gate.settle_model_call(
                        operation_key,
                        input_tokens=usage.input_tokens,
                        output_tokens=usage.output_tokens,
                        model_name=self._model_name,
                    )
                    self._settled_budget_keys.add(operation_key)
                return
        usage = (
            _estimated_usage(
                self._model,
                self._messages,
                candidates[0] if candidates else response,
            )
            if self._estimation_enabled
            else TokenUsage(usage_source="missing")
        )
        self.records.append(usage)
        operation_key = self._budget_keys.get(callback_run_id)
        if operation_key and self._budget_gate is not None:
            self._budget_gate.settle_model_call(
                operation_key,
                input_tokens=usage.estimated_input_tokens,
                output_tokens=usage.estimated_output_tokens,
                model_name=self._model_name,
            )
            self._settled_budget_keys.add(operation_key)

    def on_llm_error(self, error: BaseException, *, run_id: Any, **_kwargs: Any) -> None:
        """Record a physical provider attempt even when a wrapper falls back."""
        self._record_failed_run(str(run_id), error)

    def _record_failed_run(self, callback_run_id: str, error: BaseException) -> None:
        if callback_run_id in self._terminal_run_ids:
            return
        if callback_run_id in self._pending_physical_ids:
            self._pending_physical_ids.remove(callback_run_id)
        self._terminal_run_ids.add(callback_run_id)
        self._seen_run_ids.add(callback_run_id)
        uncertain = _is_uncertain_model_failure(error)
        self.records.append(
            TokenUsage(
                usage_source="missing",
                response_status="unknown_failed" if uncertain else "rejected",
            )
        )
        operation_key = self._budget_keys.get(callback_run_id)
        if operation_key and self._budget_gate is not None:
            self._budget_gate.fail_model_call(operation_key, uncertain=uncertain)
            self._settled_budget_keys.add(operation_key)

    def settle_outer_failure(self, error: BaseException) -> None:
        """Finalize callback reservations when cancellation bypasses callbacks."""
        unresolved = [
            callback_run_id
            for callback_run_id in self._budget_keys
            if callback_run_id not in self._terminal_run_ids
        ]
        for callback_run_id in unresolved:
            self._record_failed_run(callback_run_id, error)
        if not self._budget_keys and not self.records:
            uncertain = _is_uncertain_model_failure(error)
            self.records.append(
                TokenUsage(
                    usage_source="missing",
                    response_status="unknown_failed" if uncertain else "rejected",
                )
            )

    def settle_outer_success(self, response: Any) -> None:
        """Capture adapters that return without dispatching LangChain callbacks."""
        unresolved = [
            callback_run_id
            for callback_run_id in self._budget_keys
            if callback_run_id not in self._terminal_run_ids
        ]
        for callback_run_id in unresolved:
            self.on_llm_end(response, run_id=callback_run_id)

    def settle_estimated_success(self, usage: TokenUsage) -> None:
        """Settle successful no-usage calls with the local fallback estimate."""
        if self._budget_gate is None:
            return
        for operation_key in self._budget_keys.values():
            if operation_key in self._settled_budget_keys:
                continue
            self._budget_gate.settle_model_call(
                operation_key,
                input_tokens=usage.estimated_input_tokens or usage.input_tokens,
                output_tokens=usage.estimated_output_tokens or usage.output_tokens,
                model_name=self._model_name,
            )
            self._settled_budget_keys.add(operation_key)


def _estimated_usage(model: Any, messages: list[BaseMessage], response: Any) -> TokenUsage:
    """Build a content-free fallback estimate for a successful model response."""
    input_tokens = 0
    output_tokens = 0
    counter = getattr(model, "get_num_tokens_from_messages", None)
    if callable(counter):
        try:
            input_tokens = max(0, int(counter(messages)))
        except Exception:  # noqa: BLE001 - tokenizer support varies by adapter
            input_tokens = 0
    if not input_tokens:
        try:
            input_tokens = max(0, int(count_tokens_approximately(messages)))
        except Exception:  # noqa: BLE001
            input_tokens = max(1, len(get_buffer_string(messages)) // 4)
    output_text = _message_preview(response, None, redact=False) or ""
    token_counter = getattr(model, "get_num_tokens", None)
    if callable(token_counter):
        try:
            output_tokens = max(0, int(token_counter(output_text)))
        except Exception:  # noqa: BLE001
            output_tokens = 0
    if not output_tokens and output_text:
        output_tokens = max(1, len(output_text) // 4)
    return TokenUsage(
        estimated_input_tokens=input_tokens,
        estimated_output_tokens=output_tokens,
        estimated_total_tokens=input_tokens + output_tokens,
        usage_source="tokenizer_estimated",
    )


class SQLiteTraceStore:
    """Small SQLite-backed store for runs, spans, and usage events."""

    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._ensure_schema()
        self._migrate()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=5)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout=5000")
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def ping(self) -> None:
        """Raise when the trace database cannot execute a lightweight query."""
        with self._lock, self._connect() as conn:
            conn.execute("SELECT 1").fetchone()

    @staticmethod
    def _delete_run_rows(conn: sqlite3.Connection, run_id: str) -> int:
        """Delete one run and all trace children within the caller transaction."""
        deleted = 0
        for table in ("retry_events", "usage_events", "spans", "runs"):
            cursor = conn.execute(f"DELETE FROM {table} WHERE run_id = ?", (run_id,))
            deleted += max(0, cursor.rowcount)
        return deleted

    def delete_run(self, run_id: str) -> int:
        """Delete one run from all observability tables and return affected rows."""
        with self._lock, self._connect() as conn:
            return self._delete_run_rows(conn, run_id)

    def delete_runs_ended_before(self, cutoff: float) -> int:
        """Delete completed trace rows older than cutoff and return run count."""
        with self._lock, self._connect() as conn:
            run_ids = [
                str(row["run_id"])
                for row in conn.execute(
                    "SELECT run_id FROM runs WHERE ended_at IS NOT NULL AND ended_at < ?",
                    (cutoff,),
                )
            ]
            for run_id in run_ids:
                self._delete_run_rows(conn, run_id)
            return len(run_ids)

    def checkpoint(self) -> None:
        """Checkpoint and truncate the WAL after lifecycle cleanup."""
        with self._lock, self._connect() as conn:
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()

    def _ensure_schema(self) -> None:
        with self._lock, self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    user_id TEXT,
                    status TEXT NOT NULL,
                    started_at REAL NOT NULL,
                    ended_at REAL,
                    duration_ms INTEGER,
                    attempt_count INTEGER NOT NULL DEFAULT 1,
                    resumed_at REAL,
                    input_tokens INTEGER NOT NULL DEFAULT 0,
                    output_tokens INTEGER NOT NULL DEFAULT 0,
                    total_tokens INTEGER NOT NULL DEFAULT 0,
                    cached_input_tokens INTEGER NOT NULL DEFAULT 0,
                    cache_creation_input_tokens INTEGER NOT NULL DEFAULT 0,
                    reasoning_tokens INTEGER NOT NULL DEFAULT 0,
                    estimated_cost_usd REAL NOT NULL DEFAULT 0,
                    error TEXT,
                    metadata_json TEXT NOT NULL DEFAULT '{}'
                );

                CREATE TABLE IF NOT EXISTS spans (
                    span_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    parent_span_id TEXT,
                    name TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    agent_role TEXT,
                    status TEXT NOT NULL,
                    started_at REAL NOT NULL,
                    ended_at REAL,
                    duration_ms INTEGER,
                    model TEXT,
                    provider TEXT,
                    input_tokens INTEGER NOT NULL DEFAULT 0,
                    output_tokens INTEGER NOT NULL DEFAULT 0,
                    total_tokens INTEGER NOT NULL DEFAULT 0,
                    cached_input_tokens INTEGER NOT NULL DEFAULT 0,
                    cache_creation_input_tokens INTEGER NOT NULL DEFAULT 0,
                    reasoning_tokens INTEGER NOT NULL DEFAULT 0,
                    estimated_cost_usd REAL NOT NULL DEFAULT 0,
                    attributes_json TEXT NOT NULL DEFAULT '{}',
                    input_preview TEXT,
                    output_preview TEXT,
                    error TEXT
                );

                CREATE TABLE IF NOT EXISTS usage_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    span_id TEXT NOT NULL,
                    provider TEXT,
                    model TEXT,
                    input_tokens INTEGER NOT NULL DEFAULT 0,
                    output_tokens INTEGER NOT NULL DEFAULT 0,
                    total_tokens INTEGER NOT NULL DEFAULT 0,
                    cached_input_tokens INTEGER NOT NULL DEFAULT 0,
                    cache_creation_input_tokens INTEGER NOT NULL DEFAULT 0,
                    reasoning_tokens INTEGER NOT NULL DEFAULT 0,
                    estimated_cost_usd REAL NOT NULL DEFAULT 0,
                    estimated_input_tokens INTEGER NOT NULL DEFAULT 0,
                    estimated_output_tokens INTEGER NOT NULL DEFAULT 0,
                    estimated_total_tokens INTEGER NOT NULL DEFAULT 0,
                    usage_source TEXT NOT NULL DEFAULT 'legacy_unclassified',
                    cost_source TEXT NOT NULL DEFAULT 'unavailable',
                    event_key TEXT,
                    attempt_index INTEGER NOT NULL DEFAULT 1,
                    stage TEXT NOT NULL DEFAULT 'unknown',
                    agent_role TEXT,
                    task_id TEXT,
                    operation TEXT,
                    duration_ms INTEGER,
                    response_status TEXT NOT NULL DEFAULT 'success',
                    raw_usage_json TEXT NOT NULL DEFAULT '{}',
                    created_at REAL NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_spans_run_started
                    ON spans(run_id, started_at);
                CREATE INDEX IF NOT EXISTS idx_spans_parent
                    ON spans(parent_span_id);
                CREATE INDEX IF NOT EXISTS idx_usage_run
                    ON usage_events(run_id);
                CREATE INDEX IF NOT EXISTS idx_runs_user_started
                    ON runs(user_id, started_at);
                CREATE INDEX IF NOT EXISTS idx_usage_run_created
                    ON usage_events(run_id, created_at);

                CREATE TABLE IF NOT EXISTS retry_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    span_id TEXT NOT NULL,
                    attempt INTEGER NOT NULL,
                    error_type TEXT NOT NULL,
                    http_status INTEGER,
                    retryable INTEGER NOT NULL DEFAULT 0,
                    delay_s REAL,
                    message TEXT,
                    created_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_retry_run
                    ON retry_events(run_id);
                CREATE INDEX IF NOT EXISTS idx_retry_span
                    ON retry_events(span_id);
                """
            )

    def _migrate(self) -> None:
        """Idempotently add columns introduced after the initial schema.

        Uses ``PRAGMA table_info`` introspection so existing databases upgrade
        transparently and partially-migrated databases are left untouched. New
        tables are created via ``CREATE TABLE IF NOT EXISTS`` in ``_ensure_schema``.
        """
        with self._lock, self._connect() as conn:
            usage_columns_before = {
                row["name"] for row in conn.execute("PRAGMA table_info(usage_events)")
            }
            legacy_usage_schema = "usage_source" not in usage_columns_before
            additions = {
                "runs": (
                    ("attempt_count", "INTEGER NOT NULL DEFAULT 1"),
                    ("resumed_at", "REAL"),
                    ("cached_input_tokens", "INTEGER NOT NULL DEFAULT 0"),
                    ("cache_creation_input_tokens", "INTEGER NOT NULL DEFAULT 0"),
                    ("reasoning_tokens", "INTEGER NOT NULL DEFAULT 0"),
                ),
                "spans": (
                    ("retry_count", "INTEGER NOT NULL DEFAULT 0"),
                    ("error_type", "TEXT"),
                    ("http_status", "INTEGER"),
                    ("cached_input_tokens", "INTEGER NOT NULL DEFAULT 0"),
                    ("cache_creation_input_tokens", "INTEGER NOT NULL DEFAULT 0"),
                    ("reasoning_tokens", "INTEGER NOT NULL DEFAULT 0"),
                ),
                "usage_events": (
                    ("cached_input_tokens", "INTEGER NOT NULL DEFAULT 0"),
                    ("cache_creation_input_tokens", "INTEGER NOT NULL DEFAULT 0"),
                    ("reasoning_tokens", "INTEGER NOT NULL DEFAULT 0"),
                    ("estimated_cost_usd", "REAL NOT NULL DEFAULT 0"),
                    ("estimated_input_tokens", "INTEGER NOT NULL DEFAULT 0"),
                    ("estimated_output_tokens", "INTEGER NOT NULL DEFAULT 0"),
                    ("estimated_total_tokens", "INTEGER NOT NULL DEFAULT 0"),
                    ("usage_source", "TEXT NOT NULL DEFAULT 'legacy_unclassified'"),
                    ("cost_source", "TEXT NOT NULL DEFAULT 'unavailable'"),
                    ("event_key", "TEXT"),
                    ("attempt_index", "INTEGER NOT NULL DEFAULT 1"),
                    ("stage", "TEXT NOT NULL DEFAULT 'unknown'"),
                    ("agent_role", "TEXT"),
                    ("task_id", "TEXT"),
                    ("operation", "TEXT"),
                    ("duration_ms", "INTEGER"),
                    ("response_status", "TEXT NOT NULL DEFAULT 'success'"),
                ),
            }
            for table, columns in additions.items():
                existing = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
                for column, ddl in columns:
                    if column not in existing:
                        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_runs_user_started ON runs(user_id, started_at)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_usage_run_created "
                "ON usage_events(run_id, created_at)"
            )
            conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_usage_event_key "
                "ON usage_events(run_id, event_key) WHERE event_key IS NOT NULL"
            )
            if legacy_usage_schema:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO usage_events (
                        run_id, span_id, provider, model,
                        input_tokens, output_tokens, total_tokens,
                        cached_input_tokens, cache_creation_input_tokens,
                        reasoning_tokens, estimated_cost_usd,
                        estimated_input_tokens, estimated_output_tokens,
                        estimated_total_tokens, usage_source, cost_source,
                        event_key, attempt_index, stage, agent_role, task_id,
                        operation, duration_ms, response_status,
                        raw_usage_json, created_at
                    )
                    SELECT
                        spans.run_id, spans.span_id, spans.provider, spans.model,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        'legacy_unclassified', 'unavailable',
                        'legacy-missing:' || spans.span_id, 1, 'unknown',
                        spans.agent_role, NULL, spans.name, spans.duration_ms,
                        'success', '{}', COALESCE(spans.ended_at, spans.started_at)
                    FROM spans
                    WHERE spans.kind = 'llm'
                      AND NOT EXISTS (
                          SELECT 1 FROM usage_events
                          WHERE usage_events.span_id = spans.span_id
                      )
                    """
                )

    def start_run(self, run_id: str, user_id: str | None, metadata: dict[str, Any]) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO runs (run_id, user_id, status, started_at, metadata_json)
                VALUES (?, ?, 'running', ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    status='running',
                    user_id=excluded.user_id,
                    metadata_json=excluded.metadata_json,
                    attempt_count=runs.attempt_count + 1,
                    resumed_at=excluded.started_at,
                    ended_at=NULL,
                    duration_ms=NULL,
                    error=NULL
                """,
                (run_id, user_id, _now(), _json(metadata)),
            )

    def finish_run(self, run_id: str, status: str, error: str | None = None) -> dict[str, Any]:
        ended_at = _now()
        usage = self.get_usage(run_id)
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT started_at FROM runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            started_at = float(row["started_at"]) if row else ended_at
            conn.execute(
                """
                UPDATE runs
                SET status = ?, ended_at = ?, duration_ms = ?,
                    input_tokens = ?, output_tokens = ?, total_tokens = ?,
                    cached_input_tokens = ?, cache_creation_input_tokens = ?,
                    reasoning_tokens = ?, estimated_cost_usd = ?,
                    error = ?
                WHERE run_id = ?
                """,
                (
                    status,
                    ended_at,
                    int((ended_at - started_at) * 1000),
                    usage["input_tokens"],
                    usage["output_tokens"],
                    usage["total_tokens"],
                    usage["cached_input_tokens"],
                    usage["cache_creation_input_tokens"],
                    usage["reasoning_tokens"],
                    usage["estimated_cost_usd"],
                    error,
                    run_id,
                ),
            )
        return usage

    def start_span(
        self,
        *,
        span_id: str,
        run_id: str,
        parent_span_id: str | None,
        name: str,
        kind: str,
        agent_role: str | None,
        attributes: dict[str, Any],
        input_preview: str | None,
        provider: str | None,
        model: str | None,
    ) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO spans (
                    span_id, run_id, parent_span_id, name, kind, agent_role,
                    status, started_at, provider, model, attributes_json, input_preview
                )
                VALUES (?, ?, ?, ?, ?, ?, 'running', ?, ?, ?, ?, ?)
                """,
                (
                    span_id,
                    run_id,
                    parent_span_id,
                    name,
                    kind,
                    agent_role,
                    _now(),
                    provider,
                    model,
                    _json(attributes),
                    input_preview,
                ),
            )

    def finish_span(
        self,
        *,
        span_id: str,
        status: str,
        usage: TokenUsage | None = None,
        output_preview: str | None = None,
        error: str | None = None,
        retry_count: int = 0,
        error_type: str | None = None,
        http_status: int | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> None:
        usage = usage or TokenUsage()
        ended_at = _now()
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT started_at FROM spans WHERE span_id = ?",
                (span_id,),
            ).fetchone()
            started_at = float(row["started_at"]) if row else ended_at
            conn.execute(
                """
                UPDATE spans
                SET status = ?, ended_at = ?, duration_ms = ?,
                    input_tokens = ?, output_tokens = ?, total_tokens = ?,
                    cached_input_tokens = ?, cache_creation_input_tokens = ?,
                    reasoning_tokens = ?, estimated_cost_usd = ?,
                    output_preview = COALESCE(?, output_preview), error = ?,
                    retry_count = ?,
                    error_type = COALESCE(?, error_type),
                    http_status = COALESCE(?, http_status)
                    , attributes_json = COALESCE(?, attributes_json)
                WHERE span_id = ?
                """,
                (
                    status,
                    ended_at,
                    int((ended_at - started_at) * 1000),
                    usage.input_tokens,
                    usage.output_tokens,
                    usage.total_tokens,
                    usage.cached_input_tokens,
                    usage.cache_creation_input_tokens,
                    usage.reasoning_tokens,
                    usage.estimated_cost_usd,
                    output_preview,
                    error,
                    retry_count,
                    error_type,
                    http_status,
                    _json(attributes) if attributes is not None else None,
                    span_id,
                ),
            )

    def record_retry_event(
        self,
        *,
        run_id: str,
        span_id: str,
        attempt: int,
        error_type: str,
        http_status: int | None = None,
        retryable: bool = False,
        delay_s: float | None = None,
        message: str | None = None,
    ) -> None:
        """Append one retry event for a span (one row per failed attempt)."""
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO retry_events (
                    run_id, span_id, attempt, error_type, http_status,
                    retryable, delay_s, message, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    span_id,
                    attempt,
                    error_type,
                    http_status,
                    1 if retryable else 0,
                    delay_s,
                    message,
                    _now(),
                ),
            )

    def get_retry_summary(self, run_id: str) -> dict[str, Any]:
        """Aggregate retry events for a run into counts by error type."""
        with self._lock, self._connect() as conn:
            total_row = conn.execute(
                "SELECT COUNT(*) AS c FROM retry_events WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            rate_limited_row = conn.execute(
                "SELECT COUNT(*) AS c FROM retry_events WHERE run_id = ? AND error_type = ?",
                (run_id, "rate_limited"),
            ).fetchone()
            by_type_rows = conn.execute(
                """
                SELECT error_type, COUNT(*) AS c
                FROM retry_events
                WHERE run_id = ?
                GROUP BY error_type
                """,
                (run_id,),
            ).fetchall()
        by_error_type = {row["error_type"]: int(row["c"]) for row in by_type_rows}
        return {
            "retry_count": int(total_row["c"] if total_row else 0),
            "rate_limited_count": int(rate_limited_row["c"] if rate_limited_row else 0),
            "by_error_type": by_error_type,
        }

    def get_metrics(self, run_id: str) -> dict[str, Any]:
        """Aggregate token usage, retry counts, and 429 rate for one run.

        ``rate_429`` is the share of LLM/tool calls that experienced a rate-limit
        (429): ``rate_limited_count / total_llm_tool_calls``. The denominator is
        the count of spans with ``kind IN ('llm', 'tool')``.
        """
        usage = self.get_usage(run_id)
        retry = self.get_retry_summary(run_id)
        with self._lock, self._connect() as conn:
            calls_row = conn.execute(
                "SELECT COUNT(*) AS c FROM spans WHERE run_id = ? AND kind IN ('llm', 'tool')",
                (run_id,),
            ).fetchone()
            span_rows = conn.execute(
                """
                SELECT span_id, name, kind, status, duration_ms,
                       input_tokens, output_tokens, cached_input_tokens,
                       reasoning_tokens, retry_count, error_type, http_status,
                       attributes_json
                FROM spans
                WHERE run_id = ? AND kind IN ('llm', 'tool')
                ORDER BY started_at ASC
                """,
                (run_id,),
            ).fetchall()
        total_calls = int(calls_row["c"] if calls_row else 0)
        rate_limited = retry["rate_limited_count"]
        with self._lock, self._connect() as conn:
            rate_limited_calls_row = conn.execute(
                """
                SELECT COUNT(DISTINCT span_id) AS c
                FROM (
                    SELECT span_id FROM retry_events
                    WHERE run_id = ? AND error_type = 'rate_limited'
                    UNION
                    SELECT span_id FROM spans
                    WHERE run_id = ? AND error_type = 'rate_limited'
                )
                """,
                (run_id, run_id),
            ).fetchone()
            terminal_rate_limited_row = conn.execute(
                "SELECT COUNT(*) AS c FROM spans WHERE run_id = ? AND error_type = 'rate_limited'",
                (run_id,),
            ).fetchone()
        rate_limited_calls = int(rate_limited_calls_row["c"] if rate_limited_calls_row else 0)
        terminal_rate_limited = int(
            terminal_rate_limited_row["c"] if terminal_rate_limited_row else 0
        )
        llm_rows = [row for row in span_rows if row["kind"] == "llm"]
        tool_rows = [row for row in span_rows if row["kind"] == "tool"]
        cache_eligible_rows = [row for row in llm_rows if int(row["input_tokens"] or 0) > 0]
        cache_hit_count = sum(
            1 for row in cache_eligible_rows if int(row["cached_input_tokens"] or 0) > 0
        )
        llm_input_tokens = sum(int(row["input_tokens"] or 0) for row in llm_rows)
        llm_output_tokens = sum(int(row["output_tokens"] or 0) for row in llm_rows)
        llm_reasoning_tokens = sum(int(row["reasoning_tokens"] or 0) for row in llm_rows)
        llm_duration_seconds = sum(float(row["duration_ms"] or 0) for row in llm_rows) / 1000
        successful_tools = sum(1 for row in tool_rows if row["status"] == "success")
        empty_tool_result_count = 0
        zero_source_search_count = 0
        for row in tool_rows:
            if row["status"] != "success":
                continue
            try:
                attributes = json.loads(row["attributes_json"] or "{}")
            except (TypeError, ValueError, json.JSONDecodeError):
                attributes = {}
            if not attributes.get("result_chars", 0):
                empty_tool_result_count += 1
            if "search" in str(row["name"]).lower() and not attributes.get(
                "source_count", 0
            ):
                zero_source_search_count += 1
        run = self.get_run(run_id) or {}
        return {
            "run_id": run_id,
            "attempt_count": int(run.get("attempt_count") or 0),
            "input_tokens": usage["input_tokens"],
            "output_tokens": usage["output_tokens"],
            "total_tokens": usage["total_tokens"],
            "cached_input_tokens": usage["cached_input_tokens"],
            "cache_creation_input_tokens": usage["cache_creation_input_tokens"],
            "reasoning_tokens": usage["reasoning_tokens"],
            "estimated_cost_usd": usage["estimated_cost_usd"],
            "retry_count": retry["retry_count"],
            "rate_limit_events": rate_limited,
            "rate_limited_count": rate_limited_calls,
            "terminal_rate_limited_count": terminal_rate_limited,
            "total_llm_tool_calls": total_calls,
            "rate_429": (rate_limited_calls / total_calls) if total_calls else 0.0,
            "llm_call_count": len(llm_rows),
            "cache_eligible_count": len(cache_eligible_rows),
            "cache_hit_count": cache_hit_count,
            "cache_hit_rate": (
                cache_hit_count / len(cache_eligible_rows) if cache_eligible_rows else 0.0
            ),
            "cache_input_ratio": (
                usage["cached_input_tokens"] / llm_input_tokens if llm_input_tokens else 0.0
            ),
            "llm_output_input_ratio": (
                llm_output_tokens / llm_input_tokens if llm_input_tokens else 0.0
            ),
            "llm_reasoning_output_ratio": (
                llm_reasoning_tokens / llm_output_tokens if llm_output_tokens else 0.0
            ),
            "llm_output_tokens_per_second": (
                llm_output_tokens / llm_duration_seconds if llm_duration_seconds else 0.0
            ),
            "tool_call_count": len(tool_rows),
            "tool_success_count": successful_tools,
            "tool_success_rate": successful_tools / len(tool_rows) if tool_rows else 0.0,
            "empty_tool_result_count": empty_tool_result_count,
            "zero_source_search_count": zero_source_search_count,
            "by_error_type": retry["by_error_type"],
            "by_span": [
                {
                    "span_id": row["span_id"],
                    "name": row["name"],
                    "kind": row["kind"],
                    "status": row["status"],
                    "retry_count": int(row["retry_count"] or 0),
                    "error_type": row["error_type"],
                    "http_status": row["http_status"],
                }
                for row in span_rows
            ],
        }

    def add_usage(
        self,
        run_id: str,
        span_id: str,
        provider: str | None,
        model: str | None,
        usage: TokenUsage,
        *,
        event_key: str | None = None,
        attempt_index: int = 1,
        stage: str = "unknown",
        agent_role: str | None = None,
        task_id: str | None = None,
        operation: str | None = None,
        duration_ms: int | None = None,
        response_status: str = "success",
    ) -> int | None:
        if (
            not usage.has_reported_tokens
            and not usage.has_estimated_tokens
            and usage.usage_source not in {"missing", "legacy_unclassified"}
        ):
            return None
        with self._lock, self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT OR IGNORE INTO usage_events (
                    run_id, span_id, provider, model,
                    input_tokens, output_tokens, total_tokens,
                    cached_input_tokens, cache_creation_input_tokens,
                    reasoning_tokens, estimated_cost_usd,
                    estimated_input_tokens, estimated_output_tokens,
                    estimated_total_tokens, usage_source, cost_source,
                    event_key, attempt_index, stage, agent_role, task_id,
                    operation, duration_ms, response_status,
                    raw_usage_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    span_id,
                    provider,
                    model,
                    usage.input_tokens,
                    usage.output_tokens,
                    usage.total_tokens,
                    usage.cached_input_tokens,
                    usage.cache_creation_input_tokens,
                    usage.reasoning_tokens,
                    usage.estimated_cost_usd,
                    usage.estimated_input_tokens,
                    usage.estimated_output_tokens,
                    usage.estimated_total_tokens,
                    usage.usage_source,
                    usage.cost_source,
                    event_key,
                    max(1, attempt_index),
                    stage or "unknown",
                    agent_role,
                    task_id,
                    operation,
                    duration_ms,
                    response_status,
                    _json(usage.raw_usage),
                    _now(),
                ),
            )
            return int(cursor.lastrowid) if cursor.rowcount else None

    def get_usage(self, run_id: str) -> dict[str, Any]:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                """
                SELECT
                    COALESCE(SUM(input_tokens), 0) AS input_tokens,
                    COALESCE(SUM(output_tokens), 0) AS output_tokens,
                    COALESCE(SUM(total_tokens), 0) AS total_tokens,
                    COALESCE(SUM(cached_input_tokens), 0) AS cached_input_tokens,
                    COALESCE(SUM(cache_creation_input_tokens), 0) AS cache_creation_input_tokens,
                    COALESCE(SUM(reasoning_tokens), 0) AS reasoning_tokens,
                    COALESCE(SUM(estimated_cost_usd), 0) AS estimated_cost_usd
                FROM usage_events
                WHERE run_id = ?
                """,
                (run_id,),
            ).fetchone()
        return {
            "input_tokens": int(row["input_tokens"] if row else 0),
            "output_tokens": int(row["output_tokens"] if row else 0),
            "total_tokens": int(row["total_tokens"] if row else 0),
            "cached_input_tokens": int(row["cached_input_tokens"] if row else 0),
            "cache_creation_input_tokens": int(row["cache_creation_input_tokens"] if row else 0),
            "reasoning_tokens": int(row["reasoning_tokens"] if row else 0),
            "estimated_cost_usd": float(row["estimated_cost_usd"] if row else 0.0),
        }

    @staticmethod
    def _token_vector(rows: list[dict[str, Any]], *, estimated: bool = False) -> dict[str, int]:
        prefix = "estimated_" if estimated else ""
        return {
            "input_tokens": sum(int(row.get(f"{prefix}input_tokens") or 0) for row in rows),
            "output_tokens": sum(int(row.get(f"{prefix}output_tokens") or 0) for row in rows),
            "total_tokens": sum(int(row.get(f"{prefix}total_tokens") or 0) for row in rows),
            "cached_input_tokens": 0 if estimated else sum(
                int(row.get("cached_input_tokens") or 0) for row in rows
            ),
            "cache_creation_input_tokens": 0 if estimated else sum(
                int(row.get("cache_creation_input_tokens") or 0) for row in rows
            ),
            "reasoning_tokens": 0 if estimated else sum(
                int(row.get("reasoning_tokens") or 0) for row in rows
            ),
        }

    def _usage_rows(
        self,
        run_id: str,
        *,
        provider: str | None = None,
        model: str | None = None,
    ) -> list[dict[str, Any]]:
        clauses = ["run_id = ?"]
        parameters: list[Any] = [run_id]
        if provider:
            clauses.append("provider = ?")
            parameters.append(provider)
        if model:
            clauses.append("model = ?")
            parameters.append(model)
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM usage_events WHERE "
                + " AND ".join(clauses)
                + " ORDER BY created_at, id",
                tuple(parameters),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_usage_accounting(
        self,
        run_id: str,
        *,
        provider: str | None = None,
        model: str | None = None,
        reserved_budget: dict[str, int] | None = None,
    ) -> dict[str, Any]:
        """Return the business-facing reported/estimated usage projection."""
        rows = self._usage_rows(run_id, provider=provider, model=model)
        run = self.get_run(run_id) or {}
        retry_rows = self._retry_rows(run_id, provider=provider, model=model)
        operations = self._accounting_operations(
            run_id,
            rows=rows,
            provider=provider,
            model=model,
            retry_rows=retry_rows,
        )
        return self._project_usage_accounting(
            run_id,
            rows=rows,
            run=run,
            reserved_budget=reserved_budget,
            operations=operations,
            retry_timestamps=[float(row.get("created_at") or 0) for row in retry_rows],
        )

    def get_usage_accounting_many(
        self,
        run_ids: list[str],
        *,
        provider: str | None = None,
        model: str | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Project many runs from one consistent, bounded-query SQLite snapshot."""
        selected = list(dict.fromkeys(str(run_id) for run_id in run_ids if run_id))
        if not selected:
            return {}

        with self._lock, self._connect() as conn:
            conn.execute(
                "CREATE TEMP TABLE selected_usage_runs (run_id TEXT PRIMARY KEY)"
            )
            conn.executemany(
                "INSERT INTO selected_usage_runs (run_id) VALUES (?)",
                ((run_id,) for run_id in selected),
            )
            run_rows = conn.execute(
                "SELECT runs.* FROM runs "
                "JOIN selected_usage_runs selected ON selected.run_id = runs.run_id"
            ).fetchall()
            usage_clauses: list[str] = []
            usage_parameters: list[Any] = []
            if provider:
                usage_clauses.append("usage_events.provider = ?")
                usage_parameters.append(provider)
            if model:
                usage_clauses.append("usage_events.model = ?")
                usage_parameters.append(model)
            usage_where = (
                " WHERE " + " AND ".join(usage_clauses) if usage_clauses else ""
            )
            usage_rows = conn.execute(
                "SELECT usage_events.* FROM usage_events "
                "JOIN selected_usage_runs selected "
                "ON selected.run_id = usage_events.run_id"
                + usage_where
                + " ORDER BY usage_events.run_id, usage_events.created_at, usage_events.id",
                tuple(usage_parameters),
            ).fetchall()
            span_rows = conn.execute(
                "SELECT spans.* FROM spans "
                "JOIN selected_usage_runs selected ON selected.run_id = spans.run_id "
                "WHERE spans.kind IN ('llm', 'tool')"
            ).fetchall()
            retry_rows = conn.execute(
                "SELECT retry_events.* FROM retry_events "
                "JOIN selected_usage_runs selected "
                "ON selected.run_id = retry_events.run_id"
            ).fetchall()

        runs_by_id = {str(row["run_id"]): dict(row) for row in run_rows}
        usage_by_run: dict[str, list[dict[str, Any]]] = {
            run_id: [] for run_id in selected
        }
        spans_by_run: dict[str, list[dict[str, Any]]] = {
            run_id: [] for run_id in selected
        }
        retries_by_run: dict[str, list[dict[str, Any]]] = {
            run_id: [] for run_id in selected
        }
        for row in usage_rows:
            usage_by_run.setdefault(str(row["run_id"]), []).append(dict(row))
        for row in span_rows:
            spans_by_run.setdefault(str(row["run_id"]), []).append(dict(row))
        for row in retry_rows:
            retries_by_run.setdefault(str(row["run_id"]), []).append(dict(row))

        result: dict[str, dict[str, Any]] = {}
        for run_id in selected:
            usage = usage_by_run.get(run_id, [])
            spans = spans_by_run.get(run_id, [])
            span_by_id = {str(row.get("span_id") or ""): row for row in spans}
            retries = [
                row
                for row in retries_by_run.get(run_id, [])
                if self._span_matches_usage_filter(
                    span_by_id.get(str(row.get("span_id") or "")),
                    provider=provider,
                    model=model,
                )
            ]
            operations = (
                self._filtered_accounting_operations(usage, retries)
                if provider or model
                else self._operations_from_loaded(spans, retries, usage)
            )
            result[run_id] = self._project_usage_accounting(
                run_id,
                rows=usage,
                run=runs_by_id.get(run_id, {}),
                reserved_budget=None,
                operations=operations,
                retry_timestamps=[
                    float(row.get("created_at") or 0) for row in retries
                ],
            )
        return result

    def _project_usage_accounting(
        self,
        run_id: str,
        *,
        rows: list[dict[str, Any]],
        run: dict[str, Any],
        reserved_budget: dict[str, int] | None,
        operations: dict[str, Any],
        retry_timestamps: list[float],
    ) -> dict[str, Any]:
        """Build the public projection from already-loaded content-free rows."""
        if not rows:
            totals = self._empty_accounting_totals()
            budget_limits = self._budget_limits(run)
            for key in ("input_tokens", "output_tokens", "model_calls", "cost_micro_usd"):
                totals["budgets"][key]["limit"] = budget_limits[key]
                totals["budgets"][key]["reserved"] = int(
                    (reserved_budget or {}).get(key, 0)
                )
            return {
                "schema_version": 1,
                "run_id": run_id,
                "status": run.get("status", "unknown"),
                "duration_ms": run.get("duration_ms"),
                "revision": 0,
                "updated_at": run.get("started_at"),
                "accounting_status": "unavailable",
                "unavailable_reason": (
                    "no_usage_events" if run else "run_not_observed"
                ),
                "totals": totals,
                "breakdowns": {"by_stage": [], "by_agent_role": [], "by_model": [], "by_task": []},
                "timeline": [],
                "operations": operations,
            }

        source_counts = {
            source: sum(1 for row in rows if row.get("usage_source") == source)
            for source in (
                "provider_reported",
                "provider_partial",
                "tokenizer_estimated",
                "missing",
                "legacy_unclassified",
            )
        }
        successful = sum(1 for row in rows if row.get("response_status") == "success")
        complete_reported = source_counts["provider_reported"]
        unknown_failed = sum(
            1 for row in rows if row.get("response_status") == "unknown_failed"
        )
        coverage = complete_reported / successful if successful else 0.0
        accounting_status = (
            "complete"
            if successful > 0
            and complete_reported == successful
            and unknown_failed == 0
            else "partial"
        )
        costs = [float(row.get("estimated_cost_usd") or 0) for row in rows]
        configured_cost = any(row.get("cost_source") == "configured_estimate" for row in rows)
        provider_cost = any(row.get("cost_source") == "provider_reported" for row in rows)
        cost_source = (
            "provider_reported" if provider_cost else "configured_estimate" if configured_cost else "unavailable"
        )
        budget_limits = self._budget_limits(run)
        totals = {
            "reported": self._token_vector(rows),
            "estimated": self._token_vector(rows, estimated=True),
            "calls": {
                "attempts": len(rows),
                "successful_responses": successful,
                "provider_reported": source_counts["provider_reported"],
                "provider_partial": source_counts["provider_partial"],
                "estimated": source_counts["tokenizer_estimated"],
                "missing": source_counts["missing"],
                "unknown_failed_attempts": unknown_failed,
                "legacy_unclassified": source_counts["legacy_unclassified"],
                "coverage_ratio": coverage,
            },
            "cost": {
                "estimated_cost_micro_usd": int(round(sum(costs) * 1_000_000))
                if cost_source != "unavailable"
                else None,
                "cost_source": cost_source,
                "price_table_hash": self._price_table_hash(run),
            },
            "budgets": {
                "input_tokens": {"settled": self._token_vector(rows)["input_tokens"], "estimated": self._token_vector(rows, estimated=True)["input_tokens"], "reserved": int((reserved_budget or {}).get("input_tokens", 0)), "limit": budget_limits["input_tokens"]},
                "output_tokens": {"settled": self._token_vector(rows)["output_tokens"], "estimated": self._token_vector(rows, estimated=True)["output_tokens"], "reserved": int((reserved_budget or {}).get("output_tokens", 0)), "limit": budget_limits["output_tokens"]},
                "model_calls": {"settled": len(rows), "estimated": 0, "reserved": int((reserved_budget or {}).get("model_calls", 0)), "limit": budget_limits["model_calls"]},
                "cost_micro_usd": {"settled": int(round(sum(costs) * 1_000_000)) if cost_source != "unavailable" else None, "estimated": 0, "reserved": int((reserved_budget or {}).get("cost_micro_usd", 0)), "limit": budget_limits["cost_micro_usd"]},
            },
        }
        breakdowns = {
            "by_stage": self._breakdown(rows, "stage"),
            "by_agent_role": self._breakdown(rows, "agent_role"),
            "by_model": self._breakdown(rows, "model", include_provider=True),
            "by_task": self._breakdown(rows, "task_id"),
        }
        return {
            "schema_version": 1,
            "run_id": run_id,
            "status": run.get("status", "unknown"),
            "duration_ms": run.get("duration_ms"),
            "revision": max(int(row.get("id") or 0) for row in rows),
            "updated_at": max(float(row.get("created_at") or 0) for row in rows),
            "accounting_status": accounting_status,
            "totals": totals,
            "breakdowns": breakdowns,
            "timeline": self._timeline(rows, retry_timestamps=retry_timestamps),
            "operations": operations,
        }

    @staticmethod
    def _price_table_hash(run: dict[str, Any]) -> str | None:
        try:
            metadata = json.loads(run.get("metadata_json") or "{}")
        except (TypeError, json.JSONDecodeError):
            return None
        value = metadata.get("model_costs_price_table_hash")
        return str(value) if value else None

    @staticmethod
    def _budget_limits(run: dict[str, Any]) -> dict[str, int | None]:
        try:
            metadata = json.loads(run.get("metadata_json") or "{}")
        except (TypeError, json.JSONDecodeError):
            metadata = {}
        limits = metadata.get("budget_limits") or {}
        return {
            key: int(value) if value is not None else None
            for key, value in {
                "input_tokens": limits.get("input_tokens"),
                "output_tokens": limits.get("output_tokens"),
                "model_calls": limits.get("model_calls"),
                "cost_micro_usd": limits.get("cost_micro_usd"),
            }.items()
        }

    @staticmethod
    def _empty_accounting_totals() -> dict[str, Any]:
        vector = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "cached_input_tokens": 0,
            "cache_creation_input_tokens": 0,
            "reasoning_tokens": 0,
        }
        return {
            "reported": dict(vector),
            "estimated": dict(vector),
            "calls": {
                "attempts": 0, "successful_responses": 0, "provider_reported": 0,
                "provider_partial": 0, "estimated": 0, "missing": 0,
                "unknown_failed_attempts": 0, "legacy_unclassified": 0,
                "coverage_ratio": 0.0,
            },
            "cost": {"estimated_cost_micro_usd": None, "cost_source": "unavailable", "price_table_hash": None},
            "budgets": {
                key: {"settled": None if key == "cost_micro_usd" else 0, "estimated": 0, "reserved": 0, "limit": None}
                for key in ("input_tokens", "output_tokens", "model_calls", "cost_micro_usd")
            },
        }

    def _breakdown(
        self, rows: list[dict[str, Any]], key: str, *, include_provider: bool = False
    ) -> list[dict[str, Any]]:
        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            value = str(row.get(key) or "unknown")
            if include_provider and row.get("provider"):
                value = f"{row['provider']}:{value}"
            grouped.setdefault(value, []).append(row)
        result = []
        for value, bucket in sorted(
            grouped.items(), key=lambda item: self._token_vector(item[1])["total_tokens"] + self._token_vector(item[1], estimated=True)["total_tokens"], reverse=True
        ):
            sources = {str(row.get("usage_source") or "legacy_unclassified") for row in bucket}
            cost_available = any(row.get("cost_source") != "unavailable" for row in bucket)
            result.append({
                "key": value,
                "label": value.replace("_", " ").title(),
                "reported": self._token_vector(bucket),
                "estimated": self._token_vector(bucket, estimated=True),
                "call_count": len(bucket),
                "estimated_cost_micro_usd": int(round(sum(float(row.get("estimated_cost_usd") or 0) for row in bucket) * 1_000_000)) if cost_available else None,
                "cost_source": "configured_estimate" if cost_available else "unavailable",
                "average_latency_ms": int(sum(int(row.get("duration_ms") or 0) for row in bucket) / len(bucket)) if bucket else 0,
                "completeness": "complete" if sources == {"provider_reported"} else "partial",
            })
        return result

    def _timeline(
        self,
        rows: list[dict[str, Any]],
        *,
        retry_timestamps: list[float],
    ) -> list[dict[str, Any]]:
        all_timestamps = [
            *(float(row.get("created_at") or 0) for row in rows),
            *retry_timestamps,
        ]
        start = min(all_timestamps)
        end = max(all_timestamps)
        width = max(1.0, (end - start) / 119) if end > start else 1.0
        buckets: dict[int, list[dict[str, Any]]] = {}
        for row in rows:
            index = min(119, int((float(row.get("created_at") or start) - start) / width))
            buckets.setdefault(index, []).append(row)
        retry_buckets: dict[int, int] = {}
        for created_at in retry_timestamps:
            index = min(119, int((created_at - start) / width))
            retry_buckets[index] = retry_buckets.get(index, 0) + 1
        reported_cumulative = 0
        estimated_cumulative = 0
        result = []
        for index in sorted(set(buckets) | set(retry_buckets)):
            bucket = buckets.get(index, [])
            reported = self._token_vector(bucket)["total_tokens"]
            estimated = self._token_vector(bucket, estimated=True)["total_tokens"]
            reported_cumulative += reported
            estimated_cumulative += estimated
            result.append({
                "timestamp": start + index * width,
                "reported_tokens": reported,
                "estimated_tokens": estimated,
                "reported_cumulative": reported_cumulative,
                "estimated_cumulative": estimated_cumulative,
                "call_count": len(bucket),
                "retry_count": retry_buckets.get(index, 0),
            })
        return result

    def _retry_rows(
        self,
        run_id: str,
        *,
        provider: str | None = None,
        model: str | None = None,
    ) -> list[dict[str, Any]]:
        clauses = ["retry_events.run_id = ?"]
        parameters: list[Any] = [run_id]
        if provider:
            clauses.append("spans.provider = ?")
            parameters.append(provider)
        if model:
            clauses.append("spans.model = ?")
            parameters.append(model)
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT retry_events.* FROM retry_events "
                "JOIN spans ON spans.span_id = retry_events.span_id WHERE "
                + " AND ".join(clauses),
                tuple(parameters),
            ).fetchall()
        return [dict(row) for row in rows]

    @staticmethod
    def _span_matches_usage_filter(
        span: dict[str, Any] | None,
        *,
        provider: str | None,
        model: str | None,
    ) -> bool:
        if provider is None and model is None:
            return True
        if span is None:
            return False
        return (provider is None or span.get("provider") == provider) and (
            model is None or span.get("model") == model
        )

    @staticmethod
    def _filtered_accounting_operations(
        usage_rows: list[dict[str, Any]],
        retry_rows: list[dict[str, Any]],
    ) -> dict[str, Any]:
        successful = [
            row for row in usage_rows if row.get("response_status") == "success"
        ]
        input_tokens = sum(int(row.get("input_tokens") or 0) for row in successful)
        cached_tokens = sum(
            int(row.get("cached_input_tokens") or 0) for row in successful
        )
        cache_hits = sum(
            1
            for row in successful
            if int(row.get("input_tokens") or 0) > 0
            and int(row.get("cached_input_tokens") or 0) > 0
        )
        cache_eligible = sum(
            1 for row in successful if int(row.get("input_tokens") or 0) > 0
        )
        output_tokens = sum(
            int(row.get("output_tokens") or 0) for row in successful
        )
        reasoning_tokens = sum(
            int(row.get("reasoning_tokens") or 0) for row in successful
        )
        duration_seconds = (
            sum(int(row.get("duration_ms") or 0) for row in successful) / 1000
        )
        rate_limited_calls = {
            str(row.get("span_id") or "")
            for row in retry_rows
            if row.get("error_type") == "rate_limited"
        }
        rate_limited_count = len(rate_limited_calls - {""})
        return {
            "llm_call_count": len(usage_rows),
            "retry_count": len(retry_rows),
            "rate_limited_count": rate_limited_count,
            "rate_429": (
                rate_limited_count / len(usage_rows) if usage_rows else 0.0
            ),
            "cache_hit_rate": cache_hits / cache_eligible if cache_eligible else 0.0,
            "cache_input_ratio": cached_tokens / input_tokens if input_tokens else 0.0,
            "reasoning_output_ratio": (
                reasoning_tokens / output_tokens if output_tokens else 0.0
            ),
            "output_tokens_per_second": (
                output_tokens / duration_seconds if duration_seconds else 0.0
            ),
            "tool_call_count": 0,
            "tool_success_rate": 0.0,
            "empty_tool_result_count": 0,
            "zero_source_search_count": 0,
        }

    @classmethod
    def _operations_from_loaded(
        cls,
        spans: list[dict[str, Any]],
        retries: list[dict[str, Any]],
        usage_rows: list[dict[str, Any]],
    ) -> dict[str, Any]:
        operations = cls._filtered_accounting_operations(usage_rows, retries)
        llm_rows = [row for row in spans if row.get("kind") == "llm"]
        tool_rows = [row for row in spans if row.get("kind") == "tool"]
        rate_limited_calls = {
            str(row.get("span_id") or "")
            for row in retries
            if row.get("error_type") == "rate_limited"
        }
        rate_limited_calls.update(
            str(row.get("span_id") or "")
            for row in spans
            if row.get("error_type") == "rate_limited"
        )
        rate_limited_count = len(rate_limited_calls - {""})
        total_calls = len(usage_rows) + len(tool_rows)
        successful_tools = sum(
            1 for row in tool_rows if row.get("status") == "success"
        )
        empty_tool_result_count = 0
        zero_source_search_count = 0
        for row in tool_rows:
            if row.get("status") != "success":
                continue
            try:
                attributes = json.loads(row.get("attributes_json") or "{}")
            except (TypeError, ValueError, json.JSONDecodeError):
                attributes = {}
            if not attributes.get("result_chars", 0):
                empty_tool_result_count += 1
            if "search" in str(row.get("name") or "").lower() and not attributes.get(
                "source_count", 0
            ):
                zero_source_search_count += 1
        operations.update(
            {
                "llm_call_count": len(usage_rows) if usage_rows else len(llm_rows),
                "retry_count": len(retries),
                "rate_limited_count": rate_limited_count,
                "rate_429": rate_limited_count / total_calls if total_calls else 0.0,
                "tool_call_count": len(tool_rows),
                "tool_success_rate": (
                    successful_tools / len(tool_rows) if tool_rows else 0.0
                ),
                "empty_tool_result_count": empty_tool_result_count,
                "zero_source_search_count": zero_source_search_count,
            }
        )
        return operations

    def _accounting_operations(
        self,
        run_id: str,
        *,
        rows: list[dict[str, Any]] | None = None,
        provider: str | None = None,
        model: str | None = None,
        retry_rows: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        usage_rows = rows or []
        retries = retry_rows or []
        if provider or model:
            return self._filtered_accounting_operations(usage_rows, retries)
        metrics = self.get_metrics(run_id)
        token_operations = self._filtered_accounting_operations(usage_rows, retries)
        return {
            "llm_call_count": token_operations["llm_call_count"],
            "retry_count": metrics.get("retry_count", 0),
            "rate_limited_count": metrics.get("rate_limited_count", 0),
            "rate_429": metrics.get("rate_429", 0.0),
            "cache_hit_rate": token_operations["cache_hit_rate"],
            "cache_input_ratio": token_operations["cache_input_ratio"],
            "reasoning_output_ratio": token_operations["reasoning_output_ratio"],
            "output_tokens_per_second": token_operations[
                "output_tokens_per_second"
            ],
            "tool_call_count": metrics.get("tool_call_count", 0),
            "tool_success_rate": metrics.get("tool_success_rate", 0.0),
            "empty_tool_result_count": metrics.get("empty_tool_result_count", 0),
            "zero_source_search_count": metrics.get("zero_source_search_count", 0),
        }

    def list_runs(self, limit: int = 100, user_id: str | None = None) -> list[dict[str, Any]]:
        with self._lock, self._connect() as conn:
            if user_id is None:
                rows = conn.execute(
                    "SELECT * FROM runs ORDER BY started_at DESC LIMIT ?", (limit,)
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM runs WHERE user_id = ? ORDER BY started_at DESC LIMIT ?",
                    (user_id, limit),
                ).fetchall()
        return [dict(row) for row in rows]

    def list_runs_for_usage(
        self,
        *,
        user_id: str,
        cutoff: float | None = None,
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        """List all retained usage runs after applying owner filtering in SQL."""
        clauses = ["user_id = ?"]
        parameters: list[Any] = [user_id]
        if cutoff is not None:
            clauses.append("started_at >= ?")
            parameters.append(cutoff)
        if status:
            clauses.append("status = ?")
            parameters.append(status)
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM runs WHERE "
                + " AND ".join(clauses)
                + " ORDER BY started_at DESC",
                tuple(parameters),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_run(self, run_id: str, user_id: str | None = None) -> dict[str, Any] | None:
        with self._lock, self._connect() as conn:
            if user_id is None:
                row = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
            else:
                row = conn.execute(
                    "SELECT * FROM runs WHERE run_id = ? AND user_id = ?", (run_id, user_id)
                ).fetchone()
        return dict(row) if row else None

    def list_spans(self, run_id: str) -> list[dict[str, Any]]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM spans WHERE run_id = ? ORDER BY started_at ASC",
                (run_id,),
            ).fetchall()
        return [dict(row) for row in rows]


class SpanContext:
    """Context manager that records one span and carries contextvars."""

    def __init__(
        self,
        recorder: "TraceRecorder",
        *,
        span_id: str,
        run_id: str,
        parent_span_id: str | None,
        name: str,
        kind: str,
        agent_role: str | None,
        attributes: dict[str, Any],
        input_preview: str | None,
        provider: str | None,
        model: str | None,
        langfuse_parent_span_id: str | None,
    ):
        self.recorder = recorder
        self.span_id = span_id
        self.run_id = run_id
        self.parent_span_id = parent_span_id
        self.name = name
        self.kind = kind
        self.agent_role = agent_role
        self.attributes = attributes
        self.input_preview = input_preview
        self.provider = provider
        self.model = model
        self.langfuse_parent_span_id = langfuse_parent_span_id
        self.langfuse_observation_id: str | None = None
        self.usage = TokenUsage()
        self.output_preview: str | None = None
        self.retry_count: int = 0
        self.error_type: str | None = None
        self.http_status: int | None = None
        self.final_status: str | None = None
        self.error_message: str | None = None
        self.started_monotonic = monotonic_time()
        self.langfuse_bridge: Any = None
        self._run_token: contextvars.Token[str | None] | None = None
        self._span_token: contextvars.Token[str | None] | None = None
        self._span_ctx_token: contextvars.Token["SpanContext | NoopSpanContext | None"] | None = None
        self._langfuse_span_token: contextvars.Token[str | None] | None = None

    def __enter__(self) -> "SpanContext":
        if self.recorder.store is not None:
            self.recorder._safe(
                self.recorder.store.start_span,
                span_id=self.span_id,
                run_id=self.run_id,
                parent_span_id=self.parent_span_id,
                name=self.name,
                kind=self.kind,
                agent_role=self.agent_role,
                attributes=self.attributes,
                input_preview=self.input_preview,
                provider=self.provider,
                model=self.model,
            )
        self._run_token = _current_run_id.set(self.run_id)
        self._span_token = _current_span_id.set(self.span_id)
        self._span_ctx_token = _current_span_ctx.set(self)
        if (
            self.recorder.langfuse is not None
            and not self.attributes.get("langfuse_callback_managed", False)
        ):
            self.langfuse_bridge = self.recorder._safe(self.recorder.langfuse.span, self)
            if self.langfuse_bridge is not None:
                self.recorder._safe(self.langfuse_bridge.enter)
        self._langfuse_span_token = _current_langfuse_span_id.set(
            self.langfuse_observation_id
        )
        return self

    def __exit__(self, exc_type: Any, exc: BaseException | None, _tb: Any) -> None:
        status = "error" if exc or self.error_type else (self.final_status or "success")
        self.error_message = _redact_text(_exc_message(exc)) if exc else None
        duration_seconds = monotonic_time() - self.started_monotonic
        if self.recorder.store is not None:
            self.recorder._safe(
                self.recorder.store.finish_span,
                span_id=self.span_id,
                status=status,
                usage=self.usage,
                output_preview=self.output_preview,
                error=self.error_message,
                retry_count=self.retry_count,
                error_type=self.error_type,
                http_status=self.http_status,
                attributes=self.attributes,
            )
        if self.recorder.prometheus is not None and self.kind in {"llm", "tool", "agent"}:
            self.recorder._safe(
                self.recorder.prometheus.observe_span,
                self,
                status,
                duration_seconds,
            )
        if self.langfuse_bridge is not None:
            self.recorder._safe(self.langfuse_bridge.exit, exc_type, exc, _tb)
        if (
            self.kind == "run"
            and self.recorder.langfuse is not None
            and self.recorder.configuration.langfuse_flush_on_run_end
        ):
            self.recorder._safe(self.recorder.langfuse.flush)
        if self._span_ctx_token is not None:
            _current_span_ctx.reset(self._span_ctx_token)
        if self._langfuse_span_token is not None:
            _current_langfuse_span_id.reset(self._langfuse_span_token)
        if self._span_token is not None:
            _current_span_id.reset(self._span_token)
        if self._run_token is not None:
            _current_run_id.reset(self._run_token)

    def add_usage(
        self,
        usage: TokenUsage,
        provider: str | None,
        model: str | None,
        *,
        event_key: str | None = None,
        attempt_index: int = 1,
        stage: str = "unknown",
        task_id: str | None = None,
        operation: str | None = None,
        duration_ms: int | None = None,
        response_status: str = "success",
    ) -> int | None:
        """Attach usage to this span and persist a usage event."""
        self.recorder.estimate_usage_cost(usage, provider, model)
        self.usage = TokenUsage(
            input_tokens=self.usage.input_tokens + usage.input_tokens,
            output_tokens=self.usage.output_tokens + usage.output_tokens,
            total_tokens=self.usage.total_tokens + usage.total_tokens,
            cached_input_tokens=self.usage.cached_input_tokens + usage.cached_input_tokens,
            cache_creation_input_tokens=(
                self.usage.cache_creation_input_tokens
                + usage.cache_creation_input_tokens
            ),
            reasoning_tokens=self.usage.reasoning_tokens + usage.reasoning_tokens,
            estimated_cost_usd=self.usage.estimated_cost_usd + usage.estimated_cost_usd,
        )
        if self.recorder.store is not None:
            return self.recorder._safe(
                self.recorder.store.add_usage,
                self.run_id,
                self.span_id,
                provider,
                model,
                usage,
                event_key=event_key,
                attempt_index=attempt_index,
                stage=stage,
                agent_role=self.agent_role,
                task_id=task_id,
                operation=operation or self.name,
                duration_ms=duration_ms,
                response_status=response_status,
            )
        return None

    def set_output(self, payload: Any) -> None:
        """Attach a redacted output preview using the recorder payload policy."""
        if (
            not self.recorder.configuration.observability_enabled
            or self.recorder.configuration.trace_payload_mode == "none"
        ):
            return
        limit = (
            None
            if self.recorder.configuration.trace_payload_mode == "full"
            else self.recorder.configuration.trace_preview_chars
        )
        self.output_preview = _message_preview(
            payload,
            limit,
            redact=self.recorder.configuration.trace_redaction_enabled,
        )

    def score(self, name: str, value: float | str | bool, comment: str | None = None) -> None:
        """Record a quality/business score locally and in configured sinks."""
        self.attributes.setdefault("scores", {})[name] = value
        if self.recorder.langfuse is not None:
            self.recorder._safe(
                self.recorder.langfuse.score,
                self,
                name,
                value,
                comment,
            )
        if self.recorder.prometheus is not None:
            self.recorder._safe(
                self.recorder.prometheus.observe_score,
                name,
                value,
                self.agent_role or "unknown",
            )

    def record_retry(
        self,
        *,
        attempt: int,
        error_type: str,
        http_status: int | None = None,
        retryable: bool = False,
        delay_s: float | None = None,
        message: str | None = None,
    ) -> None:
        """Record one failed attempt on this span and persist a retry event.

        Increments the span's retry counter and appends a ``retry_events`` row
        (which carries the per-attempt ``error_type``/``http_status``). The span's
        terminal ``error_type``/``http_status`` are set separately via
        :meth:`record_outcome`, so a span that eventually succeeds keeps a clean
        outcome while still reflecting how many retries it took.
        """
        self.retry_count += 1
        if self.recorder.store is not None:
            self.recorder._safe(
                self.recorder.store.record_retry_event,
                run_id=self.run_id,
                span_id=self.span_id,
                attempt=attempt,
                error_type=error_type,
                http_status=http_status,
                retryable=retryable,
                delay_s=delay_s,
                message=_redact_text(message) if message else None,
            )
        if self.recorder.prometheus is not None:
            self.recorder._safe(self.recorder.prometheus.observe_retry, self, error_type)

    def record_outcome(
        self,
        *,
        error_type: str | None = None,
        http_status: int | None = None,
        retry_count: int | None = None,
        status: str | None = None,
    ) -> None:
        """Record the final outcome of a span (used on terminal failure)."""
        if error_type is not None:
            self.error_type = error_type
        if http_status is not None:
            self.http_status = http_status
        if retry_count is not None:
            self.retry_count = retry_count
        if status is not None:
            self.final_status = status


class NoopSpanContext:
    """Span-like context manager used when observability is disabled."""

    span_id: str | None = None
    run_id: str | None = None

    def __init__(self):
        self.started_monotonic = monotonic_time()
        self.attributes: dict[str, Any] = {}
        self.output_preview: str | None = None
        self.usage = TokenUsage()
        self.retry_count = 0
        self.error_type: str | None = None
        self.http_status: int | None = None

    def __enter__(self) -> "NoopSpanContext":
        return self

    def __exit__(self, *_args: Any) -> None:
        return None

    def add_usage(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def set_output(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def score(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def record_retry(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def record_outcome(self, *_args: Any, **_kwargs: Any) -> None:
        return None


class TraceRecorder:
    """Fail-open facade for tracing one runtime config."""

    def __init__(self, config: RunnableConfig | None):
        self.config = config or {"configurable": {}, "metadata": {}}
        self.configuration = Configuration.from_runnable_config(self.config)
        self.store = (
            _get_store(self.configuration.trace_store_path)
            if self.configuration.token_usage_accounting_enabled
            or (
                self.configuration.observability_enabled
                and self.configuration.sqlite_observability_enabled
            )
            else None
        )
        self.langfuse = create_langfuse_sink(self.configuration, self.config)
        self.prometheus = get_prometheus_metrics(self.configuration)
        self.enabled = any(
            (self.store is not None, self.langfuse is not None, self.prometheus is not None)
        )

    def _safe(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Observability write failed: %s", exc)
            if self.prometheus is not None:
                component = type(getattr(fn, "__self__", None)).__name__ or "unknown"
                operation = getattr(fn, "__name__", "unknown")
                try:
                    self.prometheus.observe_export_error(component, operation)
                except Exception:  # noqa: BLE001
                    pass
            return None

    def estimate_usage_cost(
        self,
        usage: TokenUsage,
        provider: str | None,
        model: str | None,
    ) -> float:
        """Populate a local cost estimate from configured per-million-token rates."""
        if usage.estimated_cost_usd > 0:
            usage.cost_source = "provider_reported"
            return usage.estimated_cost_usd
        prices = self.configuration.model_costs_per_million
        candidates = [model, f"{provider}:{model}" if provider and model else None]
        rates = next((prices[key] for key in candidates if key and key in prices), None)
        if not rates:
            usage.cost_source = "unavailable"
            return 0.0
        priced_input = usage.input_tokens or usage.estimated_input_tokens
        priced_output = usage.output_tokens or usage.estimated_output_tokens
        cached = min(usage.cached_input_tokens, priced_input)
        uncached = max(0, priced_input - cached)
        reasoning = min(usage.reasoning_tokens, usage.output_tokens)
        normal_output = max(0, priced_output - reasoning)
        cost = (
            uncached * float(rates.get("input", 0))
            + cached * float(rates.get("cached_input", rates.get("input", 0)))
            + usage.cache_creation_input_tokens
            * float(rates.get("cache_creation_input", rates.get("input", 0)))
            + normal_output * float(rates.get("output", 0))
            + reasoning * float(rates.get("reasoning", rates.get("output", 0)))
        ) / 1_000_000
        usage.estimated_cost_usd = cost
        usage.cost_source = "configured_estimate"
        return cost

    def active_span(self) -> SpanContext | NoopSpanContext:
        """Return the currently-entered span, or a noop span if none.

        Fail-open: governance retry code calls this to record retries on the span
        opened by ``observe_tool_call`` without holding the span handle directly.
        """
        return _current_span_ctx.get() or NoopSpanContext()

    def start_run(
        self,
        run_id: str,
        *,
        name: str = "run",
        user_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        input_payload: Any = None,
    ) -> SpanContext | NoopSpanContext:
        if not self.enabled:
            return NoopSpanContext()
        run_attributes = dict(metadata or {})
        price_table = self.configuration.model_costs_per_million
        if price_table:
            run_attributes["model_costs_price_table_hash"] = hashlib.sha256(
                json.dumps(price_table, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
        run_attributes["budget_limits"] = {
            "input_tokens": self.configuration.max_run_input_tokens,
            "output_tokens": self.configuration.max_run_output_tokens,
            "model_calls": self.configuration.max_run_model_calls,
            "cost_micro_usd": self.configuration.max_run_cost_micro_usd,
        }
        if self.store is not None:
            self._safe(self.store.start_run, run_id, user_id, run_attributes)
            stored_run = self._safe(self.store.get_run, run_id) or {}
            run_attributes["attempt_count"] = int(stored_run.get("attempt_count") or 1)
        return self.start_span(
            name=name,
            kind="run",
            run_id=run_id,
            parent_span_id=None,
            attributes=run_attributes,
            input_payload=input_payload,
        )

    def finish_run(self, run_id: str, status: str, error: str | None = None) -> dict[str, Any]:
        """Finalize the run and return a usage+metrics summary dict.

        Token keys (``input_tokens``/``output_tokens``/``total_tokens``) are kept
        for backward compatibility; retry/metrics keys are added on top. All
        metric computation is fail-open via ``_safe`` so a query error never
        interrupts run completion.
        """
        empty = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "cached_input_tokens": 0,
            "cache_creation_input_tokens": 0,
            "reasoning_tokens": 0,
            "estimated_cost_usd": 0.0,
            "retry_count": 0,
            "rate_limit_events": 0,
            "rate_limited_count": 0,
            "terminal_rate_limited_count": 0,
            "rate_429": 0.0,
            "total_llm_tool_calls": 0,
            "attempt_count": 0,
            "llm_call_count": 0,
            "cache_eligible_count": 0,
            "cache_hit_count": 0,
            "cache_hit_rate": 0.0,
            "cache_input_ratio": 0.0,
            "llm_output_input_ratio": 0.0,
            "llm_reasoning_output_ratio": 0.0,
            "llm_output_tokens_per_second": 0.0,
            "tool_call_count": 0,
            "tool_success_count": 0,
            "tool_success_rate": 0.0,
            "empty_tool_result_count": 0,
            "zero_source_search_count": 0,
        }
        if not self.enabled:
            return empty
        usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        metrics: dict[str, Any] = {}
        safe_error = (
            _redact_text(error)
            if error and self.configuration.trace_redaction_enabled
            else error
        )
        if self.store is not None:
            usage = self._safe(self.store.finish_run, run_id, status, safe_error) or usage
            metrics = self._safe(self.store.get_metrics, run_id) or {}
        stored_run = self._safe(self.store.get_run, run_id) if self.store is not None else None
        reserved_budget: dict[str, int] = {}
        if stored_run:
            gate = BudgetGate.from_config(
                self.configuration,
                run_id,
                started_at=float(stored_run.get("started_at") or time.time()),
            )
            reserved_budget = self._safe(gate.outstanding_reservations) or {}
        active = self.active_span()
        if isinstance(active, SpanContext) and active.kind == "run":
            if error:
                active.record_outcome(error_type="run_error", status="error")
            else:
                active.record_outcome(status=status)
            if self.prometheus is not None:
                self._safe(
                    self.prometheus.observe_run,
                    status,
                    monotonic_time() - active.started_monotonic,
                )
        return {
            "input_tokens": usage.get("input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
            "cached_input_tokens": usage.get("cached_input_tokens", 0),
            "cache_creation_input_tokens": usage.get("cache_creation_input_tokens", 0),
            "reasoning_tokens": usage.get("reasoning_tokens", 0),
            "estimated_cost_usd": usage.get("estimated_cost_usd", 0.0),
            "retry_count": metrics.get("retry_count", 0),
            "rate_limit_events": metrics.get("rate_limit_events", 0),
            "rate_limited_count": metrics.get("rate_limited_count", 0),
            "terminal_rate_limited_count": metrics.get("terminal_rate_limited_count", 0),
            "rate_429": metrics.get("rate_429", 0.0),
            "total_llm_tool_calls": metrics.get("total_llm_tool_calls", 0),
            "attempt_count": metrics.get("attempt_count", 0),
            "llm_call_count": metrics.get("llm_call_count", 0),
            "cache_eligible_count": metrics.get("cache_eligible_count", 0),
            "cache_hit_count": metrics.get("cache_hit_count", 0),
            "cache_hit_rate": metrics.get("cache_hit_rate", 0.0),
            "cache_input_ratio": metrics.get("cache_input_ratio", 0.0),
            "llm_output_input_ratio": metrics.get("llm_output_input_ratio", 0.0),
            "llm_reasoning_output_ratio": metrics.get(
                "llm_reasoning_output_ratio", 0.0
            ),
            "llm_output_tokens_per_second": metrics.get(
                "llm_output_tokens_per_second", 0.0
            ),
            "tool_call_count": metrics.get("tool_call_count", 0),
            "tool_success_count": metrics.get("tool_success_count", 0),
            "tool_success_rate": metrics.get("tool_success_rate", 0.0),
            "empty_tool_result_count": metrics.get("empty_tool_result_count", 0),
            "zero_source_search_count": metrics.get("zero_source_search_count", 0),
            "usage_accounting": (
                self._safe(
                    self.store.get_usage_accounting,
                    run_id,
                    reserved_budget=reserved_budget,
                )
                if self.store is not None
                else None
            ),
        }

    def start_span(
        self,
        *,
        name: str,
        kind: str,
        run_id: str | None = None,
        parent_span_id: str | None = None,
        agent_role: str | None = None,
        attributes: dict[str, Any] | None = None,
        input_payload: Any = None,
        provider: str | None = None,
        model: str | None = None,
    ) -> SpanContext | NoopSpanContext:
        if not self.enabled:
            return NoopSpanContext()
        resolved_run_id = run_id or _current_run_id.get()
        if not resolved_run_id:
            resolved_run_id = self.config.get("metadata", {}).get("run_id")
        if not resolved_run_id:
            return NoopSpanContext()
        resolved_parent = parent_span_id if parent_span_id is not None else _current_span_id.get()
        resolved_langfuse_parent = _current_langfuse_span_id.get()
        preview = None
        if (
            self.configuration.observability_enabled
            and self.configuration.trace_payload_mode != "none"
        ):
            limit = None if self.configuration.trace_payload_mode == "full" else self.configuration.trace_preview_chars
            preview = _message_preview(
                input_payload,
                limit,
                redact=self.configuration.trace_redaction_enabled,
            )
        return SpanContext(
            self,
            span_id=uuid.uuid4().hex,
            run_id=resolved_run_id,
            parent_span_id=resolved_parent,
            name=name,
            kind=kind,
            agent_role=agent_role,
            attributes=attributes or {},
            input_preview=preview,
            provider=provider,
            model=model,
            langfuse_parent_span_id=resolved_langfuse_parent,
        )


def _get_store(path: str) -> SQLiteTraceStore:
    with _stores_lock:
        store = _stores.get(path)
        if store is None:
            store = SQLiteTraceStore(path)
            _stores[path] = store
        return store


def get_trace_recorder(config: RunnableConfig | None) -> TraceRecorder:
    """Build a trace recorder for a runnable config."""
    return TraceRecorder(config)


def _langchain_invoke_config(
    recorder: TraceRecorder,
    config: RunnableConfig | None,
    capture: UsageCaptureCallback,
) -> RunnableConfig:
    """Attach usage and optional Langfuse callbacks without mutating caller config."""
    invoke_config: dict[str, Any] = dict(config or {})
    callbacks = list(invoke_config.get("callbacks") or [])
    callbacks.append(capture)
    if (
        recorder.langfuse is not None
        and recorder.configuration.langfuse_langchain_callback_enabled
    ):
        handler = recorder._safe(recorder.langfuse.callback_handler)
        if handler is not None:
            callbacks.append(handler)
    invoke_config["callbacks"] = callbacks
    return cast(RunnableConfig, invoke_config)


@dataclass(frozen=True, slots=True)
class _ModelInvocationResult:
    """Carry a model response with optional first-packet probe metadata."""

    response: Any
    ttft_seconds: float | None = None
    probe_status: str = "off"
    usage_records: tuple[TokenUsage, ...] = ()
    usage_capture: UsageCaptureCallback | None = None


def _observe_first_packet_metrics(
    recorder: TraceRecorder,
    result: _ModelInvocationResult,
    *,
    provider: str | None,
    model: str | None,
    agent_role: str | None,
    operation: str,
) -> None:
    """Best-effort record TTFT and safe streaming downgrade metrics."""
    metrics = recorder.prometheus
    if metrics is None:
        return
    if result.ttft_seconds is not None:
        recorder._safe(  # noqa: SLF001 - shared fail-open recorder boundary
            metrics.observe_first_token,
            provider=provider or "unknown",
            model=model or "unknown",
            agent_role=agent_role or "unknown",
            operation=operation,
            duration_seconds=result.ttft_seconds,
            probe_mode=recorder.configuration.model_first_packet_probe,
            slow=(
                result.ttft_seconds
                > recorder.configuration.model_slow_first_packet_threshold_seconds
            ),
        )
    if result.probe_status == "fallback":
        recorder._safe(  # noqa: SLF001
            metrics.observe_streaming_fallback,
            provider or "unknown",
            model or "unknown",
            "unsupported",
        )


async def _call_model_ainvoke(
    model: Any,
    messages: list[BaseMessage],
    invoke_config: RunnableConfig | None,
) -> Any:
    """Invoke a model while preserving the optional callback config."""
    try:
        signature = inspect.signature(model.ainvoke)
        accepts_config = "config" in signature.parameters or any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )
    except (TypeError, ValueError):
        accepts_config = True
    if invoke_config is None or not accepts_config:
        return await model.ainvoke(messages)
    return await model.ainvoke(messages, config=invoke_config)


def _streaming_unsupported(exc: BaseException) -> bool:
    """Recognize bounded capability errors that are safe to downgrade."""
    if isinstance(exc, NotImplementedError | AttributeError):
        return True
    text = _exc_message(exc).lower()
    return any(
        marker in text
        for marker in (
            "astream not implemented",
            "streaming is not supported",
            "streaming not supported",
            "does not support streaming",
        )
    )


async def _close_async_iterator(iterator: Any) -> None:
    """Best-effort close a model stream after success, error, or timeout."""
    close = getattr(iterator, "aclose", None)
    if callable(close):
        try:
            await close()
        except Exception:  # noqa: BLE001 - cleanup must preserve the real outcome
            pass


async def _ainvoke_model(
    model: Any,
    messages: list[BaseMessage],
    recorder: TraceRecorder,
    config: RunnableConfig | None,
    *,
    span_id: str | None = None,
    attempt_index: int = 1,
    model_name: str | None = None,
    agent_role: str | None = None,
    usage_capture: UsageCaptureCallback | None = None,
) -> _ModelInvocationResult:
    """Invoke with optional TTFT streaming and conservative fallback."""
    capture = usage_capture or UsageCaptureCallback(
        recorder=recorder,
        config=config,
        messages=messages,
        model=model,
        model_name=model_name,
        span_id=span_id,
        attempt_index=attempt_index,
        agent_role=agent_role,
    )
    invoke_config = _langchain_invoke_config(recorder, config, capture)

    async def call_model() -> Any:
        capture.begin_physical_attempt()
        try:
            response = await _call_model_ainvoke(model, messages, invoke_config)
            capture.settle_outer_success(response)
            return response
        except BaseException as exc:
            capture.settle_outer_failure(exc)
            setattr(exc, "usage_capture_records", tuple(capture.records))
            raise

    configuration = recorder.configuration
    probe_mode = (
        configuration.model_first_packet_probe
        if configuration.model_circuit_breaker_enabled
        else "off"
    )
    if probe_mode == "off":
        return _ModelInvocationResult(
            await call_model(),
            usage_records=tuple(capture.records),
            usage_capture=capture,
        )

    stream_method = getattr(model, "astream", None)
    if not callable(stream_method):
        return _ModelInvocationResult(
            await call_model(),
            probe_status="fallback",
            usage_records=tuple(capture.records),
            usage_capture=capture,
        )

    iterator: Any = None
    received_first = False
    started = monotonic_time()
    capture.begin_physical_attempt()
    try:
        try:
            stream_signature = inspect.signature(stream_method)
            stream_accepts_config = "config" in stream_signature.parameters or any(
                parameter.kind is inspect.Parameter.VAR_KEYWORD
                for parameter in stream_signature.parameters.values()
            )
        except (TypeError, ValueError):
            stream_accepts_config = True
        iterator = (
            stream_method(messages)
            if invoke_config is None or not stream_accepts_config
            else stream_method(messages, config=invoke_config)
        )
        if inspect.isawaitable(iterator):
            iterator = await iterator
        try:
            first = (
                await asyncio.wait_for(
                    anext(iterator),
                    timeout=configuration.model_first_packet_timeout_seconds,
                )
                if probe_mode == "enforced"
                else await anext(iterator)
            )
        except StopAsyncIteration as exc:
            raise RuntimeError("model stream returned no chunks") from exc
        received_first = True
        first_elapsed = monotonic_time() - started

        if isinstance(first, AIMessageChunk):
            merged = first
            shape_ok = True
            async for chunk in iterator:
                if not isinstance(chunk, AIMessageChunk):
                    shape_ok = False
                    break
                merged = merged + chunk
            if shape_ok:
                response = message_chunk_to_message(merged)
                capture.settle_outer_success(response)
                return _ModelInvocationResult(
                    response,
                    ttft_seconds=max(0.0, first_elapsed),
                    probe_status="streamed",
                    usage_records=tuple(capture.records),
                    usage_capture=capture,
                )
        else:
            trailing = [item async for item in iterator]
            if not trailing:
                capture.settle_outer_success(first)
                return _ModelInvocationResult(
                    first,
                    probe_status="non_streaming_wrapper",
                    usage_records=tuple(capture.records),
                    usage_capture=capture,
                )
            # Structured-output runnables stream incremental parser partials
            # of one result; every item is the same pydantic type and the
            # final item is the complete parsed value (verified against
            # ``with_structured_output(..., method="function_calling")``).
            if isinstance(first, BaseModel) and all(
                isinstance(item, type(first)) for item in trailing
            ):
                response = trailing[-1]
                capture.settle_outer_success(response)
                return _ModelInvocationResult(
                    response,
                    ttft_seconds=max(0.0, first_elapsed),
                    probe_status="non_streaming_wrapper",
                    usage_records=tuple(capture.records),
                    usage_capture=capture,
                )
        # The stream shape cannot be merged faithfully (mixed chunk types or
        # fragment-style non-message items). The probe must never change call
        # semantics, so degrade to the plain non-streaming invoke instead of
        # raising — this also keeps shadow mode observation-only.
        if iterator is not None:
            await _close_async_iterator(iterator)
            iterator = None
        capture.settle_outer_failure(
            RuntimeError("stream disconnected before final usage metadata")
        )
        return _ModelInvocationResult(
            await call_model(),
            probe_status="fallback",
            usage_records=tuple(capture.records),
            usage_capture=capture,
        )
    except Exception as exc:
        if not received_first and (
            probe_mode == "shadow" or _streaming_unsupported(exc)
        ):
            capture.settle_outer_failure(exc)
            return _ModelInvocationResult(
                await call_model(),
                probe_status="fallback",
                usage_records=tuple(capture.records),
                usage_capture=capture,
            )
        capture.settle_outer_failure(exc)
        setattr(exc, "usage_capture_records", tuple(capture.records))
        raise
    finally:
        if iterator is not None:
            await _close_async_iterator(iterator)


def _usage_stage(agent_role: str | None, attributes: dict[str, Any]) -> str:
    stage = str(attributes.get("stage") or "")
    if stage in {"preparing", "planning", "researching", "synthesizing", "writing", "finalizing"}:
        return stage
    return _CIRCUIT_ROLE_STAGE.get(agent_role or "", "preparing")


def _usage_attributes(
    attributes: dict[str, Any] | None,
    stage: str | None,
) -> dict[str, Any]:
    result = dict(attributes or {})
    if stage is not None:
        if stage not in {
            "preparing",
            "planning",
            "researching",
            "synthesizing",
            "writing",
            "finalizing",
        }:
            raise ValueError(f"Unsupported token accounting stage: {stage}")
        result["stage"] = stage
    return result


def _sum_usage(records: list[TokenUsage]) -> TokenUsage:
    return TokenUsage(
        input_tokens=sum(item.input_tokens for item in records),
        output_tokens=sum(item.output_tokens for item in records),
        total_tokens=sum(item.total_tokens for item in records),
        cached_input_tokens=sum(item.cached_input_tokens for item in records),
        cache_creation_input_tokens=sum(
            item.cache_creation_input_tokens for item in records
        ),
        reasoning_tokens=sum(item.reasoning_tokens for item in records),
        estimated_input_tokens=sum(item.estimated_input_tokens for item in records),
        estimated_output_tokens=sum(item.estimated_output_tokens for item in records),
        estimated_total_tokens=sum(item.estimated_total_tokens for item in records),
        estimated_cost_usd=sum(item.estimated_cost_usd for item in records),
        usage_source=(records[0].usage_source if len(records) == 1 else "provider_reported"),
    )


async def _publish_usage_revision(
    config: RunnableConfig | None, revision: int | None, accounting_status: str
) -> None:
    metadata = (config or {}).get("metadata") or {}
    if not revision or not metadata.get("run_id"):
        return
    try:
        await event_publisher_from_config(config or {}).publish(
            "run.usage.updated",
            stage=None,
            payload={"revision": revision, "accounting_status": accounting_status},
            dedupe_key=f"run-usage:{revision}",
        )
    except Exception as exc:  # noqa: BLE001 - accounting is fail-open
        logger.debug("Usage update event failed open: %s", exc)


def _run_accounting_status(
    recorder: TraceRecorder,
    run_id: str,
    fallback: str = "partial",
) -> str:
    if recorder.store is None:
        return fallback
    projection = recorder._safe(recorder.store.get_usage_accounting, run_id) or {}
    return str(projection.get("accounting_status") or fallback)


async def _record_successful_invocation(
    *,
    recorder: TraceRecorder,
    span: Any,
    invocation_result: _ModelInvocationResult,
    response: Any,
    model: Any,
    messages: list[BaseMessage],
    config: RunnableConfig | None,
    provider: str | None,
    model_id: str | None,
    agent_role: str | None,
    operation: str,
    attributes: dict[str, Any],
    attempt_index: int,
    duration_ms: int | None,
) -> TokenUsage:
    records = list(invocation_result.usage_records)
    if not records:
        fallback = TokenUsage.from_response(response)
        if fallback.has_reported_tokens:
            fallback.usage_source = (
                "provider_reported"
                if fallback.input_tokens > 0 and fallback.output_tokens > 0
                else "provider_partial"
            )
            records = [fallback]
    if not records and recorder.configuration.token_usage_estimation_enabled:
        records = [_estimated_usage(model, messages, response)]
    if not records:
        records = [TokenUsage(usage_source="missing")]
    task_id = str((config or {}).get("metadata", {}).get("task_id") or "") or None
    stage = _usage_stage(agent_role, attributes)
    revisions: list[int] = []
    for physical_index, usage in enumerate(records, start=1):
        physical_attempt_index = attempt_index + physical_index - 1
        revision = span.add_usage(
            usage,
            provider,
            model_id,
            event_key=f"{span.span_id}:{physical_attempt_index}:success",
            attempt_index=physical_attempt_index,
            stage=stage,
            task_id=task_id,
            operation=operation,
            duration_ms=duration_ms,
            response_status=usage.response_status,
        )
        if revision:
            revisions.append(int(revision))
    aggregate = _sum_usage(records)
    if invocation_result.usage_capture is not None:
        invocation_result.usage_capture.settle_estimated_success(aggregate)
    status = _run_accounting_status(recorder, span.run_id)
    await _publish_usage_revision(config, max(revisions, default=None), status)
    return aggregate


async def _record_failed_invocation(
    *,
    recorder: TraceRecorder,
    span: Any,
    exc: BaseException,
    config: RunnableConfig | None,
    provider: str | None,
    model_id: str | None,
    agent_role: str | None,
    operation: str,
    attributes: dict[str, Any],
    attempt_index: int,
    usage_capture: UsageCaptureCallback | None = None,
) -> int:
    if usage_capture is not None:
        usage_capture.settle_outer_failure(exc)
    captured = list(
        usage_capture.records
        if usage_capture is not None
        else (getattr(exc, "usage_capture_records", ()) or ())
    )
    task_id = str((config or {}).get("metadata", {}).get("task_id") or "") or None
    status = "unknown_failed" if _is_uncertain_model_failure(exc) else "rejected"
    records = captured or [
        TokenUsage(
            usage_source="missing",
            response_status=status,
        )
    ]
    revisions: list[int] = []
    for physical_index, usage in enumerate(records, start=1):
        physical_attempt_index = attempt_index + physical_index - 1
        revision = span.add_usage(
            usage,
            provider,
            model_id,
            event_key=f"{span.span_id}:{physical_attempt_index}:failed",
            attempt_index=physical_attempt_index,
            stage=_usage_stage(agent_role, attributes),
            task_id=task_id,
            operation=operation,
            response_status=usage.response_status if captured else status,
        )
        if revision:
            revisions.append(int(revision))
    await _publish_usage_revision(
        config,
        max(revisions, default=None),
        _run_accounting_status(recorder, span.run_id),
    )
    return len(records)


def apply_helicone_config(
    model_config: dict[str, Any],
    runtime_config: RunnableConfig | None,
    *,
    span_name: str,
    agent_role: str | None = None,
) -> dict[str, Any]:
    """Attach Helicone headers/base URL when configured.

    LangChain providers differ in which transport fields they accept. This
    function only enriches the config dict; callers and model providers may
    ignore unsupported fields without affecting local SQLite tracing.
    """
    configuration = Configuration.from_runnable_config(runtime_config)
    if not configuration.helicone_enabled or not configuration.helicone_api_key:
        return dict(model_config)

    run_id, parent_span_id = current_span_ids()
    headers = {
        "Helicone-Auth": f"Bearer {configuration.helicone_api_key}",
        "Helicone-Session-Id": run_id or runtime_config.get("metadata", {}).get("run_id", "default")
        if runtime_config
        else "default",
        "Helicone-Session-Path": f"/{agent_role or 'agent'}/{span_name}",
        "Helicone-Property-Run-Id": run_id or "",
        "Helicone-Property-Parent-Span-Id": parent_span_id or "",
        "Helicone-Property-Agent-Role": agent_role or "",
        "Helicone-Property-Stage": span_name,
        "Helicone-Property-Model": str(model_config.get("model", "")),
    }
    enriched = dict(model_config)
    if configuration.helicone_headers_enabled:
        existing_headers = dict(enriched.get("default_headers") or enriched.get("headers") or {})
        existing_headers.update(headers)
        enriched["default_headers"] = existing_headers
        enriched["headers"] = existing_headers
    if configuration.helicone_base_url:
        enriched["base_url"] = configuration.helicone_base_url
    return enriched


async def invoke_model_with_observability(
    model: Any,
    messages: list[BaseMessage],
    config: RunnableConfig | None,
    *,
    span_name: str,
    agent_role: str | None = None,
    model_name: str | None = None,
    stage: str | None = None,
    attributes: dict[str, Any] | None = None,
    budget_gate: BudgetGate | None = None,
) -> Any:
    """Invoke a model while recording a local LLM span and token usage."""
    provider, model_id = _provider_model(model_name)
    recorder = get_trace_recorder(config)
    span_attributes = _usage_attributes(attributes, stage)
    if recorder.langfuse is not None and recorder.configuration.langfuse_langchain_callback_enabled:
        span_attributes["langfuse_callback_managed"] = True
    with recorder.start_span(
        name=span_name,
        kind="llm",
        agent_role=agent_role,
        attributes=span_attributes,
        input_payload=messages,
        provider=provider,
        model=model_id or model_name,
    ) as span:
        usage_capture = UsageCaptureCallback(
            recorder=recorder,
            config=config,
            messages=messages,
            model=model,
            model_name=model_name,
            span_id=span.span_id,
            attempt_index=1,
            agent_role=agent_role,
            budget_gate=budget_gate,
        )
        try:
            invocation_result = await _ainvoke_model(
                model,
                messages,
                recorder,
                config,
                span_id=span.span_id,
                attempt_index=1,
                model_name=model_name,
                agent_role=agent_role,
                usage_capture=usage_capture,
            )
        except Exception as exc:
            await _record_failed_invocation(
                recorder=recorder,
                span=span,
                exc=exc,
                config=config,
                provider=provider,
                model_id=model_id or model_name,
                agent_role=agent_role,
                operation=span_name,
                attributes=span_attributes,
                attempt_index=1,
                usage_capture=usage_capture,
            )
            raise
        response = invocation_result.response
        span.attributes["llm.first_token_probe_status"] = (
            invocation_result.probe_status
        )
        if invocation_result.ttft_seconds is not None:
            span.attributes["llm.first_token_latency_seconds"] = (
                invocation_result.ttft_seconds
            )
        _observe_first_packet_metrics(
            recorder,
            invocation_result,
            provider=provider,
            model=model_id or model_name,
            agent_role=agent_role,
            operation=span_name,
        )
        await _record_successful_invocation(
            recorder=recorder,
            span=span,
            invocation_result=invocation_result,
            response=response,
            model=model,
            messages=messages,
            config=config,
            provider=provider,
            model_id=model_id or model_name,
            agent_role=agent_role,
            operation=span_name,
            attributes=span_attributes,
            attempt_index=1,
            duration_ms=int((monotonic_time() - span.started_monotonic) * 1000),
        )
        if (
            recorder.configuration.observability_enabled
            and getattr(recorder.configuration, "trace_payload_mode", "preview") != "none"
        ):
            span.output_preview = _message_preview(
                response,
                None if recorder.configuration.trace_payload_mode == "full" else recorder.configuration.trace_preview_chars,
                redact=recorder.configuration.trace_redaction_enabled,
            )
        return response


async def invoke_model_with_retry_observability(
    model: Any,
    messages: list[BaseMessage],
    config: RunnableConfig | None,
    *,
    span_name: str,
    agent_role: str | None = None,
    model_name: str | None = None,
    stage: str | None = None,
    attributes: dict[str, Any] | None = None,
    budget_gate: BudgetGate | None = None,
    max_attempts: int | None = None,
    base_delay: float | None = None,
    max_delay: float | None = None,
    sleeper: Callable[[float], Awaitable[Any]] | None = None,
) -> Any:
    """Invoke a model with retry + a local LLM span, recording each retry.

    Replaces LangChain ``.with_retry`` so that retries and 429s land in the
    observability store. The retry budget defaults to
    ``configurable.model_transport_max_attempts`` (total attempts). Structured
    output parsing and context recovery use independent counters. The backoff
    reuses the tool-retry delay settings. Non-retryable and exhausted errors
    are surfaced unchanged.

    The classification helper is imported lazily from ``tools.governance`` to
    avoid a circular import (governance imports observability at module level).
    """
    from open_deep_research.tools.governance import classify_llm_retryable_error

    provider, model_id = _provider_model(model_name)
    recorder = get_trace_recorder(config)
    configurable = recorder.configuration
    if max_attempts is None:
        max_attempts = configurable.model_transport_max_attempts
    if base_delay is None:
        base_delay = configurable.tool_retry_base_delay
    if max_delay is None:
        max_delay = configurable.tool_retry_max_delay
    sleeper = sleeper or asyncio.sleep

    span_attributes = _usage_attributes(attributes, stage)
    if recorder.langfuse is not None and recorder.configuration.langfuse_langchain_callback_enabled:
        span_attributes["langfuse_callback_managed"] = True
    task_id = str((config or {}).get("metadata", {}).get("task_id") or "")
    activity_call_id = uuid.uuid4().hex
    activity_started = monotonic_time()
    activity_is_quality = agent_role == "quality_evaluator"
    if task_id:
        await publish_task_activity(
            config or {},
            "quality.started" if activity_is_quality else "model.started",
            kind="quality" if activity_is_quality else "model",
            phase="quality_check" if activity_is_quality else "reasoning",
            status="running",
            title="质量复核" if activity_is_quality else "模型规划",
            summary=(
                "正在依据质量合同复核当前证据。"
                if activity_is_quality
                else "Subagent 正在分析证据并规划下一步动作。"
            ),
            iteration=None,
            duration_ms=None,
            payload={
                "evaluation_type": span_name,
                "provider": provider,
                "model": model_id or model_name,
                "attempt": 1,
            },
            dedupe_key=f"activity:model:{activity_call_id}:started",
            update_run_summary=True,
        )
    with recorder.start_span(
        name=span_name,
        kind="llm",
        agent_role=agent_role,
        attributes=span_attributes,
        input_payload=messages,
        provider=provider,
        model=model_id or model_name,
    ) as span:
        circuit_breaker: ModelCircuitBreaker | None = None
        circuit_permit: CircuitPermit | None = None
        if configurable.model_circuit_breaker_enabled and model_name:
            try:
                circuit_breaker = get_model_circuit_registry().get_or_create(
                    model_name,
                    model_circuit_policy_from_configuration(configurable),
                )
                if circuit_breaker is not None:
                    circuit_permit, transition = await circuit_breaker.before_call()
                    await observe_model_circuit_transition(
                        transition,
                        config,
                        agent_role=agent_role,
                    )
            except CircuitOpenError as exc:
                span.record_outcome(error_type="model_circuit_open")
                if recorder.prometheus is not None:
                    recorder._safe(  # noqa: SLF001
                        recorder.prometheus.observe_model_circuit_rejection,
                        provider or "unknown",
                        model_id or model_name or "unknown",
                        exc.reason,
                    )
                raise
            except Exception as exc:  # noqa: BLE001 - circuit governance fails open
                circuit_breaker = None
                circuit_permit = None
                logger.debug("Model circuit before_call failed open: %s", exc)
        attempt = 0  # attempts made so far (0 == first try in progress)
        physical_attempts_recorded = 0
        while True:
            usage_capture = UsageCaptureCallback(
                recorder=recorder,
                config=config,
                messages=messages,
                model=model,
                model_name=model_name,
                span_id=span.span_id,
                attempt_index=attempt + 1,
                agent_role=agent_role,
                budget_gate=budget_gate,
            )
            try:
                invocation = _ainvoke_model(
                    model,
                    messages,
                    recorder,
                    config,
                    span_id=span.span_id,
                    attempt_index=attempt + 1,
                    model_name=model_name,
                    agent_role=agent_role,
                    usage_capture=usage_capture,
                )
                invocation_result = await asyncio.wait_for(
                    invocation,
                    timeout=configurable.model_call_timeout_seconds,
                )
                response = invocation_result.response
            except Exception as exc:  # noqa: BLE001 -- classify then decide
                physical_attempts_recorded += await _record_failed_invocation(
                    recorder=recorder,
                    span=span,
                    exc=exc,
                    config=config,
                    provider=provider,
                    model_id=model_id or model_name,
                    agent_role=agent_role,
                    operation=span_name,
                    attributes=span_attributes,
                    attempt_index=physical_attempts_recorded + 1,
                    usage_capture=usage_capture,
                )
                error_type, retryable = classify_llm_retryable_error(exc)
                attempts_made = attempt + 1
                if not retryable or attempts_made >= max_attempts:
                    if circuit_breaker is not None and circuit_permit is not None:
                        try:
                            from open_deep_research.models.fallback import (
                                classify_model_error,
                            )

                            circuit_kind = classify_model_error(exc, model_name)
                            transition = None
                            if isinstance(exc, CircuitOpenError):
                                pass
                            elif circuit_kind.value in {
                                kind.value for kind in CircuitFailureKind
                            }:
                                transition = await circuit_breaker.record_failure(
                                    circuit_permit,
                                    failure_kind=CircuitFailureKind(
                                        circuit_kind.value
                                    ),
                                )
                            else:
                                transition = await circuit_breaker.record_inconclusive(
                                    circuit_permit
                                )
                            await observe_model_circuit_transition(
                                transition,
                                config,
                                agent_role=agent_role,
                            )
                        except Exception as circuit_exc:  # noqa: BLE001
                            logger.debug(
                                "Model circuit failure recording failed open: %s",
                                circuit_exc,
                            )
                    span.record_outcome(
                        error_type=error_type.value,
                        http_status=_safe_http_status(exc),
                    )
                    if task_id:
                        await publish_task_activity(
                            config or {},
                            "quality.failed" if activity_is_quality else "model.failed",
                            kind="quality" if activity_is_quality else "error",
                            phase="quality_check" if activity_is_quality else "reasoning",
                            status="error",
                            title="质量复核失败" if activity_is_quality else "模型调用失败",
                            summary="调用未能在重试预算内完成。",
                            iteration=None,
                            duration_ms=int((monotonic_time() - activity_started) * 1000),
                            payload={
                                "evaluation_type": span_name,
                                "provider": provider,
                                "model": model_id or model_name,
                                "error_code": error_type.value,
                                "retry_count": attempt,
                            },
                            dedupe_key=f"activity:model:{activity_call_id}:failed",
                            update_run_summary=True,
                        )
                    raise
                delay = min(max_delay, base_delay * (2 ** attempt)) + random.uniform(0, base_delay)
                span.record_retry(
                    attempt=attempts_made,
                    error_type=error_type.value,
                    http_status=_safe_http_status(exc),
                    retryable=True,
                    delay_s=delay,
                    message=_exc_message(exc),
                )
                if task_id:
                    await publish_task_activity(
                        config or {},
                        "model.retrying",
                        kind="quality" if activity_is_quality else "model",
                        phase="quality_check" if activity_is_quality else "reasoning",
                        status="warning",
                        title="质量复核重试" if activity_is_quality else "模型调用重试",
                        summary="遇到可恢复错误，正在按退避策略重试。",
                        iteration=None,
                        duration_ms=None,
                        payload={
                            "provider": provider,
                            "model": model_id or model_name,
                            "attempt": attempts_made,
                            "error_code": error_type.value,
                            "delay_ms": int(delay * 1000),
                        },
                        dedupe_key=f"activity:model:{activity_call_id}:retry:{attempts_made}",
                        update_run_summary=True,
                    )
                await sleeper(delay)
                attempt += 1
                continue
            if circuit_breaker is not None and circuit_permit is not None:
                try:
                    transition = await circuit_breaker.record_success(
                        circuit_permit,
                        ttft_seconds=invocation_result.ttft_seconds,
                    )
                    await observe_model_circuit_transition(
                        transition,
                        config,
                        agent_role=agent_role,
                    )
                except Exception as exc:  # noqa: BLE001 - circuit governance fails open
                    logger.debug("Model circuit success recording failed open: %s", exc)
            span.attributes["llm.first_token_probe_status"] = (
                invocation_result.probe_status
            )
            if invocation_result.ttft_seconds is not None:
                span.attributes["llm.first_token_latency_seconds"] = (
                    invocation_result.ttft_seconds
                )
            _observe_first_packet_metrics(
                recorder,
                invocation_result,
                provider=provider,
                model=model_id or model_name,
                agent_role=agent_role,
                operation=span_name,
            )
            usage = await _record_successful_invocation(
                recorder=recorder,
                span=span,
                invocation_result=invocation_result,
                response=response,
                model=model,
                messages=messages,
                config=config,
                provider=provider,
                model_id=model_id or model_name,
                agent_role=agent_role,
                operation=span_name,
                attributes=span_attributes,
                attempt_index=physical_attempts_recorded + 1,
                duration_ms=int((monotonic_time() - activity_started) * 1000),
            )
            if (
                recorder.configuration.observability_enabled
                and getattr(recorder.configuration, "trace_payload_mode", "preview") != "none"
            ):
                span.output_preview = _message_preview(
                    response,
                    None if recorder.configuration.trace_payload_mode == "full" else recorder.configuration.trace_preview_chars,
                    redact=recorder.configuration.trace_redaction_enabled,
                )
            if task_id:
                await publish_task_activity(
                    config or {},
                    "quality.completed" if activity_is_quality else "model.completed",
                    kind="quality" if activity_is_quality else "model",
                    phase="quality_check" if activity_is_quality else "reasoning",
                    status="success",
                    title="质量复核响应完成" if activity_is_quality else "模型规划完成",
                    summary=(
                        "质量评估模型已返回结构化结果。"
                        if activity_is_quality
                        else "模型已完成本轮分析。"
                    ),
                    iteration=None,
                    duration_ms=int((monotonic_time() - activity_started) * 1000),
                    payload={
                        "evaluation_type": span_name,
                        "provider": provider,
                        "model": model_id or model_name,
                        "input_tokens": usage.input_tokens,
                        "output_tokens": usage.output_tokens,
                        "reasoning_tokens": usage.reasoning_tokens,
                        "retry_count": attempt,
                    },
                    dedupe_key=f"activity:model:{activity_call_id}:completed",
                    update_run_summary=True,
                )
            return response


async def observe_tool_call(
    tool_call: dict[str, Any],
    role: str,
    config: RunnableConfig,
    invoke: Callable[[], Awaitable[Any]],
) -> Any:
    """Run a tool call inside a tool span."""
    recorder = get_trace_recorder(config)
    name = tool_call.get("name", "unknown_tool")
    task_id = str(config.get("metadata", {}).get("task_id") or "")
    activity_call_id = str(tool_call.get("id") or uuid.uuid4().hex)
    activity_started = monotonic_time()
    category = (
        "search" if "search" in str(name).lower() else
        "fetch" if any(token in str(name).lower() for token in ("fetch", "browser")) else
        "completion" if str(name) == "ResearchComplete" else
        "reasoning" if str(name) == "think_tool" else
        "tool"
    )
    if task_id:
        args = tool_call.get("args") or {}
        args_summary = ""
        for key in ("query", "search_query", "url"):
            value = args.get(key) if isinstance(args, dict) else None
            if isinstance(value, str) and value.strip():
                args_summary = " ".join(value.split())[:240]
                break
        await publish_task_activity(
            config,
            "tool.started",
            kind="tool",
            phase="tool_execution",
            status="running",
            title=f"执行工具 · {name}",
            summary=(args_summary or f"正在调用 {name}。"),
            iteration=None,
            duration_ms=None,
            payload={
                "tool_call_id": activity_call_id,
                "tool_name": name,
                "tool_category": category,
                "args_summary": args_summary,
                "args_keys": sorted(args.keys()) if isinstance(args, dict) else [],
            },
            dedupe_key=f"activity:tool:{activity_call_id}:started",
            update_run_summary=True,
        )
    with recorder.start_span(
        name=f"tool.{name}",
        kind="tool",
        agent_role=role,
        attributes={
            "tool_call_id": tool_call.get("id"),
            "tool_name": name,
            "args_keys": sorted((tool_call.get("args") or {}).keys()),
        },
        input_payload=tool_call,
    ) as span:
        try:
            result = await invoke()
        except Exception as exc:
            if task_id:
                await publish_task_activity(
                    config,
                    "tool.failed",
                    kind="error",
                    phase="tool_execution",
                    status="error",
                    title=f"工具失败 · {name}",
                    summary="工具调用未能完成。",
                    iteration=None,
                    duration_ms=int((monotonic_time() - activity_started) * 1000),
                    payload={
                        "tool_call_id": activity_call_id,
                        "tool_name": name,
                        "tool_category": category,
                        "error_code": type(exc).__name__,
                        "retry_count": getattr(span, "retry_count", 0),
                    },
                    dedupe_key=f"activity:tool:{activity_call_id}:failed",
                    update_run_summary=True,
                )
            raise
        result_content = getattr(getattr(result, "message", result), "content", result)
        result_text = str(result_content)
        result_urls = set(re.findall(r"https?://[^\s\]\)>'\"}]+", result_text))
        span.attributes["result_chars"] = len(result_text)
        span.attributes["source_count"] = len(result_urls)
        task_id = str(config.get("metadata", {}).get("task_id", ""))
        if role == "researcher" and task_id:
            try:
                from open_deep_research.tasks.registry import get_task_registry

                record = get_task_registry().get(task_id)
                expected_run = str(config.get("metadata", {}).get("run_id", "default"))
                if record is not None and record.run_id == expected_run:
                    if "search" in str(name).lower():
                        record.query_count += 1
                    record.source_urls.update(result_urls)
                    record.source_count = max(record.source_count, len(record.source_urls))
                    record.citation_count += len(result_urls)
                    record.retry_count += getattr(span, "retry_count", 0)
            except Exception as exc:  # noqa: BLE001 - metrics must stay fail-open
                logger.debug("Unable to update task research metrics: %s", exc)
        if getattr(recorder.configuration, "trace_payload_mode", "preview") != "none":
            if (
                role == "researcher"
                and getattr(
                    recorder.configuration,
                    "prompt_injection_protection_enabled",
                    True,
                )
            ):
                span.output_preview = json.dumps(
                    {
                        "content_hash": hashlib.sha256(
                            result_text.encode("utf-8", errors="replace")
                        ).hexdigest(),
                        "result_chars": len(result_text),
                        "source_count": len(result_urls),
                    },
                    sort_keys=True,
                )
            else:
                span.output_preview = _message_preview(
                    result,
                    None
                    if recorder.configuration.trace_payload_mode == "full"
                    else recorder.configuration.trace_preview_chars,
                    redact=recorder.configuration.trace_redaction_enabled,
                )
        if task_id:
            governed_error = getattr(result, "error", None)
            await publish_task_activity(
                config,
                "tool.failed" if governed_error is not None else "tool.completed",
                kind="error" if governed_error is not None else "tool",
                phase="tool_execution",
                status="error" if governed_error is not None else "success",
                title=(f"工具失败 · {name}" if governed_error is not None else f"工具完成 · {name}"),
                summary=(
                    "工具返回了受治理的错误结果。"
                    if governed_error is not None
                    else f"工具已完成，识别到 {len(result_urls)} 个来源链接。"
                ),
                iteration=None,
                duration_ms=int((monotonic_time() - activity_started) * 1000),
                payload={
                    "tool_call_id": activity_call_id,
                    "tool_name": name,
                    "tool_category": category,
                    "source_count": len(result_urls),
                    "result_chars": len(result_text),
                    "retry_count": getattr(span, "retry_count", 0),
                    "error_code": getattr(getattr(governed_error, "error_type", None), "value", None),
                    "urls": sorted(result_urls),
                },
                dedupe_key=(
                    f"activity:tool:{activity_call_id}:failed"
                    if governed_error is not None
                    else f"activity:tool:{activity_call_id}:completed"
                ),
                update_run_summary=True,
            )
        return result


async def await_with_observability_timeout(awaitable: Awaitable[Any], timeout: float) -> Any:
    """Tiny wrapper to keep timeout call sites readable."""
    return await asyncio.wait_for(awaitable, timeout=timeout)
