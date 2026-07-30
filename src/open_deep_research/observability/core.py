"""Fail-open tracing and usage collection for Deep Research runs."""

# ruff: noqa: D102,D105,D107,UP037

from __future__ import annotations

import asyncio
import contextvars
import hashlib
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

from langchain_core.messages import BaseMessage, get_buffer_string
from langchain_core.runnables import RunnableConfig

from open_deep_research.configuration import Configuration
from open_deep_research.observability.telemetry import (
    create_langfuse_sink,
    get_prometheus_metrics,
    monotonic_time,
)

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

_SENSITIVE_TEXT_PATTERNS = (
    re.compile(r"(?i)(authorization\s*[:=]\s*['\"]?bearer\s+)[^\s,'\"}]+"),
    re.compile(
        r"(?i)((?:api[_-]?key|access[_-]?token|refresh[_-]?token|secret|password)"
        r"['\"]?\s*[:=]\s*['\"]?)[^\s,'\"}]+"
    ),
    re.compile(r"\b(?:sk|pk)-[A-Za-z0-9_-]{12,}\b"),
)
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


def _redact_text(text: str) -> str:
    """Best-effort redaction for credentials that reach trace payloads."""
    redacted = text
    for pattern in _SENSITIVE_TEXT_PATTERNS:
        if pattern.groups:
            redacted = pattern.sub(r"\1[REDACTED]", redacted)
        else:
            redacted = pattern.sub("[REDACTED]", redacted)
    return redacted


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


class SQLiteTraceStore:
    """Small SQLite-backed store for runs, spans, and usage events."""

    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._ensure_schema()
        self._migrate()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

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
                    raw_usage_json TEXT NOT NULL DEFAULT '{}',
                    created_at REAL NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_spans_run_started
                    ON spans(run_id, started_at);
                CREATE INDEX IF NOT EXISTS idx_spans_parent
                    ON spans(parent_span_id);
                CREATE INDEX IF NOT EXISTS idx_usage_run
                    ON usage_events(run_id);

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
                ),
            }
            for table, columns in additions.items():
                existing = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
                for column, ddl in columns:
                    if column not in existing:
                        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")

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
    ) -> None:
        if usage.total_tokens <= 0 and usage.input_tokens <= 0 and usage.output_tokens <= 0:
            return
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO usage_events (
                    run_id, span_id, provider, model,
                    input_tokens, output_tokens, total_tokens,
                    cached_input_tokens, cache_creation_input_tokens,
                    reasoning_tokens, estimated_cost_usd,
                    raw_usage_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    _json(usage.raw_usage),
                    _now(),
                ),
            )

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

    def add_usage(self, usage: TokenUsage, provider: str | None, model: str | None) -> None:
        """Attach usage to this span and persist a usage event."""
        self.recorder.estimate_usage_cost(usage, provider, model)
        self.usage = usage
        if self.recorder.store is not None:
            self.recorder._safe(
                self.recorder.store.add_usage,
                self.run_id,
                self.span_id,
                provider,
                model,
                usage,
            )

    def set_output(self, payload: Any) -> None:
        """Attach a redacted output preview using the recorder payload policy."""
        if self.recorder.configuration.trace_payload_mode == "none":
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
            if self.configuration.observability_enabled and self.configuration.sqlite_observability_enabled
            else None
        )
        self.langfuse = create_langfuse_sink(self.configuration, self.config)
        self.prometheus = get_prometheus_metrics(self.configuration)
        self.enabled = self.configuration.observability_enabled and any(
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
            return usage.estimated_cost_usd
        prices = self.configuration.model_costs_per_million
        candidates = [model, f"{provider}:{model}" if provider and model else None]
        rates = next((prices[key] for key in candidates if key and key in prices), None)
        if not rates:
            return 0.0
        cached = min(usage.cached_input_tokens, usage.input_tokens)
        uncached = max(0, usage.input_tokens - cached)
        reasoning = min(usage.reasoning_tokens, usage.output_tokens)
        normal_output = max(0, usage.output_tokens - reasoning)
        cost = (
            uncached * float(rates.get("input", 0))
            + cached * float(rates.get("cached_input", rates.get("input", 0)))
            + usage.cache_creation_input_tokens
            * float(rates.get("cache_creation_input", rates.get("input", 0)))
            + normal_output * float(rates.get("output", 0))
            + reasoning * float(rates.get("reasoning", rates.get("output", 0)))
        ) / 1_000_000
        usage.estimated_cost_usd = cost
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
        if self.store is not None:
            self._safe(self.store.start_run, run_id, user_id, metadata or {})
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
        if self.configuration.trace_payload_mode != "none":
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


def _langchain_invoke_config(recorder: TraceRecorder, config: RunnableConfig | None) -> RunnableConfig | None:
    """Attach the optional Langfuse LangChain callback without mutating caller config."""
    if (
        recorder.langfuse is None
        or not recorder.configuration.langfuse_langchain_callback_enabled
    ):
        return None
    handler = recorder._safe(recorder.langfuse.callback_handler)
    if handler is None:
        return None
    invoke_config: dict[str, Any] = dict(config or {})
    callbacks = list(invoke_config.get("callbacks") or [])
    callbacks.append(handler)
    invoke_config["callbacks"] = callbacks
    return cast(RunnableConfig, invoke_config)


async def _ainvoke_model(
    model: Any,
    messages: list[BaseMessage],
    recorder: TraceRecorder,
    config: RunnableConfig | None,
) -> Any:
    """Invoke a LangChain model with optional supplemental callback tracing."""
    invoke_config = _langchain_invoke_config(recorder, config)
    if invoke_config is None:
        return await model.ainvoke(messages)
    return await model.ainvoke(messages, config=invoke_config)


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
    attributes: dict[str, Any] | None = None,
) -> Any:
    """Invoke a model while recording a local LLM span and token usage."""
    provider, model_id = _provider_model(model_name)
    recorder = get_trace_recorder(config)
    span_attributes = dict(attributes or {})
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
        response = await _ainvoke_model(model, messages, recorder, config)
        usage = TokenUsage.from_response(response)
        if hasattr(span, "add_usage"):
            span.add_usage(usage, provider, model_id or model_name)
        if getattr(recorder.configuration, "trace_payload_mode", "preview") != "none":
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
    attributes: dict[str, Any] | None = None,
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

    span_attributes = dict(attributes or {})
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
        attempt = 0  # attempts made so far (0 == first try in progress)
        while True:
            try:
                invocation = _ainvoke_model(model, messages, recorder, config)
                response = await asyncio.wait_for(
                    invocation,
                    timeout=configurable.model_call_timeout_seconds,
                )
            except Exception as exc:  # noqa: BLE001 -- classify then decide
                error_type, retryable = classify_llm_retryable_error(exc)
                attempts_made = attempt + 1
                if not retryable or attempts_made >= max_attempts:
                    span.record_outcome(
                        error_type=error_type.value,
                        http_status=_safe_http_status(exc),
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
                await sleeper(delay)
                attempt += 1
                continue
            usage = TokenUsage.from_response(response)
            if hasattr(span, "add_usage"):
                span.add_usage(usage, provider, model_id or model_name)
            if getattr(recorder.configuration, "trace_payload_mode", "preview") != "none":
                span.output_preview = _message_preview(
                    response,
                    None if recorder.configuration.trace_payload_mode == "full" else recorder.configuration.trace_preview_chars,
                    redact=recorder.configuration.trace_redaction_enabled,
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
        result = await invoke()
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
        return result


async def await_with_observability_timeout(awaitable: Awaitable[Any], timeout: float) -> Any:
    """Tiny wrapper to keep timeout call sites readable."""
    return await asyncio.wait_for(awaitable, timeout=timeout)
