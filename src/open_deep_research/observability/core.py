"""Fail-open tracing and usage collection for Deep Research runs."""

# ruff: noqa: D102,D105,D107,UP037

from __future__ import annotations

import asyncio
import contextvars
import json
import logging
import sqlite3
import threading
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langchain_core.messages import BaseMessage, get_buffer_string
from langchain_core.runnables import RunnableConfig

from open_deep_research.configuration import Configuration

logger = logging.getLogger(__name__)

_current_run_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "open_deep_research_run_id",
    default=None,
)
_current_span_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "open_deep_research_span_id",
    default=None,
)
_stores: dict[str, "SQLiteTraceStore"] = {}
_stores_lock = threading.Lock()


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


def _provider_model(model_name: str | None) -> tuple[str | None, str | None]:
    if not model_name:
        return None, None
    if ":" not in model_name:
        return None, model_name
    provider, model = model_name.split(":", 1)
    return provider, model


def _message_preview(messages: Any, limit: int | None) -> str | None:
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
    if limit is None or len(text) <= limit:
        return text
    return text[:limit] + "\n[truncated]"


def current_span_ids() -> tuple[str | None, str | None]:
    """Return the current run/span ids carried by contextvars."""
    return _current_run_id.get(), _current_span_id.get()


@dataclass
class TokenUsage:
    """Normalized token usage across LangChain/provider metadata variants."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
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
            if not total_tokens:
                total_tokens = input_tokens + output_tokens
            return cls(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                raw_usage=dict(usage),
            )
        return cls()

    def as_dict(self) -> dict[str, int]:
        """Return the token counters as a plain dict."""
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
        }


class SQLiteTraceStore:
    """Small SQLite-backed store for runs, spans, and usage events."""

    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._ensure_schema()

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
                    input_tokens INTEGER NOT NULL DEFAULT 0,
                    output_tokens INTEGER NOT NULL DEFAULT 0,
                    total_tokens INTEGER NOT NULL DEFAULT 0,
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
                    raw_usage_json TEXT NOT NULL DEFAULT '{}',
                    created_at REAL NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_spans_run_started
                    ON spans(run_id, started_at);
                CREATE INDEX IF NOT EXISTS idx_spans_parent
                    ON spans(parent_span_id);
                CREATE INDEX IF NOT EXISTS idx_usage_run
                    ON usage_events(run_id);
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
                    metadata_json=excluded.metadata_json
                """,
                (run_id, user_id, _now(), _json(metadata)),
            )

    def finish_run(self, run_id: str, status: str, error: str | None = None) -> dict[str, int]:
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
                    output_preview = COALESCE(?, output_preview), error = ?
                WHERE span_id = ?
                """,
                (
                    status,
                    ended_at,
                    int((ended_at - started_at) * 1000),
                    usage.input_tokens,
                    usage.output_tokens,
                    usage.total_tokens,
                    output_preview,
                    error,
                    span_id,
                ),
            )

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
                    raw_usage_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    span_id,
                    provider,
                    model,
                    usage.input_tokens,
                    usage.output_tokens,
                    usage.total_tokens,
                    _json(usage.raw_usage),
                    _now(),
                ),
            )

    def get_usage(self, run_id: str) -> dict[str, int]:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                """
                SELECT
                    COALESCE(SUM(input_tokens), 0) AS input_tokens,
                    COALESCE(SUM(output_tokens), 0) AS output_tokens,
                    COALESCE(SUM(total_tokens), 0) AS total_tokens
                FROM usage_events
                WHERE run_id = ?
                """,
                (run_id,),
            ).fetchone()
        return {
            "input_tokens": int(row["input_tokens"] if row else 0),
            "output_tokens": int(row["output_tokens"] if row else 0),
            "total_tokens": int(row["total_tokens"] if row else 0),
        }

    def list_runs(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM runs ORDER BY started_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
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
        self.usage = TokenUsage()
        self.output_preview: str | None = None
        self._run_token: contextvars.Token[str | None] | None = None
        self._span_token: contextvars.Token[str | None] | None = None

    def __enter__(self) -> "SpanContext":
        self.recorder._safe(
            self.recorder.store.start_span,  # type: ignore[union-attr]
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
        return self

    def __exit__(self, exc_type: Any, exc: BaseException | None, _tb: Any) -> None:
        status = "error" if exc else "success"
        self.recorder._safe(
            self.recorder.store.finish_span,  # type: ignore[union-attr]
            span_id=self.span_id,
            status=status,
            usage=self.usage,
            output_preview=self.output_preview,
            error=str(exc) if exc else None,
        )
        if self._span_token is not None:
            _current_span_id.reset(self._span_token)
        if self._run_token is not None:
            _current_run_id.reset(self._run_token)

    def add_usage(self, usage: TokenUsage, provider: str | None, model: str | None) -> None:
        """Attach usage to this span and persist a usage event."""
        self.usage = usage
        self.recorder._safe(
            self.recorder.store.add_usage,  # type: ignore[union-attr]
            self.run_id,
            self.span_id,
            provider,
            model,
            usage,
        )


class NoopSpanContext:
    """Span-like context manager used when observability is disabled."""

    span_id: str | None = None
    run_id: str | None = None

    def __enter__(self) -> "NoopSpanContext":
        return self

    def __exit__(self, *_args: Any) -> None:
        return None

    def add_usage(self, *_args: Any, **_kwargs: Any) -> None:
        return None


class TraceRecorder:
    """Fail-open facade for tracing one runtime config."""

    def __init__(self, config: RunnableConfig | None):
        self.config = config or {"configurable": {}, "metadata": {}}
        self.configuration = Configuration.from_runnable_config(self.config)
        self.enabled = self.configuration.observability_enabled
        self.store = _get_store(self.configuration.trace_store_path) if self.enabled else None

    def _safe(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Observability write failed: %s", exc)
            return None

    def start_run(
        self,
        run_id: str,
        *,
        name: str = "run",
        user_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> SpanContext | NoopSpanContext:
        if not self.enabled:
            return NoopSpanContext()
        self._safe(self.store.start_run, run_id, user_id, metadata or {})  # type: ignore[union-attr]
        return self.start_span(
            name=name,
            kind="run",
            run_id=run_id,
            parent_span_id=None,
            attributes=metadata or {},
        )

    def finish_run(self, run_id: str, status: str, error: str | None = None) -> dict[str, int]:
        if not self.enabled:
            return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        usage = self._safe(self.store.finish_run, run_id, status, error)  # type: ignore[union-attr]
        return usage or {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

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
        preview = None
        if self.configuration.trace_payload_mode != "none":
            limit = None if self.configuration.trace_payload_mode == "full" else self.configuration.trace_preview_chars
            preview = _message_preview(input_payload, limit)
        return SpanContext(
            self,
            span_id=str(uuid.uuid4()),
            run_id=resolved_run_id,
            parent_span_id=resolved_parent,
            name=name,
            kind=kind,
            agent_role=agent_role,
            attributes=attributes or {},
            input_preview=preview,
            provider=provider,
            model=model,
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
    with recorder.start_span(
        name=span_name,
        kind="llm",
        agent_role=agent_role,
        attributes=attributes or {},
        input_payload=messages,
        provider=provider,
        model=model_id or model_name,
    ) as span:
        response = await model.ainvoke(messages)
        usage = TokenUsage.from_response(response)
        if hasattr(span, "add_usage"):
            span.add_usage(usage, provider, model_id or model_name)
        if getattr(recorder.configuration, "trace_payload_mode", "preview") != "none":
            span.output_preview = _message_preview(
                response,
                None if recorder.configuration.trace_payload_mode == "full" else recorder.configuration.trace_preview_chars,
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
        if getattr(recorder.configuration, "trace_payload_mode", "preview") != "none":
            span.output_preview = _message_preview(
                result,
                None if recorder.configuration.trace_payload_mode == "full" else recorder.configuration.trace_preview_chars,
            )
        return result


async def await_with_observability_timeout(awaitable: Awaitable[Any], timeout: float) -> Any:
    """Tiny wrapper to keep timeout call sites readable."""
    return await asyncio.wait_for(awaitable, timeout=timeout)
