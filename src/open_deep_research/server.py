"""FastAPI service entrypoint for the LangGraph-free runtime."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import html
import json
import logging
import os
import re
import shutil
import sqlite3
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

from open_deep_research.agents.query_engine import QueryEngine
from open_deep_research.api_governance import ConnectionLimiter, FixedWindowRateLimiter
from open_deep_research.budgets import RunBudgetLedger
from open_deep_research.configuration import Configuration
from open_deep_research.events.public import (
    PUBLIC_EVENT_SCHEMA_VERSION,
    PublicEvent,
    RunEventStore,
    event_publisher_from_config,
    is_terminal_event,
)
from open_deep_research.events.task_activity import (
    PUBLIC_TASK_ACTIVITY_SCHEMA_VERSION,
    TASK_TERMINAL_TYPES,
    PublicTaskActivityEvent,
    TaskActivityStore,
    activity_summary,
    derive_trace_activity,
)
from open_deep_research.logging_config import (
    bind_request_id,
    configure_logging,
    current_request_id,
)
from open_deep_research.models.circuit import get_model_circuit_registry
from open_deep_research.observability import SQLiteTraceStore, get_trace_recorder
from open_deep_research.observability.telemetry import get_prometheus_metrics
from open_deep_research.run_context import (
    JournalCorruptedError,
    RunContextError,
    RunContextStore,
)
from open_deep_research.run_control import RunControlStore
from open_deep_research.sandbox.approvals import SecurityApprovalStore
from open_deep_research.sandbox.internal_api import (
    InternalRunContext,
    build_internal_sandbox_router,
)
from open_deep_research.security.inputs import (
    validate_http_configurable,
    validate_http_metadata,
)
from open_deep_research.tasks.lease import LeaderLeaseManager, LeaseConflictError
from open_deep_research.tasks.registry import TaskStatus, get_task_registry
from security.rbac import (
    Principal,
    apply_principal_to_config,
    check_database_connection,
    mount_rbac,
    register_ownership_checker,
    require_active_user,
    require_permissions,
    require_run_owner_or_any,
    shutdown_rbac,
    startup_checks,
)
from security.rbac.app_extension import assert_schema_current
from security.rbac.database import session_scope
from security.rbac.dependencies import reauthorize_session
from security.rbac.permissions import (
    RESEARCH_DIAGNOSTICS_PREVIEW,
    RESEARCH_OBSERVABILITY_READ_OWN,
    RESEARCH_RUN_CONTROL_OWN,
    RESEARCH_RUN_CREATE,
    RESEARCH_RUN_INTERACT_OWN,
    RESEARCH_RUN_READ_OWN,
    RESEARCH_SECURITY_APPROVAL_READ_ANY,
    RESEARCH_SECURITY_APPROVAL_READ_OWN,
    RESEARCH_SECURITY_APPROVAL_RESOLVE_ANY,
    RESEARCH_SECURITY_APPROVAL_RESOLVE_OWN,
    RESEARCH_TASK_ACTIVITY_READ_OWN,
)
from security.rbac.settings import get_settings as get_iam_settings
from security.rbac.settings import local_dev_bypass_enabled

load_dotenv()
configure_logging()


class RunRequest(BaseModel):
    """HTTP request body for a research run."""

    messages: list[dict[str, Any]]
    configurable: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    title: str | None = Field(default=None, max_length=160)


class ResumeRunRequest(BaseModel):
    """Runtime overrides and credentials for explicitly resuming a run."""

    configurable: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class HumanActionRequest(BaseModel):
    """Approval/revision/cancellation response for a pending HITL action."""

    action: Literal["approve", "revise", "answer", "cancel"]
    message: str | None = None


class SecurityApprovalDecisionRequest(BaseModel):
    """Resolve one sandbox security approval without changing permanent policy."""

    decision: Literal["allow_once", "allow_run", "deny"]
    reason: str = Field(default="", max_length=1000)


class HumanFeedbackRequest(BaseModel):
    """Mid-run human direction or evidence follow-up."""

    type: Literal["direction", "evidence_question"]
    message: str
    task_id: str | None = None
    source_url: str | None = None
    claim_text: str | None = None
    command_id: str | None = None


@dataclass
class RunRecord:
    """In-memory HTTP run state."""

    run_id: str
    engine: QueryEngine
    status: str = "pending"
    events: deque[dict[str, Any]] = field(default_factory=lambda: deque(maxlen=200))
    result: dict[str, Any] | None = None
    task: asyncio.Task | None = None
    finished_at: float | None = None


@contextlib.asynccontextmanager
async def _lifespan(_app: FastAPI):
    """Run startup recovery and gracefully interrupt live work on shutdown."""
    global _retention_sweep_task
    _shutting_down.clear()
    _sse_shutdown.clear()
    await startup_checks()
    await assert_schema_current("0002_sandbox_permissions")
    configurable = Configuration.from_runnable_config(None)
    if configurable.sandbox_enabled:
        from open_deep_research.sandbox.controller_client import (
            SandboxControllerClient,
        )
        from open_deep_research.sandbox.doctor import diagnose
        from open_deep_research.sandbox.schema import load_policy_bundle

        startup_grace = min(
            120.0,
            max(0.0, float(os.getenv("SANDBOX_STARTUP_GRACE_SECONDS", "30"))),
        )
        startup_deadline = time.monotonic() + startup_grace
        while True:
            sandbox_report = await asyncio.to_thread(diagnose)
            if sandbox_report.get("ready"):
                break
            if time.monotonic() >= startup_deadline:
                raise RuntimeError(
                    "sandbox_unavailable:"
                    + ",".join(
                        str(item)
                        for item in sandbox_report.get("failures", [])
                    )
                )
            await asyncio.sleep(1)
        # A fresh API process owns no live task leases yet. Stop every Worker
        # from this deployment before recovery can schedule replacements; the
        # Controller label scope prevents touching unrelated Docker resources.
        bundle = load_policy_bundle(configurable.sandbox_policy_path)
        await SandboxControllerClient(configurable, bundle).reconcile_tasks([])
    if configurable.run_recovery_sweep_on_startup:
        try:
            await _run_recovery_sweep(configurable)
        except Exception as exc:  # noqa: BLE001 - recovery is fail-open
            logger.warning("run recovery sweep failed: %s", exc)
    if configurable.retention_sweep_interval_seconds > 0:
        _retention_sweep_task = asyncio.create_task(
            _retention_sweep_loop(configurable)
        )
    try:
        yield
    finally:
        _shutting_down.set()
        try:
            await _drain_inflight_runs(
                configurable.shutdown_drain_timeout_seconds
            )
        finally:
            _sse_shutdown.set()
            if _retention_sweep_task is not None:
                _retention_sweep_task.cancel()
                await asyncio.gather(_retention_sweep_task, return_exceptions=True)
                _retention_sweep_task = None
            for task in list(_run_eviction_tasks.values()):
                task.cancel()
            if _run_eviction_tasks:
                await asyncio.gather(
                    *_run_eviction_tasks.values(),
                    return_exceptions=True,
                )
            _run_eviction_tasks.clear()
            await shutdown_rbac()


app = FastAPI(title="Open Deep Research", version="0.1.0", lifespan=_lifespan)


@app.middleware("http")
async def request_id_middleware(request: Request, call_next: Any) -> Response:
    """Bind and echo a gateway request ID for logs, runs, and traces."""
    supplied = str(request.headers.get("X-Request-ID") or "").strip()
    request_id = (
        supplied
        if 0 < len(supplied) <= 128
        and re.fullmatch(r"[A-Za-z0-9._:-]+", supplied)
        else str(uuid.uuid4())
    )
    bind_request_id(request_id)
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


@app.middleware("http")
async def request_body_limit_middleware(request: Request, call_next: Any) -> Response:
    """Reject declared and streaming request bodies above the configured cap."""
    if request.method not in {"POST", "PUT", "PATCH"}:
        return await call_next(request)
    limit = Configuration.from_runnable_config(None).max_request_body_bytes
    content_length = request.headers.get("Content-Length")
    if content_length is not None:
        try:
            if int(content_length) > limit:
                return JSONResponse(
                    {"detail": "request_body_too_large"},
                    status_code=413,
                )
        except ValueError:
            return JSONResponse({"detail": "invalid_content_length"}, status_code=400)
    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > limit:
            return JSONResponse(
                {"detail": "request_body_too_large"},
                status_code=413,
            )
    request._body = bytes(body)  # noqa: SLF001 - Starlette replays the bounded body
    return await call_next(request)
_allowed_origins = [
    item.strip()
    for item in os.environ.get(
        "FRONTEND_ALLOWED_ORIGINS",
        "http://localhost:3000,http://127.0.0.1:3000",
    ).split(",")
    if item.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "Idempotency-Key", "Last-Event-ID"],
)

# Self-hosted identity & RBAC subsystem. Supabase is intentionally not mounted.
mount_rbac(app)


async def _rbac_run_owner_checker(_db, principal, run_id: str) -> bool:
    """Ownership bridge used by ``require_run_owner`` (prepared for cutover)."""
    record = _runs.get(run_id)
    if record is not None:
        metadata = getattr(record.engine, "config", {}).get("metadata", {})
        owner = metadata.get("owner") or metadata.get("user_id")
        return bool(owner) and str(owner) == principal.user_id
    configurable = Configuration.from_runnable_config(None)
    try:
        manifest = RunContextStore(run_id, runs_dir=configurable.runs_dir).load_manifest()
    except (ValueError, JournalCorruptedError, OSError):
        return False
    return bool(manifest.owner_id) and manifest.owner_id == principal.user_id


async def _rbac_task_owner_checker(_db, principal, key: tuple[str, str]) -> bool:
    """Ownership bridge used by ``require_task_owner`` (prepared for cutover)."""
    run_id, task_id = key
    if not await _rbac_run_owner_checker(_db, principal, run_id):
        return False
    configurable = Configuration.from_runnable_config(None)
    projection = RunEventStore(run_id, runs_dir=configurable.runs_dir).project()
    return task_id in projection.task_items


register_ownership_checker("run", _rbac_run_owner_checker)
register_ownership_checker("task", _rbac_task_owner_checker)

logger = logging.getLogger(__name__)
_runs: dict[str, RunRecord] = {}
_run_eviction_tasks: dict[str, asyncio.Task[None]] = {}
_retention_sweep_task: asyncio.Task[None] | None = None
_api_rate_limiter = FixedWindowRateLimiter()
_sse_connection_limiter = ConnectionLimiter()
_shutting_down = asyncio.Event()
_sse_shutdown = asyncio.Event()
_TERMINAL_RUN_STATUSES = frozenset(
    {"success", "completed", "failed", "interrupted", "cancelled"}
)
_metrics_path = Configuration.from_runnable_config(None).prometheus_metrics_path


def _resolve_internal_sandbox_run(run_id: str) -> InternalRunContext | None:
    """Resolve the live, fenced API authority for trusted sandbox services."""
    record = _runs.get(run_id)
    if record is None or record.engine.run_fence_token is None:
        return None
    config = record.engine.config
    return InternalRunContext(
        config=config,
        configurable=Configuration.from_runnable_config(config),
        fence_token=int(record.engine.run_fence_token),
        started_at=float(record.engine.started_at),
    )


app.include_router(build_internal_sandbox_router(_resolve_internal_sandbox_run))


def _new_run_record(
    *,
    run_id: str,
    engine: QueryEngine,
    status: str,
    config: dict[str, Any] | None,
) -> RunRecord:
    """Build a run record with the configured bounded event buffer."""
    configurable = Configuration.from_runnable_config(config)
    return RunRecord(
        run_id=run_id,
        engine=engine,
        status=status,
        events=deque(maxlen=configurable.inflight_event_buffer_size),
    )


def _evict_run_record(
    run_id: str,
    *,
    expected_finished_at: float,
    reason: str,
) -> bool:
    """Evict the matching terminal record without touching a resumed replacement."""
    record = _runs.get(run_id)
    if (
        record is None
        or record.status not in _TERMINAL_RUN_STATUSES
        or record.finished_at != expected_finished_at
    ):
        return False
    _runs.pop(run_id, None)
    eviction_task = _run_eviction_tasks.pop(run_id, None)
    try:
        current_task = asyncio.current_task()
    except RuntimeError:
        current_task = None
    if eviction_task is not None and eviction_task is not current_task:
        eviction_task.cancel()
    logger.info(
        "run.evicted actor=system action=run.evicted run_id=%s reason=%s",
        run_id,
        reason,
        extra={
            "actor": "system",
            "action": "run.evicted",
            "run_id": run_id,
            "reason": reason,
        },
    )
    return True


def _enforce_run_memory_limit(maximum: int) -> None:
    """Evict least-recently-finished terminal runs until the soft cap is met."""
    while len(_runs) > maximum:
        candidates = [
            record
            for record in _runs.values()
            if record.status in _TERMINAL_RUN_STATUSES
            and record.finished_at is not None
        ]
        if not candidates:
            return
        oldest = min(candidates, key=lambda item: item.finished_at or 0.0)
        _evict_run_record(
            oldest.run_id,
            expected_finished_at=oldest.finished_at or 0.0,
            reason="capacity",
        )


def _remember_run(record: RunRecord, config: dict[str, Any] | None) -> None:
    """Register a live run and enforce the process-local record cap."""
    configurable = Configuration.from_runnable_config(config)
    if record.events.maxlen != configurable.inflight_event_buffer_size:
        record.events = deque(
            record.events,
            maxlen=configurable.inflight_event_buffer_size,
        )
    stale_task = _run_eviction_tasks.pop(record.run_id, None)
    if stale_task is not None:
        stale_task.cancel()
    _runs[record.run_id] = record
    _enforce_run_memory_limit(configurable.max_inflight_runs_in_memory)


async def _evict_run_after_delay(
    run_id: str,
    expected_finished_at: float,
    delay_seconds: float,
) -> None:
    """Evict a terminal run after its configured in-memory grace period."""
    try:
        if delay_seconds > 0:
            await asyncio.sleep(delay_seconds)
        _evict_run_record(
            run_id,
            expected_finished_at=expected_finished_at,
            reason="retention",
        )
    finally:
        try:
            current_task = asyncio.current_task()
        except RuntimeError:
            current_task = None
        if _run_eviction_tasks.get(run_id) is current_task:
            _run_eviction_tasks.pop(run_id, None)


def _schedule_run_eviction(
    record: RunRecord,
    config: dict[str, Any] | None,
) -> None:
    """Mark a terminal record finished and schedule bounded retention."""
    if record.status not in _TERMINAL_RUN_STATUSES:
        return
    if record.finished_at is None:
        record.finished_at = time.time()
    configurable = Configuration.from_runnable_config(config)
    _enforce_run_memory_limit(configurable.max_inflight_runs_in_memory)
    if _runs.get(record.run_id) is not record:
        return
    previous = _run_eviction_tasks.pop(record.run_id, None)
    if previous is not None:
        previous.cancel()
    _run_eviction_tasks[record.run_id] = asyncio.create_task(
        _evict_run_after_delay(
            record.run_id,
            record.finished_at,
            configurable.inflight_run_memory_retention_seconds,
        )
    )

FRONTEND_EDITABLE_CONFIG_KEYS = (
    "allow_clarification", "enable_async_research", "enable_human_in_loop",
    "summarization_model", "summarization_model_max_tokens",
    "research_model", "research_model_max_tokens",
    "compression_model", "compression_model_max_tokens",
    "final_report_model", "final_report_model_max_tokens",
    "search_api", "web_pipeline_mode", "web_pipeline_shadow_sample_rate",
    "web_min_source_authority", "search_candidate_limit",
    "max_fetches_per_researcher", "max_concurrent_research_units",
    "max_researcher_iterations", "max_react_tool_calls",
    "hitl_require_plan_approval", "hitl_require_outline_approval",
    "hitl_max_plan_revisions", "hitl_feedback_mode", "report_type", "output_format",
    "quality_evaluation_enabled", "quality_evaluation_model",
    "quality_evaluation_model_max_tokens", "quality_evaluation_rigor",
    "quality_evaluation_min_sources", "quality_evaluation_max_input_chars",
    "quality_risk_mode", "quality_evaluation_fail_open",
    "quality_caveat_admission_enabled", "quality_gap_recovery_max_attempts",
    "enable_memory", "memory_top_k", "memory_min_confidence", "memory_auto_write",
    "memory_write_after_report", "memory_fail_open", "memory_advanced_enabled",
    "memory_decay_enabled", "memory_reflection_enabled", "memory_profile_enabled",
    "memory_legacy_recall_enabled", "memory_run_end_maintenance_enabled",
    "memory_mutation_lock_timeout_seconds",
    "memory_soft_forgetting_enabled", "memory_verified_insights_enabled",
    "memory_search_threshold", "memory_search_rerank", "memory_importance_weight",
    "memory_relevance_weight", "memory_recency_weight",
    "memory_reflection_observation_threshold", "memory_reflection_importance_threshold",
    "memory_reflection_max_age_hours", "memory_maintenance_max_input_chars",
    "memory_profile_max_chars", "memory_half_life_days",
)


def _runs_dir_size_bytes(root: Path) -> int:
    """Return durable artifact bytes without following symlinks outside runs_dir."""
    if not root.exists():
        return 0
    total = 0
    for path in root.rglob("*"):
        try:
            if path.is_symlink() or not path.is_file():
                continue
            total += path.stat().st_size
        except OSError:
            continue
    return total


async def _refresh_operational_metrics() -> None:
    """Refresh scrape-time gauges while keeping metrics export fail-open."""
    configurable = Configuration.from_runnable_config(None)
    metrics = get_prometheus_metrics(configurable)
    if metrics is None:
        return
    try:
        used_bytes = await asyncio.to_thread(
            _runs_dir_size_bytes,
            Path(configurable.runs_dir),
        )
        metrics.set_runs_dir_usage(used_bytes, configurable.runs_dir_max_bytes)
    except Exception as exc:  # noqa: BLE001 - metrics must never block the API
        with contextlib.suppress(Exception):
            metrics.observe_export_error("prometheus", "runs_dir_usage")
        logger.debug("Unable to refresh runs_dir metrics: %s", exc)
    try:
        metrics.set_model_circuit_states(
            await get_model_circuit_registry().snapshots()
        )
    except Exception as exc:  # noqa: BLE001 - metrics must never block the API
        with contextlib.suppress(Exception):
            metrics.observe_export_error("prometheus", "model_circuit_state")
        logger.debug("Unable to refresh model circuit metrics: %s", exc)


@app.get(_metrics_path, include_in_schema=False)
async def prometheus_metrics() -> Response:
    """Expose process-wide aggregate metrics for Prometheus scraping."""
    await _refresh_operational_metrics()
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/healthz", include_in_schema=False)
async def healthz() -> dict[str, str]:
    """Report process liveness without consulting dependencies."""
    return {"status": "ok"}


def _probe_runs_directory(configurable: Configuration) -> None:
    """Raise unless the configured run directory supports create and delete."""
    root = Path(configurable.runs_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    probe = root / f".readiness-{uuid.uuid4().hex}.tmp"
    try:
        probe.write_bytes(b"ok")
    finally:
        with contextlib.suppress(OSError):
            probe.unlink()


def _search_readiness(configurable: Configuration) -> dict[str, str]:
    """Describe search credential availability without making network calls."""
    provider = getattr(configurable.search_api, "value", str(configurable.search_api))
    if provider == "none":
        return {"status": "disabled", "provider": provider}
    key_name = {
        "tavily": "TAVILY_API_KEY",
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
    }.get(provider)
    if key_name and os.environ.get(key_name):
        return {"status": "ok", "provider": provider}
    return {"status": "degraded", "provider": provider, "reason": "api_key_missing"}


async def _readiness_report() -> tuple[dict[str, Any], bool]:
    """Probe critical local dependencies and non-critical search credentials."""
    configurable = Configuration.from_runnable_config(None)
    components: dict[str, dict[str, Any]] = {}
    critical_ok = True

    if _shutting_down.is_set():
        components["server"] = {"status": "failed", "reason": "shutting_down"}
        critical_ok = False
    else:
        components["server"] = {"status": "ok"}

    try:
        await asyncio.to_thread(_probe_runs_directory, configurable)
        components["runs_dir"] = {"status": "ok"}
    except Exception as exc:  # noqa: BLE001 - readiness reports the failure
        components["runs_dir"] = {
            "status": "failed",
            "error_type": type(exc).__name__,
        }
        critical_ok = False

    if configurable.observability_enabled and configurable.sqlite_observability_enabled:
        try:
            store = await asyncio.to_thread(
                SQLiteTraceStore,
                configurable.trace_store_path,
            )
            await asyncio.to_thread(store.ping)
            components["trace_store"] = {"status": "ok"}
        except Exception as exc:  # noqa: BLE001 - readiness reports the failure
            components["trace_store"] = {
                "status": "failed",
                "error_type": type(exc).__name__,
            }
            critical_ok = False
    else:
        components["trace_store"] = {"status": "disabled"}

    iam_settings = get_iam_settings()
    if iam_settings.database_url:
        try:
            await check_database_connection(iam_settings)
            components["iam_database"] = {"status": "ok"}
        except Exception as exc:  # noqa: BLE001 - readiness reports the failure
            components["iam_database"] = {
                "status": "failed",
                "error_type": type(exc).__name__,
            }
            critical_ok = False
    else:
        components["iam_database"] = {"status": "disabled"}

    if configurable.sandbox_enabled:
        from open_deep_research.sandbox.doctor import diagnose

        sandbox_report = await asyncio.to_thread(diagnose)
        components["sandbox"] = sandbox_report
        if not sandbox_report.get("ready"):
            critical_ok = False
    else:
        components["sandbox"] = {"status": "disabled"}

    components["search"] = _search_readiness(configurable)
    overall_status = "ok" if critical_ok else "failed"
    if critical_ok and components["search"]["status"] == "degraded":
        overall_status = "degraded"
    return {"status": overall_status, "components": components}, critical_ok


@app.get("/readyz", include_in_schema=False)
async def readyz() -> JSONResponse:
    """Report whether this process can safely receive new traffic."""
    report, ready = await _readiness_report()
    return JSONResponse(report, status_code=200 if ready else 503)


def _sse(event: PublicEvent) -> str:
    return (
        f"id: {event.sequence}\n"
        f"event: {event.type}\n"
        f"data: {json.dumps(event.public_dict(), ensure_ascii=False, default=str)}\n\n"
    )


def _task_activity_sse(event: PublicTaskActivityEvent) -> str:
    """Serialize one browser-safe task activity as SSE."""
    return (
        f"id: {event.sequence}\n"
        f"event: {event.type}\n"
        f"data: {json.dumps(event.public_dict(), ensure_ascii=False, default=str)}\n\n"
    )


def _sse_headers() -> dict[str, str]:
    return {
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
        "Connection": "keep-alive",
    }


async def _public_event_iterator(
    store: RunEventStore, *, after: int = 0, principal: Principal | None = None
):
    """Replay then tail a durable public stream, including cross-worker writes."""
    configurable = Configuration.from_runnable_config(None)
    poll_seconds = configurable.sse_poll_interval_ms / 1000
    heartbeat_seconds = configurable.sse_heartbeat_seconds
    cursor = after
    last_output_at = asyncio.get_running_loop().time()
    last_auth_at = last_output_at
    while True:
        if _sse_shutdown.is_set():
            return
        now = asyncio.get_running_loop().time()
        if (
            principal is not None
            and principal.session_id is not None
            and now - last_auth_at >= get_iam_settings().sse_reauth_interval
        ):
            async with session_scope() as db:
                if await reauthorize_session(db, principal) is None:
                    return
            last_auth_at = now
        events = await asyncio.to_thread(store.read, cursor)
        for event in events:
            yield _sse(event)
            cursor = event.sequence
            last_output_at = asyncio.get_running_loop().time()
            if is_terminal_event(event):
                return
        if not events:
            last_sequence = await asyncio.to_thread(store.last_sequence)
            if last_sequence and cursor >= last_sequence:
                latest = await asyncio.to_thread(store.read, last_sequence - 1)
                if latest and is_terminal_event(latest[-1]):
                    return
            now = asyncio.get_running_loop().time()
            if now - last_output_at >= heartbeat_seconds:
                yield ": keep-alive\n\n"
                last_output_at = now
        await asyncio.sleep(poll_seconds)


async def _task_activity_iterator(
    store: TaskActivityStore, *, after: int = 0, principal: Principal | None = None
):
    """Replay then tail one task-local durable activity stream."""
    configurable = Configuration.from_runnable_config(None)
    poll_seconds = configurable.sse_poll_interval_ms / 1000
    heartbeat_seconds = configurable.sse_heartbeat_seconds
    cursor = after
    last_output_at = asyncio.get_running_loop().time()
    last_auth_at = last_output_at
    while True:
        if _sse_shutdown.is_set():
            return
        now = asyncio.get_running_loop().time()
        if (
            principal is not None
            and principal.session_id is not None
            and now - last_auth_at >= get_iam_settings().sse_reauth_interval
        ):
            async with session_scope() as db:
                if await reauthorize_session(db, principal) is None:
                    return
            last_auth_at = now
        events = await asyncio.to_thread(store.read, cursor)
        terminal_seen = False
        for event in events:
            yield _task_activity_sse(event)
            cursor = event.sequence
            last_output_at = asyncio.get_running_loop().time()
            if event.type in TASK_TERMINAL_TYPES:
                terminal_seen = True
        if terminal_seen:
            return
        if not events:
            last_sequence = await asyncio.to_thread(store.last_sequence)
            if last_sequence and cursor >= last_sequence:
                history = await asyncio.to_thread(store.read)
                if any(event.type in TASK_TERMINAL_TYPES for event in history):
                    return
            now = asyncio.get_running_loop().time()
            if now - last_output_at >= heartbeat_seconds:
                yield ": keep-alive\n\n"
                last_output_at = now
        await asyncio.sleep(poll_seconds)


def _config_from_request(request: RunRequest, principal: Principal) -> dict[str, Any]:
    try:
        validate_http_configurable(request.configurable)
        validate_http_metadata(request.metadata)
    except ValueError as exc:
        logger.warning("security.unsafe_config_rejected: %s", exc)
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    config = {
        "configurable": dict(request.configurable),
        "metadata": {
            **request.metadata,
            "deployment_surface": "http",
            "request_id": current_request_id(),
        },
    }
    return apply_principal_to_config(config, principal)


def _observability_store() -> SQLiteTraceStore:
    configurable = Configuration.from_runnable_config(None)
    return SQLiteTraceStore(configurable.trace_store_path)


def _user_identity(user: Principal) -> str:
    """Return the normalized authenticated identity used by run ownership checks."""
    return user.user_id


def _principal_kind(user: Principal) -> str:
    """Return a bounded metric label for authenticated principal provenance."""
    return "development" if user.user_id == "local-dev-user" else "authenticated"


def _api_governance_metrics(configurable: Configuration) -> Any:
    return get_prometheus_metrics(configurable)


def _observe_rate_limited(
    configurable: Configuration,
    dimension: str,
    user: Principal,
) -> None:
    metrics = _api_governance_metrics(configurable)
    if metrics is not None:
        with contextlib.suppress(Exception):
            metrics.observe_api_rate_limited(dimension, _principal_kind(user))
    logger.info(
        "API request rate limited",
        extra={
            "actor": _user_identity(user),
            "action": "api.rate_limited",
            "dimension": dimension,
            "principal_kind": _principal_kind(user),
        },
    )


def _observe_limiter_error(
    configurable: Configuration,
    dimension: str,
    exc: BaseException,
) -> None:
    metrics = _api_governance_metrics(configurable)
    if metrics is not None:
        with contextlib.suppress(Exception):
            metrics.observe_rate_limiter_error(dimension)
    logger.warning(
        "API rate limiter failed open",
        extra={"action": "api.rate_limiter_error", "dimension": dimension},
        exc_info=exc,
    )


def _active_runs_for_user(user_id: str) -> int:
    """Count process-local non-terminal runs owned by one principal."""
    active = 0
    for record in _runs.values():
        if record.status in _TERMINAL_RUN_STATUSES:
            continue
        metadata = getattr(record.engine, "config", {}).get("metadata", {})
        owner = metadata.get("owner") or metadata.get("user_id")
        if str(owner or "") == user_id:
            active += 1
    return active


def _enforce_run_create_limits(user: Principal, configurable: Configuration) -> None:
    """Apply per-principal creation and active-run limits, failing open on bugs."""
    identity = _user_identity(user)
    try:
        allowed, retry_after = _api_rate_limiter.allow(
            f"run-create:{identity}",
            configurable.api_run_create_per_minute,
        )
        if not allowed:
            _observe_rate_limited(configurable, "run_create_rate", user)
            raise HTTPException(
                status_code=429,
                detail="run_create_rate_limited",
                headers={"Retry-After": str(retry_after)},
            )
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001 - business limiter is fail-open
        _observe_limiter_error(configurable, "run_create_rate", exc)

    try:
        maximum = configurable.max_concurrent_runs_per_user
        if maximum > 0 and _active_runs_for_user(identity) >= maximum:
            _observe_rate_limited(configurable, "concurrent_runs", user)
            raise HTTPException(
                status_code=429,
                detail="concurrent_run_limit_reached",
                headers={"Retry-After": "5"},
            )
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001 - business limiter is fail-open
        _observe_limiter_error(configurable, "concurrent_runs", exc)


async def _reserve_sse_connection(
    user: Principal,
    configurable: Configuration,
) -> int:
    """Reserve one global SSE slot and return the release token (configured cap)."""
    limit = configurable.max_concurrent_sse_connections
    try:
        allowed = await _sse_connection_limiter.acquire(limit)
    except Exception as exc:  # noqa: BLE001 - business limiter is fail-open
        _observe_limiter_error(configurable, "sse_connections", exc)
        return 0
    if not allowed:
        _observe_rate_limited(configurable, "sse_connections", user)
        raise HTTPException(
            status_code=429,
            detail="sse_connection_limit_reached",
            headers={"Retry-After": "5"},
        )
    return limit


async def _limited_sse(source: Any, release_token: int):
    """Release a global SSE slot whenever iteration ends or disconnects."""
    try:
        async for item in source:
            yield item
    finally:
        with contextlib.suppress(Exception):
            await _sse_connection_limiter.release(release_token)


def _request_query_preview(request: RunRequest) -> str:
    for message in request.messages:
        if str(message.get("role", "")).lower() in {"user", "human"}:
            content = message.get("content", "")
            if isinstance(content, list):
                content = " ".join(
                    str(item.get("text", "")) if isinstance(item, dict) else str(item)
                    for item in content
                )
            return " ".join(str(content).split())[:280]
    return ""


def _run_title(request: RunRequest, run_id: str) -> str:
    explicit = " ".join((request.title or "").split())
    preview = _request_query_preview(request)
    return (explicit or preview[:80] or run_id)[:160]


def _runs_root(config: dict[str, Any] | None = None) -> Path:
    return Path(Configuration.from_runnable_config(config).runs_dir).resolve()


def _load_manifests(config: dict[str, Any] | None = None) -> list[Any]:
    return _load_manifests_from_root(_runs_root(config))


def _load_manifests_from_root(root: Path) -> list[Any]:
    """Load manifests from an already resolved root without re-reading env config."""
    if not root.exists():
        return []
    manifests = []
    for path in root.glob("*/context/manifest.json"):
        try:
            manifests.append(
                RunContextStore(path.parent.parent.name, runs_dir=str(root)).load_manifest()
            )
        except (ValueError, JournalCorruptedError, OSError):
            continue
    return manifests


def _fenced_recovery_config(
    manifest: Any,
    configurable: Configuration,
    lease_manager: LeaderLeaseManager,
    fence_token: int,
) -> dict[str, Any]:
    """Build the minimal config needed for a fenced recovery event write."""
    stored = manifest.config if isinstance(manifest.config, dict) else {}
    return {
        "configurable": {
            **dict(stored.get("configurable") or {}),
            "runs_dir": configurable.runs_dir,
        },
        "metadata": {
            **dict(stored.get("metadata") or {}),
            "run_id": manifest.run_id,
            "run_fence_token": fence_token,
            "run_lease_owner_id": lease_manager.owner_id,
        },
    }


async def _renew_sweep_lease(
    lease_manager: LeaderLeaseManager,
    fence_token: int,
    interval_seconds: float,
) -> None:
    """Keep the global recovery-sweep lease live for the whole scan."""
    while True:
        await asyncio.sleep(interval_seconds)
        await lease_manager.renew(expected_fence_token=fence_token)


async def _run_recovery_sweep(configurable: Configuration) -> int:
    """Interrupt orphaned non-terminal manifests while preserving live owners."""
    started_at = time.perf_counter()
    sweep_lease = LeaderLeaseManager(
        runs_dir=configurable.runs_dir,
        run_id="system-recovery-sweep",
        lease_seconds=configurable.leader_lease_seconds,
        lock_timeout=configurable.mailbox_lock_timeout_seconds,
    )
    try:
        global_lease = await sweep_lease.acquire()
    except LeaseConflictError:
        logger.info(
            "run recovery sweep skipped actor=system action=run.recovery_sweep "
            "reason=live_sweep_owner"
        )
        return 0

    renew_task = asyncio.create_task(
        _renew_sweep_lease(
            sweep_lease,
            global_lease.fence_token,
            max(0.1, configurable.leader_heartbeat_seconds),
        )
    )
    interrupted = 0
    try:
        for manifest in await asyncio.to_thread(
            _load_manifests,
            {"configurable": {"runs_dir": configurable.runs_dir}},
        ):
            if manifest.status in _TERMINAL_RUN_STATUSES:
                continue
            run_lease = LeaderLeaseManager(
                runs_dir=configurable.runs_dir,
                run_id=manifest.run_id,
                lease_seconds=configurable.leader_lease_seconds,
                lock_timeout=configurable.mailbox_lock_timeout_seconds,
            )
            try:
                lease = await run_lease.acquire()
            except LeaseConflictError:
                continue
            try:
                store = RunContextStore(
                    manifest.run_id,
                    runs_dir=configurable.runs_dir,
                )
                await asyncio.to_thread(
                    store.bind_fence_token,
                    lease.fence_token,
                    run_lease.owner_id,
                )
                await asyncio.to_thread(
                    store._update_manifest,  # noqa: SLF001
                    status="interrupted",
                )
                event_config = _fenced_recovery_config(
                    manifest,
                    configurable,
                    run_lease,
                    lease.fence_token,
                )
                interrupted += 1
                try:
                    await event_publisher_from_config(event_config).publish(
                        "run.interrupted",
                        payload={
                            "status": "interrupted",
                            "error_code": "startup_recovery_sweep",
                            "message": "The previous process stopped before this run completed.",
                            "termination_reason": "orphaned_run",
                            "result_status": "interrupted",
                        },
                        dedupe_key=f"run:interrupted:startup:{manifest.run_id}",
                    )
                except Exception as exc:  # noqa: BLE001 - event export is fail-open
                    logger.warning(
                        "run recovery event write failed actor=system "
                        "action=run.recovery_sweep run_id=%s error_type=%s",
                        manifest.run_id,
                        type(exc).__name__,
                    )
                try:
                    await asyncio.to_thread(
                        get_trace_recorder(event_config).finish_run,
                        manifest.run_id,
                        "interrupted",
                    )
                except Exception as exc:  # noqa: BLE001 - trace export is fail-open
                    logger.warning(
                        "run recovery trace write failed actor=system "
                        "action=run.recovery_sweep run_id=%s error_type=%s",
                        manifest.run_id,
                        type(exc).__name__,
                    )
            except Exception as exc:  # noqa: BLE001 - one bad run cannot block startup
                logger.warning(
                    "run recovery failed actor=system action=run.recovery_sweep "
                    "run_id=%s error_type=%s",
                    manifest.run_id,
                    type(exc).__name__,
                )
            finally:
                with contextlib.suppress(Exception):
                    await run_lease.release(
                        expected_fence_token=lease.fence_token
                    )
    finally:
        renew_task.cancel()
        await asyncio.gather(renew_task, return_exceptions=True)
        with contextlib.suppress(Exception):
            await sweep_lease.release(
                expected_fence_token=global_lease.fence_token
            )

    logger.info(
        "run recovery sweep completed actor=system action=run.recovery_sweep "
        "interrupted=%s duration_seconds=%.3f",
        interrupted,
        time.perf_counter() - started_at,
    )
    return interrupted


def _run_finished_at(manifest: Any) -> float:
    """Return the durable terminal timestamp, including legacy manifests."""
    return float(manifest.ended_at or manifest.updated_at or manifest.created_at)


def _run_directory(configurable: Configuration, run_id: str) -> Path:
    """Resolve a run directory while preventing cleanup path traversal."""
    root = Path(configurable.runs_dir).resolve()
    target = (root / run_id).resolve()
    if target == root or root not in target.parents:
        raise ValueError("run_id escapes runs_dir")
    return target


def _lifecycle_metrics(configurable: Configuration) -> Any:
    """Return enabled Prometheus collectors for fail-open lifecycle reporting."""
    return get_prometheus_metrics(configurable)


def _record_lifecycle_error(
    configurable: Configuration,
    operation: str,
    exc: BaseException,
) -> None:
    """Report a cleanup failure without allowing metrics to mask the cause."""
    metrics = _lifecycle_metrics(configurable)
    if metrics is not None:
        with contextlib.suppress(Exception):
            metrics.observe_export_error("retention", operation)
    logger.warning(
        "run lifecycle cleanup failed actor=system action=%s error_type=%s",
        operation,
        type(exc).__name__,
    )


async def _purge_run_artifacts(
    run_id: str,
    configurable: Configuration,
    *,
    reason: str,
    actor: str,
    require_terminal: bool = True,
) -> dict[str, Any]:
    """Delete disk, trace, and memory state through one idempotent path."""
    target = _run_directory(configurable, run_id)
    record = _runs.get(run_id)
    manifest = None
    if target.exists():
        try:
            manifest = await asyncio.to_thread(
                RunContextStore(run_id, runs_dir=configurable.runs_dir).load_manifest
            )
        except (JournalCorruptedError, OSError, ValueError):
            manifest = None
    status = record.status if record is not None else getattr(manifest, "status", None)
    if require_terminal and status not in _TERMINAL_RUN_STATUSES:
        raise RuntimeError("run_not_terminal")

    logger.info(
        "run purge started actor=%s action=run.purge run_id=%s reason=%s",
        actor,
        run_id,
        reason,
        extra={"actor": actor, "action": "run.purge", "run_id": run_id, "reason": reason},
    )
    trace_rows = await asyncio.to_thread(
        SQLiteTraceStore(configurable.trace_store_path).delete_run,
        run_id,
    )
    directory_existed = target.exists()
    if directory_existed:
        await asyncio.to_thread(shutil.rmtree, target)
    eviction_task = _run_eviction_tasks.pop(run_id, None)
    if eviction_task is not None:
        eviction_task.cancel()
    _runs.pop(run_id, None)
    metrics = _lifecycle_metrics(configurable)
    if metrics is not None:
        with contextlib.suppress(Exception):
            metrics.observe_run_purged(reason)
    logger.info(
        "run purge completed actor=%s action=run.purge run_id=%s reason=%s "
        "trace_rows=%s directory_existed=%s",
        actor,
        run_id,
        reason,
        trace_rows,
        directory_existed,
        extra={"actor": actor, "action": "run.purge", "run_id": run_id, "reason": reason},
    )
    return {
        "run_id": run_id,
        "status": "deleted",
        "reason": reason,
        "directory_deleted": directory_existed,
        "trace_rows_deleted": trace_rows,
    }


async def _run_retention_sweep(configurable: Configuration) -> dict[str, Any]:
    """Apply age retention, trace retention, and quota fallback once."""
    started_at = time.perf_counter()
    metrics = _lifecycle_metrics(configurable)
    deleted_by_age = 0
    deleted_by_quota = 0
    trace_runs_deleted = 0
    sweep_lease = LeaderLeaseManager(
        runs_dir=configurable.runs_dir,
        run_id="system-retention-sweep",
        lease_seconds=configurable.leader_lease_seconds,
        lock_timeout=configurable.mailbox_lock_timeout_seconds,
    )
    try:
        lease = await sweep_lease.acquire()
    except LeaseConflictError:
        return {
            "status": "skipped",
            "reason": "live_sweep_owner",
            "deleted_by_age": 0,
            "deleted_by_quota": 0,
        }

    try:
        manifests = await asyncio.to_thread(
            _load_manifests_from_root,
            Path(configurable.runs_dir).resolve(),
        )
        terminal = [
            item for item in manifests if item.status in _TERMINAL_RUN_STATUSES
        ]
        if configurable.run_retention_days > 0:
            cutoff = time.time() - configurable.run_retention_days * 86400
            for manifest in sorted(terminal, key=_run_finished_at):
                if _run_finished_at(manifest) >= cutoff:
                    continue
                try:
                    await _purge_run_artifacts(
                        manifest.run_id,
                        configurable,
                        reason="retention",
                        actor="system",
                    )
                    deleted_by_age += 1
                except Exception as exc:  # noqa: BLE001 - sweep is fail-open
                    _record_lifecycle_error(configurable, "retention_purge", exc)

        trace_days = (
            configurable.run_retention_days
            if configurable.trace_retention_days is None
            else configurable.trace_retention_days
        )
        trace_store = SQLiteTraceStore(configurable.trace_store_path)
        if trace_days > 0:
            trace_cutoff = time.time() - trace_days * 86400
            try:
                trace_runs_deleted = await asyncio.to_thread(
                    trace_store.delete_runs_ended_before,
                    trace_cutoff,
                )
            except Exception as exc:  # noqa: BLE001 - sweep is fail-open
                _record_lifecycle_error(configurable, "trace_retention", exc)

        quota = configurable.runs_dir_max_bytes
        if quota > 0:
            target_bytes = int(quota * 0.9)
            used_bytes = await asyncio.to_thread(
                _runs_dir_size_bytes,
                Path(configurable.runs_dir),
            )
            quota_triggered = used_bytes > quota
            if quota_triggered:
                remaining = [
                    item
                    for item in terminal
                    if _run_directory(configurable, item.run_id).exists()
                ]
                for manifest in sorted(remaining, key=_run_finished_at):
                    if used_bytes <= target_bytes:
                        break
                    try:
                        await _purge_run_artifacts(
                            manifest.run_id,
                            configurable,
                            reason="quota",
                            actor="system",
                        )
                        deleted_by_quota += 1
                        await asyncio.to_thread(trace_store.checkpoint)
                        used_bytes = await asyncio.to_thread(
                            _runs_dir_size_bytes,
                            Path(configurable.runs_dir),
                        )
                    except Exception as exc:  # noqa: BLE001 - sweep is fail-open
                        _record_lifecycle_error(configurable, "quota_purge", exc)
            quota_exceeded = quota_triggered and used_bytes > target_bytes
            if metrics is not None:
                with contextlib.suppress(Exception):
                    metrics.set_runs_dir_usage(used_bytes, quota)
                    metrics.set_retention_quota_exceeded(quota_exceeded)
            if quota_exceeded:
                logger.error(
                    "run retention quota remains exceeded actor=system "
                    "action=run.retention_sweep used_bytes=%s quota_bytes=%s",
                    used_bytes,
                    quota,
                )
        elif metrics is not None:
            with contextlib.suppress(Exception):
                metrics.set_retention_quota_exceeded(False)

        with contextlib.suppress(Exception):
            await asyncio.to_thread(trace_store.checkpoint)
    finally:
        with contextlib.suppress(Exception):
            await sweep_lease.release(expected_fence_token=lease.fence_token)
        duration = time.perf_counter() - started_at
        if metrics is not None:
            with contextlib.suppress(Exception):
                metrics.observe_retention_sweep(duration)

    logger.info(
        "run retention sweep completed actor=system action=run.retention_sweep "
        "deleted_by_age=%s deleted_by_quota=%s trace_runs_deleted=%s "
        "duration_seconds=%.3f",
        deleted_by_age,
        deleted_by_quota,
        trace_runs_deleted,
        time.perf_counter() - started_at,
        extra={"actor": "system", "action": "run.retention_sweep"},
    )
    return {
        "status": "completed",
        "deleted_by_age": deleted_by_age,
        "deleted_by_quota": deleted_by_quota,
        "trace_runs_deleted": trace_runs_deleted,
    }


async def _retention_sweep_loop(configurable: Configuration) -> None:
    """Run lifecycle cleanup periodically until service shutdown."""
    interval = configurable.retention_sweep_interval_seconds
    while True:
        await asyncio.sleep(interval)
        try:
            await _run_retention_sweep(configurable)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - maintenance remains fail-open
            _record_lifecycle_error(configurable, "retention_sweep", exc)


async def _interrupt_inflight_record(record: RunRecord) -> None:
    """Persist and cancel one process-owned run during graceful shutdown."""
    config = getattr(record.engine, "config", None) or {}
    event_config = {
        "configurable": dict(config.get("configurable") or {}),
        "metadata": {
            **dict(config.get("metadata") or {}),
            "run_id": record.run_id,
        },
    }
    store = getattr(record.engine, "context_store", None)
    if store is not None and store.manifest_path.exists():
        try:
            await asyncio.to_thread(
                store._update_manifest,  # noqa: SLF001
                status="interrupted",
            )
        except Exception as exc:  # noqa: BLE001 - shutdown remains best-effort
            logger.warning(
                "run interrupt manifest write failed actor=system "
                "action=run.interrupted run_id=%s error_type=%s",
                record.run_id,
                type(exc).__name__,
            )
    try:
        await event_publisher_from_config(event_config).publish(
            "run.interrupted",
            payload={
                "status": "interrupted",
                "error_code": "server_shutdown",
                "message": "The server stopped before this run completed.",
                "termination_reason": "server_shutdown",
                "result_status": "interrupted",
            },
            dedupe_key=f"run:interrupted:shutdown:{record.run_id}",
        )
    except Exception:
        pass
    try:
        await asyncio.to_thread(
            get_trace_recorder(event_config).finish_run,
            record.run_id,
            "interrupted",
        )
    except Exception as exc:  # noqa: BLE001 - observability must not block drain
        logger.warning(
            "run interrupt trace write failed actor=system "
            "action=run.interrupted run_id=%s error_type=%s",
            record.run_id,
            type(exc).__name__,
        )
    record.status = "interrupted"
    record.finished_at = time.time()
    logger.info(
        "run interrupted actor=system action=run.interrupted run_id=%s "
        "reason=server_shutdown",
        record.run_id,
    )
    if record.task is not None and not record.task.done():
        record.task.cancel()
        await asyncio.gather(record.task, return_exceptions=True)


async def _drain_inflight_runs(timeout_seconds: float) -> None:
    """Interrupt all live in-memory runs within one shutdown budget."""
    records = [
        record
        for record in list(_runs.values())
        if record.status not in _TERMINAL_RUN_STATUSES
    ]

    async def drain() -> None:
        for record in records:
            await _interrupt_inflight_record(record)

    try:
        async with asyncio.timeout(timeout_seconds):
            await drain()
    except TimeoutError:
        logger.error(
            "run shutdown drain timed out actor=system action=run.interrupted "
            "timeout_seconds=%s",
            timeout_seconds,
        )
        for record in records:
            if record.task is not None and not record.task.done():
                record.task.cancel()
        await asyncio.gather(
            *(record.task for record in records if record.task is not None),
            return_exceptions=True,
        )


def _find_idempotent_run(
    config: dict[str, Any],
    owner_id: str | None,
    idempotency_key: str,
) -> Any | None:
    return next(
        (
            manifest
            for manifest in _load_manifests(config)
            if manifest.owner_id == owner_id and manifest.idempotency_key == idempotency_key
        ),
        None,
    )


def _encode_cursor(created_at: float, run_id: str) -> str:
    raw = json.dumps([created_at, run_id], separators=(",", ":")).encode()
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def _decode_cursor(cursor: str) -> tuple[float, str]:
    try:
        padded = cursor + "=" * (-len(cursor) % 4)
        value = json.loads(base64.urlsafe_b64decode(padded).decode())
        return float(value[0]), str(value[1])
    except Exception as exc:
        raise HTTPException(status_code=400, detail="invalid_cursor") from exc


def _stable_output(result: dict[str, Any] | None, report: str = "") -> dict[str, Any]:
    state = result or {}
    outcome = state.get("result") if isinstance(state.get("result"), dict) else state
    outcome = outcome if isinstance(outcome, dict) else {}
    markdown = report or str(state.get("final_report") or outcome.get("result") or "")
    return {
        "markdown": markdown,
        "artifacts": state.get("artifacts") or outcome.get("artifacts") or [],
        "quality_gate": state.get("quality_gate") or outcome.get("quality_gate"),
        "termination_reason": outcome.get("termination_reason"),
        "status": outcome.get("status"),
        "usage": outcome.get("usage") or {},
        "usage_accounting": outcome.get("usage_accounting"),
        "metrics": outcome.get("metrics") or {},
    }


def _require_record_owner(record: RunRecord, user: Principal) -> None:
    """Hide in-memory runs from users other than their owner."""
    metadata = getattr(record.engine, "config", {}).get("metadata", {})
    owner = metadata.get("owner") or metadata.get("user_id")
    if not owner or str(owner) != _user_identity(user):
        raise HTTPException(status_code=404, detail="Run not found")


def _require_run_owner(run_id: str, user: Principal) -> tuple[RunRecord | None, Configuration]:
    """Authorize an active or persisted run and return its effective config."""
    record = _runs.get(run_id)
    if record is not None:
        _require_record_owner(record, user)
        return record, Configuration.from_runnable_config(getattr(record.engine, "config", None))
    configurable = Configuration.from_runnable_config(None)
    try:
        manifest = RunContextStore(run_id, runs_dir=configurable.runs_dir).load_manifest()
    except (ValueError, JournalCorruptedError, OSError):
        raise HTTPException(status_code=404, detail="Run not found") from None
    if not manifest.owner_id or manifest.owner_id != _user_identity(user):
        raise HTTPException(status_code=404, detail="Run not found")
    return None, Configuration.from_runnable_config(manifest.config)


def _task_activity_preview_allowed(user: Principal) -> bool:
    """Authorize bounded diagnostic previews without trusting the browser."""
    if local_dev_bypass_enabled():
        return True
    enabled = os.environ.get("TASK_ACTIVITY_PREVIEW_ENABLED", "false").lower() in {
        "1", "true", "yes", "on",
    }
    return enabled and user.has(RESEARCH_DIAGNOSTICS_PREVIEW.code)


def _augment_run_projection(
    run_id: str,
    projection: Any,
    configurable: Configuration,
) -> Any:
    """Attach task activity summaries without changing the public event reducer."""
    if projection is None:
        return None
    if projection.status in _TERMINAL_RUN_STATUSES:
        try:
            SecurityApprovalStore(
                run_id,
                runs_dir=configurable.runs_dir,
            ).deny_pending(actor="system", reason="run_terminal")
        except (OSError, RuntimeError, ValueError):
            pass
        projection.pending_security_approvals = []
    else:
        try:
            _version, approvals = SecurityApprovalStore(
                run_id,
                runs_dir=configurable.runs_dir,
            ).list(status="pending")
            projection.pending_security_approvals = [
                {
                    "approval_id": item.approval_id,
                    "task_id": item.task_id,
                    "kind": item.kind,
                    "capability": item.capability,
                    "target": item.target,
                    "status": item.status,
                    "expires_at": item.expires_at,
                }
                for item in approvals
            ]
        except (OSError, RuntimeError, ValueError):
            pass
    for task_id, task in projection.task_items.items():
        try:
            store = TaskActivityStore(run_id, task_id, runs_dir=configurable.runs_dir)
            if store.exists:
                task.update(activity_summary(store.read()))
        except (OSError, RuntimeError, ValueError):
            task.setdefault("activity_available", False)
    return projection


def _require_task_in_run(
    run_id: str,
    task_id: str,
    configurable: Configuration,
) -> Any:
    """Return the public task projection or hide unknown/cross-run task IDs."""
    projection = RunEventStore(run_id, runs_dir=configurable.runs_dir).project()
    task = projection.task_items.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    return task


def _span_tree_rows(spans: list[dict[str, Any]]) -> str:
    children: dict[str | None, list[dict[str, Any]]] = {}
    for span in spans:
        children.setdefault(span.get("parent_span_id"), []).append(span)

    rows: list[str] = []

    def visit(span: dict[str, Any], depth: int) -> None:
        status = html.escape(str(span.get("status") or ""))
        name = html.escape(str(span.get("name") or ""))
        kind = html.escape(str(span.get("kind") or ""))
        duration = span.get("duration_ms") or 0
        tokens = span.get("total_tokens") or 0
        retry_count = span.get("retry_count") or 0
        error_type = html.escape(str(span.get("error_type") or ""))
        error = html.escape(str(span.get("error") or ""))
        indent = "&nbsp;" * depth * 4
        cls = "error" if status == "error" else "ok"
        rows.append(
            f"<tr class='{cls}'><td>{indent}{name}</td><td>{kind}</td>"
            f"<td>{status}</td><td>{duration}</td><td>{tokens}</td>"
            f"<td>{retry_count}</td><td>{error_type}</td><td>{error}</td></tr>"
        )
        for child in children.get(span.get("span_id"), []):
            visit(child, depth + 1)

    roots = children.get(None, [])
    for root in roots:
        visit(root, 0)
    for span in spans:
        if span.get("parent_span_id") and span.get("parent_span_id") not in {s.get("span_id") for s in spans}:
            visit(span, 0)
    return "".join(rows)


async def _release_gateway_run(record: RunRecord, config: dict[str, Any]) -> None:
    """Erase ephemeral Gateway credentials after all run work has terminated."""
    configurable = Configuration.from_runnable_config(config)
    fence_token = getattr(record.engine, "run_fence_token", None)
    if not configurable.sandbox_enabled or fence_token is None:
        return
    try:
        from open_deep_research.sandbox.gateway_client import (
            SandboxGatewayControlClient,
        )

        await SandboxGatewayControlClient(configurable).unregister_run(
            run_id=record.run_id,
            fence_token=int(fence_token),
        )
    except Exception as exc:  # noqa: BLE001 - task tokens still expire independently
        logger.warning(
            "sandbox gateway credential cleanup failed run_id=%s error=%s",
            record.run_id,
            str(exc)[:500],
        )


async def _run_background(record: RunRecord, request: RunRequest, config: dict[str, Any]) -> None:
    record.status = "running"
    control_task = asyncio.create_task(_run_control_listener(record, config))
    try:
        async for event in record.engine.stream_message(request.messages, config):
            record.events.append(event)
            status = event.get("data", {}).get("status")
            if status in {
                "running",
                "awaiting_clarification",
                "awaiting_plan_approval",
                "awaiting_outline_approval",
                "completed",
                "failed",
                "cancelled",
            }:
                record.status = status
        record.result = record.engine.final_state
        record.status = record.engine.status
    except Exception as exc:  # noqa: BLE001 - surface in run state
        event = {"event": "run.failed", "data": {"run_id": record.run_id, "error": str(exc)}}
        record.events.append(event)
        record.result = {"result": {"status": "error", "error": str(exc)}}
        record.status = "failed"
        try:
            await event_publisher_from_config(config).publish(
                "run.failed",
                payload={"status": "failed", "error_code": "run_execution_failed", "message": "Research failed."},
                dedupe_key="run:terminal",
            )
        except Exception:
            pass
    finally:
        control_task.cancel()
        await asyncio.gather(control_task, return_exceptions=True)
        await _release_gateway_run(record, config)
        _schedule_run_eviction(record, config)


async def _run_resumed_background(record: RunRecord) -> None:
    """Continue a persisted Query run in the background."""
    record.status = "running"
    config = getattr(record.engine, "config", None)
    control_task = asyncio.create_task(_run_control_listener(record, config or {}))
    try:
        async for event in record.engine.stream_resume():
            record.events.append(event)
            status = event.get("data", {}).get("status")
            if status in {"running", "completed", "failed", "cancelled"}:
                record.status = status
        record.result = record.engine.final_state
        record.status = "cancelled" if record.engine.status == "cancelled" else record.engine.status
    except Exception as exc:  # noqa: BLE001 - surface through run state
        record.events.append({"event": "run.failed", "data": {"run_id": record.run_id, "error": str(exc)}})
        record.result = {"result": {"status": "error", "error": str(exc)}}
        record.status = "failed"
        try:
            await event_publisher_from_config(config).publish(
                "run.failed",
                payload={"status": "failed", "error_code": "run_execution_failed", "message": "Research failed."},
                dedupe_key="run:terminal",
            )
        except Exception:
            pass
    finally:
        control_task.cancel()
        await asyncio.gather(control_task, return_exceptions=True)
        await _release_gateway_run(record, config or {})
        _schedule_run_eviction(record, config)


async def _run_control_listener(record: RunRecord, config: dict[str, Any]) -> None:
    """Consume durable commands for the worker that owns a live run."""
    configurable = Configuration.from_runnable_config(config)
    store = RunControlStore(record.run_id, runs_dir=configurable.runs_dir)
    publisher = event_publisher_from_config(config)
    poll_seconds = configurable.sse_poll_interval_ms / 1000
    while True:
        for command in await store.pending():
            try:
                if command.type == "cancel":
                    record.engine.interrupt()
                    record.status = "cancelling"
                elif command.type == "human_action":
                    record.engine.handle_human_action(
                        str(command.payload.get("action_id", "")),
                        str(command.payload.get("action", "")),
                        str(command.payload.get("message", "")),
                    )
                elif command.type == "feedback":
                    await record.engine.submit_feedback(dict(command.payload))
                await store.ack(command)
            except Exception:
                await publisher.publish(
                    "system.warning",
                    payload={
                        "warning_code": "control_command_rejected",
                        "message": "A run control command could not be applied.",
                    },
                    dedupe_key=f"control:{command.command_id}:rejected",
                )
        await asyncio.sleep(poll_seconds)


@app.get("/capabilities")
async def get_capabilities(
    user: Principal = Depends(require_active_user),
) -> dict[str, Any]:
    """Return the safe, explicit browser-editable runtime contract."""
    schema = Configuration.model_json_schema()
    properties = schema.get("properties", {})
    selected = {
        key: properties[key]
        for key in FRONTEND_EDITABLE_CONFIG_KEYS
        if key in properties
    }
    defaults = Configuration.from_runnable_config(None).model_dump(mode="json")
    return {
        "public_event_schema_version": PUBLIC_EVENT_SCHEMA_VERSION,
        "public_task_activity_schema_version": PUBLIC_TASK_ACTIVITY_SCHEMA_VERSION,
        "accepted_event_schema_versions": [1, PUBLIC_EVENT_SCHEMA_VERSION],
        "features": {
            "clarification": True,
            "human_in_loop": True,
            "feedback": True,
            "artifacts": True,
            "memory": True,
            "subagent_activity": True,
            "subagent_activity_preview": _task_activity_preview_allowed(user),
            "local_dev_auth_bypass": os.environ.get("LOCAL_DEV_AUTH_BYPASS", "").lower()
            in {"1", "true", "yes"},
        },
        "editable_config_keys": list(selected),
        "config_schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": selected,
            "$defs": schema.get("$defs", {}),
        },
        "defaults": {key: defaults[key] for key in selected if key in defaults},
        "ui": {
            key: (Configuration.model_fields[key].json_schema_extra or {})
            for key in selected
        },
    }


@app.get("/runs")
async def list_runs(
    limit: int = 30,
    cursor: str | None = None,
    status: str | None = None,
    user: Principal = Depends(require_permissions(RESEARCH_RUN_READ_OWN.code)),
) -> dict[str, Any]:
    """List the authenticated user's persisted run manifests newest first."""
    if limit < 1 or limit > 100:
        raise HTTPException(status_code=422, detail="limit_must_be_between_1_and_100")
    owner = _user_identity(user)
    manifests = [
        item
        for item in _load_manifests()
        if item.owner_id == owner
        and (status is None or item.status == status)
    ]
    manifests.sort(key=lambda item: (item.created_at, item.run_id), reverse=True)
    if cursor:
        cursor_key = _decode_cursor(cursor)
        manifests = [
            item for item in manifests if (item.created_at, item.run_id) < cursor_key
        ]
    page = manifests[: limit + 1]
    has_more = len(page) > limit
    page = page[:limit]
    items = [
        {
            "run_id": item.run_id,
            "title": item.title or item.run_id,
            "query_preview": item.query_preview or item.run_id,
            "status": item.status,
            "created_at": item.created_at,
            "updated_at": item.updated_at,
            "last_event_id": item.last_public_event_seq,
        }
        for item in page
    ]
    next_cursor = (
        _encode_cursor(page[-1].created_at, page[-1].run_id)
        if has_more and page
        else None
    )
    return {"items": items, "next_cursor": next_cursor}


@app.post("/runs/stream")
async def stream_run(
    request: RunRequest,
    user: Principal = Depends(require_permissions(RESEARCH_RUN_CREATE.code)),
) -> StreamingResponse:
    """Run a research request and stream events with SSE."""
    config = _config_from_request(request, user)
    configurable = Configuration.from_runnable_config(config)
    _enforce_run_create_limits(user, configurable)
    release_token = await _reserve_sse_connection(user, configurable)
    try:
        engine = QueryEngine(config)
        record = _new_run_record(
            run_id=engine.run_id,
            engine=engine,
            status="running",
            config=config,
        )
        if engine.context_store is not None:
            engine.context_store.initialize(_user_identity(user), config)
            engine.context_store._update_manifest(  # noqa: SLF001
                title=_run_title(request, engine.run_id),
                query_preview=_request_query_preview(request),
            )
        await event_publisher_from_config(config).publish(
            "run.created",
            payload={"status": "pending"},
            dedupe_key="run:created",
        )
        record.task = asyncio.create_task(_run_background(record, request, config))
        _remember_run(record, config)
        logger.info(
            "run created",
            extra={
                "actor": _user_identity(user),
                "action": "run.created",
                "run_id": record.run_id,
            },
        )
        store = RunEventStore(record.run_id, runs_dir=configurable.runs_dir)
    except Exception:
        await _sse_connection_limiter.release(release_token)
        raise
    return StreamingResponse(
        _limited_sse(
            _public_event_iterator(store, principal=user),
            release_token,
        ),
        media_type="text/event-stream",
        headers=_sse_headers(),
    )


@app.post("/runs")
async def create_run(
    request: RunRequest,
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    user: Principal = Depends(require_permissions(RESEARCH_RUN_CREATE.code)),
) -> dict[str, Any]:
    """Create a background research run."""
    config = _config_from_request(request, user)
    if idempotency_key:
        existing = _find_idempotent_run(config, _user_identity(user), idempotency_key)
        if existing is not None:
            return {
                "run_id": existing.run_id,
                "status": existing.status,
                "events_url": f"/runs/{existing.run_id}/events",
                "last_event_id": existing.last_public_event_seq,
                "idempotent_replay": True,
            }
    _enforce_run_create_limits(user, Configuration.from_runnable_config(config))
    engine = QueryEngine(config)
    record = _new_run_record(
        run_id=engine.run_id,
        engine=engine,
        status="running",
        config=config,
    )
    if engine.context_store is not None:
        engine.context_store.initialize(_user_identity(user), config)
        engine.context_store._update_manifest(  # noqa: SLF001
            title=_run_title(request, engine.run_id),
            query_preview=_request_query_preview(request),
            idempotency_key=idempotency_key,
        )
    created = await event_publisher_from_config(config).publish(
        "run.created",
        payload={"status": "pending"},
        dedupe_key="run:created",
    )
    record.task = asyncio.create_task(_run_background(record, request, config))
    _remember_run(record, config)
    logger.info(
        "run created",
        extra={"actor": _user_identity(user), "action": "run.created", "run_id": record.run_id},
    )
    return {
        "run_id": record.run_id,
        "status": record.status,
        "events_url": f"/runs/{record.run_id}/events",
        "last_event_id": created.sequence,
        "idempotent_replay": False,
    }


@app.get("/runs/{run_id}")
async def get_run(
    run_id: str,
    user: Principal = Depends(require_permissions(RESEARCH_RUN_READ_OWN.code)),
) -> dict[str, Any]:
    """Return the latest status/result for a run."""
    record = _runs.get(run_id)
    if record is None:
        configurable = Configuration.from_runnable_config(None)
        try:
            store = RunContextStore(run_id, runs_dir=configurable.runs_dir)
            manifest = store.load_manifest()
        except (ValueError, JournalCorruptedError, OSError):
            raise HTTPException(status_code=404, detail="Run not found") from None
        if not manifest.owner_id or manifest.owner_id != user.user_id:
            raise HTTPException(status_code=404, detail="Run not found")
        result = manifest.result
        report_text = ""
        if manifest.status == "completed":
            report_path = store.context_dir / "final_report.md"
            if report_path.exists():
                report_text = report_path.read_text(encoding="utf-8")
                result = {"status": "success", "result": report_text}
        event_store = RunEventStore(run_id, runs_dir=configurable.runs_dir)
        projection = event_store.project() if event_store.exists else None
        projection = _augment_run_projection(run_id, projection, configurable)
        return {
            "run_id": run_id,
            "title": manifest.title or run_id,
            "status": manifest.status,
            "created_at": manifest.created_at,
            "updated_at": manifest.updated_at,
            "runtime_seconds": max(0.0, manifest.updated_at - manifest.created_at),
            "pending_human_action": manifest.pending_human_action,
            "pending_security_approvals": (
                projection.pending_security_approvals if projection else []
            ),
            "result": result,
            "output": _stable_output(manifest.result, report_text),
            "event_count": manifest.last_journal_seq,
            "persistence_degraded": manifest.persistence_degraded,
            "progress": projection.model_dump() if projection else None,
            "events_url": f"/runs/{run_id}/events",
            "last_event_id": projection.last_event_id if projection else 0,
        }
    _require_record_owner(record, user)
    configurable = Configuration.from_runnable_config(record.engine.config)
    event_store = RunEventStore(run_id, runs_dir=configurable.runs_dir)
    projection = event_store.project()
    projection = _augment_run_projection(run_id, projection, configurable)
    manifest = (
        record.engine.context_store.load_manifest()
        if getattr(record.engine, "context_store", None)
        and record.engine.context_store.manifest_path.exists()
        else None
    )
    now = time.time()
    manifest_status = manifest.status if manifest is not None else None
    if (
        manifest_status in {"completed", "failed", "cancelled", "interrupted"}
        and record.status != manifest_status
    ):
        # The durable manifest is authoritative once terminal; the in-memory
        # record can lag behind a just-finished run.
        run_status = manifest_status
    else:
        run_status = record.status
    return {
        "run_id": run_id,
        "title": (manifest.title if manifest else None) or run_id,
        "status": run_status,
        "created_at": manifest.created_at if manifest else record.engine.started_at,
        "updated_at": manifest.updated_at if manifest else now,
        "runtime_seconds": max(0.0, now - record.engine.started_at),
        "pending_human_action": (
            getattr(record.engine, "pending_human_action", None)
            or (manifest.pending_human_action if manifest else None)
            or projection.pending_human_action
        ),
        "pending_security_approvals": projection.pending_security_approvals,
        "result": record.result,
        "output": _stable_output(record.result),
        "event_count": projection.last_event_id,
        "progress": projection.model_dump(),
        "events_url": f"/runs/{run_id}/events",
        "last_event_id": projection.last_event_id,
    }


@app.post("/runs/{run_id}/resume", status_code=202)
async def resume_run(
    run_id: str,
    request: ResumeRunRequest,
    user: Principal = Depends(require_permissions(RESEARCH_RUN_CONTROL_OWN.code)),
) -> dict[str, str]:
    """Explicitly resume an interrupted file-backed Query run."""
    active = _runs.get(run_id)
    if active is not None:
        _require_record_owner(active, user)
        if active.status == "completed":
            raise HTTPException(status_code=409, detail="run_already_completed")
        if active.status == "cancelled":
            raise HTTPException(status_code=409, detail="run_not_recoverable")
        if active.status not in {"failed", "cancelled"}:
            raise HTTPException(status_code=409, detail="run_already_active")

    try:
        validate_http_configurable(request.configurable)
        validate_http_metadata(request.metadata)
    except ValueError as exc:
        logger.warning("security.unsafe_config_rejected: %s", exc)
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    config = apply_principal_to_config(
        {
            "configurable": dict(request.configurable),
            "metadata": {
                **request.metadata,
                "run_id": run_id,
                "deployment_surface": "http",
                "request_id": current_request_id(),
            },
        },
        user,
    )
    runs_dir = str(request.configurable.get("runs_dir") or Configuration.from_runnable_config(None).runs_dir)
    try:
        engine = QueryEngine.load(run_id, runs_dir=runs_dir, config=config)
        if engine.context_store is None:
            raise JournalCorruptedError("run_not_recoverable")
        replay = engine.context_store.replay()
    except RunContextError as exc:
        if str(exc).startswith("run_schema_not_resumable:"):
            raise HTTPException(
                status_code=409,
                detail=str(exc),
            ) from None
        raise HTTPException(status_code=409, detail="run_not_recoverable") from None
    except (ValueError, OSError):
        raise HTTPException(status_code=409, detail="run_not_recoverable") from None
    if not replay.manifest.owner_id or replay.manifest.owner_id != user.user_id:
        raise HTTPException(status_code=404, detail="Run not found")
    if replay.manifest.status == "completed":
        raise HTTPException(status_code=409, detail="run_already_completed")
    if replay.manifest.status == "cancelled" or replay.manifest.next_stage == "cancelled":
        raise HTTPException(status_code=409, detail="run_not_recoverable")
    try:
        await engine.acquire_run_lease()
    except Exception as exc:
        from open_deep_research.tasks.lease import LeaseConflictError

        if isinstance(exc, LeaseConflictError):
            raise HTTPException(status_code=409, detail="run_already_active") from None
        raise

    effective_config = getattr(engine, "config", config)
    record = _new_run_record(
        run_id=run_id,
        engine=engine,
        status="running",
        config=effective_config,
    )
    record.task = asyncio.create_task(_run_resumed_background(record))
    _remember_run(record, effective_config)
    logger.info(
        "run resumed",
        extra={"actor": _user_identity(user), "action": "run.resumed", "run_id": run_id},
    )
    return {"run_id": run_id, "status": "running"}


@app.post("/runs/{run_id}/human-actions/{action_id}")
async def submit_human_action(
    run_id: str,
    action_id: str,
    request: HumanActionRequest,
    user: Principal = Depends(require_permissions(RESEARCH_RUN_INTERACT_OWN.code)),
) -> dict[str, Any]:
    """Resolve a pending human approval, revision, or cancellation action."""
    record, configurable = _require_run_owner(run_id, user)
    if record is not None:
        try:
            result = record.engine.handle_human_action(action_id, request.action, request.message or "")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if request.action == "cancel":
            record.status = "cancelled"
        return result
    manifest = RunContextStore(run_id, runs_dir=configurable.runs_dir).load_manifest()
    pending = manifest.pending_human_action or {}
    if pending.get("action_id") != action_id:
        raise HTTPException(status_code=400, detail="No matching pending human action")
    allowed = (
        {"answer", "cancel"}
        if pending.get("type") == "clarification"
        else {"approve", "revise", "cancel"}
    )
    if request.action not in allowed:
        raise HTTPException(
            status_code=400,
            detail="Human action does not match the pending action type",
        )
    if request.action in {"answer", "revise"} and not (request.message or "").strip():
        raise HTTPException(status_code=400, detail="A message is required for this human action")
    command = await RunControlStore(run_id, runs_dir=configurable.runs_dir).enqueue(
        "human_action",
        {"action_id": action_id, "action": request.action, "message": request.message or ""},
        command_id=f"human-action-{action_id}",
    )
    return {"status": "accepted", "command_id": command.command_id, "action": request.action}


def _sandbox_store_context(
    run_id: str,
    *,
    require_live_fence: bool = False,
) -> tuple[Configuration, int, dict[str, Any]]:
    """Resolve store configuration and current fence after RBAC authorization."""
    record = _runs.get(run_id)
    if record is not None:
        if require_live_fence and record.status in _TERMINAL_RUN_STATUSES:
            raise HTTPException(status_code=409, detail="stale_fence")
        token = record.engine.run_fence_token
        if token is None:
            raise HTTPException(status_code=409, detail="run_not_active")
        return (
            Configuration.from_runnable_config(record.engine.config),
            int(token),
            record.engine.config,
        )
    if require_live_fence:
        raise HTTPException(status_code=409, detail="stale_fence")
    configurable = Configuration.from_runnable_config(None)
    try:
        manifest = RunContextStore(run_id, runs_dir=configurable.runs_dir).load_manifest()
    except Exception as exc:
        raise HTTPException(status_code=404, detail="Run not found") from exc
    if not manifest.fence_token:
        raise HTTPException(status_code=409, detail="run_has_no_fence")
    config = {
        "configurable": {"runs_dir": configurable.runs_dir},
        "metadata": {"run_id": run_id, "run_fence_token": manifest.fence_token},
    }
    return configurable, int(manifest.fence_token), config


@app.get("/runs/{run_id}/security-approvals")
async def list_security_approvals(
    run_id: str,
    status: Literal["pending", "resolved", "expired", "consumed"] | None = "pending",
    user: Principal = Depends(
        require_run_owner_or_any(
            RESEARCH_SECURITY_APPROVAL_READ_OWN.code,
            RESEARCH_SECURITY_APPROVAL_READ_ANY.code,
        )
    ),
) -> dict[str, Any]:
    """List the caller-authorized run's durable sandbox approval queue."""
    del user
    configurable, _fence_token, _config = _sandbox_store_context(run_id)
    version, approvals = await asyncio.to_thread(
        SecurityApprovalStore(run_id, runs_dir=configurable.runs_dir).list,
        status=status,
    )
    return {
        "run_id": run_id,
        "version": version,
        "approvals": [approval.model_dump(mode="json") for approval in approvals],
    }


@app.post("/runs/{run_id}/security-approvals/{approval_id}")
async def resolve_security_approval(
    run_id: str,
    approval_id: str,
    request: SecurityApprovalDecisionRequest,
    user: Principal = Depends(
        require_run_owner_or_any(
            RESEARCH_SECURITY_APPROVAL_RESOLVE_OWN.code,
            RESEARCH_SECURITY_APPROVAL_RESOLVE_ANY.code,
        )
    ),
) -> dict[str, Any]:
    """Resolve one approval for exactly the live run ownership epoch."""
    configurable, fence_token, config = _sandbox_store_context(
        run_id,
        require_live_fence=True,
    )
    try:
        approval = await asyncio.to_thread(
            SecurityApprovalStore(run_id, runs_dir=configurable.runs_dir).resolve,
            approval_id,
            decision=request.decision,
            actor=user.user_id,
            reason=request.reason,
            expected_fence_token=fence_token,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="security_approval_not_found") from exc
    except ValueError as exc:
        status_code = 409 if str(exc) == "stale_fence" else 400
        raise HTTPException(status_code=status_code, detail=str(exc)) from exc
    await event_publisher_from_config(config).publish(
        "security.approval.resolved",
        stage="researching",
        payload={
            "approval_id": approval.approval_id,
            "task_id": approval.task_id,
            "kind": approval.kind,
            "capability": approval.capability,
            "decision": approval.decision,
            "status": approval.status,
        },
        dedupe_key=f"security-approval:{approval.approval_id}:resolved:{approval.version}",
    )
    task = get_task_registry().get(approval.task_id)
    if task is not None and task.run_id == run_id:
        _version, pending = await asyncio.to_thread(
            SecurityApprovalStore(run_id, runs_dir=configurable.runs_dir).list,
            status="pending",
        )
        task_pending = [item for item in pending if item.task_id == approval.task_id]
        if task_pending:
            task.pending_domain = str(
                task_pending[0].target.get("domain") or ""
            ) or None
            task.pending_domain_tool = task_pending[0].capability
        else:
            task.pending_domain = None
            task.pending_domain_tool = None
        if (
            not task_pending
            and task.status == TaskStatus.WAITING_FOR_CONFIRMATION
        ):
            get_task_registry().update_status(approval.task_id, TaskStatus.RUNNING)
    return approval.model_dump(mode="json")


@app.post("/runs/{run_id}/feedback")
async def submit_feedback(
    run_id: str,
    request: HumanFeedbackRequest,
    user: Principal = Depends(require_permissions(RESEARCH_RUN_INTERACT_OWN.code)),
) -> dict[str, Any]:
    """Accept mid-run human direction or evidence questions."""
    record, configurable = _require_run_owner(run_id, user)
    payload = request.model_dump(exclude_none=True)
    if record is not None:
        try:
            return await record.engine.submit_feedback(payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    command = await RunControlStore(run_id, runs_dir=configurable.runs_dir).enqueue(
        "feedback",
        payload,
        command_id=request.command_id,
    )
    return {"status": "accepted", "command_id": command.command_id}


@app.get("/runs/{run_id}/tasks/{task_id}/activity")
async def get_task_activity(
    run_id: str,
    task_id: str,
    before: int | None = None,
    limit: int = 100,
    kind: str | None = None,
    user: Principal = Depends(require_permissions(RESEARCH_TASK_ACTIVITY_READ_OWN.code)),
) -> dict[str, Any]:
    """Return one reverse-page of safe task activity in chronological order."""
    _record, configurable = _require_run_owner(run_id, user)
    _require_task_in_run(run_id, task_id, configurable)
    if before is not None and before < 1:
        raise HTTPException(status_code=400, detail="invalid_activity_cursor")
    if not 1 <= limit <= 200:
        raise HTTPException(status_code=422, detail="activity_limit_out_of_range")
    valid_kinds = {
        "lifecycle", "model", "tool", "source", "quality", "checkpoint",
        "control", "security", "error",
    }
    if kind is not None and kind not in valid_kinds:
        raise HTTPException(status_code=400, detail="invalid_activity_kind")

    store = TaskActivityStore(run_id, task_id, runs_dir=configurable.runs_dir)
    source = "native"
    if store.exists:
        items, has_more, last_event_id = await asyncio.to_thread(
            store.page,
            before=before,
            limit=limit,
            kind=kind,
        )
    else:
        observed = _observability_store()
        run = observed.get_run(run_id, user_id=_user_identity(user))
        all_derived = derive_trace_activity(
            run_id,
            task_id,
            observed.list_spans(run_id) if run is not None else [],
        )
        last_event_id = all_derived[-1].sequence if all_derived else 0
        source = "derived_trace" if all_derived else "summary_only"
        derived = all_derived
        if kind is not None:
            derived = [event for event in derived if event.kind == kind]
        if before is not None:
            derived = [event for event in derived if event.sequence < before]
        has_more = len(derived) > limit
        items = derived[-limit:]
    return {
        "run_id": run_id,
        "task_id": task_id,
        "items": [event.public_dict() for event in items],
        "oldest_sequence": items[0].sequence if items else 0,
        "last_event_id": last_event_id,
        "has_more": has_more,
        "detail_level": "preview" if _task_activity_preview_allowed(user) else "summary",
        "source": source,
        "stream_url": f"/runs/{run_id}/tasks/{task_id}/activity/stream",
    }


@app.get("/runs/{run_id}/tasks/{task_id}/activity/stream")
async def stream_task_activity(
    run_id: str,
    task_id: str,
    after: int = 0,
    last_event_id: str | None = Header(default=None, alias="Last-Event-ID"),
    user: Principal = Depends(require_permissions(RESEARCH_TASK_ACTIVITY_READ_OWN.code)),
) -> StreamingResponse:
    """Replay and tail a task-local activity stream while the drawer is open."""
    record, configurable = _require_run_owner(run_id, user)
    _require_task_in_run(run_id, task_id, configurable)
    store = TaskActivityStore(run_id, task_id, runs_dir=configurable.runs_dir)
    cursor = after
    if last_event_id is not None:
        try:
            cursor = int(last_event_id)
        except ValueError:
            raise HTTPException(status_code=400, detail="invalid_activity_cursor") from None
    if cursor < 0:
        raise HTTPException(status_code=400, detail="invalid_activity_cursor")
    current = await asyncio.to_thread(store.last_sequence)
    if cursor > current:
        raise HTTPException(status_code=409, detail="activity_cursor_ahead")
    if not store.exists and (
        record is None or record.status in {"completed", "failed", "cancelled"}
    ):
        raise HTTPException(status_code=409, detail="activity_stream_unavailable_legacy_run")
    release_token = await _reserve_sse_connection(user, configurable)
    return StreamingResponse(
        _limited_sse(
            _task_activity_iterator(store, after=cursor, principal=user),
            release_token,
        ),
        media_type="text/event-stream",
        headers=_sse_headers(),
    )


@app.get("/runs/{run_id}/events")
async def stream_run_events(
    run_id: str,
    after: int = 0,
    last_event_id: str | None = Header(default=None, alias="Last-Event-ID"),
    user: Principal = Depends(require_permissions(RESEARCH_RUN_READ_OWN.code)),
) -> StreamingResponse:
    """Replay and tail the durable public event stream for a run."""
    record = _runs.get(run_id)
    if record is not None:
        _require_record_owner(record, user)
        configurable = Configuration.from_runnable_config(record.engine.config)
    else:
        configurable = Configuration.from_runnable_config(None)
        try:
            manifest = RunContextStore(run_id, runs_dir=configurable.runs_dir).load_manifest()
        except (ValueError, JournalCorruptedError, OSError):
            raise HTTPException(status_code=404, detail="Run not found") from None
        if not manifest.owner_id or manifest.owner_id != _user_identity(user):
            raise HTTPException(status_code=404, detail="Run not found")

    store = RunEventStore(run_id, runs_dir=configurable.runs_dir)
    if not store.exists:
        raise HTTPException(status_code=409, detail="event_stream_unavailable_legacy_run")
    cursor = after
    if last_event_id is not None:
        try:
            cursor = int(last_event_id)
        except ValueError:
            raise HTTPException(status_code=400, detail="invalid_event_cursor") from None
    if cursor < 0:
        raise HTTPException(status_code=400, detail="invalid_event_cursor")
    current = await asyncio.to_thread(store.last_sequence)
    if cursor > current:
        raise HTTPException(status_code=409, detail="event_cursor_ahead")
    release_token = await _reserve_sse_connection(user, configurable)
    return StreamingResponse(
        _limited_sse(
            _public_event_iterator(store, after=cursor, principal=user),
            release_token,
        ),
        media_type="text/event-stream",
        headers=_sse_headers(),
    )


def _unavailable_usage_response(
    run_id: str,
    *,
    status: str = "unknown",
    configurable: Configuration,
    reason: str = "storage_unavailable",
) -> dict[str, Any]:
    vector = {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "cached_input_tokens": 0,
        "cache_creation_input_tokens": 0,
        "reasoning_tokens": 0,
    }
    limits = {
        "input_tokens": configurable.max_run_input_tokens,
        "output_tokens": configurable.max_run_output_tokens,
        "model_calls": configurable.max_run_model_calls,
        "cost_micro_usd": configurable.max_run_cost_micro_usd,
    }
    return {
        "schema_version": 1,
        "run_id": run_id,
        "status": status,
        "duration_ms": None,
        "revision": 0,
        "updated_at": None,
        "accounting_status": "unavailable",
        "unavailable_reason": reason,
        "totals": {
            "reported": dict(vector),
            "estimated": dict(vector),
            "calls": {
                "attempts": 0,
                "successful_responses": 0,
                "provider_reported": 0,
                "provider_partial": 0,
                "estimated": 0,
                "missing": 0,
                "unknown_failed_attempts": 0,
                "legacy_unclassified": 0,
                "coverage_ratio": 0.0,
            },
            "cost": {
                "estimated_cost_micro_usd": None,
                "cost_source": "unavailable",
                "price_table_hash": None,
            },
            "budgets": {
                key: {
                    "settled": None if key == "cost_micro_usd" else 0,
                    "estimated": 0,
                    "reserved": 0,
                    "limit": limit,
                }
                for key, limit in limits.items()
            },
        },
        "breakdowns": {
            "by_stage": [],
            "by_agent_role": [],
            "by_model": [],
            "by_task": [],
        },
        "timeline": [],
        "operations": {
            "llm_call_count": 0,
            "retry_count": 0,
            "rate_limited_count": 0,
            "rate_429": 0.0,
            "cache_hit_rate": 0.0,
            "cache_input_ratio": 0.0,
            "reasoning_output_ratio": 0.0,
            "output_tokens_per_second": 0.0,
            "tool_call_count": 0,
            "tool_success_rate": 0.0,
            "empty_tool_result_count": 0,
            "zero_source_search_count": 0,
        },
    }


def _outstanding_usage_budget(
    configurable: Configuration,
    run_id: str,
) -> dict[str, int]:
    try:
        return RunBudgetLedger(
            run_id,
            runs_dir=configurable.runs_dir,
        ).outstanding_by_dimension()
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError):
        logger.warning(
            "Token budget ledger unavailable for run %s",
            run_id,
            exc_info=True,
        )
        return {}


def _load_run_usage_response(
    run_id: str,
    *,
    status: str,
    configurable: Configuration,
    duration_ms: int | None = None,
) -> dict[str, Any]:
    if not configurable.token_usage_accounting_enabled:
        response = _unavailable_usage_response(
            run_id,
            status=status,
            configurable=configurable,
            reason="accounting_disabled",
        )
        response["duration_ms"] = duration_ms
        return response
    try:
        store = SQLiteTraceStore(configurable.trace_store_path)
        reserved = _outstanding_usage_budget(configurable, run_id)
        response = store.get_usage_accounting(
            run_id,
            reserved_budget=reserved,
        )
    except (OSError, RuntimeError, sqlite3.Error):
        logger.warning(
            "Token accounting storage unavailable for run %s",
            run_id,
            exc_info=True,
        )
        response = _unavailable_usage_response(
            run_id,
            status=status,
            configurable=configurable,
            reason="storage_unavailable",
        )
    if response["status"] == "unknown":
        response["status"] = status
    if not response.get("duration_ms"):
        response["duration_ms"] = duration_ms
    return response


@app.get("/runs/{run_id}/usage")
async def get_run_usage_accounting(
    run_id: str,
    user: Principal = Depends(require_permissions(RESEARCH_RUN_READ_OWN.code)),
) -> dict[str, Any]:
    """Return content-free token accounting for one owned research run."""
    record, configurable = _require_run_owner(run_id, user)
    manifest = None
    with contextlib.suppress(ValueError, JournalCorruptedError, OSError):
        manifest = RunContextStore(
            run_id,
            runs_dir=configurable.runs_dir,
        ).load_manifest()
    manifest_status = manifest.status if manifest is not None else None
    if record is None:
        status = manifest_status or "unknown"
    elif (
        manifest_status in {"completed", "failed", "cancelled", "interrupted"}
        and record.status != manifest_status
    ):
        # The durable manifest is authoritative once terminal; an in-memory
        # record can lag behind a just-finished run.
        status = manifest_status
    else:
        status = record.status
    duration_ms = (
        max(0, int((manifest.updated_at - manifest.created_at) * 1000))
        if manifest is not None and status in {
            "completed", "failed", "cancelled", "interrupted",
        }
        else None
    )
    return await asyncio.to_thread(
        _load_run_usage_response,
        run_id,
        status=status,
        configurable=configurable,
        duration_ms=duration_ms,
    )


def _analytics_cursor_offset(cursor: str | None) -> int:
    if not cursor:
        return 0
    try:
        return max(0, int(base64.urlsafe_b64decode(cursor.encode()).decode()))
    except (ValueError, UnicodeDecodeError, base64.binascii.Error):
        raise HTTPException(status_code=400, detail="invalid_cursor") from None


def _sum_accounting_vectors(
    reports: list[dict[str, Any]], track: Literal["reported", "estimated"]
) -> dict[str, int]:
    keys = (
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "cached_input_tokens",
        "cache_creation_input_tokens",
        "reasoning_tokens",
    )
    return {
        key: sum(int(report["totals"][track].get(key) or 0) for report in reports)
        for key in keys
    }


def _normalize_analytics_status(status: str | None) -> str | None:
    aliases = {
        "completed": "success",
        "failed": "error",
    }
    normalized = (status or "").strip().lower()
    return aliases.get(normalized, normalized) or None


def _usage_manifest_title(
    run_id: str,
    *,
    configurable: Configuration,
    fallback: str,
) -> str:
    """Read a display title from the content store, never the usage database."""
    try:
        manifest = RunContextStore(
            run_id,
            runs_dir=configurable.runs_dir,
        ).load_manifest()
    except (ValueError, JournalCorruptedError, OSError):
        return fallback
    return str(manifest.title or fallback)


def _build_usage_analytics_response(
    *,
    range_name: Literal["7d", "30d", "retained"],
    status: str | None,
    provider: str | None,
    model: str | None,
    query: str | None,
    timezone_name: str,
    timezone_info: ZoneInfo,
    limit: int,
    offset: int,
    user_id: str,
) -> dict[str, Any]:
    """Build historical usage off the event loop using owner-filtered SQL."""
    configurable = Configuration.from_runnable_config(None)
    retention_days = float(
        configurable.run_retention_days
        if configurable.trace_retention_days is None
        else configurable.trace_retention_days
    )
    unlimited_retention = retention_days <= 0
    now = time.time()
    if range_name == "retained":
        actual_days = 0.0 if unlimited_retention else retention_days
        cutoff = None if unlimited_retention else now - actual_days * 86400
    else:
        requested_days = 7.0 if range_name == "7d" else 30.0
        actual_days = (
            requested_days
            if unlimited_retention
            else min(requested_days, retention_days)
        )
        cutoff = now - max(0.0, actual_days) * 86400

    store = SQLiteTraceStore(configurable.trace_store_path)
    owned_runs = store.list_runs_for_usage(
        user_id=user_id,
        cutoff=cutoff,
        status=_normalize_analytics_status(status),
    )
    reports: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []
    selected_runs: list[tuple[dict[str, Any], str]] = []
    normalized_query = (query or "").strip().lower()
    for run in owned_runs:
        title = str(run["run_id"])
        try:
            metadata = json.loads(run.get("metadata_json") or "{}")
        except (TypeError, json.JSONDecodeError):
            metadata = {}
        title = str(metadata.get("title") or title)
        if normalized_query:
            if normalized_query not in f"{title} {run['run_id']}".lower():
                title = _usage_manifest_title(
                    str(run["run_id"]),
                    configurable=configurable,
                    fallback=title,
                )
            if normalized_query not in f"{title} {run['run_id']}".lower():
                continue
        selected_runs.append((run, title))

    reports_by_run = store.get_usage_accounting_many(
        [str(run["run_id"]) for run, _title in selected_runs],
        provider=provider,
        model=model,
    )
    for run, title in selected_runs:
        report = reports_by_run[str(run["run_id"])]
        if (provider or model) and not report["totals"]["calls"]["attempts"]:
            continue
        reports.append(report)
        run_rows.append(
            {
                "run_id": run["run_id"],
                "title": title,
                "status": run.get("status"),
                "started_at": run.get("started_at"),
                "ended_at": run.get("ended_at"),
                "duration_ms": run.get("duration_ms"),
                "accounting_status": report["accounting_status"],
                "reported": report["totals"]["reported"],
                "estimated": report["totals"]["estimated"],
                "calls": report["totals"]["calls"],
                "cost": report["totals"]["cost"],
                "operations": report["operations"],
            }
        )

    if unlimited_retention and range_name == "retained" and run_rows:
        oldest = min(float(row["started_at"] or now) for row in run_rows)
        actual_days = max(0.0, (now - oldest) / 86400)

    distributions: dict[str, dict[str, dict[str, Any]]] = {
        "provider": {},
        "model": {},
        "status": {},
    }
    daily: dict[str, dict[str, Any]] = {}
    for report, row in zip(reports, run_rows, strict=True):
        date_key = datetime.fromtimestamp(
            float(row["started_at"]),
            timezone_info,
        ).date().isoformat()
        day = daily.setdefault(
            date_key,
            {
                "date": date_key,
                "reported_tokens": 0,
                "estimated_tokens": 0,
                "run_count": 0,
                "provider_reported_responses": 0,
                "successful_responses": 0,
                "rate_429_sum": 0.0,
                "throughput_sum": 0.0,
                "cache_hit_rate_sum": 0.0,
            },
        )
        day["reported_tokens"] += int(row["reported"]["total_tokens"])
        day["estimated_tokens"] += int(row["estimated"]["total_tokens"])
        day["run_count"] += 1
        day["provider_reported_responses"] += int(
            row["calls"]["provider_reported"]
        )
        day["successful_responses"] += int(row["calls"]["successful_responses"])
        day["rate_429_sum"] += float(row["operations"]["rate_429"])
        day["throughput_sum"] += float(
            row["operations"]["output_tokens_per_second"]
        )
        day["cache_hit_rate_sum"] += float(row["operations"]["cache_hit_rate"])
        status_key = str(row["status"] or "unknown")
        status_bucket = distributions["status"].setdefault(
            status_key,
            {
                "key": status_key,
                "reported_tokens": 0,
                "estimated_tokens": 0,
                "run_count": 0,
            },
        )
        status_bucket["reported_tokens"] += int(row["reported"]["total_tokens"])
        status_bucket["estimated_tokens"] += int(row["estimated"]["total_tokens"])
        status_bucket["run_count"] += 1
        for bucket in report["breakdowns"]["by_model"]:
            full_key = str(bucket["key"])
            provider_key, _, model_key = full_key.partition(":")
            if not model_key:
                model_key = provider_key
                provider_key = "unknown"
            for dimension, key in (
                ("provider", provider_key),
                ("model", model_key),
            ):
                target = distributions[dimension].setdefault(
                    key,
                    {
                        "key": key,
                        "reported_tokens": 0,
                        "estimated_tokens": 0,
                        "call_count": 0,
                    },
                )
                target["reported_tokens"] += int(
                    bucket["reported"]["total_tokens"]
                )
                target["estimated_tokens"] += int(
                    bucket["estimated"]["total_tokens"]
                )
                target["call_count"] += int(bucket["call_count"])

    daily_rows: list[dict[str, Any]] = []
    for day in sorted(daily.values(), key=lambda item: item["date"]):
        count = max(1, int(day["run_count"]))
        successful = int(day.pop("successful_responses"))
        provider_reported = int(day.pop("provider_reported_responses"))
        day["coverage_ratio"] = (
            provider_reported / successful if successful else 0.0
        )
        day["rate_429"] = float(day.pop("rate_429_sum")) / count
        day["output_tokens_per_second"] = (
            float(day.pop("throughput_sum")) / count
        )
        day["cache_hit_rate"] = float(day.pop("cache_hit_rate_sum")) / count
        daily_rows.append(day)

    page = run_rows[offset : offset + limit]
    for item in page:
        item["title"] = _usage_manifest_title(
            str(item["run_id"]),
            configurable=configurable,
            fallback=str(item["title"]),
        )
    next_offset = offset + len(page)
    next_cursor = (
        base64.urlsafe_b64encode(str(next_offset).encode()).decode()
        if next_offset < len(run_rows)
        else None
    )
    costs = [
        row["cost"]["estimated_cost_micro_usd"]
        for row in run_rows
        if row["cost"]["estimated_cost_micro_usd"] is not None
    ]
    successful = sum(int(row["calls"]["successful_responses"]) for row in run_rows)
    provider_reported = sum(
        int(row["calls"]["provider_reported"]) for row in run_rows
    )
    return {
        "schema_version": 1,
        "range": range_name,
        "timezone": timezone_name,
        "retention_days": retention_days,
        "actual_range_days": actual_days,
        "summary": {
            "run_count": len(run_rows),
            "reported": _sum_accounting_vectors(reports, "reported"),
            "estimated": _sum_accounting_vectors(reports, "estimated"),
            "estimated_cost_micro_usd": (
                sum(int(value) for value in costs) if costs else None
            ),
            "coverage_ratio": (
                provider_reported / successful if successful else 0.0
            ),
        },
        "daily": daily_rows,
        "distributions": {
            key: list(value.values()) for key, value in distributions.items()
        },
        "runs": page,
        "next_cursor": next_cursor,
    }


@app.get("/usage/analytics")
async def get_usage_analytics(
    range: Literal["7d", "30d", "retained"] = "30d",  # noqa: A002
    status: str | None = None,
    provider: str | None = None,
    model: str | None = None,
    query: str | None = None,
    timezone: str = "Asia/Shanghai",
    limit: int = 50,
    cursor: str | None = None,
    user: Principal = Depends(require_permissions(RESEARCH_RUN_READ_OWN.code)),
) -> dict[str, Any]:
    """Aggregate retained token usage for the current run owner only."""
    if limit < 1 or limit > 100:
        raise HTTPException(status_code=422, detail="limit must be between 1 and 100")
    try:
        timezone_info = ZoneInfo(timezone)
    except ZoneInfoNotFoundError:
        raise HTTPException(status_code=422, detail="invalid timezone") from None
    offset = _analytics_cursor_offset(cursor)
    return await asyncio.to_thread(
        _build_usage_analytics_response,
        range_name=range,
        status=status,
        provider=provider,
        model=model,
        query=query,
        timezone_name=timezone,
        timezone_info=timezone_info,
        limit=limit,
        offset=offset,
        user_id=_user_identity(user),
    )


@app.get("/observability/runs")
async def list_observed_runs(
    limit: int = 100,
    user: Principal = Depends(require_permissions(RESEARCH_OBSERVABILITY_READ_OWN.code)),
) -> dict[str, Any]:
    """Return persisted observed run summaries."""
    store = _observability_store()
    return {"runs": store.list_runs(limit=limit, user_id=_user_identity(user))}


@app.get("/observability/runs/{run_id}")
async def get_observed_run(
    run_id: str,
    user: Principal = Depends(require_permissions(RESEARCH_OBSERVABILITY_READ_OWN.code)),
) -> dict[str, Any]:
    """Return one persisted observed run summary."""
    store = _observability_store()
    run = store.get_run(run_id, user_id=_user_identity(user))
    if run is None:
        raise HTTPException(status_code=404, detail="Observed run not found")
    return {"run": run}


@app.get("/observability/runs/{run_id}/spans")
async def get_observed_run_spans(
    run_id: str,
    user: Principal = Depends(require_permissions(RESEARCH_OBSERVABILITY_READ_OWN.code)),
) -> dict[str, Any]:
    """Return ordered spans for a persisted observed run."""
    store = _observability_store()
    if store.get_run(run_id, user_id=_user_identity(user)) is None:
        raise HTTPException(status_code=404, detail="Observed run not found")
    return {"run_id": run_id, "spans": store.list_spans(run_id)}


@app.get("/observability/runs/{run_id}/usage")
async def get_observed_run_usage(
    run_id: str,
    user: Principal = Depends(require_permissions(RESEARCH_OBSERVABILITY_READ_OWN.code)),
) -> dict[str, Any]:
    """Return token usage aggregate for a persisted observed run."""
    store = _observability_store()
    if store.get_run(run_id, user_id=_user_identity(user)) is None:
        raise HTTPException(status_code=404, detail="Observed run not found")
    return {"run_id": run_id, "usage": store.get_usage(run_id)}


@app.get("/observability/runs/{run_id}/metrics")
async def get_observed_run_metrics(
    run_id: str,
    user: Principal = Depends(require_permissions(RESEARCH_OBSERVABILITY_READ_OWN.code)),
) -> dict[str, Any]:
    """Return token usage, retry counts, and 429 rate for a persisted observed run."""
    store = _observability_store()
    if store.get_run(run_id, user_id=_user_identity(user)) is None:
        raise HTTPException(status_code=404, detail="Observed run not found")
    return {"run_id": run_id, "metrics": store.get_metrics(run_id)}


@app.get("/observability/ui", response_class=HTMLResponse)
async def observability_ui(
    run_id: str | None = None,
    user: Principal = Depends(require_permissions(RESEARCH_OBSERVABILITY_READ_OWN.code)),
) -> HTMLResponse:
    """Render a small server-side observability page."""
    store = _observability_store()
    identity = _user_identity(user)
    runs = store.list_runs(limit=50, user_id=identity)
    selected = run_id or (runs[0]["run_id"] if runs else None)
    selected_run = store.get_run(selected, user_id=identity) if selected else None
    if selected and selected_run is None:
        raise HTTPException(status_code=404, detail="Observed run not found")
    spans = store.list_spans(selected) if selected else []
    run_links = "".join(
        "<li><a href='/observability/ui?run_id="
        + html.escape(str(run["run_id"]))
        + "'>"
        + html.escape(str(run["run_id"]))
        + "</a> "
        + html.escape(str(run.get("status") or ""))
        + " "
        + str(run.get("total_tokens") or 0)
        + " tokens</li>"
        for run in runs
    )
    usage = store.get_usage(selected) if selected else {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    metrics = store.get_metrics(selected) if selected else {
        "retry_count": 0,
        "rate_limited_count": 0,
        "rate_429": 0.0,
        "total_llm_tool_calls": 0,
        "cache_hit_rate": 0.0,
        "tool_success_rate": 0.0,
    }
    run_title = html.escape(str(selected or "No observed runs"))
    run_status = html.escape(str((selected_run or {}).get("status") or ""))
    rows = _span_tree_rows(spans)
    rate_pct = f"{(metrics.get('rate_429') or 0) * 100:.1f}%"
    cache_hit_pct = f"{(metrics.get('cache_hit_rate') or 0) * 100:.1f}%"
    cache_input_pct = f"{(metrics.get('cache_input_ratio') or 0) * 100:.1f}%"
    tool_success_pct = f"{(metrics.get('tool_success_rate') or 0) * 100:.1f}%"
    body = f"""
    <!doctype html>
    <html>
    <head>
      <meta charset='utf-8'>
      <title>Open Deep Research Observability</title>
      <style>
        body {{ font-family: system-ui, sans-serif; margin: 24px; color: #1f2937; }}
        main {{ display: grid; grid-template-columns: 320px 1fr; gap: 24px; }}
        a {{ color: #0f766e; text-decoration: none; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 16px; }}
        th, td {{ border-bottom: 1px solid #e5e7eb; padding: 8px; text-align: left; vertical-align: top; }}
        th {{ background: #f8fafc; }}
        .metric {{ display: inline-block; margin-right: 16px; padding: 8px 10px; background: #f8fafc; border: 1px solid #e5e7eb; border-radius: 6px; }}
        .error td {{ background: #fef2f2; }}
        .ok td {{ background: #ffffff; }}
        aside {{ border-right: 1px solid #e5e7eb; padding-right: 16px; }}
        ul {{ padding-left: 18px; }}
      </style>
    </head>
    <body>
      <h1>Open Deep Research Observability</h1>
      <main>
        <aside>
          <h2>Runs</h2>
          <ul>{run_links}</ul>
        </aside>
        <section>
          <h2>{run_title}</h2>
          <div class='metric'>status: {run_status}</div>
          <div class='metric'>input: {usage['input_tokens']}</div>
          <div class='metric'>output: {usage['output_tokens']}</div>
          <div class='metric'>total: {usage['total_tokens']}</div>
          <div class='metric'>cached input: {usage.get('cached_input_tokens', 0)}</div>
          <div class='metric'>reasoning: {usage.get('reasoning_tokens', 0)}</div>
          <div class='metric'>estimated cost: ${usage.get('estimated_cost_usd', 0):.6f}</div>
          <div class='metric'>attempts: {metrics.get('attempt_count', 0)}</div>
          <div class='metric'>retries: {metrics.get('retry_count', 0)}</div>
          <div class='metric'>429 call rate: {rate_pct} ({metrics.get('rate_limited_count', 0)}/{metrics.get('total_llm_tool_calls', 0)} calls; {metrics.get('rate_limit_events', 0)} events)</div>
          <div class='metric'>cache hit: {cache_hit_pct} ({metrics.get('cache_hit_count', 0)}/{metrics.get('cache_eligible_count', 0)} calls)</div>
          <div class='metric'>cached input ratio: {cache_input_pct}</div>
          <div class='metric'>output throughput: {metrics.get('llm_output_tokens_per_second', 0):.1f} token/s</div>
          <div class='metric'>tool success: {tool_success_pct} ({metrics.get('tool_success_count', 0)}/{metrics.get('tool_call_count', 0)})</div>
          <div class='metric'>empty tool results: {metrics.get('empty_tool_result_count', 0)}</div>
          <div class='metric'>zero-source searches: {metrics.get('zero_source_search_count', 0)}</div>
          <table>
            <thead><tr><th>Span</th><th>Kind</th><th>Status</th><th>Duration ms</th><th>Tokens</th><th>Retries</th><th>Error type</th><th>Error</th></tr></thead>
            <tbody>{rows}</tbody>
          </table>
        </section>
      </main>
    </body>
    </html>
    """
    return HTMLResponse(body)


@app.post("/runs/{run_id}/cancel")
async def cancel_run(
    run_id: str,
    user: Principal = Depends(require_permissions(RESEARCH_RUN_CONTROL_OWN.code)),
) -> dict[str, str]:
    """Cancel a background run."""
    record, configurable = _require_run_owner(run_id, user)
    terminal_statuses = {"completed", "failed", "cancelled"}
    if record is not None:
        if record.status in terminal_statuses:
            return {"run_id": run_id, "status": record.status}
        record.engine.interrupt()
        record.status = "cancelling"
    else:
        manifest = RunContextStore(
            run_id,
            runs_dir=configurable.runs_dir,
        ).load_manifest()
        if manifest.status in terminal_statuses:
            return {"run_id": run_id, "status": manifest.status}
        await RunControlStore(run_id, runs_dir=configurable.runs_dir).enqueue(
            "cancel",
            {},
            command_id=f"cancel-{run_id}",
        )
    logger.info(
        "run cancellation requested",
        extra={"actor": _user_identity(user), "action": "run.cancel", "run_id": run_id},
    )
    return {"run_id": run_id, "status": "cancelling"}


async def _cancel_before_forced_purge(
    run_id: str,
    record: RunRecord | None,
    configurable: Configuration,
) -> None:
    """Use the existing interrupt/control mechanisms before destructive purge."""
    if record is not None:
        record.engine.interrupt()
        if record.task is not None and not record.task.done():
            record.task.cancel()
            await asyncio.gather(record.task, return_exceptions=True)
        record.status = "cancelled"
        record.finished_at = time.time()
        store = getattr(record.engine, "context_store", None)
        if store is not None and store.manifest_path.exists():
            with contextlib.suppress(Exception):
                await asyncio.to_thread(
                    store._update_manifest,  # noqa: SLF001
                    status="cancelled",
                )
        return
    await RunControlStore(run_id, runs_dir=configurable.runs_dir).enqueue(
        "cancel",
        {},
        command_id=f"cancel-before-purge-{run_id}",
    )


@app.delete("/runs/{run_id}")
async def delete_run(
    run_id: str,
    force: bool = False,
    dry_run: bool = False,
    user: Principal = Depends(require_permissions(RESEARCH_RUN_CONTROL_OWN.code)),
) -> dict[str, Any]:
    """Permanently delete an owned run and all durable observability rows."""
    is_admin = "admin" in user.roles
    if is_admin:
        record = _runs.get(run_id)
        configurable = Configuration.from_runnable_config(
            getattr(record.engine, "config", None) if record is not None else None
        )
        target = _run_directory(configurable, run_id)
        trace_exists = await asyncio.to_thread(
            SQLiteTraceStore(configurable.trace_store_path).get_run,
            run_id,
        )
        if record is None and not target.exists() and trace_exists is None:
            raise HTTPException(status_code=404, detail="Run not found")
    else:
        record, configurable = _require_run_owner(run_id, user)
        target = _run_directory(configurable, run_id)

    manifest = None
    if target.exists():
        with contextlib.suppress(JournalCorruptedError, OSError, ValueError):
            manifest = await asyncio.to_thread(
                RunContextStore(run_id, runs_dir=configurable.runs_dir).load_manifest
            )
    status = record.status if record is not None else getattr(manifest, "status", None)
    if status not in _TERMINAL_RUN_STATUSES and not force:
        raise HTTPException(status_code=409, detail="run_is_active")
    if dry_run:
        return {
            "run_id": run_id,
            "status": "would_delete",
            "run_status": status,
            "force": force,
            "directory": str(target),
            "trace_store": configurable.trace_store_path,
        }
    if status not in _TERMINAL_RUN_STATUSES:
        await _cancel_before_forced_purge(run_id, record, configurable)
    return await _purge_run_artifacts(
        run_id,
        configurable,
        reason="manual",
        actor=_user_identity(user),
        require_terminal=not force,
    )
