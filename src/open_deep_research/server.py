"""FastAPI service entrypoint for the LangGraph-free runtime."""

from __future__ import annotations

import asyncio
import html
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Literal

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

from open_deep_research.agents.query_engine import QueryEngine
from open_deep_research.configuration import Configuration
from open_deep_research.observability import SQLiteTraceStore
from open_deep_research.public_events import (
    PublicEvent,
    RunEventStore,
    event_publisher_from_config,
    is_terminal_event,
)
from open_deep_research.run_context import (
    JournalCorruptedError,
    RunContextError,
    RunContextStore,
)
from open_deep_research.run_control import RunControlStore
from open_deep_research.security.inputs import (
    validate_http_configurable,
    validate_http_metadata,
)
from security.auth import apply_user_to_config, get_current_user


class RunRequest(BaseModel):
    """HTTP request body for a research run."""

    messages: list[dict[str, Any]]
    configurable: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ResumeRunRequest(BaseModel):
    """Runtime overrides and credentials for explicitly resuming a run."""

    configurable: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class HumanActionRequest(BaseModel):
    """Approval/revision/cancellation response for a pending HITL action."""

    action: Literal["approve", "revise", "cancel"]
    message: str | None = None


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
    events: list[dict[str, Any]] = field(default_factory=list)
    result: dict[str, Any] | None = None
    task: asyncio.Task | None = None


app = FastAPI(title="Open Deep Research", version="0.1.0")
logger = logging.getLogger(__name__)
_runs: dict[str, RunRecord] = {}
_metrics_path = Configuration.from_runnable_config(None).prometheus_metrics_path


@app.get(_metrics_path, include_in_schema=False)
async def prometheus_metrics() -> Response:
    """Expose process-wide aggregate metrics for Prometheus scraping."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


def _sse(event: PublicEvent) -> str:
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


async def _public_event_iterator(store: RunEventStore, *, after: int = 0):
    """Replay then tail a durable public stream, including cross-worker writes."""
    configurable = Configuration.from_runnable_config(None)
    poll_seconds = configurable.sse_poll_interval_ms / 1000
    heartbeat_seconds = configurable.sse_heartbeat_seconds
    cursor = after
    last_output_at = asyncio.get_running_loop().time()
    while True:
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


def _config_from_request(request: RunRequest, user: dict[str, Any]) -> dict[str, Any]:
    try:
        validate_http_configurable(request.configurable)
        validate_http_metadata(request.metadata)
    except ValueError as exc:
        logger.warning("security.unsafe_config_rejected: %s", exc)
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    config = {
        "configurable": dict(request.configurable),
        "metadata": {**request.metadata, "deployment_surface": "http"},
    }
    return apply_user_to_config(config, user)


def _observability_store() -> SQLiteTraceStore:
    configurable = Configuration.from_runnable_config(None)
    return SQLiteTraceStore(configurable.trace_store_path)


def _user_identity(user: dict[str, Any]) -> str | None:
    """Return the normalized authenticated identity used by run ownership checks."""
    value = user.get("identity") or user.get("id")
    return str(value) if value else None


def _require_record_owner(record: RunRecord, user: dict[str, Any]) -> None:
    """Hide in-memory runs from users other than their owner."""
    metadata = getattr(record.engine, "config", {}).get("metadata", {})
    owner = metadata.get("owner") or metadata.get("user_id")
    if owner and str(owner) != _user_identity(user):
        raise HTTPException(status_code=404, detail="Run not found")


def _require_run_owner(run_id: str, user: dict[str, Any]) -> tuple[RunRecord | None, Configuration]:
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
    if manifest.owner_id and manifest.owner_id != _user_identity(user):
        raise HTTPException(status_code=404, detail="Run not found")
    return None, configurable


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


async def _run_background(record: RunRecord, request: RunRequest, config: dict[str, Any]) -> None:
    record.status = "running"
    control_task = asyncio.create_task(_run_control_listener(record, config))
    try:
        async for event in record.engine.stream_message(request.messages, config):
            record.events.append(event)
            status = event.get("data", {}).get("status")
            if status in {
                "running",
                "awaiting_plan_approval",
                "awaiting_outline_approval",
                "completed",
                "failed",
                "cancelled",
            }:
                record.status = status
        record.result = record.engine.final_state
        record.status = "cancelled" if record.engine.status == "cancelled" else "completed"
    except Exception as exc:  # noqa: BLE001 - surface in run state
        event = {"event": "run.failed", "data": {"run_id": record.run_id, "error": str(exc)}}
        record.events.append(event)
        record.result = {"result": {"status": "error", "error": str(exc)}}
        record.status = "failed"
        try:
            await event_publisher_from_config(config).publish(
                "run.failed",
                payload={"status": "failed", "error_code": "run_execution_failed", "message": "Research failed."},
                dedupe_key="run:failed",
            )
        except Exception:
            pass
    finally:
        control_task.cancel()
        await asyncio.gather(control_task, return_exceptions=True)


async def _run_resumed_background(record: RunRecord) -> None:
    """Continue a persisted Query run in the background."""
    record.status = "running"
    control_task = asyncio.create_task(_run_control_listener(record, record.engine.config))
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
            await event_publisher_from_config(record.engine.config).publish(
                "run.failed",
                payload={"status": "failed", "error_code": "run_execution_failed", "message": "Research failed."},
                dedupe_key="run:failed",
            )
        except Exception:
            pass
    finally:
        control_task.cancel()
        await asyncio.gather(control_task, return_exceptions=True)


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
                    record.status = "cancelled"
                    await publisher.publish(
                        "run.cancelled",
                        payload={"status": "cancelled"},
                        dedupe_key="run:cancelled",
                    )
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
                await store.ack(command)
        await asyncio.sleep(poll_seconds)


@app.post("/runs/stream")
async def stream_run(
    request: RunRequest,
    user: dict[str, Any] = Depends(get_current_user),
) -> StreamingResponse:
    """Run a research request and stream events with SSE."""
    config = _config_from_request(request, user)
    engine = QueryEngine(config)
    record = RunRecord(run_id=engine.run_id, engine=engine, status="running")
    await event_publisher_from_config(config).publish(
        "run.created",
        payload={"status": "pending"},
        dedupe_key="run:created",
    )
    record.task = asyncio.create_task(_run_background(record, request, config))
    _runs[record.run_id] = record
    store = RunEventStore(
        record.run_id,
        runs_dir=Configuration.from_runnable_config(config).runs_dir,
    )
    return StreamingResponse(
        _public_event_iterator(store),
        media_type="text/event-stream",
        headers=_sse_headers(),
    )


@app.post("/runs")
async def create_run(
    request: RunRequest,
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    """Create a background research run."""
    config = _config_from_request(request, user)
    engine = QueryEngine(config)
    record = RunRecord(run_id=engine.run_id, engine=engine, status="running")
    created = await event_publisher_from_config(config).publish(
        "run.created",
        payload={"status": "pending"},
        dedupe_key="run:created",
    )
    record.task = asyncio.create_task(_run_background(record, request, config))
    _runs[record.run_id] = record
    return {
        "run_id": record.run_id,
        "status": record.status,
        "events_url": f"/runs/{record.run_id}/events",
        "last_event_id": created.sequence,
    }


@app.get("/runs/{run_id}")
async def get_run(
    run_id: str,
    user: dict[str, Any] = Depends(get_current_user),
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
        if manifest.owner_id and manifest.owner_id != user.get("identity"):
            raise HTTPException(status_code=404, detail="Run not found")
        result = manifest.result
        if manifest.status == "completed":
            report_path = store.context_dir / "final_report.md"
            if report_path.exists():
                result = {"status": "success", "result": report_path.read_text(encoding="utf-8")}
        event_store = RunEventStore(run_id, runs_dir=configurable.runs_dir)
        projection = event_store.project() if event_store.exists else None
        return {
            "run_id": run_id,
            "status": manifest.status,
            "pending_human_action": None,
            "result": result,
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
    return {
        "run_id": run_id,
        "status": record.status,
        "pending_human_action": getattr(record.engine, "pending_human_action", None),
        "result": record.result,
        "event_count": projection.last_event_id,
        "progress": projection.model_dump(),
        "events_url": f"/runs/{run_id}/events",
        "last_event_id": projection.last_event_id,
    }


@app.post("/runs/{run_id}/resume", status_code=202)
async def resume_run(
    run_id: str,
    request: ResumeRunRequest,
    user: dict[str, Any] = Depends(get_current_user),
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
    config = apply_user_to_config(
        {
            "configurable": dict(request.configurable),
            "metadata": {
                **request.metadata,
                "run_id": run_id,
                "deployment_surface": "http",
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
    except (ValueError, RunContextError, OSError):
        raise HTTPException(status_code=409, detail="run_not_recoverable") from None
    if replay.manifest.owner_id and replay.manifest.owner_id != user.get("identity"):
        raise HTTPException(status_code=404, detail="Run not found")
    if replay.manifest.status == "completed":
        raise HTTPException(status_code=409, detail="run_already_completed")
    if replay.manifest.status == "cancelled" or replay.manifest.next_stage == "cancelled":
        raise HTTPException(status_code=409, detail="run_not_recoverable")

    record = RunRecord(run_id=run_id, engine=engine, status="running")
    record.task = asyncio.create_task(_run_resumed_background(record))
    _runs[run_id] = record
    return {"run_id": run_id, "status": "running"}


@app.post("/runs/{run_id}/human-actions/{action_id}")
async def submit_human_action(
    run_id: str,
    action_id: str,
    request: HumanActionRequest,
    user: dict[str, Any] = Depends(get_current_user),
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
    command = await RunControlStore(run_id, runs_dir=configurable.runs_dir).enqueue(
        "human_action",
        {"action_id": action_id, "action": request.action, "message": request.message or ""},
        command_id=f"human-action-{action_id}",
    )
    return {"status": "accepted", "command_id": command.command_id, "action": request.action}


@app.post("/runs/{run_id}/feedback")
async def submit_feedback(
    run_id: str,
    request: HumanFeedbackRequest,
    user: dict[str, Any] = Depends(get_current_user),
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


@app.get("/runs/{run_id}/events")
async def stream_run_events(
    run_id: str,
    after: int = 0,
    last_event_id: str | None = Header(default=None, alias="Last-Event-ID"),
    user: dict[str, Any] = Depends(get_current_user),
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
        if manifest.owner_id and manifest.owner_id != _user_identity(user):
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
    return StreamingResponse(
        _public_event_iterator(store, after=cursor),
        media_type="text/event-stream",
        headers=_sse_headers(),
    )


@app.get("/observability/runs")
async def list_observed_runs(
    limit: int = 100,
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    """Return persisted observed run summaries."""
    store = _observability_store()
    return {"runs": store.list_runs(limit=limit, user_id=_user_identity(user))}


@app.get("/observability/runs/{run_id}")
async def get_observed_run(
    run_id: str,
    user: dict[str, Any] = Depends(get_current_user),
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
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    """Return ordered spans for a persisted observed run."""
    store = _observability_store()
    if store.get_run(run_id, user_id=_user_identity(user)) is None:
        raise HTTPException(status_code=404, detail="Observed run not found")
    return {"run_id": run_id, "spans": store.list_spans(run_id)}


@app.get("/observability/runs/{run_id}/usage")
async def get_observed_run_usage(
    run_id: str,
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    """Return token usage aggregate for a persisted observed run."""
    store = _observability_store()
    if store.get_run(run_id, user_id=_user_identity(user)) is None:
        raise HTTPException(status_code=404, detail="Observed run not found")
    return {"run_id": run_id, "usage": store.get_usage(run_id)}


@app.get("/observability/runs/{run_id}/metrics")
async def get_observed_run_metrics(
    run_id: str,
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    """Return token usage, retry counts, and 429 rate for a persisted observed run."""
    store = _observability_store()
    if store.get_run(run_id, user_id=_user_identity(user)) is None:
        raise HTTPException(status_code=404, detail="Observed run not found")
    return {"run_id": run_id, "metrics": store.get_metrics(run_id)}


@app.get("/observability/ui", response_class=HTMLResponse)
async def observability_ui(
    run_id: str | None = None,
    user: dict[str, Any] = Depends(get_current_user),
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
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, str]:
    """Cancel a background run."""
    record, configurable = _require_run_owner(run_id, user)
    if record is not None:
        record.engine.interrupt()
        record.status = "cancelled"
    else:
        await RunControlStore(run_id, runs_dir=configurable.runs_dir).enqueue(
            "cancel",
            {},
            command_id=f"cancel-{run_id}",
        )
    config = record.engine.config if record is not None else {
        "configurable": {"runs_dir": configurable.runs_dir},
        "metadata": {"run_id": run_id},
    }
    await event_publisher_from_config(config).publish(
        "run.cancelled",
        payload={"status": "cancelled"},
        dedupe_key="run:cancelled",
    )
    try:
        RunContextStore(run_id, runs_dir=configurable.runs_dir)._update_manifest(status="cancelled")  # noqa: SLF001
    except Exception:
        pass
    return {"run_id": run_id, "status": "cancelled"}
