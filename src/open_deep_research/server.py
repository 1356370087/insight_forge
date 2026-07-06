"""FastAPI service entrypoint for the LangGraph-free runtime."""

from __future__ import annotations

import asyncio
import html
import json
from dataclasses import dataclass, field
from typing import Any, Literal

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

from open_deep_research.agents.query_engine import QueryEngine
from open_deep_research.configuration import Configuration
from open_deep_research.observability import SQLiteTraceStore
from security.auth import apply_user_to_config, get_current_user


class RunRequest(BaseModel):
    """HTTP request body for a research run."""

    messages: list[dict[str, Any]]
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
_runs: dict[str, RunRecord] = {}
_metrics_path = Configuration.from_runnable_config(None).prometheus_metrics_path


@app.get(_metrics_path, include_in_schema=False)
async def prometheus_metrics() -> Response:
    """Expose process-wide aggregate metrics for Prometheus scraping."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


def _sse(event: dict[str, Any]) -> str:
    return (
        f"event: {event.get('event', 'message')}\n"
        f"data: {json.dumps(event.get('data', {}), ensure_ascii=False, default=str)}\n\n"
    )


def _config_from_request(request: RunRequest, user: dict[str, Any]) -> dict[str, Any]:
    config = {
        "configurable": dict(request.configurable),
        "metadata": dict(request.metadata),
    }
    return apply_user_to_config(config, user)


def _observability_store() -> SQLiteTraceStore:
    configurable = Configuration.from_runnable_config(None)
    return SQLiteTraceStore(configurable.trace_store_path)


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


@app.post("/runs/stream")
async def stream_run(
    request: RunRequest,
    user: dict[str, Any] = Depends(get_current_user),
) -> StreamingResponse:
    """Run a research request and stream events with SSE."""
    config = _config_from_request(request, user)
    engine = QueryEngine(config)

    async def iterator():
        async for event in engine.stream_message(request.messages, config):
            yield _sse(event)

    return StreamingResponse(iterator(), media_type="text/event-stream")


@app.post("/runs")
async def create_run(
    request: RunRequest,
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, str]:
    """Create a background research run."""
    config = _config_from_request(request, user)
    engine = QueryEngine(config)
    record = RunRecord(run_id=engine.run_id, engine=engine)
    record.task = asyncio.create_task(_run_background(record, request, config))
    _runs[record.run_id] = record
    return {"run_id": record.run_id, "status": record.status}


@app.get("/runs/{run_id}")
async def get_run(
    run_id: str,
    _user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    """Return the latest status/result for a run."""
    record = _runs.get(run_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return {
        "run_id": run_id,
        "status": record.status,
        "pending_human_action": getattr(record.engine, "pending_human_action", None),
        "result": record.result,
        "event_count": len(record.events),
    }


@app.post("/runs/{run_id}/human-actions/{action_id}")
async def submit_human_action(
    run_id: str,
    action_id: str,
    request: HumanActionRequest,
    _user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    """Resolve a pending human approval, revision, or cancellation action."""
    record = _runs.get(run_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Run not found")
    try:
        result = record.engine.handle_human_action(action_id, request.action, request.message or "")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if request.action == "cancel":
        record.status = "cancelled"
    return result


@app.post("/runs/{run_id}/feedback")
async def submit_feedback(
    run_id: str,
    request: HumanFeedbackRequest,
    _user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    """Accept mid-run human direction or evidence questions."""
    record = _runs.get(run_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Run not found")
    try:
        result = await record.engine.submit_feedback(request.model_dump(exclude_none=True))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    event_name = "hitl.evidence_question_received" if request.type == "evidence_question" else "hitl.feedback_received"
    record.events.append({"event": event_name, "data": {"run_id": run_id, **result}})
    return result


@app.get("/runs/{run_id}/events")
async def stream_run_events(
    run_id: str,
    _user: dict[str, Any] = Depends(get_current_user),
) -> StreamingResponse:
    """Subscribe to stored and future events for a background run."""
    record = _runs.get(run_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Run not found")

    async def iterator():
        index = 0
        while True:
            while index < len(record.events):
                yield _sse(record.events[index])
                index += 1
            if record.status in {"completed", "failed", "cancelled"}:
                break
            await asyncio.sleep(0.25)

    return StreamingResponse(iterator(), media_type="text/event-stream")


@app.get("/observability/runs")
async def list_observed_runs(
    limit: int = 100,
    _user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    """Return persisted observed run summaries."""
    store = _observability_store()
    return {"runs": store.list_runs(limit=limit)}


@app.get("/observability/runs/{run_id}")
async def get_observed_run(
    run_id: str,
    _user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    """Return one persisted observed run summary."""
    store = _observability_store()
    run = store.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Observed run not found")
    return {"run": run}


@app.get("/observability/runs/{run_id}/spans")
async def get_observed_run_spans(
    run_id: str,
    _user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    """Return ordered spans for a persisted observed run."""
    store = _observability_store()
    if store.get_run(run_id) is None:
        raise HTTPException(status_code=404, detail="Observed run not found")
    return {"run_id": run_id, "spans": store.list_spans(run_id)}


@app.get("/observability/runs/{run_id}/usage")
async def get_observed_run_usage(
    run_id: str,
    _user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    """Return token usage aggregate for a persisted observed run."""
    store = _observability_store()
    if store.get_run(run_id) is None:
        raise HTTPException(status_code=404, detail="Observed run not found")
    return {"run_id": run_id, "usage": store.get_usage(run_id)}


@app.get("/observability/runs/{run_id}/metrics")
async def get_observed_run_metrics(
    run_id: str,
    _user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    """Return token usage, retry counts, and 429 rate for a persisted observed run."""
    store = _observability_store()
    if store.get_run(run_id) is None:
        raise HTTPException(status_code=404, detail="Observed run not found")
    return {"run_id": run_id, "metrics": store.get_metrics(run_id)}


@app.get("/observability/ui", response_class=HTMLResponse)
async def observability_ui(
    run_id: str | None = None,
    _user: dict[str, Any] = Depends(get_current_user),
) -> HTMLResponse:
    """Render a small server-side observability page."""
    store = _observability_store()
    runs = store.list_runs(limit=50)
    selected = run_id or (runs[0]["run_id"] if runs else None)
    selected_run = store.get_run(selected) if selected else None
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
        "retry_count": 0, "rate_limited_count": 0, "rate_429": 0.0, "total_llm_tool_calls": 0,
    }
    run_title = html.escape(str(selected or "No observed runs"))
    run_status = html.escape(str((selected_run or {}).get("status") or ""))
    rows = _span_tree_rows(spans)
    rate_pct = f"{(metrics.get('rate_429') or 0) * 100:.1f}%"
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
          <div class='metric'>retries: {metrics.get('retry_count', 0)}</div>
          <div class='metric'>429 rate: {rate_pct} ({metrics.get('rate_limited_count', 0)}/{metrics.get('total_llm_tool_calls', 0)})</div>
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
    _user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, str]:
    """Cancel a background run."""
    record = _runs.get(run_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Run not found")
    record.engine.interrupt()
    if record.task and not record.task.done():
        record.task.cancel()
    record.status = "cancelled"
    record.events.append({"event": "run.cancelled", "data": {"run_id": run_id}})
    return {"run_id": run_id, "status": "cancelled"}
