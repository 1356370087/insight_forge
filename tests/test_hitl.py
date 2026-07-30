from __future__ import annotations

import asyncio
import tempfile
from typing import Any

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import HumanMessage, SystemMessage

from open_deep_research.agents.query_engine import QueryEngine
from open_deep_research.configuration import Configuration
from open_deep_research.observability import SQLiteTraceStore
from open_deep_research.runtime import RuntimeCommand


def _config(**configurable: Any) -> dict[str, Any]:
    return {
        "configurable": {
            "event_log_enabled": False,
            "query_session_persistence_enabled": False,
            "search_api": "none",
            "runs_dir": tempfile.mkdtemp(prefix="open-deep-research-hitl-"),
            **configurable,
        },
        "metadata": {"run_id": "hitl-test"},
    }


async def _install_basic_graph(monkeypatch, *, final_report: str = "final report") -> dict[str, int]:
    from open_deep_research.agents import deep_researcher as graph

    calls = {"supervisor": 0, "final_report": 0}

    async def summarize_messages(_state, _config):
        return {}

    async def memory_recall(_state, _config):
        return {}

    async def clarify_with_user(_state, _config):
        return RuntimeCommand(goto="write_research_brief")

    async def write_research_brief(_state, _config):
        return RuntimeCommand(
            goto="research_supervisor",
            update={
                "research_brief": "Research brief: compare browser HITL options.",
                "supervisor_messages": [SystemMessage(content="supervisor base")],
            },
        )

    async def final_report_generation(_state, _config):
        calls["final_report"] += 1
        return {"final_report": final_report}

    async def memory_extract_and_write(_state, _config):
        return RuntimeCommand()

    async def fake_supervisor(self, main_state):
        calls["supervisor"] += 1
        assert main_state.get("approved_research_plan")
        return {
            "notes": {"type": "override", "value": ["research complete"]},
            "raw_notes": {"type": "override", "value": ["raw evidence"]},
            "supervisor_messages": {
                "type": "override",
                "value": main_state.get("supervisor_messages", []),
            },
        }

    monkeypatch.setattr(graph, "summarize_messages", summarize_messages)
    monkeypatch.setattr(graph, "memory_recall", memory_recall)
    monkeypatch.setattr(graph, "clarify_with_user", clarify_with_user)
    monkeypatch.setattr(graph, "write_research_brief", write_research_brief)
    monkeypatch.setattr(graph, "final_report_generation", final_report_generation)
    monkeypatch.setattr(graph, "memory_extract_and_write", memory_extract_and_write)
    monkeypatch.setattr(QueryEngine, "_run_supervisor", fake_supervisor)
    return calls


async def _collect_until(events: list[dict[str, Any]], event_name: str, queue: asyncio.Queue):
    while True:
        event = await queue.get()
        events.append(event)
        if event["event"] == event_name:
            return event


def _drain_available(events: list[dict[str, Any]], queue: asyncio.Queue) -> None:
    while not queue.empty():
        events.append(queue.get_nowait())


@pytest.mark.asyncio
async def test_hitl_plan_approval_pauses_before_supervisor(monkeypatch):
    calls = await _install_basic_graph(monkeypatch)
    engine = QueryEngine(
        _config(
            enable_human_in_loop=True,
            hitl_require_outline_approval=False,
        )
    )
    events: list[dict[str, Any]] = []
    queue: asyncio.Queue = asyncio.Queue()

    async def run():
        async for event in engine.stream_message([HumanMessage(content="research HITL")]):
            await queue.put(event)

    task = asyncio.create_task(run())
    plan_event = await asyncio.wait_for(_collect_until(events, "hitl.plan_pending", queue), 2)

    assert engine.status == "awaiting_plan_approval"
    assert calls["supervisor"] == 0
    action_id = plan_event["data"]["pending_human_action"]["action_id"]

    result = engine.handle_human_action(action_id, "approve")
    assert result["status"] == "accepted"

    await asyncio.wait_for(task, 2)
    _drain_available(events, queue)
    assert any(event["event"] == "hitl.plan_approved" for event in events)
    assert engine.final_state["approved_research_plan"] == engine.final_state["research_plan"]
    assert calls["supervisor"] == 1
    assert engine.final_state["final_report"] == "final report"


@pytest.mark.asyncio
async def test_hitl_plan_revision_regenerates_plan_before_approval(monkeypatch):
    await _install_basic_graph(monkeypatch)
    engine = QueryEngine(
        _config(
            enable_human_in_loop=True,
            hitl_require_outline_approval=False,
        )
    )
    queue: asyncio.Queue = asyncio.Queue()

    async def run():
        async for event in engine.stream_message([HumanMessage(content="research HITL")]):
            await queue.put(event)

    task = asyncio.create_task(run())
    first = await asyncio.wait_for(_collect_until([], "hitl.plan_pending", queue), 2)
    first_plan = first["data"]["research_plan"]
    engine.handle_human_action(first["data"]["pending_human_action"]["action_id"], "revise", "focus on enterprise users")

    second = await asyncio.wait_for(_collect_until([], "hitl.plan_pending", queue), 2)
    revised_plan = second["data"]["research_plan"]
    assert revised_plan != first_plan
    assert "focus on enterprise users" in revised_plan

    engine.handle_human_action(second["data"]["pending_human_action"]["action_id"], "approve")
    await asyncio.wait_for(task, 2)
    assert any(item["type"] == "plan_revision" for item in engine.final_state["human_feedback"])


@pytest.mark.asyncio
async def test_hitl_cancellation_finalizes_observability_run(monkeypatch, tmp_path):
    await _install_basic_graph(monkeypatch)
    trace_path = tmp_path / "cancel.sqlite3"
    engine = QueryEngine(
        _config(
            enable_human_in_loop=True,
            hitl_require_outline_approval=False,
            trace_store_path=str(trace_path),
        )
    )
    queue: asyncio.Queue = asyncio.Queue()
    events: list[dict[str, Any]] = []

    async def run():
        async for event in engine.stream_message([HumanMessage(content="research HITL")]):
            events.append(event)
            await queue.put(event)

    task = asyncio.create_task(run())
    pending = await asyncio.wait_for(_collect_until([], "hitl.plan_pending", queue), 2)
    engine.handle_human_action(
        pending["data"]["pending_human_action"]["action_id"], "cancel"
    )
    await asyncio.wait_for(task, 2)

    observed = SQLiteTraceStore(str(trace_path)).get_run("hitl-test")
    assert observed is not None
    assert observed["status"] == "cancelled"
    assert engine.final_state["result"]["status"] == "cancelled"
    assert any(event["event"] == "run.cancelled" for event in events)


@pytest.mark.asyncio
async def test_interrupt_wakes_hitl_without_human_action(monkeypatch):
    await _install_basic_graph(monkeypatch)
    engine = QueryEngine(
        _config(
            enable_human_in_loop=True,
            hitl_require_outline_approval=False,
        )
    )
    queue: asyncio.Queue = asyncio.Queue()
    events: list[dict[str, Any]] = []

    async def run():
        async for event in engine.stream_message([HumanMessage(content="research HITL")]):
            events.append(event)
            await queue.put(event)

    task = asyncio.create_task(run())
    await asyncio.wait_for(_collect_until([], "hitl.plan_pending", queue), 2)

    engine.interrupt()
    assert engine.status == "cancelling"
    await asyncio.wait_for(task, 2)

    assert engine.status == "cancelled"
    assert engine.pending_human_action is None
    assert sum(event["event"] == "run.cancelled" for event in events) == 1


@pytest.mark.asyncio
async def test_success_terminal_claim_rejects_late_cancellation(monkeypatch):
    await _install_basic_graph(monkeypatch)
    engine = QueryEngine(
        _config(
            hitl_require_plan_approval=False,
            web_pipeline_mode="legacy",
        )
    )
    terminal_persist_started = asyncio.Event()
    allow_terminal_persist = asyncio.Event()
    original_persist_checkpoint = engine._persist_checkpoint

    async def pause_terminal_persistence(*args, **kwargs):
        if kwargs.get("status") == "completed":
            terminal_persist_started.set()
            await allow_terminal_persist.wait()
        return await original_persist_checkpoint(*args, **kwargs)

    monkeypatch.setattr(engine, "_persist_checkpoint", pause_terminal_persistence)
    events: list[dict[str, Any]] = []

    async def run():
        async for event in engine.stream_message([HumanMessage(content="research")]):
            events.append(event)

    task = asyncio.create_task(run())
    await asyncio.wait_for(terminal_persist_started.wait(), 10)
    engine.interrupt()
    allow_terminal_persist.set()
    await asyncio.wait_for(task, 10)

    assert engine.status == "completed"
    assert engine.cancelled is False
    assert engine.cancellation_scope.is_cancelled is False
    assert sum(event["event"] == "run.completed" for event in events) == 1
    assert not any(event["event"] == "run.cancelled" for event in events)


@pytest.mark.asyncio
async def test_hitl_outline_approval_pauses_before_final_report(monkeypatch):
    calls = await _install_basic_graph(monkeypatch, final_report="approved outline report")
    engine = QueryEngine(
        _config(
            enable_human_in_loop=True,
            hitl_require_plan_approval=False,
            hitl_require_outline_approval=True,
        )
    )
    events: list[dict[str, Any]] = []
    queue: asyncio.Queue = asyncio.Queue()

    async def run():
        async for event in engine.stream_message([HumanMessage(content="research HITL")]):
            await queue.put(event)

    task = asyncio.create_task(run())
    outline_event = await asyncio.wait_for(_collect_until(events, "hitl.outline_pending", queue), 2)

    assert engine.status == "awaiting_outline_approval"
    assert calls["supervisor"] == 1
    assert calls["final_report"] == 0

    engine.handle_human_action(outline_event["data"]["pending_human_action"]["action_id"], "approve")
    await asyncio.wait_for(task, 2)
    _drain_available(events, queue)
    assert any(event["event"] == "hitl.outline_approved" for event in events)
    assert calls["final_report"] == 1
    assert engine.final_state["final_report"] == "approved outline report"


@pytest.mark.asyncio
async def test_run_level_feedback_is_injected_into_supervisor_context():
    engine = QueryEngine(_config(enable_human_in_loop=True))
    await engine.submit_feedback({
        "type": "direction",
        "message": "Prioritize official filings.",
    })
    supervisor_state = {"supervisor_messages": []}

    engine._drain_human_feedback(supervisor_state)

    assert len(supervisor_state["supervisor_messages"]) == 1
    assert "[User Feedback]" in supervisor_state["supervisor_messages"][0].content
    assert "official filings" in supervisor_state["supervisor_messages"][0].content


def test_hitl_configuration_defaults_are_disabled():
    cfg = Configuration()

    assert cfg.enable_human_in_loop is False
    assert cfg.hitl_require_plan_approval is True
    assert cfg.hitl_require_outline_approval is True
    assert cfg.hitl_feedback_mode == "safe_points"

def test_hitl_api_accepts_human_action():
    from open_deep_research import server
    from security.auth import get_current_user

    class FakeEngine:
        pending_human_action = {"action_id": "act-1", "type": "plan_approval"}

        def handle_human_action(self, action_id: str, action: str, message: str = ""):
            assert action_id == "act-1"
            assert action == "approve"
            assert message == ""
            self.pending_human_action = None
            return {"status": "accepted", "action": "approve"}

    server._runs.clear()
    server._runs["run-1"] = server.RunRecord(run_id="run-1", engine=FakeEngine(), status="awaiting_plan_approval")
    server.app.dependency_overrides[get_current_user] = lambda: {"identity": "u1", "permissions": []}
    client = TestClient(server.app)
    try:
        response = client.post("/runs/run-1/human-actions/act-1", json={"action": "approve"})
    finally:
        server.app.dependency_overrides.clear()
        server._runs.clear()

    assert response.status_code == 200
    assert response.json()["status"] == "accepted"


def test_cancel_api_requests_cancellation_without_terminal_event(monkeypatch):
    from open_deep_research import server
    from security.auth import get_current_user

    class FakeEngine:
        pending_human_action = None
        status = "running"
        config = _config(runs_dir=".runs")

        def interrupt(self):
            self.status = "cancelling"

    async def terminal_publish_must_not_run(*_args, **_kwargs):
        raise AssertionError("cancel endpoint must not publish a terminal event")

    fake_engine = FakeEngine()
    server._runs.clear()
    server._runs["run-1"] = server.RunRecord(
        run_id="run-1",
        engine=fake_engine,
        status="running",
    )
    server.app.dependency_overrides[get_current_user] = lambda: {
        "identity": "u1",
        "permissions": [],
    }
    monkeypatch.setattr(
        server,
        "event_publisher_from_config",
        terminal_publish_must_not_run,
    )
    client = TestClient(server.app)
    try:
        response = client.post("/runs/run-1/cancel")
    finally:
        server.app.dependency_overrides.clear()
        server._runs.clear()

    assert response.status_code == 200
    assert response.json()["status"] == "cancelling"
    assert fake_engine.status == "cancelling"


@pytest.mark.asyncio
async def test_background_run_preserves_engine_failed_status():
    from open_deep_research import server

    class FakeEngine:
        status = "failed"
        final_state = {"result": {"status": "error"}}

        async def stream_message(self, _messages, _config):
            if False:
                yield None

    record = server.RunRecord(run_id="failed-run", engine=FakeEngine())
    await server._run_background(
        record,
        server.RunRequest(messages=[]),
        _config(),
    )

    assert record.status == "failed"
    assert record.result == {"result": {"status": "error"}}


def test_hitl_api_records_feedback():
    from open_deep_research import server
    from security.auth import get_current_user

    class FakeEngine:
        pending_human_action = None

        async def submit_feedback(self, feedback):
            assert feedback["type"] == "evidence_question"
            assert feedback["message"] == "Which source supports this?"
            return {"status": "accepted", "feedback_id": "fb-1"}

    server._runs.clear()
    server._runs["run-1"] = server.RunRecord(run_id="run-1", engine=FakeEngine(), status="running")
    server.app.dependency_overrides[get_current_user] = lambda: {"identity": "u1", "permissions": []}
    client = TestClient(server.app)
    try:
        response = client.post(
            "/runs/run-1/feedback",
            json={"type": "evidence_question", "message": "Which source supports this?"},
        )
    finally:
        server.app.dependency_overrides.clear()
        server._runs.clear()

    assert response.status_code == 200
    assert response.json()["status"] == "accepted"

