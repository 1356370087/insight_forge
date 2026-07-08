"""Integration tests for QueryEngine file-backed persistence and recovery."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import HumanMessage, SystemMessage

from open_deep_research.agents.query_engine import QueryEngine
from open_deep_research.run_context import ResearchBriefPersistenceError
from open_deep_research.runtime import RuntimeCommand


def _config(tmp_path, run_id: str, **overrides: Any) -> dict[str, Any]:
    return {
        "configurable": {
            "runs_dir": str(tmp_path),
            "event_log_enabled": False,
            "observability_enabled": False,
            "query_context_compaction_enabled": False,
            "search_api": "none",
            **overrides,
        },
        "metadata": {"run_id": run_id, "owner": "user-1"},
    }


def _install_graph(monkeypatch, *, supervisor_error: Exception | None = None) -> dict[str, int]:
    from open_deep_research.agents import deep_researcher as graph

    calls = {"summarize": 0, "memory": 0, "brief": 0, "supervisor": 0, "report": 0}

    async def summarize(_state, _config):
        calls["summarize"] += 1
        return RuntimeCommand(goto="memory_recall")

    async def memory(_state, _config):
        calls["memory"] += 1
        return RuntimeCommand(goto="clarify_with_user")

    async def clarify(_state, _config):
        return RuntimeCommand(goto="write_research_brief")

    async def brief(_state, _config):
        calls["brief"] += 1
        return RuntimeCommand(
            goto="research_supervisor",
            update={
                "research_brief": "完整且不可压缩的研究目标。",
                "supervisor_messages": [SystemMessage(content="supervisor")],
                "enable_async_research": False,
            },
        )

    async def supervisor(_self, state):
        calls["supervisor"] += 1
        if supervisor_error is not None:
            raise supervisor_error
        return {
            "supervisor_messages": {"type": "override", "value": state.get("supervisor_messages", [])},
            "notes": {"type": "override", "value": ["finding"]},
        }

    async def report(_state, _config):
        calls["report"] += 1
        return {"final_report": "# Final\n\nReport"}

    async def write_memory(_state, _config):
        return RuntimeCommand()

    monkeypatch.setattr(graph, "summarize_messages", summarize)
    monkeypatch.setattr(graph, "memory_recall", memory)
    monkeypatch.setattr(graph, "clarify_with_user", clarify)
    monkeypatch.setattr(graph, "write_research_brief", brief)
    monkeypatch.setattr(graph, "final_report_generation", report)
    monkeypatch.setattr(graph, "memory_extract_and_write", write_memory)
    monkeypatch.setattr(QueryEngine, "_run_supervisor", supervisor)
    return calls


@pytest.mark.asyncio
async def test_query_run_persists_authoritative_context(tmp_path, monkeypatch) -> None:
    _install_graph(monkeypatch)
    engine = QueryEngine(_config(tmp_path, "persist-run"))

    result = await engine.submit_message([HumanMessage(content="研究请求")])

    context = tmp_path / "persist-run" / "context"
    assert result["result"]["status"] == "success"
    assert (context / "research_brief.md").read_text(encoding="utf-8") == "完整且不可压缩的研究目标。"
    assert (context / "session_memory.jsonl").exists()
    assert (context / "final_report.md").read_text(encoding="utf-8") == "# Final\n\nReport"
    manifest = engine.context_store.load_manifest()
    assert manifest.status == "completed"
    assert manifest.next_stage == "completed"


@pytest.mark.asyncio
async def test_resume_skips_completed_outer_nodes(tmp_path, monkeypatch) -> None:
    first_calls = _install_graph(monkeypatch, supervisor_error=RuntimeError("interrupted"))
    config = _config(tmp_path, "resume-run")
    first = QueryEngine(config)

    failed = await first.submit_message([HumanMessage(content="研究请求")])

    assert failed["result"]["status"] == "error"
    assert first.context_store.load_manifest().next_stage == "supervisor.supervisor"
    assert first_calls["brief"] == 1

    resumed_calls = _install_graph(monkeypatch)
    resumed = QueryEngine.load("resume-run", runs_dir=str(tmp_path), config=config)
    completed = await resumed.resume()

    assert completed["result"]["status"] == "success"
    assert resumed_calls["summarize"] == 0
    assert resumed_calls["memory"] == 0
    assert resumed_calls["brief"] == 0
    assert resumed_calls["supervisor"] == 1
    assert resumed_calls["report"] == 1
    assert completed["research_brief"] == "完整且不可压缩的研究目标。"


@pytest.mark.asyncio
async def test_research_brief_write_failure_stops_run(tmp_path, monkeypatch) -> None:
    _install_graph(monkeypatch)
    engine = QueryEngine(_config(tmp_path, "brief-failure"))
    assert engine.context_store is not None

    def fail(_content: str) -> str:
        raise ResearchBriefPersistenceError("research_brief_persistence_failed")

    monkeypatch.setattr("open_deep_research.run_context.RunContextStore.persist_research_brief", lambda _self, content: fail(content))
    result = await engine.submit_message([HumanMessage(content="研究请求")])

    assert result["result"]["status"] == "error"
    assert result["result"]["error"] == "research_brief_persistence_failed"


def test_resume_api_launches_explicit_recovery(monkeypatch) -> None:
    from open_deep_research import server
    from security.auth import get_current_user

    manifest = SimpleNamespace(owner_id="user-1", status="failed", next_stage="supervisor.supervisor")

    class FakeStore:
        def replay(self):
            return SimpleNamespace(manifest=manifest)

    class FakeEngine:
        context_store = FakeStore()
        final_state = None
        status = "running"

        async def stream_resume(self):
            self.status = "completed"
            self.final_state = {"result": {"status": "success"}}
            yield {"event": "run.completed", "data": {"status": "completed"}}

    monkeypatch.setattr(server.QueryEngine, "load", lambda *_args, **_kwargs: FakeEngine())
    server._runs.clear()
    server.app.dependency_overrides[get_current_user] = lambda: {
        "identity": "user-1",
        "permissions": [],
    }
    client = TestClient(server.app)
    try:
        response = client.post("/runs/resume-api/resume", json={})
    finally:
        server.app.dependency_overrides.clear()
        server._runs.clear()

    assert response.status_code == 202
    assert response.json() == {"run_id": "resume-api", "status": "running"}


def test_resume_api_rejects_completed_run(monkeypatch) -> None:
    from open_deep_research import server
    from security.auth import get_current_user

    class CompletedEngine:
        pass

    server._runs.clear()
    server._runs["done"] = server.RunRecord(
        run_id="done",
        engine=CompletedEngine(),
        status="completed",
    )
    server.app.dependency_overrides[get_current_user] = lambda: {
        "identity": "user-1",
        "permissions": [],
    }
    client = TestClient(server.app)
    try:
        response = client.post("/runs/done/resume", json={})
    finally:
        server.app.dependency_overrides.clear()
        server._runs.clear()

    assert response.status_code == 409
    assert response.json()["detail"] == "run_already_completed"
