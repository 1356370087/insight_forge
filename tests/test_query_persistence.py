"""Integration tests for QueryEngine file-backed persistence and recovery."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import HumanMessage, SystemMessage

from open_deep_research.agents.query_engine import QueryEngine
from open_deep_research.configuration import (
    RUN_CONFIG_FROZEN_FIELDS,
    RUN_CONFIG_FROZEN_FIELDS_V3,
    Configuration,
    freeze_run_config,
    run_config_fingerprint,
)
from open_deep_research.quality.gate import HandoffAssessment
from open_deep_research.run_context import (
    ResearchBriefPersistenceError,
    RunConfigurationError,
    RunContextStore,
)
from open_deep_research.runtime import RuntimeCommand
from tests.auth_helpers import research_principal


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


def test_v3_frozen_run_does_not_require_v4_quality_fields() -> None:
    all_values = Configuration().model_dump(mode="json")
    v3_values = {
        field_name: all_values[field_name]
        for field_name in RUN_CONFIG_FROZEN_FIELDS_V3
    }
    config = {
        "configurable": v3_values,
        "metadata": {
            "runtime_config_frozen": True,
            "run_config_schema_version": 3,
            "quality_policy_version": "quality-gate-v3",
        },
    }
    config["metadata"]["run_config_fingerprint"] = run_config_fingerprint(
        config
    )

    restored = freeze_run_config(config)

    assert restored["metadata"]["quality_policy_version"] == "quality-gate-v3"
    assert "quality_risk_mode" not in restored["configurable"]
    assert "quality_caveat_admission_enabled" not in restored["configurable"]
    assert "quality_gap_recovery_max_attempts" not in restored["configurable"]


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
    assert resumed.context_store.load_manifest().status == "completed"


@pytest.mark.asyncio
async def test_resume_uses_frozen_models_after_environment_switch(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("RESEARCH_MODEL", "openai:research-original")
    monkeypatch.setenv("QUALITY_EVALUATION_MODEL", "openai:qwen3.7-flash")
    monkeypatch.setenv("QUALITY_EVALUATION_RIGOR", "strict")
    _install_graph(monkeypatch, supervisor_error=RuntimeError("interrupted"))
    config = _config(
        tmp_path,
        "frozen-model-run",
        apiKeys={"DASHSCOPE_API_KEY": "must-not-be-persisted"},
    )
    first = QueryEngine(config)

    await first.submit_message([HumanMessage(content="研究请求")])
    manifest = first.context_store.load_manifest()
    serialized_manifest = manifest.model_dump_json()

    assert manifest.config_fingerprint
    assert manifest.quality_policy_version == "quality-gate-v4"
    assert manifest.quality_evaluation_epoch
    assert manifest.quality_evaluation_rigor == "strict"
    assert manifest.quality_rigor_policy["runtime_average_floor"] == 4.0
    assert "must-not-be-persisted" not in serialized_manifest
    assert "apiKeys" not in serialized_manifest

    monkeypatch.setenv("RESEARCH_MODEL", "openai:research-switched")
    monkeypatch.setenv("QUALITY_EVALUATION_MODEL", "openai:qwen3.7-max")
    monkeypatch.setenv("QUALITY_EVALUATION_RIGOR", "very_relaxed")
    resumed = QueryEngine.load(
        "frozen-model-run",
        runs_dir=str(tmp_path),
        config={"metadata": {"owner": "user-1"}},
    )
    resolved = Configuration.from_runnable_config(resumed.config)

    assert resolved.research_model == "openai:research-original"
    assert resolved.quality_evaluation_model == "openai:qwen3.7-flash"
    assert resolved.quality_evaluation_rigor.value == "strict"


@pytest.mark.asyncio
async def test_resume_rejects_explicit_frozen_model_conflict(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("QUALITY_EVALUATION_MODEL", "openai:qwen3.7-flash")
    _install_graph(monkeypatch, supervisor_error=RuntimeError("interrupted"))
    first = QueryEngine(_config(tmp_path, "config-mismatch-run"))
    await first.submit_message([HumanMessage(content="研究请求")])

    with pytest.raises(RuntimeError, match="run_config_mismatch"):
        QueryEngine.load(
            "config-mismatch-run",
            runs_dir=str(tmp_path),
            config={
                "configurable": {
                    "quality_evaluation_model": "openai:qwen3.7-max",
                    "quality_evaluation_rigor": "very_strict",
                },
                "metadata": {"owner": "user-1"},
            },
        )


def test_legacy_resume_requires_explicit_full_migration_config(tmp_path) -> None:
    store = RunContextStore("legacy-unfrozen", runs_dir=str(tmp_path))
    store.initialize(
        "user-1",
        {
            "configurable": {"runs_dir": str(tmp_path)},
            "metadata": {"run_id": "legacy-unfrozen", "owner": "user-1"},
        },
    )

    with pytest.raises(RuntimeError, match="legacy_run_config_not_frozen"):
        QueryEngine.load("legacy-unfrozen", runs_dir=str(tmp_path))

    full_config = Configuration().model_dump(mode="json")
    assert set(RUN_CONFIG_FROZEN_FIELDS) <= set(full_config)
    migrated = QueryEngine.load(
        "legacy-unfrozen",
        runs_dir=str(tmp_path),
        config={
            "configurable": full_config,
            "metadata": {"owner": "user-1"},
        },
        legacy_migration=True,
    )

    assert migrated.config["metadata"]["legacy_config_migration"] is True
    assert migrated.config["metadata"]["runtime_config_frozen"] is True


@pytest.mark.asyncio
async def test_safe_rejected_evidence_finishes_once_as_nonempty_partial(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("QUALITY_EVALUATION_ENABLED", "true")
    monkeypatch.setenv("QUALITY_EVALUATION_FAIL_OPEN", "false")
    calls = _install_graph(monkeypatch)
    engine = QueryEngine(_config(tmp_path, "partial-recovery-run"))
    artifact = {
        "research_topic": "Verify two sources.",
        "compressed_research": "REJECTED_COMPRESSED_TEXT " * 20,
        "raw_notes": ["REJECTED_RAW_TEXT"],
        "metrics": {"sources_read": 2},
        "evidence_registry": [
            {
                "evidence_id": "ev-a",
                "claim": "Supported claim A.",
                "supporting_excerpt": "Complete excerpt A.",
                "source_url": "https://a.example/source",
                "locator": "section A",
                "security_status": "accepted",
            },
            {
                "evidence_id": "ev-b",
                "claim": "Supported claim B.",
                "supporting_excerpt": "Complete excerpt B.",
                "source_url": "https://b.example/source",
                "locator": "section B",
                "security_status": "accepted",
            },
        ],
    }
    digest = engine.context_store.persist_task_result("task-1", artifact)

    async def terminate_with_rejected_artifact(_self, _state):
        return {
            "research_artifact_refs": {
                "type": "override",
                "value": {
                    "task-1": {
                        "path": (
                            "context/artifacts/research_tasks/task-1.json"
                        ),
                        "sha256": digest,
                    }
                },
            },
            "handoff_assessments": {
                "type": "override",
                "value": [
                    {
                        "tool_call_id": "task-1",
                        "accepted": False,
                        "reason": "Initial rejection.",
                    }
                ],
            },
            "completion_decision": {
                "type": "override",
                "value": {
                    "action": "terminate",
                    "reason": "max_turns",
                    "gaps": ["accepted_evidence"],
                },
            },
        }

    async def reject_reassessment(*_args, **_kwargs):
        return HandoffAssessment(
            accepted=False,
            relevance=4,
            source_quality=4,
            evidence_coverage=2,
            groundedness=4,
            missing_information=["Full handoff was not accepted."],
            reason="Synthetic reassessment rejection.",
        )

    monkeypatch.setattr(QueryEngine, "_run_supervisor", terminate_with_rejected_artifact)
    monkeypatch.setattr(
        "open_deep_research.agents.query_engine.evaluate_subagent_handoff",
        reject_reassessment,
    )

    state = await engine.submit_message([HumanMessage(content="研究请求")])

    assert state["result"]["status"] == "partial"
    assert state["result"]["termination_reason"] == "quality_gate_recovery"
    assert state["result"]["quality_gate"]["status"] == "degraded"
    assert state["final_report"].startswith("# 质量门禁未通过的证据恢复报告")
    assert "Complete excerpt A." in state["final_report"]
    assert "REJECTED_COMPRESSED_TEXT" not in state["final_report"]
    assert calls["report"] == 0
    assert engine.context_store.load_manifest().result["status"] == "partial"
    event_names = [item["event"] for item in engine.transcript]
    assert event_names.count("report.completed") == 1
    assert event_names.count("run.completed") == 1
    assert event_names.count("run.failed") == 0


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

        async def acquire_run_lease(self):
            return 1

        async def stream_resume(self):
            self.status = "completed"
            self.final_state = {"result": {"status": "success"}}
            yield {"event": "run.completed", "data": {"status": "completed"}}

    monkeypatch.setattr(server.QueryEngine, "load", lambda *_args, **_kwargs: FakeEngine())
    server._runs.clear()
    server.app.dependency_overrides[get_current_user] = lambda: research_principal("user-1")
    client = TestClient(server.app)
    try:
        response = client.post("/runs/resume-api/resume", json={})
    finally:
        server.app.dependency_overrides.clear()
        server._runs.clear()

    assert response.status_code == 202
    assert response.json() == {"run_id": "resume-api", "status": "running"}


def test_resume_api_maps_frozen_config_conflict_to_409(monkeypatch) -> None:
    from open_deep_research import server
    from security.auth import get_current_user

    def reject_conflict(*_args, **_kwargs):
        raise RunConfigurationError("run_config_mismatch:research_model")

    monkeypatch.setattr(server.QueryEngine, "load", reject_conflict)
    server._runs.clear()
    server.app.dependency_overrides[get_current_user] = lambda: research_principal("user-1")
    client = TestClient(server.app, raise_server_exceptions=False)
    try:
        response = client.post("/runs/config-conflict/resume", json={})
    finally:
        server.app.dependency_overrides.clear()
        server._runs.clear()

    assert response.status_code == 409
    assert response.json()["detail"] == "run_not_recoverable"


def test_resume_api_maps_legacy_schema_to_specific_409(monkeypatch) -> None:
    from open_deep_research import server
    from security.auth import get_current_user

    def reject_legacy_schema(*_args, **_kwargs):
        raise RunConfigurationError("run_schema_not_resumable")

    monkeypatch.setattr(server.QueryEngine, "load", reject_legacy_schema)
    server._runs.clear()
    server.app.dependency_overrides[get_current_user] = lambda: research_principal("user-1")
    client = TestClient(server.app, raise_server_exceptions=False)
    try:
        response = client.post("/runs/legacy-schema/resume", json={})
    finally:
        server.app.dependency_overrides.clear()
        server._runs.clear()

    assert response.status_code == 409
    assert response.json()["detail"] == "run_schema_not_resumable"


@pytest.mark.asyncio
async def test_legacy_migration_restarts_before_stale_report(
    tmp_path,
    monkeypatch,
) -> None:
    from open_deep_research.agents import deep_researcher as graph

    run_id = "legacy-stale-report"
    store = RunContextStore(run_id, runs_dir=str(tmp_path))
    store.initialize(
        "user-1",
        {
            "configurable": {"runs_dir": str(tmp_path)},
            "metadata": {"run_id": run_id, "owner": "user-1"},
        },
    )
    await store.append(
        channel="lead",
        record_type="state_delta",
        stage="report_generated",
        payload={
            "scope": "main",
            "update": {
                "messages": {
                    "type": "override",
                    "value": [HumanMessage(content="研究请求")],
                },
                "research_brief": "Verify accepted evidence.",
                "final_report": "STALE REPORT FROM REJECTED EVIDENCE",
                "completion_decision": {
                    "action": "complete",
                    "reason": "legacy_quality_gate",
                    "gaps": [],
                },
            },
        },
    )
    await store.checkpoint(
        "report_generated",
        "memory_extract_and_write",
        status="failed",
    )

    async def memory(_state, _config):
        return RuntimeCommand()

    supervisor_calls = 0

    async def terminate(_self, _state, **_kwargs):
        nonlocal supervisor_calls
        supervisor_calls += 1
        return {
            "completion_decision": {
                "type": "override",
                "value": {
                    "action": "terminate",
                    "reason": "insufficient_evidence",
                    "gaps": ["accepted_evidence"],
                },
            }
        }

    monkeypatch.setattr(graph, "memory_extract_and_write", memory)
    monkeypatch.setattr(QueryEngine, "_run_supervisor", terminate)
    full_config = Configuration(
        quality_evaluation_enabled=True,
        quality_evaluation_fail_open=False,
    ).model_dump(mode="json")
    engine = QueryEngine.load(
        run_id,
        runs_dir=str(tmp_path),
        config={
            "configurable": full_config,
            "metadata": {"owner": "user-1"},
        },
        legacy_migration=True,
    )

    result = await engine.resume()

    assert supervisor_calls == 1
    assert result["result"]["status"] == "failed"
    assert "STALE REPORT" not in str(result.get("final_report", ""))
    assert (
        "legacy_config_migration"
        not in engine.context_store.load_manifest().config["metadata"]
    )


def test_resume_api_rejects_completed_run(monkeypatch) -> None:
    from open_deep_research import server
    from security.auth import get_current_user

    class CompletedEngine:
        config = {"metadata": {"owner": "user-1"}}

    server._runs.clear()
    server._runs["done"] = server.RunRecord(
        run_id="done",
        engine=CompletedEngine(),
        status="completed",
    )
    server.app.dependency_overrides[get_current_user] = lambda: research_principal("user-1")
    client = TestClient(server.app)
    try:
        response = client.post("/runs/done/resume", json={})
    finally:
        server.app.dependency_overrides.clear()
        server._runs.clear()

    assert response.status_code == 409
    assert response.json()["detail"] == "run_already_completed"
