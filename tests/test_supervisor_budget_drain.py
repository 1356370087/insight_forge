"""Regression tests for async supervisor budget/drain semantics.

Covers the E2E failure where WaitForResearchUpdates turns consumed the whole
research-turn budget before in-flight Researchers could hand off evidence:
- passive coordination turns no longer advance the turn counter,
- budget exhaustion drains unfinished tasks before terminating,
- run-terminal cleanup closes still-running task snapshots,
- the usage endpoint response can carry authoritative terminal status/duration.
"""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool as lc_tool
from pydantic import ValidationError

from open_deep_research.agents import deep_researcher
from open_deep_research.agents.deep_researcher import (
    drain_unfinished_async_tasks,
    finalize_interrupted_task_snapshots,
)
from open_deep_research.agents.query_engine import _supervisor_turn_advance_policy
from open_deep_research.configuration import Configuration
from open_deep_research.tasks.registry import TaskStatus
from open_deep_research.tasks.state import TaskSnapshot
from open_deep_research.tools.adapters import adapt_langchain_tool
from open_deep_research.tools.base import ToolOrigin
from open_deep_research.tools.supervisor.wait_for_research_updates.definition import (
    WaitForResearchUpdates,
)


@lc_tool("WaitForResearchUpdates")
async def _wait_stub(timeout_seconds: int = 30) -> str:
    """Wait stub."""
    return "no updates"


wait_stub = adapt_langchain_tool(
    _wait_stub,
    origin=ToolOrigin.SYSTEM,
)


def _config(*, async_research: bool = True, runs_dir: str = ".runs-test-drain") -> dict:
    return {
        "configurable": {
            "enable_async_research": async_research,
            "runs_dir": runs_dir,
        },
        "metadata": {"run_id": "run-drain-test"},
    }


def _ai_message(*tool_names: str) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {"name": name, "args": {}, "id": f"call-{index}"}
            for index, name in enumerate(tool_names)
        ],
    )


class TestPassiveTurnPolicy:
    @pytest.mark.asyncio
    async def test_all_passive_calls_with_unfinished_tasks_do_not_advance(self, monkeypatch):
        async def unfinished(_config):
            return True

        monkeypatch.setattr(
            deep_researcher, "has_unfinished_async_tasks", unfinished
        )
        delta = await _supervisor_turn_advance_policy(
            [_ai_message("WaitForResearchUpdates")],
            None,
            _config(),
        )
        assert delta == 0

    @pytest.mark.asyncio
    async def test_passive_calls_without_unfinished_tasks_still_advance(self, monkeypatch):
        async def finished(_config):
            return False

        monkeypatch.setattr(
            deep_researcher, "has_unfinished_async_tasks", finished
        )
        delta = await _supervisor_turn_advance_policy(
            [_ai_message("WaitForResearchUpdates", "CheckResearchTask")],
            None,
            _config(),
        )
        assert delta == 1

    @pytest.mark.asyncio
    async def test_active_delegation_turns_always_advance(self, monkeypatch):
        async def unfinished(_config):
            return True

        monkeypatch.setattr(
            deep_researcher, "has_unfinished_async_tasks", unfinished
        )
        for message in (
            _ai_message("StartResearchTask"),
            _ai_message("StartResearchTask", "WaitForResearchUpdates"),
        ):
            delta = await _supervisor_turn_advance_policy(
                [message],
                None,
                _config(),
            )
            assert delta == 1

    @pytest.mark.asyncio
    async def test_plain_answer_turn_advances(self, monkeypatch):
        async def unfinished(_config):
            return True

        monkeypatch.setattr(
            deep_researcher, "has_unfinished_async_tasks", unfinished
        )
        delta = await _supervisor_turn_advance_policy(
            [AIMessage(content="done")],
            None,
            _config(),
        )
        assert delta == 1


class TestDrainUnfinishedAsyncTasks:
    @pytest.mark.asyncio
    async def test_waits_until_terminal_then_finalizes(self, monkeypatch):
        checks = {"count": 0}

        async def unfinished(_config):
            checks["count"] += 1
            return checks["count"] < 2

        finalized: dict[str, Any] = {}

        async def fake_finalize(state, config, configurable, publisher):
            finalized["called"] = True
            return {"evidence_registry": [{"evidence_id": "e-1"}]}

        monkeypatch.setattr(
            deep_researcher, "has_unfinished_async_tasks", unfinished
        )
        monkeypatch.setattr(
            deep_researcher, "_finalize_async_research_outputs", fake_finalize
        )
        result = await drain_unfinished_async_tasks(
            {},
            _config(),
            Configuration.from_runnable_config(_config()),
            None,
            timeout_seconds=5,
            poll_seconds=0.01,
        )
        assert result == {"evidence_registry": [{"evidence_id": "e-1"}]}
        assert finalized["called"] is True

    @pytest.mark.asyncio
    async def test_timeout_still_finalizes_partial_results(self, monkeypatch):
        async def unfinished(_config):
            return True

        async def fake_finalize(state, config, configurable, publisher):
            return {"raw_notes": ["partial"]}

        monkeypatch.setattr(
            deep_researcher, "has_unfinished_async_tasks", unfinished
        )
        monkeypatch.setattr(
            deep_researcher, "_finalize_async_research_outputs", fake_finalize
        )
        result = await drain_unfinished_async_tasks(
            {},
            _config(),
            Configuration.from_runnable_config(_config()),
            None,
            timeout_seconds=0.05,
            poll_seconds=0.01,
        )
        assert result == {"raw_notes": ["partial"]}

    @pytest.mark.asyncio
    async def test_noop_when_nothing_unfinished(self, monkeypatch):
        async def unfinished(_config):
            return False

        async def fail_finalize(state, config, configurable, publisher):
            raise AssertionError("finalizer must not run without unfinished tasks")

        monkeypatch.setattr(
            deep_researcher, "has_unfinished_async_tasks", unfinished
        )
        monkeypatch.setattr(
            deep_researcher, "_finalize_async_research_outputs", fail_finalize
        )
        result = await drain_unfinished_async_tasks(
            {},
            _config(),
            Configuration.from_runnable_config(_config()),
            None,
            timeout_seconds=1,
        )
        assert result is None


class TestFinalizeInterruptedTaskSnapshots:
    @pytest.mark.asyncio
    async def test_running_snapshots_become_cancelled_and_publish(self, monkeypatch, tmp_path):
        running = TaskSnapshot(task_id="task-running", run_id="run-drain-test")
        done = TaskSnapshot(
            task_id="task-done",
            run_id="run-drain-test",
            status=TaskStatus.COMPLETED,
        )
        upserted: list[TaskSnapshot] = []
        published: list[dict[str, Any]] = []

        class FakeStore:
            async def list(self, run_id: str):
                return [running, done]

            async def upsert(self, snapshot: TaskSnapshot):
                upserted.append(snapshot)

        async def fake_publish(config, **kwargs):
            published.append(kwargs)

        monkeypatch.setattr(
            deep_researcher, "get_task_state_store", lambda _c: FakeStore()
        )
        monkeypatch.setattr(
            deep_researcher, "publish_task_activity", fake_publish
        )
        config = _config(runs_dir=str(tmp_path))
        await finalize_interrupted_task_snapshots(config, reason="run_terminal")

        assert [snapshot.task_id for snapshot in upserted] == ["task-running"]
        assert upserted[0].status is TaskStatus.CANCELLED
        assert running.status is TaskStatus.CANCELLED
        assert done.status is TaskStatus.COMPLETED
        assert len(published) == 1
        assert published[0]["event_type"] == "task.cancelled"
        assert published[0]["task_id"] == "task-running"

    @pytest.mark.asyncio
    async def test_noop_without_async_research(self, monkeypatch):
        def fail_store(_configurable):
            raise AssertionError("task store must not be touched")

        monkeypatch.setattr(
            deep_researcher, "get_task_state_store", fail_store
        )
        await finalize_interrupted_task_snapshots(
            _config(async_research=False), reason="run_terminal"
        )


class TestWaitToolSchema:
    def test_default_and_bounds_relaxed(self):
        assert WaitForResearchUpdates().timeout_seconds == 30
        assert WaitForResearchUpdates(timeout_seconds=300).timeout_seconds == 300
        with pytest.raises(ValidationError):
            WaitForResearchUpdates(timeout_seconds=301)


class TestUsageResponseDurationFallback:
    def test_unavailable_response_carries_manifest_duration(self, tmp_path):
        from open_deep_research.server import _load_run_usage_response

        configurable = Configuration.from_runnable_config(
            {
                "configurable": {
                    "token_usage_accounting_enabled": False,
                    "runs_dir": str(tmp_path),
                }
            }
        )
        response = _load_run_usage_response(
            "run-drain-test",
            status="failed",
            configurable=configurable,
            duration_ms=12345,
        )
        assert response["status"] == "failed"
        assert response["duration_ms"] == 12345


class TestTurnAdvancePolicyWiring:
    def test_query_params_exposes_optional_policy(self):
        from open_deep_research.agents.query import QueryParams

        assert "turn_advance_policy" in QueryParams.__dataclass_fields__
        instance = QueryParams(
            messages=[],
            system_prompt=None,
            model=None,
            config={},
            turn_advance_policy=_supervisor_turn_advance_policy,
        )
        assert instance.turn_advance_policy is _supervisor_turn_advance_policy

    @pytest.mark.asyncio
    async def test_query_loop_does_not_advance_turn_on_passive_batch(self, monkeypatch):
        from open_deep_research.agents.query import QueryParams, query

        def build_responses():
            return [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "WaitForResearchUpdates",
                            "args": {"timeout_seconds": 1},
                            "id": "call-0",
                        }
                    ],
                ),
                AIMessage(content="final answer"),
            ]

        async def unfinished(_config):
            return True

        monkeypatch.setattr(
            deep_researcher, "has_unfinished_async_tasks", unfinished
        )

        async def completed_turn(*, with_policy: bool) -> int:
            responses = build_responses()

            async def fake_model(_messages):
                return responses.pop(0)

            terminal = None
            async for event in query(QueryParams(
                messages=[HumanMessage(content="research this")],
                system_prompt=None,
                model=None,
                config=_config(),
                tools=[wait_stub],
                call_model=fake_model,
                turn_advance_policy=(
                    _supervisor_turn_advance_policy if with_policy else None
                ),
            )):
                if event.type == "query.completed":
                    terminal = event.data["transition"]["turn"]
            assert terminal is not None
            return terminal

        # Turn 1 is consumed by the initial planning entry; the passive wait
        # batch must not consume a second one, while without the policy the
        # same transcript burns an extra research turn.
        assert await completed_turn(with_policy=True) == 1
        assert await completed_turn(with_policy=False) == 2


class _CapturePublisher:
    def __init__(self):
        self.published: list[tuple[str, dict[str, Any]]] = []

    async def publish(self, event_type, stage=None, payload=None, **kwargs):
        self.published.append((event_type, dict(payload or {})))


class _FakeTaskStateStore:
    def __init__(self, snapshots):
        self._snapshots = snapshots
        self.upserted: list[TaskSnapshot] = []

    async def get(self, task_id, run_id=None):
        return self._snapshots.get(task_id)

    async def list(self, run_id=None):
        return list(self._snapshots.values())

    async def upsert(self, snapshot):
        self.upserted.append(snapshot)


def _rejected_assessment():
    from types import SimpleNamespace

    return SimpleNamespace(
        accepted=False,
        admission_status=None,
        missing_information=["gap-1"],
        unsupported_claims=[],
        follow_up_tasks=[],
        model_dump=lambda: {
            "missing_information": ["gap-1"],
            "unsupported_claims": [],
            "follow_up_tasks": [],
        },
    )


class TestFinalizeAsyncOutputs:
    def _patch(self, monkeypatch, tmp_path, *, outputs, snapshots):
        published: list[dict[str, Any]] = []

        async def fake_collect(registry, *, run_id, state_store):
            return list(outputs)

        async def fake_evaluate(*args, **kwargs):
            return _rejected_assessment()

        async def fake_publish_activity(config, **kwargs):
            published.append(kwargs)

        monkeypatch.setattr(
            deep_researcher, "collect_completed_task_outputs", fake_collect
        )
        monkeypatch.setattr(
            deep_researcher, "get_task_state_store", lambda _c: _FakeTaskStateStore(snapshots)
        )
        monkeypatch.setattr(
            deep_researcher,
            "_load_handoff_artifact_for_quality",
            lambda output, *, task_id, run_id, configurable: {},
        )
        monkeypatch.setattr(
            deep_researcher, "evaluate_subagent_handoff", fake_evaluate
        )
        monkeypatch.setattr(
            deep_researcher, "publish_task_activity", fake_publish_activity
        )
        monkeypatch.setattr(
            deep_researcher, "summarize_public_findings", _async_none
        )
        monkeypatch.setattr(
            deep_researcher, "extract_public_sources", lambda output, *, limit: []
        )
        return published

    @pytest.mark.asyncio
    async def test_rejected_handoff_reads_source_count_from_metrics(self, monkeypatch, tmp_path):
        snapshot = TaskSnapshot(
            task_id="task-1",
            status=TaskStatus.COMPLETED,
            wave_id="wave-1",
            metrics={"source_count": 36},
        )
        published = self._patch(
            monkeypatch,
            tmp_path,
            outputs=[{"task_id": "task-1", "research_topic": "t", "requirement_ids": []}],
            snapshots={"task-1": snapshot},
        )
        finalize_config = _config(runs_dir=str(tmp_path))
        finalize_config["configurable"]["quality_evaluation_enabled"] = True
        update = await deep_researcher._finalize_async_research_outputs(
            {"coverage_ledger": {}, "coverage_contract": None, "research_risk_profile": None},
            finalize_config,
            Configuration.from_runnable_config(finalize_config),
            _CapturePublisher(),
        )
        warnings = [
            p for p in published
            if p.get("event_type") == "task.completed" and "source_count" in p.get("payload", {})
        ]
        assert warnings, "rejected handoff must publish a warning activity"
        assert warnings[0]["payload"]["source_count"] == 36
        assert update["completed_task_outputs"] == []

    @pytest.mark.asyncio
    async def test_single_output_failure_does_not_abort_remaining(self, monkeypatch, tmp_path):
        snapshots = {
            "task-bad": TaskSnapshot(task_id="task-bad", status=TaskStatus.COMPLETED, wave_id="w"),
            "task-good": TaskSnapshot(
                task_id="task-good",
                status=TaskStatus.COMPLETED,
                wave_id="w",
                metrics={"source_count": 5},
            ),
        }
        published = self._patch(
            monkeypatch,
            tmp_path,
            outputs=[
                {"task_id": "task-bad", "research_topic": "t", "requirement_ids": []},
                {"task_id": "task-good", "research_topic": "t", "requirement_ids": []},
            ],
            snapshots=snapshots,
        )
        original_evaluate = deep_researcher.evaluate_subagent_handoff

        async def failing_evaluate(*args, **kwargs):
            # 第一个输出在评估阶段崩溃，第二个正常评估
            if not hasattr(failing_evaluate, "calls"):
                failing_evaluate.calls = 0
            failing_evaluate.calls += 1
            if failing_evaluate.calls == 1:
                raise RuntimeError("assessment exploded")
            return await original_evaluate(*args, **kwargs)

        monkeypatch.setattr(
            deep_researcher, "evaluate_subagent_handoff", failing_evaluate
        )
        finalize_config = _config(runs_dir=str(tmp_path))
        finalize_config["configurable"]["quality_evaluation_enabled"] = True
        await deep_researcher._finalize_async_research_outputs(
            {"coverage_ledger": {}, "coverage_contract": None, "research_risk_profile": None},
            finalize_config,
            Configuration.from_runnable_config(finalize_config),
            _CapturePublisher(),
        )
        good_warnings = [
            p for p in published
            if p.get("payload", {}).get("admission_status") == "rejected"
            and p.get("payload", {}).get("source_count") == 5
        ]
        assert good_warnings, "second output must still be finalized"


async def _async_none(*args, **kwargs):
    return None


class TestTerminalProjectionOverrides:
    @pytest.mark.asyncio
    async def test_stuck_running_tasks_render_cancelled_on_failed_run(self, tmp_path):
        from open_deep_research.events.public import RunEventStore

        store = RunEventStore("run-terminal-override", runs_dir=str(tmp_path))
        await store.append("run.created", payload={"status": "pending"}, dedupe_key="run:created")
        await store.append(
            "research.task.created",
            payload={"task_id": "t1", "wave_id": "w1", "mode": "async", "status": "pending"},
            dedupe_key="task:t1:created",
        )
        await store.append(
            "research.task.started",
            payload={"task_id": "t1", "wave_id": "w1", "mode": "async", "status": "researching"},
            dedupe_key="task:t1:started",
        )
        await store.append(
            "run.failed",
            payload={
                "status": "failed",
                "error_code": "internal_error",
                "result_status": "error",
            },
            dedupe_key="run:terminal",
        )
        projection = store.project()
        assert projection.status == "failed"
        assert projection.tasks["running"] == 0
        assert projection.tasks["cancelled"] == 1
        assert projection.task_items["t1"]["status"] == "cancelled"

    @pytest.mark.asyncio
    async def test_live_run_keeps_running_tasks(self, tmp_path):
        from open_deep_research.events.public import RunEventStore

        store = RunEventStore("run-live-override", runs_dir=str(tmp_path))
        await store.append("run.created", payload={"status": "pending"}, dedupe_key="run:created")
        await store.append(
            "research.task.started",
            payload={"task_id": "t1", "wave_id": "w1", "mode": "async", "status": "researching"},
            dedupe_key="task:t1:started",
        )
        projection = store.project()
        assert projection.tasks["running"] == 1
        assert projection.tasks["cancelled"] == 0


class TestGetRunTerminalManifestPreference:
    def test_stale_memory_record_yields_to_terminal_manifest(self, tmp_path, monkeypatch):
        import time as time_module
        from types import SimpleNamespace

        from fastapi.testclient import TestClient

        from open_deep_research import server
        from open_deep_research.run_context import RunContextStore

        run_id = "stale-record-run"
        monkeypatch.setenv("RUNS_DIR", str(tmp_path))
        monkeypatch.setenv("APP_ENV", "development")
        monkeypatch.setenv("LOCAL_DEV_AUTH_BYPASS", "true")
        monkeypatch.delenv("IAM_DATABASE_URL", raising=False)
        context = RunContextStore(run_id, runs_dir=str(tmp_path))
        context.initialize(
            "local-dev-user",
            {
                "configurable": {"runs_dir": str(tmp_path)},
                "metadata": {"run_id": run_id, "owner": "local-dev-user"},
            },
        )
        context._update_manifest(  # noqa: SLF001
            status="failed",
            result={"status": "error", "error": "boom"},
        )
        engine = SimpleNamespace(
            config={
                "configurable": {"runs_dir": str(tmp_path)},
                "metadata": {"run_id": run_id, "owner": "local-dev-user"},
            },
            context_store=context,
            started_at=time_module.time() - 60,
        )
        server._runs.clear()
        server._runs[run_id] = server.RunRecord(
            run_id=run_id,
            engine=engine,
            status="running",
        )
        client = TestClient(server.app, raise_server_exceptions=False)
        try:
            response = client.get(f"/runs/{run_id}")
        finally:
            server._runs.clear()
        assert response.status_code == 200
        assert response.json()["status"] == "failed"


