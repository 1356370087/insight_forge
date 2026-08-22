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
