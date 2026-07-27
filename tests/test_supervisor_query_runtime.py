from __future__ import annotations

import asyncio
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from open_deep_research.agents import deep_researcher
from open_deep_research.agents.query_engine import QueryEngine
from open_deep_research.completion import accepted_evidence
from open_deep_research.quality import HandoffAssessment
from open_deep_research.run_context import RunContextStore
from open_deep_research.runtime import apply_update_to_state
from open_deep_research.runtime_control import RunCancelled


class FakeSupervisorModel:
    def __init__(self, responses: list[AIMessage]):
        self.responses = list(responses)
        self.calls: list[list[Any]] = []

    def bind_tools(self, _tools):
        return self

    def with_config(self, _config):
        return self

    async def ainvoke(self, messages):
        self.calls.append(list(messages))
        return self.responses.pop(0)


def _config(**overrides: Any) -> dict[str, Any]:
    return {
        "configurable": {
            "search_api": "none",
            "observability_enabled": False,
            "quality_evaluation_enabled": False,
            "query_session_persistence_enabled": False,
            "event_log_enabled": False,
            "max_researcher_iterations": 3,
            **overrides,
        },
        "metadata": {"run_id": "supervisor-query-runtime"},
    }


def _main_state(*, with_evidence: bool = False) -> dict[str, Any]:
    return {
        "supervisor_messages": [
            SystemMessage(content="supervisor system"),
            HumanMessage(content="research brief"),
        ],
        "research_brief": "research brief",
        "enable_async_research": False,
        "evidence_registry": (
            [{"id": "ev-1", "source_url": "https://example.com"}]
            if with_evidence
            else []
        ),
    }


@pytest.mark.asyncio
async def test_supervisor_uses_unified_query_loop(monkeypatch) -> None:
    model = FakeSupervisorModel([
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "think_tool",
                    "args": {"reflection": "check coverage"},
                    "id": "think-1",
                }
            ],
        ),
        AIMessage(
            content="",
            tool_calls=[{"name": "ResearchComplete", "args": {}, "id": "done-1"}],
        ),
    ])

    async def legacy_node_must_not_run(*_args, **_kwargs):
        raise AssertionError("legacy supervisor node loop was invoked")

    monkeypatch.setattr(deep_researcher, "configurable_model", model)
    monkeypatch.setattr(deep_researcher, "supervisor", legacy_node_must_not_run)
    monkeypatch.setattr(deep_researcher, "supervisor_tools", legacy_node_must_not_run)

    result = await QueryEngine(_config())._run_supervisor(_main_state(with_evidence=True))

    messages = result["supervisor_messages"]["value"]
    assert len(model.calls) == 2
    assert any(getattr(message, "name", None) == "think_tool" for message in messages)
    assert result["research_brief"] == "research brief"


@pytest.mark.asyncio
async def test_supervisor_preserves_research_complete_on_last_allowed_turn(
    monkeypatch,
) -> None:
    model = FakeSupervisorModel([
        AIMessage(
            content="",
            tool_calls=[{
                "name": "think_tool",
                "args": {"reflection": "first coverage check"},
                "id": "think-1",
            }],
        ),
        AIMessage(
            content="",
            tool_calls=[{
                "name": "think_tool",
                "args": {"reflection": "final coverage check"},
                "id": "think-2",
            }],
        ),
        AIMessage(
            content="",
            tool_calls=[{"name": "ResearchComplete", "args": {}, "id": "done-1"}],
        ),
    ])
    monkeypatch.setattr(deep_researcher, "configurable_model", model)

    result = await QueryEngine(
        _config(max_researcher_iterations=3)
    )._run_supervisor(_main_state(with_evidence=True))

    assert len(model.calls) == 3
    assert result["completion_decision"]["value"] == {
        "action": "complete",
        "reason": "explicit_completion",
        "gaps": [],
    }, (
        "ResearchComplete succeeded on the final legal turn, so max_turns "
        "must not downgrade the run to complete_partial."
    )


@pytest.mark.asyncio
async def test_supervisor_research_batch_uses_task_timeout_not_hook_timeout(
    monkeypatch,
) -> None:
    model = FakeSupervisorModel([
        AIMessage(
            content="",
            tool_calls=[{
                "name": "think_tool",
                "args": {"reflection": "check coverage"},
                "id": "think-1",
            }],
        ),
        AIMessage(
            content="",
            tool_calls=[{"name": "ResearchComplete", "args": {}, "id": "done-1"}],
        ),
    ])
    original = deep_researcher._execute_supervisor_tools

    async def slow_supervisor_batch(state, config):
        await asyncio.sleep(0.05)
        return await original(state, config)

    monkeypatch.setattr(deep_researcher, "configurable_model", model)
    monkeypatch.setattr(
        deep_researcher,
        "_execute_supervisor_tools",
        slow_supervisor_batch,
    )

    result = await QueryEngine(_config(
        hook_timeout_seconds=0.01,
        task_timeout_seconds=1,
    ))._run_supervisor(_main_state(with_evidence=True))

    messages = result["supervisor_messages"]["value"]
    assert len(model.calls) == 2
    assert not any("runtime_hook_error" in str(message.content) for message in messages)


@pytest.mark.asyncio
async def test_supervisor_model_call_is_cancelled_by_outer_engine(monkeypatch) -> None:
    model_started = asyncio.Event()
    model_drained = asyncio.Event()

    class BlockingSupervisorModel(FakeSupervisorModel):
        async def ainvoke(self, messages):
            self.calls.append(list(messages))
            model_started.set()
            try:
                await asyncio.Event().wait()
            finally:
                model_drained.set()

    engine = QueryEngine(_config(model_call_timeout_seconds=180))
    monkeypatch.setattr(
        deep_researcher,
        "configurable_model",
        BlockingSupervisorModel([]),
    )
    task = asyncio.create_task(engine._run_supervisor(_main_state(with_evidence=True)))
    await asyncio.wait_for(model_started.wait(), 2)

    engine.interrupt()

    with pytest.raises(RunCancelled, match="cancel_requested"):
        await asyncio.wait_for(task, 2)
    assert model_drained.is_set()


@pytest.mark.asyncio
async def test_supervisor_resume_from_tool_boundary_does_not_repeat_model(monkeypatch) -> None:
    model = FakeSupervisorModel([])
    restored = {
        **_main_state(with_evidence=True),
        "supervisor_messages": [
            *_main_state(with_evidence=True)["supervisor_messages"],
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "ResearchComplete", "args": {}, "id": "resume-done"}
                ],
            ),
        ],
        "research_iterations": 1,
    }
    monkeypatch.setattr(deep_researcher, "configurable_model", model)

    result = await QueryEngine(_config())._run_supervisor(
        _main_state(with_evidence=True),
        restored_state=restored,
        start_step="supervisor_tools",
    )

    assert model.calls == []
    assert result["research_brief"] == "research brief"


@pytest.mark.asyncio
async def test_supervisor_no_tool_response_uses_domain_exit_policy(monkeypatch) -> None:
    model = FakeSupervisorModel([
        AIMessage(content="research complete"),
        AIMessage(
            content="",
            tool_calls=[{"name": "ResearchComplete", "args": {}, "id": "done-1"}],
        ),
    ])

    async def legacy_node_must_not_run(*_args, **_kwargs):
        raise AssertionError("legacy supervisor node loop was invoked")

    monkeypatch.setattr(deep_researcher, "configurable_model", model)
    monkeypatch.setattr(deep_researcher, "supervisor", legacy_node_must_not_run)
    monkeypatch.setattr(deep_researcher, "supervisor_tools", legacy_node_must_not_run)

    result = await QueryEngine(_config())._run_supervisor(_main_state(with_evidence=True))

    messages = result["supervisor_messages"]["value"]
    assert len(model.calls) == 2
    assert messages[2].content == "research complete"
    assert result["notes"]["value"] == []


@pytest.mark.asyncio
async def test_supervisor_rejects_explicit_completion_without_evidence(monkeypatch) -> None:
    model = FakeSupervisorModel([
        AIMessage(
            content="",
            tool_calls=[{"name": "ResearchComplete", "args": {}, "id": "done-1"}],
        ),
        AIMessage(
            content="",
            tool_calls=[{"name": "ResearchComplete", "args": {}, "id": "done-2"}],
        ),
    ])
    calls = {"count": 0}
    original = deep_researcher._execute_supervisor_tools

    async def execute_with_evidence(state, config):
        command = await original(state, config)
        calls["count"] += 1
        if calls["count"] == 1:
            return command
        return command.__class__(
            goto=command.goto,
            update={
                **command.update,
                "evidence_registry": [
                    {"id": "ev-1", "source_url": "https://example.com"}
                ],
            },
        )

    monkeypatch.setattr(deep_researcher, "configurable_model", model)
    monkeypatch.setattr(deep_researcher, "_execute_supervisor_tools", execute_with_evidence)

    result = await QueryEngine(_config())._run_supervisor(_main_state())

    assert len(model.calls) == 2
    assert result["completion_decision"]["value"]["action"] == "complete"
    assert result["evidence_registry"]["value"]


@pytest.mark.asyncio
async def test_supervisor_can_readmit_rejected_artifact_after_verified_read(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("QUALITY_EVALUATION_ENABLED", "true")
    run_id = "supervisor-readmission"
    task_id = "research-1"
    artifact = {
        "schema_version": 1,
        "task_id": task_id,
        "research_topic": "Verify the official standard.",
        "compressed_research": "Official evidence was collected.",
        "researcher_messages": [],
        "raw_notes": ["Official evidence from the persisted artifact."],
        "candidate_registry": [],
        "document_registry": [],
        "evidence_registry": [
            {
                "evidence_id": "ev-readmitted",
                "claim": "The official standard defines the requirement.",
                "supporting_excerpt": (
                    "The official standard defines the requirement."
                ),
                "source_url": "https://official.example/standard",
                "security_status": "accepted",
            }
        ],
        "web_research_iterations": [],
        "result_assessment": {},
        "metrics": {"sources_read": 1},
    }
    store = RunContextStore(run_id, runs_dir=str(tmp_path))
    digest = store.persist_task_result(task_id, artifact)
    artifact_ref = {
        "path": f"context/artifacts/research_tasks/{task_id}.json",
        "sha256": digest,
        "available_sections": ["raw_notes", "evidence_registry"],
    }
    assessments = [
        HandoffAssessment(
            accepted=False,
            relevance=4,
            source_quality=5,
            evidence_coverage=2,
            groundedness=2,
            missing_information=["Inspect the exact persisted excerpt."],
            reason="The compact handoff is insufficient.",
        ),
        HandoffAssessment(
            accepted=True,
            relevance=5,
            source_quality=5,
            evidence_coverage=5,
            groundedness=5,
            reason="The verified artifact excerpt resolves the gap.",
        ),
    ]
    assessment_calls: list[dict[str, Any]] = []

    async def fake_evaluate_handoff(
        research_topic: str,
        handoff: dict[str, Any],
        _config: dict[str, Any],
    ) -> HandoffAssessment:
        assessment_calls.append({
            "research_topic": research_topic,
            "handoff": handoff,
        })
        return assessments[len(assessment_calls) - 1]

    monkeypatch.setattr(
        deep_researcher,
        "evaluate_subagent_handoff",
        fake_evaluate_handoff,
    )
    state = _main_state()
    state["research_artifact_refs"] = {task_id: artifact_ref}
    config = _config(
        quality_evaluation_enabled=True,
        runs_dir=str(tmp_path),
    )
    config["metadata"]["run_id"] = run_id

    async def execute_turn(tool_call: dict[str, Any]) -> None:
        state["supervisor_messages"].append(
            AIMessage(content="", tool_calls=[tool_call])
        )
        command = await deep_researcher._execute_supervisor_tools(state, config)
        apply_update_to_state(state, command.update)

    await execute_turn({
        "name": "think_tool",
        "args": {"reflection": "plan one bounded task"},
        "id": "think-1",
    })
    await execute_turn({
        "name": "ConductResearch",
        "args": {
            "research_topic": "Verify the official standard.",
            "display_title": "Official standard",
        },
        "id": task_id,
    })
    assert not accepted_evidence(state)
    await execute_turn({
        "name": "ReadResearchArtifact",
        "args": {
            "task_id": task_id,
            "artifact_sha256": digest,
            "section": "evidence_registry",
        },
        "id": "read-1",
    })

    assert len(assessment_calls) == 2
    assert accepted_evidence(state)[0]["evidence_id"] == "ev-readmitted"
    assert "accepted_after_artifact_reassessment" in str(
        state["supervisor_messages"][-1].content
    )


@pytest.mark.asyncio
async def test_legacy_supervisor_uses_compatibility_completion_policy(monkeypatch) -> None:
    model = FakeSupervisorModel([
        AIMessage(
            content="",
            tool_calls=[{"name": "ResearchComplete", "args": {}, "id": "done-1"}],
        )
    ])
    monkeypatch.setattr(deep_researcher, "configurable_model", model)

    result = await QueryEngine(
        _config(web_pipeline_mode="legacy")
    )._run_supervisor(_main_state())

    assert len(model.calls) == 1
    assert result["completion_decision"]["value"]["action"] == "complete"
