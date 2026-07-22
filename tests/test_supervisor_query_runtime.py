from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from open_deep_research.agents import deep_researcher
from open_deep_research.agents.query_engine import QueryEngine


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
