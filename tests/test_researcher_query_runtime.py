from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool as lc_tool

from open_deep_research.agents import deep_researcher
from open_deep_research.agents.query_engine import ResearcherQueryEngine
from open_deep_research.completion import CompletionDecision
from open_deep_research.tasks import registry as task_registry
from open_deep_research.tools.adapters import adapt_langchain_tool
from open_deep_research.tools.base import ToolOrigin
from open_deep_research.tools.governance import (
    GovernedToolCallResult,
    ToolError,
    ToolErrorType,
)


@lc_tool("research_echo")
async def _research_echo_impl(text: str) -> str:
    """Echo research evidence."""
    return f"evidence:{text}"


research_echo = adapt_langchain_tool(
    _research_echo_impl,
    origin=ToolOrigin.SEARCH,
)


class FakeResearchModel:
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
            "max_react_tool_calls": 5,
            **overrides,
        },
        "metadata": {"run_id": "research-query-runtime", "task_id": "researcher-1"},
    }


@pytest.mark.asyncio
async def test_researcher_engine_uses_unified_query_loop(monkeypatch) -> None:
    model = FakeResearchModel([
        AIMessage(
            content="",
            tool_calls=[
                {"name": "research_echo", "args": {"text": "fact"}, "id": "tool-1"},
            ],
        ),
        AIMessage(
            content="",
            tool_calls=[{"name": "ResearchComplete", "args": {}, "id": "done-1"}],
        ),
    ])

    async def fake_get_all_tools(_config):
        return [research_echo, *deep_researcher.build_supervisor_tools({})[-2:-1]]

    async def old_node_must_not_run(*_args, **_kwargs):
        raise AssertionError("legacy researcher node loop was invoked")

    async def fake_compress(state, _config):
        contents = [str(message.content) for message in state["researcher_messages"]]
        return {
            "compressed_research": "compressed",
            "raw_notes": contents,
        }

    monkeypatch.setattr(deep_researcher, "configurable_model", model)
    monkeypatch.setattr(deep_researcher, "get_all_tools", fake_get_all_tools)
    monkeypatch.setattr(deep_researcher, "researcher", old_node_must_not_run)
    monkeypatch.setattr(deep_researcher, "researcher_tools", old_node_must_not_run)
    monkeypatch.setattr(deep_researcher, "compress_research", fake_compress)

    result = await ResearcherQueryEngine(_config()).ainvoke(
        {
            "researcher_messages": [HumanMessage(content="topic")],
            "research_topic": "topic",
            "memory_context": None,
            "evidence_registry": [{"id": "ev-1", "source_url": "https://example.com"}],
        },
    )

    assert result["compressed_research"] == "compressed"
    assert result["tool_call_iterations"] == 2
    assert any("evidence:fact" in note for note in result["raw_notes"])
    assert len(model.calls) == 2


@pytest.mark.asyncio
async def test_researcher_observes_cancellation_at_terminal_boundary(monkeypatch) -> None:
    registry = task_registry.TaskRegistry()
    record = registry.restore(task_registry.TaskRecord(
        task_id="researcher-1",
        run_id="research-query-runtime",
        research_topic="topic",
    ))

    class TerminalCancellingModel(FakeResearchModel):
        async def ainvoke(self, messages):
            response = await super().ainvoke(messages)
            record.cancelled.set()
            return response

    model = TerminalCancellingModel([AIMessage(content="research complete")])

    async def fake_get_all_tools(_config):
        return [research_echo]

    async def compress_must_not_run(*_args, **_kwargs):
        raise AssertionError("cancelled research must not be compressed")

    monkeypatch.setattr(task_registry, "get_task_registry", lambda: registry)
    monkeypatch.setattr(deep_researcher, "configurable_model", model)
    monkeypatch.setattr(deep_researcher, "get_all_tools", fake_get_all_tools)
    monkeypatch.setattr(deep_researcher, "compress_research", compress_must_not_run)

    result = await ResearcherQueryEngine(_config()).ainvoke(
        {
            "researcher_messages": [HumanMessage(content="topic")],
            "research_topic": "topic",
            "memory_context": None,
        },
    )

    assert result["cancelled"] is True
    assert result["compressed_research"] == ""
    assert result["raw_notes"] == []


@pytest.mark.asyncio
async def test_failed_research_complete_is_not_a_successful_completion():
    call = {"name": "ResearchComplete", "args": {}, "id": "done-1"}
    error = ToolError(
        error_type=ToolErrorType.permission_denied,
        tool_name="ResearchComplete",
        message="not permitted",
    )
    outcome = GovernedToolCallResult(
        message=error.to_tool_message("done-1"),
        error=error,
    )

    _messages, update = await deep_researcher.prepare_researcher_tool_outcomes(
        [call],
        [outcome],
        {},
        _config(),
    )

    assert update["research_complete_requested"] is True
    assert update["research_complete_succeeded"] is False


@pytest.mark.asyncio
async def test_no_tool_response_without_evidence_does_not_complete(monkeypatch) -> None:
    model = FakeResearchModel([
        AIMessage(content="I think this is done"),
        AIMessage(
            content="",
            tool_calls=[
                {"name": "research_echo", "args": {"text": "fact"}, "id": "tool-1"},
            ],
        ),
        AIMessage(
            content="",
            tool_calls=[{"name": "ResearchComplete", "args": {}, "id": "done-1"}],
        ),
    ])

    async def fake_get_all_tools(_config):
        return [research_echo, *deep_researcher.build_supervisor_tools({})[-2:-1]]

    async def fake_compress(state, _config):
        return {"compressed_research": "compressed", "raw_notes": ["evidence"]}

    monkeypatch.setattr(deep_researcher, "configurable_model", model)
    monkeypatch.setattr(deep_researcher, "get_all_tools", fake_get_all_tools)
    monkeypatch.setattr(deep_researcher, "compress_research", fake_compress)

    result = await ResearcherQueryEngine(_config()).ainvoke({
        "researcher_messages": [HumanMessage(content="topic")],
        "research_topic": "topic",
        "evidence_registry": [{"id": "ev-1", "source_url": "https://example.com"}],
    })

    assert len(model.calls) == 3
    assert result["completion_decision"]["action"] == CompletionDecision.COMPLETE.value


@pytest.mark.asyncio
async def test_researcher_collects_permission_denials(monkeypatch) -> None:
    model = FakeResearchModel([
        AIMessage(
            content="",
            tool_calls=[
                {"name": "research_echo", "args": {"text": "fact"}, "id": "denied-1"},
            ],
        ),
    ])

    async def fake_get_all_tools(_config):
        return [research_echo, deep_researcher.think_tool]

    async def fake_compress(state, _config):
        return {"compressed_research": "", "raw_notes": []}

    monkeypatch.setattr(deep_researcher, "configurable_model", model)
    monkeypatch.setattr(deep_researcher, "get_all_tools", fake_get_all_tools)
    monkeypatch.setattr(deep_researcher, "compress_research", fake_compress)

    result = await ResearcherQueryEngine(_config(
        researcher_tool_whitelist=["think_tool"],
        max_react_tool_calls=1,
    )).ainvoke({
        "researcher_messages": [HumanMessage(content="topic")],
        "research_topic": "topic",
    })

    assert result["permission_denials"] == [{
        "tool_call_id": "denied-1",
        "tool_name": "research_echo",
        "role": "researcher",
        "reason_code": "permission_denied",
        "turn": 1,
        "task_id": "researcher-1",
    }]
