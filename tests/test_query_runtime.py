from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool as lc_tool

from open_deep_research.agents.query import (
    BeforeTurnHookResult,
    ContextPolicy,
    QueryParams,
    StopHookResult,
    ToolResultsHookResult,
    prepare_messages_for_query,
    query,
)
from open_deep_research.tools.adapters import adapt_langchain_tool
from open_deep_research.tools.base import ToolOrigin


def _config(**configurable: Any) -> dict[str, Any]:
    return {"configurable": configurable, "metadata": {"run_id": "query-test"}}


@lc_tool("echo_tool")
async def _echo_tool_impl(text: str) -> str:
    """Echo text."""
    return f"echo:{text}"


echo_tool = adapt_langchain_tool(
    _echo_tool_impl,
    origin=ToolOrigin.SYSTEM,
)


class FakeModel:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls: list[list[Any]] = []

    def bind_tools(self, _tools):
        return self

    def with_config(self, _config):
        return self

    async def ainvoke(self, messages):
        self.calls.append(messages)
        return self.responses.pop(0)


@pytest.mark.asyncio
async def test_query_continues_after_tool_results_then_completes():
    model = FakeModel([
        AIMessage(
            content="",
            tool_calls=[{"name": "echo_tool", "args": {"text": "hi"}, "id": "tc1"}],
        ),
        AIMessage(content="done"),
    ])
    events = [
        event
        async for event in query(QueryParams(
            messages=[HumanMessage(content="start")],
            system_prompt="system",
            model=model,
            tools=[echo_tool],
            config=_config(),
            max_turns=5,
        ))
    ]

    assert any(event.type == "query.tool_result" for event in events)
    completed = [event for event in events if event.type == "query.completed"][-1]
    assert completed.data["transition"]["reason"] == "completed"
    assert len(model.calls) == 2


def test_messages_for_query_is_trimmed_view_not_full_transcript():
    messages = [HumanMessage(content=f"m{i}") for i in range(6)]
    view = prepare_messages_for_query(messages, ContextPolicy(keep_last_messages=2))

    assert len(messages) == 6
    assert [m.content for m in view] == ["m4", "m5"]


def test_messages_for_query_preserves_tool_call_boundary():
    messages = [
        HumanMessage(content="start"),
        AIMessage(
            content="",
            tool_calls=[{"name": "echo_tool", "args": {"text": "hi"}, "id": "tc1"}],
        ),
        ToolMessage(content="echo:hi", name="echo_tool", tool_call_id="tc1"),
    ]

    view = prepare_messages_for_query(messages, ContextPolicy(keep_last_messages=1))

    assert isinstance(view[0], AIMessage)
    assert isinstance(view[1], ToolMessage)
    assert view[0].tool_calls[0]["id"] == view[1].tool_call_id


@pytest.mark.asyncio
async def test_tool_results_hook_can_transform_results_and_complete():
    model = FakeModel([
        AIMessage(
            content="",
            tool_calls=[{"name": "echo_tool", "args": {"text": "hi"}, "id": "tc1"}],
        ),
    ])

    async def hook(_messages, _tool_calls, outcomes, _tools, _turn, _config):
        transformed = outcomes[0].message.model_copy()
        transformed.content = "protected:echo:hi"
        return ToolResultsHookResult(
            messages=[transformed],
            updates={"evidence_registry": [{"id": "ev-1"}]},
            should_continue=False,
        )

    events = [
        event
        async for event in query(QueryParams(
            messages=[HumanMessage(content="start")],
            system_prompt=None,
            model=model,
            tools=[echo_tool],
            config=_config(),
            tool_results_hook=hook,
            max_turns=5,
        ))
    ]

    result_event = next(event for event in events if event.type == "query.tool_result")
    assert result_event.data["messages"][0].content == "protected:echo:hi"
    assert result_event.data["updates"] == {"evidence_registry": [{"id": "ev-1"}]}
    completed = events[-1]
    assert completed.type == "query.completed"
    assert completed.data["transition"]["reason"] == "completed"


@pytest.mark.asyncio
async def test_tool_batch_hook_can_own_execution_and_complete():
    model = FakeModel([
        AIMessage(
            content="",
            tool_calls=[{"name": "echo_tool", "args": {"text": "hi"}, "id": "tc1"}],
        ),
    ])

    async def batch_hook(_messages, tool_calls, _tools, _turn, _config):
        return ToolResultsHookResult(
            messages=[ToolMessage(
                content="batch:hi",
                name="echo_tool",
                tool_call_id=tool_calls[0]["id"],
            )],
            updates={"batch_owned": True},
            should_continue=False,
        )

    events = [
        event
        async for event in query(QueryParams(
            messages=[HumanMessage(content="start")],
            system_prompt=None,
            model=model,
            tools=[echo_tool],
            config=_config(),
            tool_batch_hook=batch_hook,
        ))
    ]

    result_event = next(event for event in events if event.type == "query.tool_result")
    assert result_event.data["messages"][0].content == "batch:hi"
    assert result_event.data["updates"] == {"batch_owned": True}
    assert result_event.data["should_continue"] is False


@pytest.mark.asyncio
async def test_stop_hook_can_complete_with_updates():
    model = FakeModel([AIMessage(content="done")])

    async def stop_with_update(_messages, _config):
        return StopHookResult(
            should_continue=False,
            updates={"notes": ["complete"]},
            reason="completed",
        )

    events = [
        event
        async for event in query(QueryParams(
            messages=[HumanMessage(content="start")],
            system_prompt=None,
            model=model,
            config=_config(),
            stop_hooks=[stop_with_update],
        ))
    ]

    assert events[-1].type == "query.completed"
    assert events[-1].data["updates"] == {"notes": ["complete"]}


@pytest.mark.asyncio
async def test_before_turn_hook_can_replace_context():
    model = FakeModel([AIMessage(content="done")])

    async def replace_context(_messages, _turn, _config):
        return BeforeTurnHookResult(
            replace_messages=[HumanMessage(content="compacted context")]
        )

    events = [
        event
        async for event in query(QueryParams(
            messages=[HumanMessage(content="old context")],
            system_prompt=None,
            model=model,
            config=_config(),
            before_turn_hooks=[replace_context],
        ))
    ]

    assert model.calls[0][0].content == "compacted context"
    assert events[-1].data["messages"][0].content == "compacted context"


@pytest.mark.asyncio
async def test_stop_hook_can_block_stop_and_continue():
    model = FakeModel([AIMessage(content="first"), AIMessage(content="done")])
    called = {"count": 0}

    async def hook(_messages, _config):
        called["count"] += 1
        if called["count"] == 1:
            return StopHookResult(
                should_continue=True,
                messages=[HumanMessage(content="continue")],
                reason="stop_hook_blocked",
            )
        return None

    events = [
        event
        async for event in query(QueryParams(
            messages=[HumanMessage(content="start")],
            system_prompt=None,
            model=model,
            config=_config(),
            stop_hooks=[hook],
            max_turns=5,
        ))
    ]

    transitions = [event for event in events if event.type == "query.transition"]
    assert transitions[0].data["transition"]["reason"] == "stop_hook_blocked"
    assert len(model.calls) == 2
