from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool as lc_tool

from open_deep_research.agents.query import (
    ContextPolicy,
    QueryParams,
    StopHookResult,
    prepare_messages_for_query,
    query,
)


def _config(**configurable: Any) -> dict[str, Any]:
    return {"configurable": configurable, "metadata": {"run_id": "query-test"}}


@lc_tool
async def echo_tool(text: str) -> str:
    """Echo text."""
    return f"echo:{text}"


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
