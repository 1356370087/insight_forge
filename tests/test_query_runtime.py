from __future__ import annotations

import asyncio
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
from open_deep_research.agents.query_state import (
    InMemoryQueryCheckpointSink,
    QueryPhase,
    TerminalReason,
)
from open_deep_research.agents.tool_protocol import validate_tool_transcript
from open_deep_research.budgets import BudgetGate, RunBudgetLedger, RunBudgetPolicy
from open_deep_research.runtime_control import CancellationScope, RunCancelled
from open_deep_research.tools.adapters import adapt_langchain_tool
from open_deep_research.tools.base import ToolOrigin
from open_deep_research.tools.governance import ToolErrorType


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
async def test_query_cancels_an_inflight_model_call_without_waiting_for_timeout():
    scope = CancellationScope()
    model_started = asyncio.Event()
    model_drained = asyncio.Event()

    async def blocking_model(_messages):
        model_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            model_drained.set()

    async def consume_query():
        return [
            event
            async for event in query(QueryParams(
                messages=[HumanMessage(content="start")],
                system_prompt="system",
                model=FakeModel([]),
                config=_config(),
                call_model=blocking_model,
                model_timeout_seconds=180,
                cancellation_scope=scope,
            ))
        ]

    task = asyncio.create_task(consume_query())
    await asyncio.wait_for(model_started.wait(), 1)
    scope.request("cancel_requested")

    with pytest.raises(RunCancelled, match="cancel_requested"):
        await asyncio.wait_for(task, 1)
    assert model_drained.is_set()


@pytest.mark.asyncio
async def test_query_persists_terminal_before_propagating_model_cancellation():
    sink = InMemoryQueryCheckpointSink()

    async def cancelled_model(_messages):
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        async for _event in query(QueryParams(
            messages=[HumanMessage(content="start")],
            system_prompt="system",
            model=FakeModel([]),
            config=_config(),
            call_model=cancelled_model,
            checkpoint_sink=sink,
        )):
            pass

    terminal = sink.states[-1]
    assert terminal.phase is QueryPhase.TERMINAL
    assert terminal.terminal is not None
    assert terminal.terminal.reason is TerminalReason.CANCELLED
    assert terminal.pending_tool_batch is None


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
async def test_tool_batch_hook_can_use_a_longer_timeout_than_control_hooks():
    model = FakeModel([
        AIMessage(
            content="",
            tool_calls=[{"name": "echo_tool", "args": {"text": "hi"}, "id": "tc1"}],
        ),
    ])

    async def slow_batch_hook(_messages, tool_calls, _tools, _turn, _config):
        await asyncio.sleep(0.05)
        return ToolResultsHookResult(
            messages=[ToolMessage(
                content="slow batch completed",
                name="echo_tool",
                tool_call_id=tool_calls[0]["id"],
            )],
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
            tool_batch_hook=slow_batch_hook,
            hook_timeout_seconds=0.01,
            tool_batch_timeout_seconds=0.2,
        ))
    ]

    result_event = next(event for event in events if event.type == "query.tool_result")
    assert result_event.data["messages"][0].content == "slow batch completed"


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


@pytest.mark.asyncio
async def test_query_closes_missing_batch_hook_results_before_termination():
    model = FakeModel([
        AIMessage(
            content="",
            tool_calls=[
                {"name": "echo_tool", "args": {"text": "first"}, "id": "tc1"},
                {"name": "echo_tool", "args": {"text": "second"}, "id": "tc2"},
            ],
        ),
    ])

    async def incomplete_hook(_messages, tool_calls, _tools, _turn, _config):
        return ToolResultsHookResult(
            messages=[ToolMessage(
                content="batch:first",
                name="echo_tool",
                tool_call_id=tool_calls[0]["id"],
            )],
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
            tool_batch_hook=incomplete_hook,
        ))
    ]

    result_event = next(event for event in events if event.type == "query.tool_result")
    assert [message.tool_call_id for message in result_event.data["messages"]] == ["tc1", "tc2"]
    assert "runtime_missing_result" in str(result_event.data["messages"][1].content)
    assert result_event.data["should_continue"] is False
    assert events[-1].data["transition"]["reason"] == "tool_protocol_violation"


@pytest.mark.asyncio
async def test_query_rejects_duplicate_and_unknown_hook_results():
    model = FakeModel([
        AIMessage(
            content="",
            tool_calls=[{"name": "echo_tool", "args": {"text": "hi"}, "id": "tc1"}],
        ),
    ])

    async def malformed_hook(_messages, _tool_calls, _tools, _turn, _config):
        return ToolResultsHookResult(
            messages=[
                ToolMessage(content="first", name="echo_tool", tool_call_id="tc1"),
                ToolMessage(content="duplicate", name="echo_tool", tool_call_id="tc1"),
                ToolMessage(content="unknown", name="echo_tool", tool_call_id="other"),
            ],
            should_continue=True,
        )

    events = [
        event
        async for event in query(QueryParams(
            messages=[HumanMessage(content="start")],
            system_prompt=None,
            model=model,
            tools=[echo_tool],
            config=_config(),
            tool_batch_hook=malformed_hook,
        ))
    ]

    result_event = next(event for event in events if event.type == "query.tool_result")
    assert len(result_event.data["messages"]) == 1
    assert result_event.data["messages"][0].tool_call_id == "tc1"
    assert "runtime_duplicate_result" in str(result_event.data["messages"][0].content)
    diagnostics = result_event.data["protocol_diagnostics"]
    assert {item["code"] for item in diagnostics} == {
        "duplicate_tool_result",
        "unknown_tool_result",
    }
    assert result_event.data["should_continue"] is False


@pytest.mark.asyncio
async def test_query_closes_batch_when_hook_raises():
    model = FakeModel([
        AIMessage(
            content="",
            tool_calls=[{"name": "echo_tool", "args": {"text": "hi"}, "id": "tc1"}],
        ),
    ])

    async def raising_hook(*_args):
        raise RuntimeError("hook failed")

    events = [
        event
        async for event in query(QueryParams(
            messages=[HumanMessage(content="start")],
            system_prompt=None,
            model=model,
            tools=[echo_tool],
            config=_config(),
            tool_batch_hook=raising_hook,
        ))
    ]

    result_event = next(event for event in events if event.type == "query.tool_result")
    assert [message.tool_call_id for message in result_event.data["messages"]] == ["tc1"]
    assert "runtime_hook_error" in str(result_event.data["messages"][0].content)
    assert events[-1].data["transition"]["reason"] == "tool_protocol_violation"


@pytest.mark.asyncio
async def test_query_closes_batch_before_propagating_cancellation():
    model = FakeModel([
        AIMessage(
            content="",
            tool_calls=[
                {"name": "echo_tool", "args": {"text": "one"}, "id": "tc1"},
                {"name": "echo_tool", "args": {"text": "two"}, "id": "tc2"},
            ],
        ),
    ])

    async def cancelled_hook(*_args):
        raise asyncio.CancelledError

    sink = InMemoryQueryCheckpointSink()
    with pytest.raises(asyncio.CancelledError):
        async for _event in query(QueryParams(
            messages=[HumanMessage(content="start")],
            system_prompt=None,
            model=model,
            tools=[echo_tool],
            config=_config(),
            tool_batch_hook=cancelled_hook,
            checkpoint_sink=sink,
        )):
            pass

    terminal = sink.states[-1]
    assert terminal.phase is QueryPhase.TERMINAL
    assert terminal.terminal is not None
    assert terminal.terminal.reason is TerminalReason.CANCELLED
    assert terminal.pending_tool_batch is None
    assert terminal.pending_query_event is not None
    assert terminal.pending_query_event.event_type == "query.tool_result"
    assert [
        message.tool_call_id
        for message in terminal.pending_query_event.messages
    ] == ["tc1", "tc2"]
    assert all(
        "cancelled" in str(message.content)
        for message in terminal.pending_query_event.messages
    )
    validate_tool_transcript(list(terminal.messages))


@pytest.mark.asyncio
async def test_query_canonicalizes_missing_and_duplicate_tool_call_ids():
    first_response = AIMessage(
        content="",
        tool_calls=[
            {"name": "echo_tool", "args": {"text": "first"}, "id": ""},
            {"name": "echo_tool", "args": {"text": "second"}, "id": "dup"},
            {"name": "echo_tool", "args": {"text": "third"}, "id": "dup"},
        ],
    )
    first_response.additional_kwargs["tool_calls"] = [
        {"id": "", "type": "function", "function": {"name": "echo_tool", "arguments": '{"text":"first"}'}},
        {"id": "dup", "type": "function", "function": {"name": "echo_tool", "arguments": '{"text":"second"}'}},
        {"id": "dup", "type": "function", "function": {"name": "echo_tool", "arguments": '{"text":"third"}'}},
    ]
    model = FakeModel([
        first_response,
        AIMessage(content="done"),
    ])

    events = [
        event
        async for event in query(QueryParams(
            messages=[HumanMessage(content="start")],
            system_prompt=None,
            model=model,
            tools=[echo_tool],
            config=_config(),
        ))
    ]

    model_event = next(event for event in events if event.type == "query.model_event")
    canonical_ids = [call["id"] for call in model_event.data["message"].tool_calls]
    assert len(set(canonical_ids)) == 3
    assert all(canonical_ids)
    assert canonical_ids[1] == "dup"
    raw_ids = [
        call["id"]
        for call in model_event.data["message"].additional_kwargs["tool_calls"]
    ]
    assert raw_ids == canonical_ids
    result_event = next(event for event in events if event.type == "query.tool_result")
    assert [message.tool_call_id for message in result_event.data["messages"]] == canonical_ids


@pytest.mark.asyncio
async def test_terminal_tool_updates_are_emitted_once():
    model = FakeModel([
        AIMessage(
            content="",
            tool_calls=[{"name": "echo_tool", "args": {"text": "hi"}, "id": "tc1"}],
        ),
    ])

    async def hook(_messages, _tool_calls, outcomes, _tools, _turn, _config):
        return ToolResultsHookResult(
            messages=[outcomes[0].message],
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
        ))
    ]

    events_with_updates = [event for event in events if event.data.get("updates")]
    assert [event.type for event in events_with_updates] == ["query.tool_result"]
    assert "updates" not in events[-1].data


@pytest.mark.asyncio
async def test_query_rejects_tool_messages_in_additional_messages():
    model = FakeModel([
        AIMessage(
            content="",
            tool_calls=[{"name": "echo_tool", "args": {"text": "hi"}, "id": "tc1"}],
        ),
    ])

    async def hook(_messages, _tool_calls, outcomes, _tools, _turn, _config):
        return ToolResultsHookResult(
            messages=[outcomes[0].message],
            additional_messages=[
                ToolMessage(content="second channel", name="echo_tool", tool_call_id="tc1")
            ],
            should_continue=True,
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
        ))
    ]

    result_event = next(event for event in events if event.type == "query.tool_result")
    assert result_event.data["additional_messages"] == []
    assert result_event.data["should_continue"] is False
    assert any(
        item["code"] == "tool_message_in_additional_messages"
        for item in result_event.data["protocol_diagnostics"]
    )


@pytest.mark.asyncio
async def test_query_bounds_tool_concurrency():
    active = 0
    peak = 0

    @lc_tool("slow_echo")
    async def slow_echo_impl(text: str) -> str:
        """Echo text after yielding control."""
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.01)
        active -= 1
        return text

    slow_echo = adapt_langchain_tool(
        slow_echo_impl,
        origin=ToolOrigin.SYSTEM,
        concurrency_safe=True,
    )
    model = FakeModel([
        AIMessage(
            content="",
            tool_calls=[
                {"name": "slow_echo", "args": {"text": str(index)}, "id": f"tc{index}"}
                for index in range(5)
            ],
        ),
        AIMessage(content="done"),
    ])

    _events = [
        event
        async for event in query(QueryParams(
            messages=[HumanMessage(content="start")],
            system_prompt=None,
            model=model,
            tools=[slow_echo],
            config=_config(),
            max_concurrent_tools=2,
        ))
    ]

    assert peak == 2


@pytest.mark.asyncio
async def test_query_stops_when_model_call_budget_is_exhausted(tmp_path):
    model = FakeModel([
        AIMessage(
            content="",
            tool_calls=[{"name": "echo_tool", "args": {"text": "hi"}, "id": "tc1"}],
        ),
        AIMessage(content="done"),
    ])
    ledger = RunBudgetLedger(
        "budget-run",
        runs_dir=str(tmp_path),
        policy=RunBudgetPolicy(max_model_calls=1),
    )
    gate = BudgetGate(ledger=ledger)

    events = [
        event
        async for event in query(QueryParams(
            messages=[HumanMessage(content="start")],
            system_prompt="system",
            model=model,
            tools=[echo_tool],
            config=_config(),
            budget_gate=gate,
        ))
    ]

    assert len(model.calls) == 1
    completed = events[-1]
    assert completed.type == "query.completed"
    assert completed.data["transition"]["reason"] == "budget_exhausted"
    assert completed.data.get("budget_exhausted") is True
    assert ledger.snapshot().model_calls == 1
    assert ledger.snapshot().exhausted is True


@pytest.mark.asyncio
async def test_model_budget_keys_are_scoped_by_researcher_task(tmp_path):
    ledger = RunBudgetLedger(
        "shared-budget-run",
        runs_dir=str(tmp_path),
        policy=RunBudgetPolicy(max_model_calls=1),
    )
    gate = BudgetGate(ledger=ledger)
    first = FakeModel([AIMessage(content="done")])
    second = FakeModel([AIMessage(content="must not execute")])

    first_events = [
        event
        async for event in query(QueryParams(
            messages=[HumanMessage(content="first")],
            system_prompt=None,
            model=first,
            config=_config(),
            execution_namespace="researcher:task-1",
            budget_gate=gate,
        ))
    ]
    second_events = [
        event
        async for event in query(QueryParams(
            messages=[HumanMessage(content="second")],
            system_prompt=None,
            model=second,
            config=_config(),
            execution_namespace="researcher:task-2",
            budget_gate=gate,
        ))
    ]

    assert first_events[-1].data["transition"]["reason"] == "completed"
    assert second_events[-1].data["transition"]["reason"] == "budget_exhausted"
    assert len(first.calls) == 1
    assert second.calls == []


@pytest.mark.asyncio
async def test_tool_timeout_preserves_successful_sibling_result():
    @lc_tool("timed_echo")
    async def timed_echo_impl(text: str, delay: float) -> str:
        """Echo text after a configurable delay."""
        await asyncio.sleep(delay)
        return text

    timed_echo = adapt_langchain_tool(timed_echo_impl, origin=ToolOrigin.SYSTEM)
    model = FakeModel([
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "timed_echo",
                    "args": {"text": "fast", "delay": 0.0},
                    "id": "fast-call",
                },
                {
                    "name": "timed_echo",
                    "args": {"text": "slow", "delay": 0.2},
                    "id": "slow-call",
                },
            ],
        ),
        AIMessage(content="done"),
    ])
    observed_outcomes = []

    async def capture(_messages, _calls, outcomes, _tools, _turn, _config):
        observed_outcomes.extend(outcomes)
        return ToolResultsHookResult(
            messages=[outcome.message for outcome in outcomes],
            should_continue=True,
        )

    events = [
        event
        async for event in query(QueryParams(
            messages=[HumanMessage(content="start")],
            system_prompt=None,
            model=model,
            tools=[timed_echo],
            config=_config(),
            tool_results_hook=capture,
            tool_timeout_seconds=0.1,
        ))
    ]

    result = next(event for event in events if event.type == "query.tool_result")
    assert result.data["messages"][0].content == "fast"
    assert "timeout" in str(result.data["messages"][1].content).lower()
    assert observed_outcomes[0].error is None
    assert observed_outcomes[1].error is not None
    assert observed_outcomes[1].error.error_type == ToolErrorType.timeout


@pytest.mark.asyncio
async def test_query_stops_on_run_deadline(tmp_path):
    model = FakeModel([AIMessage(content="done")])
    gate = BudgetGate(
        ledger=None,
        deadline_at=0.0,
    )

    events = [
        event
        async for event in query(QueryParams(
            messages=[HumanMessage(content="start")],
            system_prompt=None,
            model=model,
            config=_config(),
            budget_gate=gate,
        ))
    ]

    assert len(model.calls) == 0
    assert events[-1].type == "query.completed"
    assert events[-1].data["transition"]["reason"] == "deadline_exceeded"


@pytest.mark.asyncio
async def test_query_retries_a_model_call_timeout_without_claiming_run_deadline():
    calls = 0

    async def flaky_model(_messages):
        nonlocal calls
        calls += 1
        if calls == 1:
            await asyncio.Event().wait()
        return AIMessage(content="recovered")

    events = [
        event
        async for event in query(QueryParams(
            messages=[HumanMessage(content="start")],
            system_prompt=None,
            model=FakeModel([]),
            config=_config(),
            call_model=flaky_model,
            model_timeout_seconds=0.01,
            model_transport_max_attempts=2,
        ))
    ]

    retry = next(event for event in events if event.type == "query.model_retry")
    assert retry.data == {
        "turn": 1,
        "attempt": 1,
        "max_attempts": 2,
        "reason": "model_timeout",
        "timeout_seconds": 0.01,
    }
    assert calls == 2
    assert events[-1].data["transition"]["reason"] == "completed"


@pytest.mark.asyncio
async def test_query_reports_model_timeout_after_bounded_attempts():
    calls = 0

    async def blocking_model(_messages):
        nonlocal calls
        calls += 1
        await asyncio.Event().wait()

    events = [
        event
        async for event in query(QueryParams(
            messages=[HumanMessage(content="start")],
            system_prompt=None,
            model=FakeModel([]),
            config=_config(),
            call_model=blocking_model,
            model_timeout_seconds=0.01,
            model_transport_max_attempts=2,
        ))
    ]

    assert calls == 2
    assert sum(event.type == "query.model_retry" for event in events) == 1
    assert events[-1].type == "query.completed"
    assert events[-1].data["transition"]["reason"] == "model_timeout"


@pytest.mark.asyncio
async def test_query_rejects_non_positive_tool_limits():
    model = FakeModel([AIMessage(content="done")])

    with pytest.raises(ValueError, match="max_concurrent_tools"):
        _events = [
            event
            async for event in query(QueryParams(
                messages=[HumanMessage(content="start")],
                system_prompt=None,
                model=model,
                tools=[echo_tool],
                config=_config(),
                max_concurrent_tools=0,
            ))
        ]

    with pytest.raises(ValueError, match="max_tool_batch_size"):
        _events = [
            event
            async for event in query(QueryParams(
                messages=[HumanMessage(content="start")],
                system_prompt=None,
                model=model,
                tools=[echo_tool],
                config=_config(),
                max_tool_batch_size=-1,
            ))
        ]


def test_validate_tool_transcript_allows_only_explicit_pending_tail():
    pending = [
        HumanMessage(content="start"),
        AIMessage(
            content="",
            tool_calls=[{"name": "echo_tool", "args": {"text": "hi"}, "id": "tc1"}],
        ),
    ]

    validate_tool_transcript(pending, allow_pending_tail=True)
    with pytest.raises(ValueError, match="tool_batch_not_closed"):
        validate_tool_transcript(pending)


def test_validate_tool_transcript_rejects_orphan_and_out_of_order_results():
    orphan = [
        HumanMessage(content="start"),
        ToolMessage(content="orphan", name="echo_tool", tool_call_id="tc1"),
    ]
    with pytest.raises(ValueError, match="orphan_tool_result"):
        validate_tool_transcript(orphan)

    out_of_order = [
        AIMessage(
            content="",
            tool_calls=[
                {"name": "echo_tool", "args": {"text": "one"}, "id": "tc1"},
                {"name": "echo_tool", "args": {"text": "two"}, "id": "tc2"},
            ],
        ),
        ToolMessage(content="two", name="echo_tool", tool_call_id="tc2"),
        ToolMessage(content="one", name="echo_tool", tool_call_id="tc1"),
    ]
    with pytest.raises(ValueError, match="tool_result_order_mismatch"):
        validate_tool_transcript(out_of_order)
