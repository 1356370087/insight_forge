from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from pydantic import BaseModel

from open_deep_research.agents.context_compiler import ContextCompiler
from open_deep_research.agents.model_recovery import (
    ModelCandidate,
    ModelErrorKind,
    resolve_model_context_window,
)
from open_deep_research.agents.query import (
    QueryParams,
    StopHookResult,
    ToolResultsHookResult,
    query,
)
from open_deep_research.agents.query_engine import QueryEngine
from open_deep_research.agents.query_state import (
    ContinueReason,
    InMemoryQueryCheckpointSink,
    PendingToolBatch,
    QualityRecoveryState,
    QueryLoopState,
    QueryPhase,
    QueryStateAction,
    TerminalReason,
    advance,
)
from open_deep_research.agents.tool_protocol import validate_tool_transcript
from open_deep_research.run_context import (
    RunConfigurationError,
    RunContextStore,
)
from open_deep_research.tools.base import (
    ToolContext,
    ToolEffect,
    ToolOrigin,
    ToolResult,
    build_tool,
)
from open_deep_research.tools.governance import (
    AgentRole,
    GovernedToolCallResult,
)


def _config() -> dict[str, Any]:
    return {
        "configurable": {
            "observability_enabled": False,
            "search_api": "none",
        },
        "metadata": {"run_id": "query-v2-test"},
    }


async def _events(params: QueryParams):
    return [event async for event in query(params)]


def test_query_state_is_immutable_and_serializes_pending_tool_results() -> None:
    initial = QueryLoopState(
        state_key="researcher:task-1",
        role=AgentRole.RESEARCHER,
        messages=(HumanMessage(content="brief"),),
    )
    calling = advance(
        initial,
        QueryStateAction(
            phase=QueryPhase.CALLING_MODEL,
            reason=ContinueReason.NEXT_TURN.value,
            changes={"turn": 1},
        ),
    )
    pending = PendingToolBatch(
        batch_id="batch-1",
        tool_calls=({"name": "read", "args": {}, "id": "tc-1"},),
        committed_tool_call_ids=("tc-1",),
        committed_results=(
            ToolMessage(
                content="result",
                name="read",
                tool_call_id="tc-1",
            ),
        ),
    )
    executing = advance(
        calling,
        QueryStateAction(
            phase=QueryPhase.EXECUTING_TOOLS,
            reason=ContinueReason.NEXT_TURN.value,
            changes={"pending_tool_batch": pending},
        ),
    )

    restored = QueryLoopState.from_snapshot(executing.to_snapshot())

    assert initial.phase is QueryPhase.PREPARING
    assert initial.revision == 0
    assert calling.revision == 1
    assert restored == executing
    assert (
        restored.pending_tool_batch.committed_results[0].content
        == "result"
    )


def test_query_state_v2_persists_quality_recovery_and_reads_v1() -> None:
    state = QueryLoopState(
        state_key="researcher:task-1",
        role=AgentRole.RESEARCHER,
        messages=(HumanMessage(content="brief"),),
        quality_recovery=QualityRecoveryState(
            attempts=1,
            active=True,
            target_requirement_ids=("COV-01",),
            triggering_assessment_revision=7,
        ),
    )

    restored = QueryLoopState.from_snapshot(state.to_snapshot())
    legacy_snapshot = state.to_snapshot()
    legacy_snapshot["schema_version"] = 1
    legacy_snapshot.pop("quality_recovery")
    legacy = QueryLoopState.from_snapshot(legacy_snapshot)

    assert restored.quality_recovery == state.quality_recovery
    assert legacy.schema_version == 1
    assert legacy.quality_recovery == QualityRecoveryState()


def test_query_state_rejects_illegal_terminal_transition() -> None:
    state = QueryLoopState(
        state_key="supervisor",
        role=AgentRole.SUPERVISOR,
        messages=(HumanMessage(content="brief"),),
    )

    with pytest.raises(ValueError, match="terminal_query_state_requires_outcome"):
        advance(
            state,
            QueryStateAction(
                phase=QueryPhase.TERMINAL,
                reason=TerminalReason.COMPLETED.value,
            ),
        )


@pytest.mark.asyncio
async def test_context_compiler_accounts_for_full_envelope_and_keeps_tool_pair() -> None:
    compiler = ContextCompiler(
        model_context_window_overrides={"test:model": 8_000},
    )
    messages = [
        HumanMessage(content="authoritative brief"),
        HumanMessage(content="old " * 2_000),
        AIMessage(
            content="",
            tool_calls=[
                {"name": "read", "args": {}, "id": "tc-1"},
            ],
        ),
        ToolMessage(
            content="evidence " * 300,
            name="read",
            tool_call_id="tc-1",
        ),
    ]

    compilation = await compiler.compile(
        messages,
        system_prompt="system policy",
        tools=[],
        model_name="test:model",
        reserved_output_tokens=1_000,
        target_ratio=0.5,
    )

    assert compilation.envelope.context_window_tokens == 8_000
    assert compilation.envelope.reserved_output_tokens == 1_000
    assert compilation.envelope.system_prompt_tokens > 0
    assert compilation.envelope.safety_margin_tokens == 2_048
    assert compilation.within_budget
    validate_tool_transcript(
        list(compilation.messages),
        allow_pending_tail=True,
    )
    retained_types = [message.type for message in compilation.messages]
    assert "ai" in retained_types
    assert "tool" in retained_types


def test_unknown_model_uses_conservative_context_default() -> None:
    assert (
        resolve_model_context_window(
            "provider:future-unknown-model",
            unknown_default=32_768,
        )
        == 32_768
    )


@pytest.mark.asyncio
async def test_prompt_too_long_reprojects_then_reactive_compacts() -> None:
    calls = 0
    compact_calls = 0

    async def call_model(_messages):
        nonlocal calls
        calls += 1
        if calls <= 2:
            raise RuntimeError("provider rejected context")
        return AIMessage(content="done")

    async def reactive_compactor(messages, _ratio):
        nonlocal compact_calls
        compact_calls += 1
        return messages

    events = await _events(QueryParams(
        messages=[HumanMessage(content="brief")],
        system_prompt="system",
        model=object(),
        config=_config(),
        call_model=call_model,
        reactive_compactor=reactive_compactor,
        error_classifier=lambda _error, _model: (
            ModelErrorKind.PROMPT_TOO_LONG
        ),
        context_recovery_max_attempts=3,
    ))

    recovery_reasons = [
        event.data["reason"]
        for event in events
        if event.type == "query.model_recovery"
    ]
    assert calls == 3
    assert compact_calls == 1
    assert recovery_reasons == [
        "context_reproject_retry",
        "reactive_compact_retry",
    ]
    assert events[-1].data["transition"]["reason"] == "completed"


@pytest.mark.asyncio
async def test_prompt_too_long_exhaustion_skips_stop_hooks() -> None:
    calls = 0
    stop_calls = 0

    async def call_model(_messages):
        nonlocal calls
        calls += 1
        raise RuntimeError("provider rejected context")

    async def stop_hook(_messages, _config):
        nonlocal stop_calls
        stop_calls += 1

    events = await _events(QueryParams(
        messages=[HumanMessage(content="brief")],
        system_prompt=None,
        model=object(),
        config=_config(),
        call_model=call_model,
        stop_hooks=[stop_hook],
        error_classifier=lambda _error, _model: (
            ModelErrorKind.PROMPT_TOO_LONG
        ),
        context_recovery_max_attempts=2,
    ))

    assert calls == 3
    assert stop_calls == 0
    assert events[-1].data["transition"]["reason"] == "prompt_too_long"


@pytest.mark.asyncio
async def test_output_escalation_then_continuation_merges_canonical_message() -> None:
    responses = [
        AIMessage(
            content="discarded-escalation-fragment",
            response_metadata={"finish_reason": "length"},
        ),
        AIMessage(
            content="part-1",
            response_metadata={"finish_reason": "max_tokens"},
        ),
        AIMessage(
            content="part-2",
            response_metadata={"finish_reason": "stop"},
        ),
    ]
    seen_messages: list[list[Any]] = []

    async def call_model(messages):
        seen_messages.append(messages)
        return responses.pop(0)

    events = await _events(QueryParams(
        messages=[HumanMessage(content="brief")],
        system_prompt=None,
        model=object(),
        config=_config(),
        call_model=call_model,
        model_config={"model": "test:model", "max_tokens": 100},
        model_max_output_tokens_overrides={"test:model": 10_000},
        output_continuation_max_attempts=2,
    ))

    completed = events[-1]
    final_message = completed.data["messages"][-1]
    assert len(seen_messages) == 3
    assert final_message.content == "part-1part-2"
    assert "discarded-escalation-fragment" not in final_message.content
    assert all(
        not message.additional_kwargs.get("query_internal")
        for message in completed.data["messages"]
    )


@pytest.mark.asyncio
async def test_output_recovery_exhaustion_is_not_completed() -> None:
    async def call_model(_messages):
        return AIMessage(
            content="partial",
            response_metadata={"finish_reason": "length"},
        )

    events = await _events(QueryParams(
        messages=[HumanMessage(content="brief")],
        system_prompt=None,
        model=object(),
        config=_config(),
        call_model=call_model,
        output_token_escalation_enabled=False,
        output_continuation_max_attempts=1,
    ))

    assert (
        events[-1].data["transition"]["reason"]
        == "output_recovery_exhausted"
    )


class _InjectedCrash(BaseException):
    pass


@dataclass
class _CrashCheckpointSink:
    crash_reason: str
    crash_phase: QueryPhase | None = None
    states: list[QueryLoopState] = field(default_factory=list)
    crashed: bool = False

    async def save(self, state: QueryLoopState) -> None:
        self.states.append(state)
        if (
            state.transition_reason == self.crash_reason
            and (
                self.crash_phase is None
                or state.phase is self.crash_phase
            )
            and not self.crashed
        ):
            self.crashed = True
            raise _InjectedCrash


class _ToolInput(BaseModel):
    value: str


@pytest.mark.asyncio
async def test_resume_after_model_checkpoint_does_not_repeat_model_call() -> None:
    model_calls = 0
    tool_calls = 0

    async def tool_call(
        tool_input: _ToolInput,
        _context: ToolContext,
        _progress,
    ) -> ToolResult[str]:
        nonlocal tool_calls
        tool_calls += 1
        return ToolResult(output=f"result:{tool_input.value}")

    read_tool = build_tool(
        name="read",
        description="Read a value.",
        input_schema=_ToolInput,
        call=tool_call,
        origin=ToolOrigin.SYSTEM,
        effect=ToolEffect.READ_ONLY,
        concurrency_safe=True,
    )

    async def first_model(_messages):
        nonlocal model_calls
        model_calls += 1
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "read",
                    "args": {"value": "x"},
                    "id": "tc-1",
                },
            ],
        )

    crash_sink = _CrashCheckpointSink(
        ContinueReason.NEXT_TURN.value,
        crash_phase=QueryPhase.EXECUTING_TOOLS,
    )
    with pytest.raises(_InjectedCrash):
        await _events(QueryParams(
            messages=[HumanMessage(content="brief")],
            system_prompt=None,
            model=object(),
            config=_config(),
            tools=[read_tool],
            call_model=first_model,
            checkpoint_sink=crash_sink,
            state_key="researcher:task-1",
        ))
    restored = QueryLoopState.from_snapshot(
        crash_sink.states[-1].to_snapshot()
    )
    assert restored.phase is QueryPhase.EXECUTING_TOOLS

    async def resumed_model(_messages):
        return AIMessage(content="done")

    events = await _events(QueryParams(
        messages=[],
        system_prompt=None,
        model=object(),
        config=_config(),
        tools=[read_tool],
        call_model=resumed_model,
        initial_state=restored,
        state_key="researcher:task-1",
        checkpoint_sink=InMemoryQueryCheckpointSink(),
    ))

    assert model_calls == 1
    assert tool_calls == 1
    assert events[-1].data["transition"]["reason"] == "completed"


@pytest.mark.asyncio
async def test_resume_after_tool_commit_does_not_repeat_committed_tool() -> None:
    executions = 0

    async def tool_call(
        tool_input: _ToolInput,
        _context: ToolContext,
        _progress,
    ) -> ToolResult[str]:
        nonlocal executions
        executions += 1
        return ToolResult(output=f"result:{tool_input.value}")

    read_tool = build_tool(
        name="read",
        description="Read a value.",
        input_schema=_ToolInput,
        call=tool_call,
        origin=ToolOrigin.SYSTEM,
        effect=ToolEffect.READ_ONLY,
        concurrency_safe=True,
    )

    async def first_model(_messages):
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "read",
                    "args": {"value": "x"},
                    "id": "tc-1",
                },
            ],
        )

    crash_sink = _CrashCheckpointSink("tool_result_committed")
    with pytest.raises(_InjectedCrash):
        await _events(QueryParams(
            messages=[HumanMessage(content="brief")],
            system_prompt=None,
            model=object(),
            config=_config(),
            tools=[read_tool],
            call_model=first_model,
            checkpoint_sink=crash_sink,
            state_key="researcher:task-2",
        ))
    restored = crash_sink.states[-1]
    assert restored.pending_tool_batch is not None
    assert restored.pending_tool_batch.committed_tool_call_ids == ("tc-1",)

    async def resumed_model(_messages):
        return AIMessage(content="done")

    events = await _events(QueryParams(
        messages=[],
        system_prompt=None,
        model=object(),
        config=_config(),
        tools=[read_tool],
        call_model=resumed_model,
        initial_state=restored,
        state_key="researcher:task-2",
    ))

    assert executions == 1
    assert events[-1].data["transition"]["reason"] == "completed"


@pytest.mark.asyncio
async def test_supervisor_batch_resume_reuses_child_result_persisted_before_crash() -> None:
    """A replayed Supervisor batch must reuse its durable child-task result."""
    child_executions = 0
    batch_calls = 0
    persisted_results: dict[str, ToolMessage] = {}

    async def tool_call(
        tool_input: _ToolInput,
        _context: ToolContext,
        _progress,
    ) -> ToolResult[str]:
        return ToolResult(output=f"result:{tool_input.value}")

    child_tool = build_tool(
        name="conduct_child",
        description="Run one child research task.",
        input_schema=_ToolInput,
        call=tool_call,
        origin=ToolOrigin.SYSTEM,
        effect=ToolEffect.READ_ONLY,
        concurrency_safe=True,
    )

    async def first_model(_messages):
        return AIMessage(
            content="",
            tool_calls=[{
                "name": "conduct_child",
                "args": {"value": "task"},
                "id": "child-task-1",
            }],
        )

    async def supervisor_batch_hook(
        _messages,
        tool_calls,
        _tools_by_name,
        _turn,
        _config,
    ):
        nonlocal batch_calls, child_executions
        batch_calls += 1
        tool_call = tool_calls[0]
        call_id = str(tool_call["id"])
        persisted = persisted_results.get(call_id)
        if persisted is None:
            child_executions += 1
            persisted = ToolMessage(
                content="persisted:task",
                name=str(tool_call["name"]),
                tool_call_id=call_id,
            )
            persisted_results[call_id] = persisted
            raise _InjectedCrash
        return ToolResultsHookResult(
            messages=[persisted],
            updates={"completed_child_task_ids": [call_id]},
            should_continue=False,
        )

    checkpoint_sink = InMemoryQueryCheckpointSink()
    with pytest.raises(_InjectedCrash):
        await _events(QueryParams(
            messages=[HumanMessage(content="brief")],
            system_prompt=None,
            model=object(),
            config=_config(),
            tools=[child_tool],
            call_model=first_model,
            tool_batch_hook=supervisor_batch_hook,
            checkpoint_sink=checkpoint_sink,
            state_key="supervisor",
            role=AgentRole.SUPERVISOR,
        ))

    restored = checkpoint_sink.states[-1]
    assert restored.phase is QueryPhase.EXECUTING_TOOLS
    assert restored.pending_tool_batch is not None
    assert restored.pending_tool_batch.tool_calls[0]["id"] == "child-task-1"

    async def resumed_model(_messages):
        raise AssertionError("the model must not run before the pending batch resumes")

    events = await _events(QueryParams(
        messages=[],
        system_prompt=None,
        model=object(),
        config=_config(),
        tools=[child_tool],
        call_model=resumed_model,
        tool_batch_hook=supervisor_batch_hook,
        initial_state=restored,
        checkpoint_sink=InMemoryQueryCheckpointSink(),
        state_key="supervisor",
        role=AgentRole.SUPERVISOR,
    ))

    assert batch_calls == 2
    assert child_executions == 1
    assert events[-1].type == "query.completed"
    assert events[-1].data["transition"]["reason"] == "completed"


@pytest.mark.parametrize("committed_before_crash", [1, 2])
@pytest.mark.asyncio
async def test_durable_supervisor_batch_resumes_after_each_committed_result(
    committed_before_crash: int,
) -> None:
    """Committed Supervisor tools must not restart after a partial-batch crash."""
    crash_enabled = True
    executions = {
        "conduct-1": 0,
        "artifact-1": 0,
        "conduct-2": 0,
    }

    async def unused_tool_call(
        tool_input: _ToolInput,
        _context: ToolContext,
        _progress,
    ) -> ToolResult[str]:
        return ToolResult(output=f"unused:{tool_input.value}")

    conduct_tool = build_tool(
        name="ConductResearch",
        description="Run one child research task.",
        input_schema=_ToolInput,
        call=unused_tool_call,
        origin=ToolOrigin.SYSTEM,
        effect=ToolEffect.READ_ONLY,
    )
    artifact_tool = build_tool(
        name="ReadResearchArtifact",
        description="Read one child research artifact.",
        input_schema=_ToolInput,
        call=unused_tool_call,
        origin=ToolOrigin.SYSTEM,
        effect=ToolEffect.READ_ONLY,
    )
    calls = [
        {
            "name": "ConductResearch",
            "args": {"value": "first"},
            "id": "conduct-1",
        },
        {
            "name": "ReadResearchArtifact",
            "args": {"value": "artifact"},
            "id": "artifact-1",
        },
        {
            "name": "ConductResearch",
            "args": {"value": "second"},
            "id": "conduct-2",
        },
    ]

    async def first_model(_messages):
        return AIMessage(content="", tool_calls=calls)

    async def durable_batch_hook(
        _messages,
        tool_calls,
        _tools_by_name,
        _turn,
        _config,
        committed,
        on_committed,
    ):
        outcomes = dict(committed)
        for call in tool_calls:
            call_id = str(call["id"])
            if call_id in outcomes:
                assert outcomes[call_id].result is not None
                continue
            executions[call_id] += 1
            payload = {
                "artifact_ref": {
                    "path": f"context/artifacts/{call_id}.json",
                    "sha256": call_id[-1] * 64,
                },
                "value": call["args"]["value"],
            }
            outcome = GovernedToolCallResult(
                message=ToolMessage(
                    content=json.dumps(payload),
                    name=str(call["name"]),
                    tool_call_id=call_id,
                ),
                result=ToolResult(output=payload),
            )
            await on_committed(call, outcome)
            outcomes[call_id] = outcome
            if crash_enabled and len(outcomes) == committed_before_crash:
                raise RuntimeError("injected durable batch crash")
        return ToolResultsHookResult(
            messages=[
                outcomes[str(call["id"])].message
                for call in tool_calls
            ],
            should_continue=False,
        )

    crash_sink = InMemoryQueryCheckpointSink()
    with pytest.raises(
        RuntimeError,
        match="injected durable batch crash",
    ):
        await _events(QueryParams(
            messages=[HumanMessage(content="brief")],
            system_prompt=None,
            model=object(),
            config=_config(),
            tools=[conduct_tool, artifact_tool],
            call_model=first_model,
            durable_tool_batch_hook=durable_batch_hook,
            checkpoint_sink=crash_sink,
            state_key="supervisor",
            role=AgentRole.SUPERVISOR,
        ))

    restored = crash_sink.states[-1]
    assert restored.phase is QueryPhase.EXECUTING_TOOLS
    assert restored.terminal is None
    assert restored.pending_tool_batch is not None
    assert len(
        restored.pending_tool_batch.committed_tool_call_ids
    ) == committed_before_crash
    assert len(
        restored.pending_tool_batch.result_refs
    ) == committed_before_crash

    async def resumed_model(_messages):
        raise AssertionError(
            "the model must not run before the pending batch resumes"
        )

    crash_enabled = False
    events = await _events(QueryParams(
        messages=[],
        system_prompt=None,
        model=object(),
        config=_config(),
        tools=[conduct_tool, artifact_tool],
        call_model=resumed_model,
        durable_tool_batch_hook=durable_batch_hook,
        initial_state=restored,
        checkpoint_sink=InMemoryQueryCheckpointSink(),
        state_key="supervisor",
        role=AgentRole.SUPERVISOR,
    ))

    assert executions == {
        "conduct-1": 1,
        "artifact-1": 1,
        "conduct-2": 1,
    }
    result_events = [
        event for event in events
        if event.type == "query.tool_result"
    ]
    assert len(result_events) == 1
    assert [
        message.tool_call_id
        for message in result_events[0].data["messages"]
    ] == ["conduct-1", "artifact-1", "conduct-2"]
    assert events[-1].type == "query.completed"
    assert events[-1].data["transition"]["reason"] == "completed"


@pytest.mark.asyncio
async def test_resume_replays_checkpointed_tool_domain_update_before_model() -> None:
    """A crash after the Query checkpoint must not lose outer state updates."""

    async def first_model(_messages):
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "read",
                    "args": {"value": "x"},
                    "id": "tc-domain-update",
                }
            ],
        )

    async def tool_batch_hook(
        _messages,
        _tool_calls,
        _tools_by_name,
        _turn,
        _config,
    ):
        return ToolResultsHookResult(
            messages=[
                ToolMessage(
                    content="result:x",
                    name="read",
                    tool_call_id="tc-domain-update",
                )
            ],
            updates={
                "coverage_ledger": {
                    "COV-01": {
                        "status": "supported",
                        "evidence_ids": ["ev-1"],
                    }
                }
            },
            should_continue=True,
        )

    crash_sink = _CrashCheckpointSink(
        ContinueReason.NEXT_TURN.value,
        crash_phase=QueryPhase.PREPARING,
    )
    with pytest.raises(_InjectedCrash):
        await _events(QueryParams(
            messages=[HumanMessage(content="brief")],
            system_prompt=None,
            model=object(),
            config=_config(),
            call_model=first_model,
            tool_batch_hook=tool_batch_hook,
            checkpoint_sink=crash_sink,
            state_key="researcher:domain-update",
        ))

    restored = crash_sink.states[-1]
    model_called = False

    async def resumed_model(_messages):
        nonlocal model_called
        model_called = True
        return AIMessage(content="unexpected")

    stream = query(QueryParams(
        messages=[],
        system_prompt=None,
        model=object(),
        config=_config(),
        call_model=resumed_model,
        tool_batch_hook=tool_batch_hook,
        initial_state=restored,
        state_key="researcher:domain-update",
        checkpoint_sink=InMemoryQueryCheckpointSink(),
    ))
    started = await anext(stream)
    recovered = await anext(stream)
    await stream.aclose()

    assert started.type == "query.started"
    assert recovered.type == "query.tool_result"
    assert recovered.data["updates"]["coverage_ledger"]["COV-01"][
        "status"
    ] == "supported"
    assert model_called is False

    pending_event = restored.pending_query_event
    assert pending_event is not None
    deduplicated_events = await _events(QueryParams(
        messages=[],
        system_prompt=None,
        model=object(),
        config=_config(),
        call_model=resumed_model,
        tool_batch_hook=tool_batch_hook,
        initial_state=restored,
        state_key="researcher:domain-update",
        max_turns=restored.turn,
        acknowledged_event_ids=(pending_event.event_id,),
        checkpoint_sink=InMemoryQueryCheckpointSink(),
    ))
    assert not any(
        event.type == "query.tool_result"
        for event in deduplicated_events
    )
    assert model_called is False


@pytest.mark.asyncio
async def test_resume_replays_checkpointed_terminal_stop_hook_update() -> None:
    """Terminal stop-hook updates must survive the checkpoint/event crash gap."""

    async def first_model(_messages):
        return AIMessage(content="done")

    async def stop_hook(_messages, _config):
        return StopHookResult(
            updates={"completion_decision": {"reason": "coverage_satisfied"}},
            reason="completion_policy_satisfied",
        )

    crash_sink = _CrashCheckpointSink(
        TerminalReason.COMPLETED.value,
        crash_phase=QueryPhase.TERMINAL,
    )
    with pytest.raises(_InjectedCrash):
        await _events(QueryParams(
            messages=[HumanMessage(content="brief")],
            system_prompt=None,
            model=object(),
            config=_config(),
            call_model=first_model,
            stop_hooks=[stop_hook],
            checkpoint_sink=crash_sink,
            state_key="supervisor",
        ))

    restored = QueryLoopState.from_snapshot(
        crash_sink.states[-1].to_snapshot()
    )
    events = await _events(QueryParams(
        messages=[],
        system_prompt=None,
        model=object(),
        config=_config(),
        call_model=first_model,
        stop_hooks=[stop_hook],
        initial_state=restored,
        state_key="supervisor",
        checkpoint_sink=InMemoryQueryCheckpointSink(),
    ))

    completed = next(event for event in events if event.type == "query.completed")
    assert completed.data["updates"] == {
        "completion_decision": {"reason": "coverage_satisfied"}
    }


class _CandidateModel:
    def __init__(self, response: BaseException | AIMessage):
        self.response = response
        self.calls: list[list[Any]] = []

    def with_config(self, _config):
        return self

    async def ainvoke(self, messages):
        self.calls.append(messages)
        if isinstance(self.response, BaseException):
            raise self.response
        return self.response


@pytest.mark.asyncio
async def test_model_fallback_only_for_allowed_error_and_sanitizes_metadata() -> None:
    primary = _CandidateModel(RuntimeError("model unavailable"))
    fallback = _CandidateModel(AIMessage(content="done"))
    messages = [
        AIMessage(
            content="prior",
            additional_kwargs={"signature": "provider-bound"},
            response_metadata={"reasoning": "provider-bound"},
        ),
        HumanMessage(content="continue"),
    ]

    events = await _events(QueryParams(
        messages=messages,
        system_prompt=None,
        model=primary,
        config=_config(),
        model_candidates=[
            ModelCandidate("primary", primary, {"model": "primary"}),
            ModelCandidate("fallback", fallback, {"model": "fallback"}),
        ],
    ))

    assert len(primary.calls) == 1
    assert len(fallback.calls) == 1
    replayed_ai = next(
        message
        for message in fallback.calls[0]
        if isinstance(message, AIMessage)
    )
    assert "signature" not in replayed_ai.additional_kwargs
    assert "reasoning" not in replayed_ai.response_metadata
    assert any(event.type == "query.model_fallback" for event in events)


@pytest.mark.asyncio
async def test_model_fallback_is_forbidden_for_authentication_error() -> None:
    primary = _CandidateModel(RuntimeError("invalid api key"))
    fallback = _CandidateModel(AIMessage(content="must not run"))

    events = await _events(QueryParams(
        messages=[HumanMessage(content="brief")],
        system_prompt=None,
        model=primary,
        config=_config(),
        model_candidates=[
            ModelCandidate("primary", primary, {"model": "primary"}),
            ModelCandidate("fallback", fallback, {"model": "fallback"}),
        ],
    ))

    assert len(primary.calls) == 1
    assert fallback.calls == []
    assert events[-1].data["transition"]["reason"] == "model_error"


@pytest.mark.asyncio
async def test_non_concurrency_safe_tools_execute_in_model_order() -> None:
    execution_order: list[str] = []

    async def tool_call(
        tool_input: _ToolInput,
        _context: ToolContext,
        _progress,
    ) -> ToolResult[str]:
        execution_order.append(f"start:{tool_input.value}")
        await asyncio.sleep(0)
        execution_order.append(f"end:{tool_input.value}")
        return ToolResult(output=tool_input.value)

    serial_tool = build_tool(
        name="serial",
        description="Serial operation.",
        input_schema=_ToolInput,
        call=tool_call,
        origin=ToolOrigin.SYSTEM,
        effect=ToolEffect.READ_ONLY,
        concurrency_safe=False,
        supports_idempotency=True,
    )
    responses = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "serial",
                    "args": {"value": "one"},
                    "id": "tc-1",
                },
                {
                    "name": "serial",
                    "args": {"value": "two"},
                    "id": "tc-2",
                },
            ],
        ),
        AIMessage(content="done"),
    ]

    async def call_model(_messages):
        return responses.pop(0)

    await _events(QueryParams(
        messages=[HumanMessage(content="brief")],
        system_prompt=None,
        model=object(),
        config=_config(),
        tools=[serial_tool],
        call_model=call_model,
        max_concurrent_tools=2,
    ))

    assert execution_order == [
        "start:one",
        "end:one",
        "start:two",
        "end:two",
    ]


@pytest.mark.asyncio
async def test_unsafe_uncommitted_tool_is_not_replayed_automatically() -> None:
    executions = 0

    async def tool_call(
        _tool_input: _ToolInput,
        _context: ToolContext,
        _progress,
    ) -> ToolResult[str]:
        nonlocal executions
        executions += 1
        return ToolResult(output="written")

    write_tool = build_tool(
        name="write",
        description="Write an external value.",
        input_schema=_ToolInput,
        call=tool_call,
        origin=ToolOrigin.SYSTEM,
        effect=ToolEffect.EXTERNAL_WRITE,
        concurrency_safe=False,
        supports_idempotency=False,
    )
    pending_ai = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "write",
                "args": {"value": "x"},
                "id": "write-1",
            }
        ],
    )
    restored = QueryLoopState(
        state_key="researcher:unsafe",
        role=AgentRole.RESEARCHER,
        phase=QueryPhase.EXECUTING_TOOLS,
        messages=(HumanMessage(content="brief"), pending_ai),
        turn=1,
        revision=2,
        transition_reason="next_turn",
        pending_tool_batch=PendingToolBatch(
            batch_id="unsafe-batch",
            tool_calls=tuple(pending_ai.tool_calls),
        ),
    )

    events = await _events(QueryParams(
        messages=[],
        system_prompt=None,
        model=object(),
        config=_config(),
        tools=[write_tool],
        initial_state=restored,
        state_key="researcher:unsafe",
    ))

    assert executions == 0
    assert any(
        event.type == "query.replay_confirmation_required"
        for event in events
    )
    assert events[-1].data["transition"]["reason"] == "hook_stopped"


@pytest.mark.asyncio
async def test_journal_v2_replays_latest_query_state(tmp_path) -> None:
    store = RunContextStore("journal-v2", runs_dir=str(tmp_path))
    store.initialize(
        "user-1",
        {
            "configurable": {"runs_dir": str(tmp_path)},
            "metadata": {"run_id": "journal-v2", "owner": "user-1"},
        },
    )
    state = QueryLoopState(
        state_key="supervisor",
        role=AgentRole.SUPERVISOR,
        messages=(HumanMessage(content="brief"),),
    )
    calling = advance(
        state,
        QueryStateAction(
            phase=QueryPhase.CALLING_MODEL,
            reason=ContinueReason.NEXT_TURN.value,
            changes={"turn": 1},
        ),
    )

    await store.save_query_state(state)
    await store.save_query_state(calling)
    replayed = QueryLoopState.from_snapshot(
        store.replay().query_states["supervisor"]
    )

    assert store.load_manifest().schema_version == 2
    assert replayed == calling


def test_journal_v1_is_readable_but_not_resumable(tmp_path) -> None:
    store = RunContextStore("journal-v1", runs_dir=str(tmp_path))
    store.initialize(
        "user-1",
        {
            "configurable": {"runs_dir": str(tmp_path)},
            "metadata": {"run_id": "journal-v1", "owner": "user-1"},
        },
    )
    manifest_payload = json.loads(
        store.manifest_path.read_text(encoding="utf-8")
    )
    manifest_payload["schema_version"] = 1
    store.manifest_path.write_text(
        json.dumps(manifest_payload, ensure_ascii=False),
        encoding="utf-8",
    )

    assert store.load_manifest().schema_version == 1
    with pytest.raises(
        RunConfigurationError,
        match="run_schema_not_resumable",
    ):
        QueryEngine.load("journal-v1", runs_dir=str(tmp_path))
