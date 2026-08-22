"""Inner agent loop: model calls, tool execution, and continuation decisions."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Literal, cast

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    ToolMessage,
)
from langchain_core.runnables import RunnableConfig

from open_deep_research.agents.context_compiler import ContextCompiler
from open_deep_research.agents.model_recovery import (
    ModelCandidate,
    ModelErrorKind,
    classify_model_error,
    classify_model_response,
    resolve_model_max_output_tokens,
    sanitize_messages_for_model_fallback,
)
from open_deep_research.agents.query_state import (
    ContextRecoveryState,
    ContinueReason,
    ModelRouteState,
    OutputRecoveryState,
    PendingQueryEvent,
    PendingToolBatch,
    QualityRecoveryState,
    QueryCheckpointSink,
    QueryLoopState,
    QueryPhase,
    QueryStateAction,
    QueryTerminal,
    StopAction,
    TerminalReason,
    advance,
)
from open_deep_research.agents.tool_protocol import (
    ToolProtocolDiagnostic,
    canonicalize_ai_tool_calls,
    close_tool_batch,
    validate_tool_transcript,
)
from open_deep_research.budgets import (
    BudgetExhausted,
    DeadlineExceeded,
)
from open_deep_research.configuration import Configuration
from open_deep_research.models.circuit import (
    get_model_circuit_registry,
    model_circuit_policy_from_configuration,
)
from open_deep_research.observability import (
    apply_helicone_config,
    get_trace_recorder,
    invoke_model_with_retry_observability,
    observe_model_circuit_transition,
    observe_tool_call,
)
from open_deep_research.runtime import normalize_messages
from open_deep_research.runtime_control import CancellationScope
from open_deep_research.tools.base import (
    Tool,
    ToolEffect,
    ToolResult,
    build_tool_registry,
    tools_to_model_definitions,
)
from open_deep_research.tools.governance import (
    AgentRole,
    GovernedToolCallResult,
    ToolError,
    ToolErrorType,
    execute_governed_tool_call,
    get_tool_concurrency_safe,
    get_tool_effect,
    get_tool_supports_idempotency,
    resolve_allowed_tools,
)

TransitionReason = Literal[
    "start",
    "tool_results",
    "stop_hook_blocked",
    "token_budget_continue",
    "external_update",
    "cancelled",
    "max_turns",
    "tool_protocol_violation",
    "explicit_completion",
    "completion_policy_satisfied",
    "budget_exhausted",
    "deadline_exceeded",
    "model_timeout",
    "prompt_too_long",
    "output_recovery_exhausted",
    "model_error",
    "hook_stopped",
    "context_reproject_retry",
    "reactive_compact_retry",
    "output_token_escalate",
    "output_continuation",
    "model_fallback",
    "completed",
]


@dataclass
class ContextPolicy:
    """Controls the per-request projection of the full loop history."""

    keep_last_messages: int | None = None
    max_tool_result_chars: int | None = 50_000


@dataclass
class QueryTransition:
    """Explains why the inner loop continued or stopped."""

    reason: TransitionReason
    turn: int


@dataclass
class QueryEvent:
    """Event yielded by the inner loop."""

    type: str
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class StopHookResult:
    """Optional stop-hook response."""

    should_continue: bool = False
    action: StopAction | None = None
    messages: list[BaseMessage] = field(default_factory=list)
    updates: dict[str, Any] = field(default_factory=dict)
    reason: TransitionReason = "stop_hook_blocked"

    @property
    def resolved_action(self) -> StopAction:
        """Resolve the new action enum while preserving legacy callers."""
        if self.action is not None:
            return self.action
        return StopAction.CONTINUE if self.should_continue else StopAction.COMPLETE


@dataclass
class BeforeTurnHookResult:
    """Control updates applied immediately before the next model request."""

    messages: list[BaseMessage] = field(default_factory=list)
    replace_messages: list[BaseMessage] | None = None
    updates: dict[str, Any] = field(default_factory=dict)
    should_stop: bool = False
    reason: TransitionReason = "external_update"


@dataclass
class ToolResultsHookResult:
    """Domain processing applied after a governed tool batch completes."""

    messages: list[BaseMessage] | None = None
    additional_messages: list[BaseMessage] = field(default_factory=list)
    updates: dict[str, Any] = field(default_factory=dict)
    should_continue: bool = True
    reason: TransitionReason | None = None
    quality_recovery: QualityRecoveryState | None = None


StopHook = Callable[[list[BaseMessage], RunnableConfig], Awaitable[StopHookResult | None]]
BeforeTurnHook = Callable[
    [list[BaseMessage], int, RunnableConfig],
    Awaitable[BeforeTurnHookResult | None],
]
ToolResultsHook = Callable[
    [
        list[BaseMessage],
        list[dict[str, Any]],
        list[GovernedToolCallResult],
        dict[str, Tool],
        int,
        RunnableConfig,
    ],
    Awaitable[ToolResultsHookResult | None],
]
ToolBatchHook = Callable[
    [
        list[BaseMessage],
        list[dict[str, Any]],
        dict[str, Tool],
        int,
        RunnableConfig,
    ],
    Awaitable[ToolResultsHookResult],
]
ToolCommitCallback = Callable[
    [dict[str, Any], GovernedToolCallResult],
    Awaitable[None],
]
DurableToolBatchHook = Callable[
    [
        list[BaseMessage],
        list[dict[str, Any]],
        dict[str, Tool],
        int,
        RunnableConfig,
        dict[str, GovernedToolCallResult],
        ToolCommitCallback,
    ],
    Awaitable[ToolResultsHookResult],
]
CallModel = Callable[[list[BaseMessage]], Awaitable[BaseMessage]]
TurnAdvancePolicy = Callable[
    [list[BaseMessage], "QueryLoopState", RunnableConfig],
    Awaitable[int],
]


@dataclass
class QueryParams:
    """Parameters for one agentic turn inside a conversation/session."""

    messages: list[Any]
    system_prompt: str | BaseMessage | None
    model: Any
    config: RunnableConfig
    tools: Sequence[Tool] = field(default_factory=list)
    execution_tools: Sequence[Tool] | None = None
    role: AgentRole = AgentRole.RESEARCHER
    model_span_name: str = "query.model"
    model_config: dict[str, Any] = field(default_factory=dict)
    max_turns: int | None = None
    initial_turn: int = 0
    turn_advance_policy: TurnAdvancePolicy | None = None
    max_tool_description_chars: int | None = None
    context_policy: ContextPolicy = field(default_factory=ContextPolicy)
    before_turn_hooks: Sequence[BeforeTurnHook] = field(default_factory=list)
    stop_hooks: Sequence[StopHook] = field(default_factory=list)
    tool_batch_hook: ToolBatchHook | None = None
    durable_tool_batch_hook: DurableToolBatchHook | None = None
    tool_results_hook: ToolResultsHook | None = None
    call_model: CallModel | None = None
    max_concurrent_tools: int | None = None
    max_tool_batch_size: int | None = None
    tool_timeout_seconds: float | None = None
    hook_timeout_seconds: float | None = None
    tool_batch_timeout_seconds: float | None = None
    model_timeout_seconds: float | None = None
    model_transport_max_attempts: int = 1
    budget_gate: Any = None
    execution_namespace: str | None = None
    cancellation_scope: CancellationScope | None = None
    initial_state: QueryLoopState | None = None
    state_key: str | None = None
    checkpoint_sink: QueryCheckpointSink | None = None
    acknowledged_event_ids: Sequence[str] = field(default_factory=tuple)
    model_candidates: Sequence[ModelCandidate] = field(default_factory=list)
    context_compiler: ContextCompiler | None = None
    error_classifier: Callable[
        [BaseException, str | None],
        ModelErrorKind,
    ] = classify_model_error
    reactive_compactor: Callable[
        [list[BaseMessage], float],
        Awaitable[list[BaseMessage]],
    ] | None = None
    context_recovery_max_attempts: int = 3
    output_token_escalation_enabled: bool = True
    output_continuation_max_attempts: int = 3
    model_max_output_tokens_overrides: dict[str, int] = field(
        default_factory=dict
    )


def prepare_messages_for_query(
    messages: list[BaseMessage],
    policy: ContextPolicy,
) -> list[BaseMessage]:
    """Create the compact request view from the full inner-loop history."""
    projected = list(messages)
    if policy.keep_last_messages is not None and len(projected) > policy.keep_last_messages:
        boundary = len(projected) - policy.keep_last_messages
        if isinstance(projected[boundary], ToolMessage):
            pending_ids: set[str] = set()
            cursor = boundary
            while cursor < len(projected):
                pending_message = projected[cursor]
                if not isinstance(pending_message, ToolMessage):
                    break
                pending_ids.add(pending_message.tool_call_id)
                cursor += 1
            for index in range(boundary - 1, -1, -1):
                message = projected[index]
                if not isinstance(message, AIMessage):
                    continue
                call_ids = {str(call.get("id", "")) for call in message.tool_calls}
                if pending_ids & call_ids:
                    boundary = index
                    break
        projected = projected[boundary:]

    if policy.max_tool_result_chars is None:
        return projected

    trimmed: list[BaseMessage] = []
    for message in projected:
        if message.type != "tool" or not isinstance(message.content, str):
            trimmed.append(message)
            continue
        if len(message.content) <= policy.max_tool_result_chars:
            trimmed.append(message)
            continue
        copied = message.model_copy()
        copied.content = (
            message.content[: policy.max_tool_result_chars]
            + "\n\n[Tool result trimmed by context policy]"
        )
        trimmed.append(copied)
    return trimmed


async def _default_call_model(
    params: QueryParams,
    messages: list[BaseMessage],
    *,
    candidate: ModelCandidate | None = None,
    max_output_tokens_override: int | None = None,
) -> BaseMessage:
    model = candidate.model if candidate is not None else params.model
    if params.tools and hasattr(model, "bind_tools"):
        if params.max_tool_description_chars is None:
            model_tool_definitions = await tools_to_model_definitions(
                list(params.tools)
            )
        else:
            model_tool_definitions = await tools_to_model_definitions(
                list(params.tools),
                max_description_chars=params.max_tool_description_chars,
            )
        model = model.bind_tools(
            model_tool_definitions
        )
    candidate_config = (
        candidate.model_config if candidate is not None else params.model_config
    )
    if max_output_tokens_override is not None:
        candidate_config = {
            **candidate_config,
            "max_tokens": max_output_tokens_override,
        }
    model_config = apply_helicone_config(
        candidate_config,
        params.config,
        span_name=params.model_span_name,
        agent_role=params.role.value,
    )
    if model_config and hasattr(model, "with_config"):
        model = model.with_config(model_config)
    return await invoke_model_with_retry_observability(
        model,
        messages,
        params.config,
        span_name=params.model_span_name,
        agent_role=params.role.value,
        model_name=model_config.get("model"),
        stage=(
            "planning"
            if params.role is AgentRole.SUPERVISOR
            else "researching"
        ),
        attributes={"tool_count": len(params.tools)},
        budget_gate=params.budget_gate,
    )


@dataclass(frozen=True, slots=True)
class _ToolBatchExecution:
    outcomes: tuple[GovernedToolCallResult, ...]
    result: ToolResultsHookResult | None
    diagnostics: tuple[ToolProtocolDiagnostic, ...] = ()
    cancelled: bool = False
    hook_failed: bool = False


def _rehydrate_committed_tool_result(
    message: ToolMessage,
) -> GovernedToolCallResult:
    """Restore the typed portion of a committed result from durable transport."""
    output: Any = message.content
    if isinstance(output, str):
        try:
            output = json.loads(output)
        except (TypeError, ValueError):
            pass
    if isinstance(output, dict) and output.get("error_type"):
        try:
            error = ToolError.model_validate(output)
        except (TypeError, ValueError):
            pass
        else:
            return GovernedToolCallResult(message=message, error=error)
    return GovernedToolCallResult(
        message=message,
        result=ToolResult(output=output),
    )


def _committed_result_ref(
    call_id: str,
    outcome: GovernedToolCallResult,
) -> str | None:
    """Return a stable artifact reference for one committed tool outcome."""
    output = outcome.result.output if outcome.result is not None else None
    if not isinstance(output, dict):
        return None
    artifact_ref = output.get("artifact_ref")
    if not isinstance(artifact_ref, dict):
        return None
    path = str(artifact_ref.get("path") or "").strip()
    digest = str(artifact_ref.get("sha256") or "").strip()
    if not path and not digest:
        return None
    suffix = f"#sha256={digest}" if digest else ""
    return f"{call_id}:{path or 'artifact'}{suffix}"


def _message_text(message: BaseMessage) -> str:
    """Render response content for output-fragment consolidation."""
    if isinstance(message.content, str):
        return message.content
    return "\n".join(
        str(item.get("text", item)) if isinstance(item, dict) else str(item)
        for item in message.content
    )


def _is_output_recovery_message(message: BaseMessage) -> bool:
    marker = str(
        (getattr(message, "additional_kwargs", {}) or {}).get(
            "query_internal", ""
        )
    )
    return marker in {"output_fragment", "output_continuation"}


def _consolidate_output_response(
    messages: list[BaseMessage],
    response: BaseMessage,
    recovery: OutputRecoveryState,
) -> tuple[list[BaseMessage], BaseMessage]:
    canonical = [
        message
        for message in messages
        if not _is_output_recovery_message(message)
    ]
    if not recovery.pending_fragments:
        return canonical, response
    merged = "".join((*recovery.pending_fragments, _message_text(response)))
    consolidated = response.model_copy(deep=True)
    consolidated.content = merged
    return canonical, consolidated


async def _persist_query_state(
    params: QueryParams,
    state: QueryLoopState,
) -> None:
    if params.checkpoint_sink is not None:
        await params.checkpoint_sink.save(state)


def _state_changed_event(state: QueryLoopState) -> QueryEvent:
    return QueryEvent(
        "query.state_changed",
        {
            "state": state,
            "state_key": state.state_key,
            "revision": state.revision,
            "phase": state.phase.value,
            "reason": state.transition_reason,
            "turn": state.turn,
        },
    )


def _pending_event_id(
    state: QueryLoopState,
    event_type: str,
) -> str:
    """Build a deterministic id for one write-ahead outer-state transition."""
    return f"{state.state_key}:{state.revision + 1}:{event_type}"


def _pending_event_to_query_event(
    pending: PendingQueryEvent,
) -> QueryEvent:
    """Rehydrate a checkpointed outbox entry without domain-side recomputation."""
    data: dict[str, Any] = {
        "event_id": pending.event_id,
        "transition": QueryTransition(
            reason=cast(TransitionReason, pending.transition_reason),
            turn=pending.turn,
        ).__dict__,
        "messages": list(pending.messages),
        "updates": dict(pending.updates),
    }
    if pending.event_type == "query.tool_result":
        data.update(
            {
                "turn": pending.turn,
                "additional_messages": list(
                    pending.additional_messages
                ),
                "should_continue": pending.should_continue,
                "protocol_diagnostics": [
                    dict(item)
                    for item in pending.protocol_diagnostics
                ],
            }
        )
    return QueryEvent(pending.event_type, data)


def _terminal_state(
    state: QueryLoopState,
    *,
    reason: TerminalReason,
    detail: str | None = None,
    messages: list[BaseMessage] | None = None,
    changes: dict[str, Any] | None = None,
) -> QueryLoopState:
    terminal_changes = dict(changes or {})
    if state.quality_recovery.active:
        terminal_changes.setdefault(
            "quality_recovery",
            replace(state.quality_recovery, active=False),
        )
    return advance(
        state,
        QueryStateAction(
            phase=QueryPhase.TERMINAL,
            reason=reason.value,
            changes={
                **terminal_changes,
                "messages": tuple(messages or state.messages),
                "pending_tool_batch": None,
                "terminal": QueryTerminal(reason=reason, detail=detail),
            },
        ),
    )


async def _execute_default_tool_calls(
    *,
    params: QueryParams,
    tool_calls: list[dict[str, Any]],
    tools_by_name: dict[str, Tool],
    allowed: set[str],
    execution_namespace: str,
    turn: int,
    configurable: Configuration,
    committed: dict[str, GovernedToolCallResult],
    on_committed: Callable[
        [dict[str, Any], GovernedToolCallResult],
        Awaitable[None],
    ],
) -> tuple[list[GovernedToolCallResult], tuple[ToolProtocolDiagnostic, ...]]:
    """Run explicitly safe tools concurrently and all others serially."""
    budget_gate = params.budget_gate
    concurrency = params.max_concurrent_tools or len(tool_calls) or 1
    semaphore = asyncio.Semaphore(concurrency)

    async def execute_one(
        tool_call: dict[str, Any],
    ) -> GovernedToolCallResult:
        call_id = str(tool_call["id"])
        if call_id in committed:
            return committed[call_id]
        async with semaphore:
            budget_op_key = (
                f"tool:{execution_namespace}:{params.role.value}:"
                f"{turn}:{call_id}"
            )
            if budget_gate is not None and budget_gate.enabled:
                try:
                    budget_gate.reserve_tool_call(budget_op_key)
                except (BudgetExhausted, DeadlineExceeded):
                    error = ToolError(
                        error_type=ToolErrorType.budget_exhausted,
                        tool_name=str(
                            tool_call.get("name", "unknown_tool")
                        ),
                        message=(
                            "The tool call was skipped because the run "
                            "budget or deadline was exhausted."
                        ),
                    )
                    get_trace_recorder(params.config).active_span() \
                        .record_outcome(error_type=error.error_type.value)
                    return GovernedToolCallResult(
                        message=error.to_tool_message(call_id),
                        error=error,
                    )
            invocation = observe_tool_call(
                tool_call,
                params.role.value,
                params.config,
                lambda: execute_governed_tool_call(
                    tool_call,
                    tools_by_name,
                    params.role,
                    params.config,
                    allowed_tools=allowed,
                    apply_retry=True,
                    max_retries=configurable.max_tool_retries,
                    base_delay=configurable.tool_retry_base_delay,
                    max_delay=configurable.tool_retry_max_delay,
                    operation_id=(
                        f"{params.config.get('metadata', {}).get('run_id', 'default')}:"
                        f"{params.state_key or params.role.value}:{turn}:{call_id}"
                    ),
                ),
            )
            try:
                return await (
                    asyncio.wait_for(
                        invocation,
                        timeout=params.tool_timeout_seconds,
                    )
                    if params.tool_timeout_seconds is not None
                    else invocation
                )
            except TimeoutError:
                error = ToolError(
                    error_type=ToolErrorType.timeout,
                    tool_name=str(
                        tool_call.get("name", "unknown_tool")
                    ),
                    message="The tool call exceeded its execution timeout.",
                    retryable=True,
                    detail={
                        "timeout_seconds": params.tool_timeout_seconds
                    },
                )
                return GovernedToolCallResult(
                    message=error.to_tool_message(call_id),
                    error=error,
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                error = ToolError(
                    error_type=ToolErrorType.internal_error,
                    tool_name=str(
                        tool_call.get("name", "unknown_tool")
                    ),
                    message="The tool call failed unexpectedly.",
                    detail={"error_type": type(exc).__name__},
                )
                return GovernedToolCallResult(
                    message=error.to_tool_message(call_id),
                    error=error,
                )

    batch_limit = params.max_tool_batch_size or len(tool_calls)
    runnable_calls = tool_calls[:batch_limit]
    by_id: dict[str, GovernedToolCallResult] = dict(committed)
    safe_group: list[dict[str, Any]] = []

    async def flush_safe_group() -> None:
        if not safe_group:
            return
        calls = list(safe_group)
        safe_group.clear()
        tasks = {
            asyncio.create_task(execute_one(call)): call for call in calls
        }
        try:
            for future in asyncio.as_completed(tasks):
                outcome = await future
                call = tasks[future] if future in tasks else None
                if call is None:
                    outcome_id = str(outcome.message.tool_call_id)
                    call = next(
                        item
                        for item in calls
                        if str(item["id"]) == outcome_id
                    )
                by_id[str(call["id"])] = outcome
                await on_committed(call, outcome)
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()

    for tool_call in runnable_calls:
        call_id = str(tool_call["id"])
        if call_id in committed:
            continue
        tool = tools_by_name.get(str(tool_call.get("name", "")))
        if tool is not None and get_tool_concurrency_safe(tool):
            safe_group.append(tool_call)
            continue
        await flush_safe_group()
        outcome = await execute_one(tool_call)
        by_id[call_id] = outcome
        await on_committed(tool_call, outcome)
    await flush_safe_group()

    diagnostics: tuple[ToolProtocolDiagnostic, ...] = ()
    if len(runnable_calls) < len(tool_calls):
        diagnostics = tuple(
            ToolProtocolDiagnostic(
                code="tool_batch_capacity_exceeded",
                tool_call_id=str(call["id"]),
            )
            for call in tool_calls[len(runnable_calls):]
        )
    return [
        by_id[str(call["id"])]
        for call in runnable_calls
        if str(call["id"]) in by_id
    ], diagnostics


async def query(params: QueryParams) -> AsyncIterator[QueryEvent]:
    """Run a recoverable immutable model/tool loop."""
    if (
        params.max_concurrent_tools is not None
        and params.max_concurrent_tools <= 0
    ):
        raise ValueError("max_concurrent_tools must be greater than zero")
    if (
        params.max_tool_batch_size is not None
        and params.max_tool_batch_size <= 0
    ):
        raise ValueError("max_tool_batch_size must be greater than zero")
    if (
        params.hook_timeout_seconds is not None
        and params.hook_timeout_seconds <= 0
    ):
        raise ValueError("hook_timeout_seconds must be greater than zero")
    if (
        params.tool_batch_timeout_seconds is not None
        and params.tool_batch_timeout_seconds <= 0
    ):
        raise ValueError("tool_batch_timeout_seconds must be greater than zero")
    if (
        params.tool_timeout_seconds is not None
        and params.tool_timeout_seconds <= 0
    ):
        raise ValueError("tool_timeout_seconds must be greater than zero")

    configurable = Configuration.from_runnable_config(params.config)
    state_key = params.state_key or params.execution_namespace or str(
        params.config.get("metadata", {}).get("task_id")
        or params.role.value
    )
    initial_messages = normalize_messages(list(params.messages))
    state = params.initial_state or QueryLoopState(
        state_key=state_key,
        role=params.role,
        messages=tuple(initial_messages),
        turn=params.initial_turn,
    )
    if state.state_key != state_key or state.role is not params.role:
        raise ValueError("query_initial_state_scope_mismatch")
    messages = list(state.messages)
    validate_tool_transcript(messages, allow_pending_tail=True)

    candidates = list(params.model_candidates) or [
        ModelCandidate(
            model_id=str(params.model_config.get("model", "")),
            model=params.model,
            model_config=dict(params.model_config),
        )
    ]
    if not candidates:
        raise ValueError("query_requires_model_candidate")
    if state.model_route.active_candidate_index >= len(candidates):
        raise ValueError("query_model_route_out_of_range")
    if configurable.model_circuit_breaker_enabled:
        try:
            selected_index, _transition = (
                await get_model_circuit_registry().select_candidate_index(
                    [candidate.model_id for candidate in candidates],
                    model_circuit_policy_from_configuration(configurable),
                    start_index=state.model_route.active_candidate_index,
                )
            )
            await observe_model_circuit_transition(
                _transition,
                params.config,
                agent_role=params.role.value,
            )
            if selected_index != state.model_route.active_candidate_index:
                state = replace(
                    state,
                    model_route=ModelRouteState(
                        active_candidate_index=selected_index
                    ),
                )
        except Exception:  # noqa: BLE001 - circuit routing fails open
            pass
    compiler = params.context_compiler or ContextCompiler(
        model_context_window_overrides=(
            configurable.model_context_window_overrides
        ),
        unknown_model_context_window_tokens=(
            configurable.unknown_model_context_window_tokens
        ),
        max_tool_result_chars=(
            params.context_policy.max_tool_result_chars or 50_000
        ),
    )
    context_recovery_limit = max(1, params.context_recovery_max_attempts)
    output_continuation_limit = max(
        0, params.output_continuation_max_attempts
    )
    output_overrides = (
        params.model_max_output_tokens_overrides
        or configurable.model_max_output_tokens_overrides
    )
    acknowledged_event_ids = frozenset(params.acknowledged_event_ids)
    execution_namespace = params.execution_namespace or state_key
    recorder = get_trace_recorder(params.config)
    resumed_tool_batch = (
        params.initial_state is not None
        and state.phase is QueryPhase.EXECUTING_TOOLS
    )

    await _persist_query_state(params, state)
    yield QueryEvent(
        "query.started",
        {
            "transition": QueryTransition(
                reason="start", turn=state.turn
            ).__dict__,
            "state_key": state.state_key,
            "revision": state.revision,
            "resumed": params.initial_state is not None,
        },
    )

    while True:
        pending_query_event = state.pending_query_event
        if pending_query_event is not None:
            if pending_query_event.event_id not in acknowledged_event_ids:
                yield _pending_event_to_query_event(pending_query_event)
            state = advance(
                state,
                QueryStateAction(
                    phase=state.phase,
                    reason="domain_event_delivered",
                    changes={"pending_query_event": None},
                ),
            )
            await _persist_query_state(params, state)
            if pending_query_event.event_type == "query.completed":
                return
            yield _state_changed_event(state)
            continue
        messages = list(state.messages)
        if state.phase is QueryPhase.TERMINAL:
            assert state.terminal is not None
            transition = QueryTransition(
                reason=state.terminal.reason.value,  # type: ignore[arg-type]
                turn=state.turn,
            )
            yield QueryEvent(
                "query.completed",
                {
                    "transition": transition.__dict__,
                    "messages": messages,
                    "terminal_detail": state.terminal.detail,
                    "budget_exhausted": (
                        state.terminal.reason
                        is TerminalReason.BUDGET_EXHAUSTED
                    ),
                },
            )
            return

        if (
            state.phase is QueryPhase.PREPARING
            and params.max_turns is not None
            and state.turn >= params.max_turns
        ):
            validate_tool_transcript(messages)
            state = _terminal_state(
                state,
                reason=TerminalReason.MAX_TURNS,
                messages=messages,
            )
            await _persist_query_state(params, state)
            yield _state_changed_event(state)
            continue

        if state.phase is QueryPhase.PREPARING:
            stopped_update: dict[str, Any] | None = None
            stopped_reason: TransitionReason = "hook_stopped"
            for before_turn_hook in params.before_turn_hooks:
                call = before_turn_hook(
                    messages,
                    state.turn + 1,
                    params.config,
                )
                result = await (
                    asyncio.wait_for(
                        call,
                        timeout=params.hook_timeout_seconds,
                    )
                    if params.hook_timeout_seconds is not None
                    else call
                )
                if result is None:
                    continue
                messages = (
                    normalize_messages(result.replace_messages)
                    if result.replace_messages is not None
                    else [*messages, *result.messages]
                )
                state = advance(
                    state,
                    QueryStateAction(
                        phase=QueryPhase.PREPARING,
                        reason=result.reason,
                        changes={"messages": tuple(messages)},
                    ),
                )
                await _persist_query_state(params, state)
                yield _state_changed_event(state)
                transition = QueryTransition(
                    reason=result.reason,
                    turn=state.turn,
                )
                yield QueryEvent(
                    "query.transition",
                    {
                        "transition": transition.__dict__,
                        "messages": result.messages,
                        "replacement_messages": result.replace_messages,
                        "updates": result.updates,
                    },
                )
                if result.should_stop:
                    stopped_update = result.updates
                    stopped_reason = result.reason
                    break
            if stopped_update is not None:
                validate_tool_transcript(messages)
                terminal_reason = (
                    TerminalReason.CANCELLED
                    if stopped_reason == "cancelled"
                    else TerminalReason.HOOK_STOPPED
                )
                state = _terminal_state(
                    state,
                    reason=terminal_reason,
                    detail=stopped_reason,
                    messages=messages,
                )
                await _persist_query_state(params, state)
                yield _state_changed_event(state)
                yield QueryEvent(
                    "query.completed",
                    {
                        "transition": QueryTransition(
                            reason=stopped_reason,
                            turn=state.turn,
                        ).__dict__,
                        "messages": messages,
                        "updates": stopped_update,
                    },
                )
                return
            turn_delta = 1
            if params.turn_advance_policy is not None:
                turn_delta = max(
                    0,
                    int(
                        await params.turn_advance_policy(
                            messages,
                            state,
                            params.config,
                        )
                    ),
                )
            state = advance(
                state,
                QueryStateAction(
                    phase=QueryPhase.CALLING_MODEL,
                    reason=ContinueReason.NEXT_TURN.value,
                    changes={
                        "messages": tuple(messages),
                        "turn": state.turn + turn_delta,
                        "stop_hook_active": False,
                    },
                ),
            )
            await _persist_query_state(params, state)
            yield _state_changed_event(state)

        if state.phase is QueryPhase.CALLING_MODEL:
            while state.phase is QueryPhase.CALLING_MODEL:
                candidate_index = state.model_route.active_candidate_index
                candidate = candidates[candidate_index]
                model_name = candidate.model_id
                requested_output = int(
                    state.output_recovery.max_output_tokens_override
                    or candidate.model_config.get("max_tokens")
                    or params.model_config.get("max_tokens")
                    or 0
                )
                with recorder.start_span(
                    name="query.request",
                    kind="agent",
                    agent_role=params.role.value,
                    attributes={
                        "turn": state.turn,
                        "state_revision": state.revision,
                        "model_candidate_index": candidate_index,
                    },
                ):
                    validate_tool_transcript(
                        list(state.messages),
                        allow_pending_tail=True,
                    )
                    compilation = await compiler.compile(
                        list(state.messages),
                        system_prompt=params.system_prompt,
                        tools=list(params.tools),
                        model_name=model_name,
                        reserved_output_tokens=requested_output,
                        target_ratio=(
                            state.context_recovery.target_ratio
                        ),
                        keep_last_messages=(
                            params.context_policy.keep_last_messages
                        ),
                        max_tool_description_chars=(
                            params.max_tool_description_chars or 2_000
                        ),
                    )
                    request_messages = list(compilation.messages)
                yield QueryEvent(
                    "query.request",
                    {
                        "turn": state.turn,
                        "message_count": len(state.messages),
                        "messages_for_query_count": len(request_messages),
                        "estimated_input_tokens": (
                            compilation.estimated_input_tokens
                        ),
                        "available_input_tokens": (
                            compilation.envelope.available_input_tokens
                        ),
                        "context_compacted": compilation.compacted,
                    },
                )
                if not compilation.within_budget:
                    state = _terminal_state(
                        state,
                        reason=TerminalReason.PROMPT_TOO_LONG,
                        detail="protected_context_exceeds_model_envelope",
                    )
                    await _persist_query_state(params, state)
                    yield _state_changed_event(state)
                    break

                model_stage = f"{params.role.value}.model"
                if params.cancellation_scope is not None:
                    params.cancellation_scope.checkpoint(model_stage)
                response: BaseMessage | None = None
                model_error: BaseException | None = None
                max_model_attempts = max(
                    1, params.model_transport_max_attempts
                )
                for model_attempt in range(1, max_model_attempts + 1):
                    if state.phase is QueryPhase.TERMINAL:
                        break
                    if params.call_model is not None:
                        with recorder.start_span(
                            name=params.model_span_name,
                            kind="llm",
                            agent_role=params.role.value,
                            attributes={
                                "custom_call_model": True,
                                "attempt": model_attempt,
                                "max_attempts": max_model_attempts,
                            },
                            input_payload=request_messages,
                        ):
                            model_call = params.call_model(
                                request_messages
                            )
                    else:
                        model_call = _default_call_model(
                            params,
                            request_messages,
                            candidate=candidate,
                            max_output_tokens_override=(
                                state.output_recovery
                                .max_output_tokens_override
                            ),
                        )
                    try:
                        response = await (
                            params.cancellation_scope.run(
                                model_call,
                                stage=model_stage,
                                timeout_seconds=(
                                    params.model_timeout_seconds
                                ),
                            )
                            if params.cancellation_scope is not None
                            else asyncio.wait_for(
                                model_call,
                                timeout=params.model_timeout_seconds,
                            )
                            if params.model_timeout_seconds is not None
                            else model_call
                        )
                    except asyncio.CancelledError:
                        state = _terminal_state(
                            state,
                            reason=TerminalReason.CANCELLED,
                            detail="model_call_cancelled",
                            messages=list(state.messages),
                        )
                        await _persist_query_state(params, state)
                        raise
                    except DeadlineExceeded:
                        state = _terminal_state(
                            state,
                            reason=TerminalReason.DEADLINE_EXCEEDED,
                        )
                        await _persist_query_state(params, state)
                        yield _state_changed_event(state)
                        break
                    except BudgetExhausted:
                        state = _terminal_state(
                            state,
                            reason=TerminalReason.BUDGET_EXHAUSTED,
                        )
                        await _persist_query_state(params, state)
                        yield _state_changed_event(state)
                        break
                    except TimeoutError as exc:
                        model_error = exc
                        if model_attempt < max_model_attempts:
                            yield QueryEvent(
                                "query.model_retry",
                                {
                                    "turn": state.turn,
                                    "attempt": model_attempt,
                                    "max_attempts": max_model_attempts,
                                    "reason": "model_timeout",
                                    "timeout_seconds": (
                                        params.model_timeout_seconds
                                    ),
                                },
                            )
                            continue
                    except Exception as exc:  # noqa: BLE001
                        model_error = exc
                    else:
                        model_error = None
                    break

                if state.phase is QueryPhase.TERMINAL:
                    break
                if model_error is not None:
                    error_kind = params.error_classifier(
                        model_error,
                        model_name,
                    )
                    if (
                        error_kind is ModelErrorKind.PROMPT_TOO_LONG
                        and state.context_recovery.attempts
                        < context_recovery_limit
                    ):
                        attempts = state.context_recovery.attempts + 1
                        reactive = (
                            attempts >= 2
                            and not state.context_recovery
                            .reactive_compact_attempted
                        )
                        recovered_messages = list(state.messages)
                        if reactive:
                            target_tokens = max(
                                1,
                                int(
                                    compilation.envelope
                                    .available_input_tokens
                                    * 0.5
                                ),
                            )
                            try:
                                recovered_messages = (
                                    await params.reactive_compactor(
                                        recovered_messages,
                                        state.context_recovery.target_ratio
                                        * 0.8,
                                    )
                                    if params.reactive_compactor is not None
                                    else compiler.deterministic_compact(
                                        recovered_messages,
                                        target_tokens=target_tokens,
                                    )
                                )
                            except Exception:  # noqa: BLE001
                                recovered_messages = (
                                    compiler.deterministic_compact(
                                        recovered_messages,
                                        target_tokens=target_tokens,
                                    )
                                )
                        recovery = ContextRecoveryState(
                            attempts=attempts,
                            reactive_compact_attempted=(
                                state.context_recovery
                                .reactive_compact_attempted
                                or reactive
                            ),
                            compaction_generation=(
                                state.context_recovery
                                .compaction_generation
                                + (1 if reactive else 0)
                            ),
                            target_ratio=max(
                                0.05,
                                state.context_recovery.target_ratio * 0.8,
                            ),
                        )
                        reason = (
                            ContinueReason.REACTIVE_COMPACT_RETRY
                            if reactive
                            else ContinueReason.CONTEXT_REPROJECT_RETRY
                        )
                        state = advance(
                            state,
                            QueryStateAction(
                                phase=QueryPhase.CALLING_MODEL,
                                reason=reason.value,
                                changes={
                                    "messages": tuple(
                                        recovered_messages
                                    ),
                                    "context_recovery": recovery,
                                },
                            ),
                        )
                        await _persist_query_state(params, state)
                        yield _state_changed_event(state)
                        yield QueryEvent(
                            "query.model_recovery",
                            {
                                "turn": state.turn,
                                "reason": reason.value,
                                "attempt": attempts,
                                "max_attempts": context_recovery_limit,
                            },
                        )
                        continue
                    fallback_allowed = error_kind in {
                        ModelErrorKind.MODEL_UNAVAILABLE,
                        ModelErrorKind.RATE_LIMITED,
                        ModelErrorKind.TRANSIENT,
                    }
                    if (
                        fallback_allowed
                        and params.call_model is None
                        and candidate_index + 1 < len(candidates)
                    ):
                        next_candidate_index = candidate_index + 1
                        if configurable.model_circuit_breaker_enabled:
                            try:
                                next_candidate_index, _transition = (
                                    await get_model_circuit_registry()
                                    .select_candidate_index(
                                        [
                                            candidate.model_id
                                            for candidate in candidates
                                        ],
                                        model_circuit_policy_from_configuration(
                                            configurable
                                        ),
                                        start_index=next_candidate_index,
                                    )
                                )
                                await observe_model_circuit_transition(
                                    _transition,
                                    params.config,
                                    agent_role=params.role.value,
                                )
                            except Exception:  # noqa: BLE001
                                next_candidate_index = candidate_index + 1
                        state = advance(
                            state,
                            QueryStateAction(
                                phase=QueryPhase.CALLING_MODEL,
                                reason=ContinueReason.MODEL_FALLBACK.value,
                                changes={
                                    "messages": tuple(
                                        sanitize_messages_for_model_fallback(
                                            list(state.messages)
                                        )
                                    ),
                                    "model_route": ModelRouteState(
                                        active_candidate_index=next_candidate_index
                                    ),
                                    "output_recovery": (
                                        OutputRecoveryState()
                                    ),
                                },
                            ),
                        )
                        await _persist_query_state(params, state)
                        yield _state_changed_event(state)
                        yield QueryEvent(
                            "query.model_fallback",
                            {
                                "turn": state.turn,
                                "from_model": model_name,
                                "to_model": candidates[
                                    next_candidate_index
                                ].model_id,
                                "reason": error_kind.value,
                            },
                        )
                        continue
                    terminal_reason = (
                        TerminalReason.PROMPT_TOO_LONG
                        if error_kind is ModelErrorKind.PROMPT_TOO_LONG
                        else TerminalReason.MODEL_TIMEOUT
                        if isinstance(model_error, TimeoutError)
                        else TerminalReason.MODEL_ERROR
                    )
                    state = _terminal_state(
                        state,
                        reason=terminal_reason,
                        detail=error_kind.value,
                    )
                    await _persist_query_state(params, state)
                    yield _state_changed_event(state)
                    break

                assert response is not None
                response_status = classify_model_response(response)
                raw_tool_calls = list(
                    getattr(response, "tool_calls", []) or []
                )
                if (
                    response_status.error_kind
                    is ModelErrorKind.OUTPUT_TRUNCATED
                    and not raw_tool_calls
                ):
                    requested = requested_output
                    maximum = resolve_model_max_output_tokens(
                        model_name,
                        requested=max(1, requested),
                        overrides=output_overrides,
                    )
                    can_escalate = (
                        params.output_token_escalation_enabled
                        and requested > 0
                        and maximum > requested
                        and not state.output_recovery.escalation_attempted
                    )
                    if can_escalate:
                        override = min(
                            maximum,
                            max(requested * 2, requested + 4_096),
                        )
                        state = advance(
                            state,
                            QueryStateAction(
                                phase=QueryPhase.CALLING_MODEL,
                                reason=(
                                    ContinueReason
                                    .OUTPUT_TOKEN_ESCALATE.value
                                ),
                                changes={
                                    "output_recovery": replace(
                                        state.output_recovery,
                                        escalation_attempted=True,
                                        max_output_tokens_override=override,
                                    ),
                                },
                            ),
                        )
                        await _persist_query_state(params, state)
                        yield _state_changed_event(state)
                        continue
                    if (
                        state.output_recovery.continuation_count
                        < output_continuation_limit
                    ):
                        fragment_text = _message_text(response)
                        fragment_message = response.model_copy(deep=True)
                        fragment_kwargs = dict(
                            getattr(
                                fragment_message,
                                "additional_kwargs",
                                {},
                            )
                            or {}
                        )
                        fragment_kwargs["query_internal"] = (
                            "output_fragment"
                        )
                        fragment_message.additional_kwargs = fragment_kwargs
                        continuation_message = HumanMessage(
                            content=(
                                "Output token limit hit. Resume directly "
                                "from the previous fragment without "
                                "repeating completed text."
                            ),
                            additional_kwargs={
                                "query_internal": "output_continuation"
                            },
                        )
                        output_recovery = replace(
                            state.output_recovery,
                            continuation_count=(
                                state.output_recovery.continuation_count + 1
                            ),
                            pending_fragments=(
                                *state.output_recovery.pending_fragments,
                                fragment_text,
                            ),
                        )
                        state = advance(
                            state,
                            QueryStateAction(
                                phase=QueryPhase.CALLING_MODEL,
                                reason=(
                                    ContinueReason
                                    .OUTPUT_CONTINUATION.value
                                ),
                                changes={
                                    "messages": (
                                        *state.messages,
                                        fragment_message,
                                        continuation_message,
                                    ),
                                    "output_recovery": output_recovery,
                                },
                            ),
                        )
                        await _persist_query_state(params, state)
                        yield _state_changed_event(state)
                        continue
                    state = _terminal_state(
                        state,
                        reason=(
                            TerminalReason.OUTPUT_RECOVERY_EXHAUSTED
                        ),
                    )
                    await _persist_query_state(params, state)
                    yield _state_changed_event(state)
                    break

                canonical_messages, response = (
                    _consolidate_output_response(
                        list(state.messages),
                        response,
                        state.output_recovery,
                    )
                )
                run_id = str(
                    params.config.get("metadata", {}).get(
                        "run_id", "default"
                    )
                )
                response, call_diagnostics = (
                    canonicalize_ai_tool_calls(
                        response,
                        run_id=run_id,
                        role=params.role.value,
                        turn=state.turn,
                    )
                )
                messages = [*canonical_messages, response]
                tool_calls = list(
                    getattr(response, "tool_calls", []) or []
                )
                if tool_calls:
                    pending = PendingToolBatch(
                        batch_id=(
                            f"{state.state_key}:{state.turn}:"
                            f"{state.revision + 1}"
                        ),
                        tool_calls=tuple(tool_calls),
                    )
                    state = advance(
                        state,
                        QueryStateAction(
                            phase=QueryPhase.EXECUTING_TOOLS,
                            reason=ContinueReason.NEXT_TURN.value,
                            changes={
                                "messages": tuple(messages),
                                "pending_tool_batch": pending,
                                "output_recovery": OutputRecoveryState(),
                                "context_recovery": (
                                    ContextRecoveryState()
                                ),
                            },
                        ),
                    )
                else:
                    state = advance(
                        state,
                        QueryStateAction(
                            phase=QueryPhase.STOP_GOVERNANCE,
                            reason="model_response",
                            changes={
                                "messages": tuple(messages),
                                "output_recovery": OutputRecoveryState(),
                                "context_recovery": (
                                    ContextRecoveryState()
                                ),
                            },
                        ),
                    )
                await _persist_query_state(params, state)
                yield _state_changed_event(state)
                yield QueryEvent(
                    "query.model_event",
                    {
                        "turn": state.turn,
                        "message": response,
                        "protocol_diagnostics": [
                            diagnostic.to_dict()
                            for diagnostic in call_diagnostics
                        ],
                    },
                )
                break
            continue

        if state.phase is QueryPhase.STOP_GOVERNANCE:
            messages = list(state.messages)
            stop_updates: dict[str, Any] = {}
            stop_messages: list[BaseMessage] = []
            completion_reason: TransitionReason = "completed"
            action = StopAction.COMPLETE
            for stop_hook in params.stop_hooks:
                hook_call = stop_hook(messages, params.config)
                stop_result = await (
                    asyncio.wait_for(
                        hook_call,
                        timeout=params.hook_timeout_seconds,
                    )
                    if params.hook_timeout_seconds is not None
                    else hook_call
                )
                if stop_result is None:
                    continue
                stop_updates.update(stop_result.updates)
                stop_messages.extend(stop_result.messages)
                completion_reason = stop_result.reason
                action = stop_result.resolved_action
                if action is not StopAction.COMPLETE:
                    break
            messages = [*messages, *stop_messages]
            if action is StopAction.CONTINUE:
                pending_query_event = PendingQueryEvent(
                    event_id=_pending_event_id(
                        state,
                        "query.transition",
                    ),
                    event_type="query.transition",
                    transition_reason=completion_reason,
                    turn=state.turn,
                    messages=tuple(stop_messages),
                    updates=dict(stop_updates),
                )
                state = advance(
                    state,
                    QueryStateAction(
                        phase=QueryPhase.PREPARING,
                        reason=(
                            ContinueReason.STOP_HOOK_BLOCKING.value
                        ),
                        changes={
                            "messages": tuple(messages),
                            "stop_hook_active": True,
                            "pending_query_event": pending_query_event,
                        },
                    ),
                )
                await _persist_query_state(params, state)
                yield _state_changed_event(state)
                continue
            terminal_reason = (
                TerminalReason.HOOK_STOPPED
                if action is StopAction.HALT
                else TerminalReason.COMPLETED
            )
            validate_tool_transcript(messages)
            pending_query_event = PendingQueryEvent(
                event_id=_pending_event_id(state, "query.completed"),
                event_type="query.completed",
                transition_reason=completion_reason,
                turn=state.turn,
                messages=tuple(messages),
                updates=dict(stop_updates),
            )
            state = _terminal_state(
                state,
                reason=terminal_reason,
                detail=(
                    completion_reason
                    if action is StopAction.HALT
                    else None
                ),
                messages=messages,
                changes={"pending_query_event": pending_query_event},
            )
            await _persist_query_state(params, state)
            yield _state_changed_event(state)
            continue

        if state.phase is QueryPhase.EXECUTING_TOOLS:
            pending_value = state.pending_tool_batch
            if pending_value is None:
                raise RuntimeError("executing_tools_without_pending_batch")
            pending = cast(PendingToolBatch, pending_value)
            tool_calls = [dict(call) for call in pending.tool_calls]
            messages = list(state.messages)
            execution_tools = list(
                params.execution_tools or params.tools
            )
            tools_by_name = build_tool_registry(execution_tools)
            allowed = resolve_allowed_tools(
                params.role,
                params.config,
                set(tools_by_name),
            )
            allowed = allowed or set(tools_by_name)
            if resumed_tool_batch:
                committed_ids = set(
                    pending.committed_tool_call_ids
                )
                unsafe = []
                for call in tool_calls:
                    if str(call["id"]) in committed_ids:
                        continue
                    tool = tools_by_name.get(str(call.get("name", "")))
                    if tool is None:
                        continue
                    if (
                        get_tool_effect(tool) is not ToolEffect.READ_ONLY
                        and not get_tool_supports_idempotency(tool)
                    ):
                        unsafe.append(str(call["id"]))
                if unsafe:
                    state = _terminal_state(
                        state,
                        reason=TerminalReason.HOOK_STOPPED,
                        detail=(
                            "unsafe_tool_replay_requires_approval:"
                            + ",".join(unsafe)
                        ),
                    )
                    await _persist_query_state(params, state)
                    yield _state_changed_event(state)
                    yield QueryEvent(
                        "query.replay_confirmation_required",
                        {
                            "turn": state.turn,
                            "tool_call_ids": unsafe,
                        },
                    )
                    continue
            yield QueryEvent(
                "query.tool_call",
                {
                    "turn": state.turn,
                    "tool_calls": tool_calls,
                    "resumed": resumed_tool_batch,
                },
            )

            committed_results = {
                str(message.tool_call_id): _rehydrate_committed_tool_result(
                    message
                )
                for message in pending.committed_results
                if isinstance(message, ToolMessage)
            }
            committed_state_events: list[QueryEvent] = []
            commit_lock = asyncio.Lock()

            async def on_committed(
                call: dict[str, Any],
                outcome: GovernedToolCallResult,
            ) -> None:
                nonlocal state, pending
                async with commit_lock:
                    call_id = str(call["id"])
                    if call_id in pending.committed_tool_call_ids:
                        return
                    result_ref = _committed_result_ref(call_id, outcome)
                    pending = replace(
                        pending,
                        committed_tool_call_ids=(
                            *pending.committed_tool_call_ids,
                            call_id,
                        ),
                        committed_results=(
                            *pending.committed_results,
                            outcome.message,
                        ),
                        result_refs=(
                            *pending.result_refs,
                            *((result_ref,) if result_ref is not None else ()),
                        ),
                    )
                    state = advance(
                        state,
                        QueryStateAction(
                            phase=QueryPhase.EXECUTING_TOOLS,
                            reason="tool_result_committed",
                            changes={"pending_tool_batch": pending},
                        ),
                    )
                    await _persist_query_state(params, state)
                    committed_state_events.append(
                        _state_changed_event(state)
                    )

            outcomes: list[GovernedToolCallResult] = []
            tool_results_result: ToolResultsHookResult | None = None
            batch_diagnostics: tuple[
                ToolProtocolDiagnostic, ...
            ] = ()
            hook_failed = False
            batch_cancelled = False
            batch_cancellation: asyncio.CancelledError | None = None
            try:
                if params.durable_tool_batch_hook is not None:
                    batch_timeout = (
                        params.tool_batch_timeout_seconds
                        if params.tool_batch_timeout_seconds is not None
                        else params.hook_timeout_seconds
                    )
                    batch_call = params.durable_tool_batch_hook(
                        messages,
                        tool_calls,
                        tools_by_name,
                        state.turn,
                        params.config,
                        dict(committed_results),
                        on_committed,
                    )
                    tool_results_result = await (
                        asyncio.wait_for(
                            batch_call,
                            timeout=batch_timeout,
                        )
                        if batch_timeout is not None
                        else batch_call
                    )
                elif params.tool_batch_hook is not None:
                    batch_timeout = (
                        params.tool_batch_timeout_seconds
                        if params.tool_batch_timeout_seconds is not None
                        else params.hook_timeout_seconds
                    )
                    batch_call = params.tool_batch_hook(
                        messages,
                        tool_calls,
                        tools_by_name,
                        state.turn,
                        params.config,
                    )
                    tool_results_result = await (
                        asyncio.wait_for(
                            batch_call,
                            timeout=batch_timeout,
                        )
                        if batch_timeout is not None
                        else batch_call
                    )
                else:
                    outcomes, batch_diagnostics = (
                        await _execute_default_tool_calls(
                            params=params,
                            tool_calls=tool_calls,
                            tools_by_name=tools_by_name,
                            allowed=allowed,
                            execution_namespace=execution_namespace,
                            turn=state.turn,
                            configurable=configurable,
                            committed=committed_results,
                            on_committed=on_committed,
                        )
                    )
                    if params.tool_results_hook is not None:
                        tool_results_hook_call = params.tool_results_hook(
                            messages,
                            tool_calls,
                            outcomes,
                            tools_by_name,
                            state.turn,
                            params.config,
                        )
                        tool_results_result = await (
                            asyncio.wait_for(
                                tool_results_hook_call,
                                timeout=params.hook_timeout_seconds,
                            )
                            if params.hook_timeout_seconds is not None
                            else tool_results_hook_call
                        )
            except asyncio.CancelledError as exc:
                batch_cancelled = True
                batch_cancellation = exc
                batch_diagnostics = (
                    *batch_diagnostics,
                    ToolProtocolDiagnostic(
                        code="tool_batch_cancelled"
                    ),
                )
            except Exception as exc:  # noqa: BLE001
                if (
                    params.durable_tool_batch_hook is not None
                    and pending.committed_tool_call_ids
                ):
                    # A partial durable batch is a recoverable checkpoint, not
                    # a protocol failure. Let the caller restart from the last
                    # committed result instead of synthesizing missing outputs.
                    raise
                hook_failed = True
                batch_diagnostics = (
                    *batch_diagnostics,
                    ToolProtocolDiagnostic(
                        code="tool_batch_hook_error",
                        detail={"error_type": type(exc).__name__},
                    ),
                )
            for state_event in committed_state_events:
                yield state_event

            candidate_messages: list[BaseMessage] = (
                list(tool_results_result.messages)
                if tool_results_result is not None
                and tool_results_result.messages is not None
                else [outcome.message for outcome in outcomes]
            )
            missing_error_type = (
                ToolErrorType.cancelled
                if batch_cancelled
                else ToolErrorType.runtime_hook_error
                if hook_failed
                else ToolErrorType.task_capacity_exceeded
                if params.max_tool_batch_size is not None
                and len(outcomes) < len(tool_calls)
                else ToolErrorType.runtime_missing_result
            )
            missing_message = {
                ToolErrorType.cancelled: (
                    "The tool call was cancelled before it completed."
                ),
                ToolErrorType.runtime_hook_error: (
                    "The tool batch hook failed before returning a result."
                ),
                ToolErrorType.task_capacity_exceeded: (
                    "The tool call exceeded the configured batch capacity."
                ),
                ToolErrorType.runtime_missing_result: (
                    "The runtime did not produce a result for this tool call."
                ),
            }[missing_error_type]
            closed_batch = close_tool_batch(
                tool_calls,
                candidate_messages,
                (
                    tool_results_result.additional_messages
                    if tool_results_result is not None
                    else []
                ),
                missing_error_type=missing_error_type,
                missing_message=missing_message,
                initial_diagnostics=batch_diagnostics,
            )
            tool_results = list(closed_batch.messages)
            additional_messages = list(
                closed_batch.additional_messages
            )
            messages = [
                *messages,
                *tool_results,
                *additional_messages,
            ]
            validate_tool_transcript(messages)
            protocol_failed = (
                not closed_batch.is_valid and not batch_cancelled
            )
            requested_continue = (
                tool_results_result.should_continue
                if tool_results_result is not None
                else True
            )
            should_continue = (
                requested_continue
                and not protocol_failed
                and not batch_cancelled
            )
            updates = (
                tool_results_result.updates
                if tool_results_result is not None
                else {}
            )
            quality_recovery = (
                tool_results_result.quality_recovery
                if tool_results_result is not None
                and tool_results_result.quality_recovery is not None
                else state.quality_recovery
            )
            if (
                quality_recovery.active
                and quality_recovery.triggering_assessment_revision is None
            ):
                quality_recovery = replace(
                    quality_recovery,
                    triggering_assessment_revision=state.revision,
                )
            pending_query_event = PendingQueryEvent(
                event_id=_pending_event_id(state, "query.tool_result"),
                event_type="query.tool_result",
                transition_reason="tool_results",
                turn=state.turn,
                messages=tuple(tool_results),
                additional_messages=tuple(additional_messages),
                updates=dict(updates),
                should_continue=should_continue,
                protocol_diagnostics=tuple(
                    diagnostic.to_dict()
                    for diagnostic in closed_batch.diagnostics
                ),
            )
            if should_continue:
                state = advance(
                    state,
                    QueryStateAction(
                        phase=QueryPhase.PREPARING,
                        reason=ContinueReason.NEXT_TURN.value,
                        changes={
                            "messages": tuple(messages),
                            "pending_tool_batch": None,
                            "pending_query_event": pending_query_event,
                            "quality_recovery": quality_recovery,
                        },
                    ),
                )
            else:
                terminal_reason = (
                    TerminalReason.CANCELLED
                    if batch_cancelled
                    else TerminalReason.TOOL_PROTOCOL_VIOLATION
                    if protocol_failed
                    else TerminalReason.COMPLETED
                )
                detail = (
                    tool_results_result.reason
                    if tool_results_result is not None
                    else None
                )
                state = _terminal_state(
                    state,
                    reason=terminal_reason,
                    detail=detail,
                    messages=messages,
                    changes={
                        "pending_query_event": pending_query_event,
                        "quality_recovery": quality_recovery,
                    },
                )
            await _persist_query_state(params, state)
            if batch_cancellation is not None:
                raise batch_cancellation
            yield _state_changed_event(state)
            resumed_tool_batch = False
            continue
