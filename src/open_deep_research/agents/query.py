"""Inner agent loop: model calls, tool execution, and continuation decisions."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.runnables import RunnableConfig

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
from open_deep_research.observability import (
    TokenUsage,
    apply_helicone_config,
    get_trace_recorder,
    invoke_model_with_retry_observability,
    observe_tool_call,
)
from open_deep_research.runtime import normalize_messages
from open_deep_research.runtime_control import CancellationScope
from open_deep_research.tools.base import (
    Tool,
    build_tool_registry,
    tools_to_model_definitions,
)
from open_deep_research.tools.governance import (
    AgentRole,
    GovernedToolCallResult,
    ToolError,
    ToolErrorType,
    execute_governed_tool_call,
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
    messages: list[BaseMessage] = field(default_factory=list)
    updates: dict[str, Any] = field(default_factory=dict)
    reason: TransitionReason = "stop_hook_blocked"


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
CallModel = Callable[[list[BaseMessage]], Awaitable[BaseMessage]]


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
    max_tool_description_chars: int | None = None
    context_policy: ContextPolicy = field(default_factory=ContextPolicy)
    before_turn_hooks: Sequence[BeforeTurnHook] = field(default_factory=list)
    stop_hooks: Sequence[StopHook] = field(default_factory=list)
    tool_batch_hook: ToolBatchHook | None = None
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


async def _default_call_model(params: QueryParams, messages: list[BaseMessage]) -> BaseMessage:
    model = params.model
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
    model_config = apply_helicone_config(
        params.model_config,
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
        attributes={"tool_count": len(params.tools)},
    )


async def query(params: QueryParams) -> AsyncIterator[QueryEvent]:
    """Run the inner model/tool loop.

    This is deliberately protocol-light: outer engines decide how to persist,
    stream, and package events. The loop owns continuation decisions.
    """
    messages: list[BaseMessage] = normalize_messages(list(params.messages))
    if params.max_concurrent_tools is not None and params.max_concurrent_tools <= 0:
        raise ValueError("max_concurrent_tools must be greater than zero")
    if params.max_tool_batch_size is not None and params.max_tool_batch_size <= 0:
        raise ValueError("max_tool_batch_size must be greater than zero")
    validate_tool_transcript(messages, allow_pending_tail=True)
    transition = QueryTransition(reason="start", turn=params.initial_turn)
    yield QueryEvent("query.started", {"transition": transition.__dict__})

    turn = params.initial_turn
    while True:
        if params.max_turns is not None and turn >= params.max_turns:
            validate_tool_transcript(messages)
            transition = QueryTransition(reason="max_turns", turn=turn)
            yield QueryEvent("query.completed", {"transition": transition.__dict__, "messages": messages})
            return

        if params.hook_timeout_seconds is not None and params.hook_timeout_seconds <= 0:
            raise ValueError("hook_timeout_seconds must be greater than zero")
        if (
            params.tool_batch_timeout_seconds is not None
            and params.tool_batch_timeout_seconds <= 0
        ):
            raise ValueError("tool_batch_timeout_seconds must be greater than zero")
        if params.tool_timeout_seconds is not None and params.tool_timeout_seconds <= 0:
            raise ValueError("tool_timeout_seconds must be greater than zero")
        before_turn_hooks = params.before_turn_hooks
        for before_turn_hook in before_turn_hooks:
            before_turn_result = await (
                asyncio.wait_for(
                    before_turn_hook(messages, turn + 1, params.config),
                    timeout=params.hook_timeout_seconds,
                )
                if params.hook_timeout_seconds is not None
                else before_turn_hook(messages, turn + 1, params.config)
            )
            if before_turn_result is None:
                continue
            if before_turn_result.replace_messages is not None:
                messages = normalize_messages(before_turn_result.replace_messages)
            else:
                messages.extend(before_turn_result.messages)
            transition = QueryTransition(reason=before_turn_result.reason, turn=turn)
            yield QueryEvent(
                "query.transition",
                {
                    "transition": transition.__dict__,
                    "messages": before_turn_result.messages,
                    "replacement_messages": before_turn_result.replace_messages,
                    "updates": before_turn_result.updates,
                },
            )
            if before_turn_result.should_stop:
                validate_tool_transcript(messages)
                yield QueryEvent(
                    "query.completed",
                    {
                        "transition": transition.__dict__,
                        "messages": messages,
                        "updates": before_turn_result.updates,
                    },
                )
                return

        turn += 1
        recorder = get_trace_recorder(params.config)
        with recorder.start_span(
            name="query.request",
            kind="agent",
            agent_role=params.role.value,
            attributes={"turn": turn},
        ):
            validate_tool_transcript(messages)
            messages_for_query = prepare_messages_for_query(messages, params.context_policy)
            request_messages = list(messages_for_query)
            if params.system_prompt:
                system_message = (
                    params.system_prompt
                    if isinstance(params.system_prompt, BaseMessage)
                    else SystemMessage(content=params.system_prompt)
                )
                request_messages = [system_message, *request_messages]

        yield QueryEvent(
            "query.request",
            {
                "turn": turn,
                "message_count": len(messages),
                "messages_for_query_count": len(messages_for_query),
            },
        )

        budget_gate = params.budget_gate
        execution_namespace = params.execution_namespace or str(
            params.config.get("metadata", {}).get("task_id")
            or params.role.value
        )
        model_name = str(params.model_config.get("model", ""))
        model_stage = f"{params.role.value}.model"
        if params.cancellation_scope is not None:
            params.cancellation_scope.checkpoint(model_stage)
        max_model_attempts = max(1, params.model_transport_max_attempts)
        response: BaseMessage | None = None
        successful_model_op_key = ""
        for model_attempt in range(1, max_model_attempts + 1):
            base_model_op_key = (
                f"model:{execution_namespace}:{params.role.value}:{turn}"
            )
            model_op_key = (
                base_model_op_key
                if model_attempt == 1
                else f"{base_model_op_key}:retry:{model_attempt}"
            )
            if budget_gate is not None and budget_gate.enabled:
                estimated_input = count_tokens_approximately(request_messages)
                estimated_output = int(params.model_config.get("max_tokens") or 0)
                try:
                    budget_gate.reserve_model_call(
                        model_op_key,
                        estimated_input_tokens=estimated_input,
                        estimated_output_tokens=estimated_output,
                        model_name=model_name,
                    )
                except DeadlineExceeded:
                    transition = QueryTransition(reason="deadline_exceeded", turn=turn)
                    yield QueryEvent(
                        "query.completed",
                        {"transition": transition.__dict__, "messages": messages},
                    )
                    return
                except BudgetExhausted:
                    transition = QueryTransition(reason="budget_exhausted", turn=turn)
                    yield QueryEvent(
                        "query.completed",
                        {
                            "transition": transition.__dict__,
                            "messages": messages,
                            "budget_exhausted": True,
                        },
                    )
                    return
            if params.call_model:
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
                    model_call = params.call_model(request_messages)
            else:
                model_call = _default_call_model(params, request_messages)
            try:
                if params.cancellation_scope is not None:
                    response = await params.cancellation_scope.run(
                        model_call,
                        stage=model_stage,
                        timeout_seconds=params.model_timeout_seconds,
                    )
                else:
                    response = await (
                        asyncio.wait_for(
                            model_call,
                            timeout=params.model_timeout_seconds,
                        )
                        if params.model_timeout_seconds is not None
                        else model_call
                    )
            except TimeoutError:
                if model_attempt < max_model_attempts:
                    yield QueryEvent(
                        "query.model_retry",
                        {
                            "turn": turn,
                            "attempt": model_attempt,
                            "max_attempts": max_model_attempts,
                            "reason": "model_timeout",
                            "timeout_seconds": params.model_timeout_seconds,
                        },
                    )
                    continue
                transition = QueryTransition(reason="model_timeout", turn=turn)
                yield QueryEvent(
                    "query.completed",
                    {"transition": transition.__dict__, "messages": messages},
                )
                return
            successful_model_op_key = model_op_key
            break
        assert response is not None
        if budget_gate is not None and budget_gate.ledger is not None:
            usage = TokenUsage.from_response(response)
            try:
                budget_gate.settle_model_call(
                    successful_model_op_key,
                    input_tokens=usage.input_tokens,
                    output_tokens=usage.output_tokens,
                    model_name=model_name,
                )
            except (BudgetExhausted, DeadlineExceeded):
                pass
        run_id = str(params.config.get("metadata", {}).get("run_id", "default"))
        response, call_diagnostics = canonicalize_ai_tool_calls(
            response,
            run_id=run_id,
            role=params.role.value,
            turn=turn,
        )
        messages.append(response)
        yield QueryEvent(
            "query.model_event",
            {
                "turn": turn,
                "message": response,
                "protocol_diagnostics": [
                    diagnostic.to_dict() for diagnostic in call_diagnostics
                ],
            },
        )

        tool_calls = list(getattr(response, "tool_calls", []) or [])
        if not tool_calls:
            stop_updates: dict[str, Any] = {}
            stop_messages: list[BaseMessage] = []
            completion_reason: TransitionReason = "completed"
            for stop_hook in params.stop_hooks:
                stop_result = await stop_hook(messages, params.config)
                if stop_result is None:
                    continue
                stop_updates.update(stop_result.updates)
                stop_messages.extend(stop_result.messages)
                completion_reason = stop_result.reason
                if stop_result.should_continue:
                    messages.extend(stop_messages)
                    transition = QueryTransition(reason=completion_reason, turn=turn)
                    yield QueryEvent(
                        "query.transition",
                        {
                            "transition": transition.__dict__,
                            "messages": stop_messages,
                            "updates": stop_updates,
                        },
                    )
                    break
            else:
                messages.extend(stop_messages)
                validate_tool_transcript(messages)
                transition = QueryTransition(reason=completion_reason, turn=turn)
                yield QueryEvent(
                    "query.completed",
                    {
                        "transition": transition.__dict__,
                        "messages": messages,
                        "updates": stop_updates,
                    },
                )
                return
            continue

        execution_tools = list(params.execution_tools or params.tools)
        tools_by_name = build_tool_registry(execution_tools)
        allowed = resolve_allowed_tools(params.role, params.config, set(tools_by_name))
        yield QueryEvent("query.tool_call", {"turn": turn, "tool_calls": tool_calls})

        outcomes: list[GovernedToolCallResult] = []
        tool_results_result: ToolResultsHookResult | None = None
        batch_diagnostics: tuple[ToolProtocolDiagnostic, ...] = ()
        hook_failed = False
        batch_cancelled = False
        try:
            if params.tool_batch_hook is not None:
                batch_timeout = (
                    params.tool_batch_timeout_seconds
                    if params.tool_batch_timeout_seconds is not None
                    else params.hook_timeout_seconds
                )
                batch_hook_call = params.tool_batch_hook(
                    messages,
                    tool_calls,
                    tools_by_name,
                    turn,
                    params.config,
                )
                tool_results_result = await (
                    asyncio.wait_for(
                        batch_hook_call,
                        timeout=batch_timeout,
                    )
                    if batch_timeout is not None
                    else batch_hook_call
                )
            else:
                configurable = Configuration.from_runnable_config(params.config)
                concurrency = params.max_concurrent_tools or len(tool_calls) or 1
                semaphore = asyncio.Semaphore(concurrency)

                async def _execute_tool(
                    tool_call: dict[str, Any],
                ) -> GovernedToolCallResult:
                    async with semaphore:
                        budget_op_key = (
                            f"tool:{execution_namespace}:{params.role.value}:"
                            f"{turn}:{tool_call['id']}"
                        )
                        if budget_gate is not None and budget_gate.enabled:
                            try:
                                budget_gate.reserve_tool_call(budget_op_key)
                            except (BudgetExhausted, DeadlineExceeded):
                                error = ToolError(
                                    error_type=ToolErrorType.budget_exhausted,
                                    tool_name=str(tool_call.get("name", "unknown_tool")),
                                    message=(
                                        "The tool call was skipped because the run "
                                        "budget or deadline was exhausted."
                                    ),
                                )
                                get_trace_recorder(params.config).active_span() \
                                    .record_outcome(error_type=error.error_type.value)
                                return GovernedToolCallResult(
                                    message=error.to_tool_message(str(tool_call["id"])),
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
                                tool_name=str(tool_call.get("name", "unknown_tool")),
                                message="The tool call exceeded its execution timeout.",
                                retryable=True,
                                detail={"timeout_seconds": params.tool_timeout_seconds},
                            )
                            return GovernedToolCallResult(
                                message=error.to_tool_message(str(tool_call["id"])),
                                error=error,
                            )
                        except asyncio.CancelledError:
                            raise
                        except Exception as exc:  # noqa: BLE001 - close this call independently
                            error = ToolError(
                                error_type=ToolErrorType.internal_error,
                                tool_name=str(tool_call.get("name", "unknown_tool")),
                                message="The tool call failed unexpectedly.",
                                detail={"error_type": type(exc).__name__},
                            )
                            return GovernedToolCallResult(
                                message=error.to_tool_message(str(tool_call["id"])),
                                error=error,
                            )

                batch_limit = params.max_tool_batch_size or len(tool_calls)
                runnable_calls = tool_calls[:batch_limit]
                outcomes = list(await asyncio.gather(*(
                    _execute_tool(tool_call) for tool_call in runnable_calls
                )))
                if len(runnable_calls) < len(tool_calls):
                    overflow = tool_calls[len(runnable_calls):]
                    batch_diagnostics = tuple(
                        ToolProtocolDiagnostic(
                            code="tool_batch_capacity_exceeded",
                            tool_call_id=str(call["id"]),
                        )
                        for call in overflow
                    )
                if params.tool_results_hook is not None:
                    results_hook_call = params.tool_results_hook(
                        messages,
                        tool_calls,
                        outcomes,
                        tools_by_name,
                        turn,
                        params.config,
                    )
                    tool_results_result = await (
                        asyncio.wait_for(
                            results_hook_call,
                            timeout=params.hook_timeout_seconds,
                        )
                        if params.hook_timeout_seconds is not None
                        else results_hook_call
                    )
        except asyncio.CancelledError:
            batch_cancelled = True
            batch_diagnostics = (*batch_diagnostics, ToolProtocolDiagnostic(
                code="tool_batch_cancelled",
            ))
        except Exception as exc:  # noqa: BLE001 - close the current batch before stopping
            hook_failed = True
            batch_diagnostics = (*batch_diagnostics, ToolProtocolDiagnostic(
                code="tool_batch_hook_error",
                detail={"error_type": type(exc).__name__},
            ))

        candidate_messages: list[BaseMessage]
        if tool_results_result is not None and tool_results_result.messages is not None:
            candidate_messages = list(tool_results_result.messages)
        else:
            candidate_messages = []
            candidate_messages.extend(outcome.message for outcome in outcomes)
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
            ToolErrorType.cancelled: "The tool call was cancelled before it completed.",
            ToolErrorType.runtime_hook_error: "The tool batch hook failed before returning a result.",
            ToolErrorType.task_capacity_exceeded: "The tool call exceeded the configured batch capacity.",
            ToolErrorType.runtime_missing_result: "The runtime did not produce a result for this tool call.",
        }[missing_error_type]
        closed_batch = close_tool_batch(
            tool_calls,
            candidate_messages,
            tool_results_result.additional_messages if tool_results_result is not None else [],
            missing_error_type=missing_error_type,
            missing_message=missing_message,
            initial_diagnostics=batch_diagnostics,
        )
        tool_results = list(closed_batch.messages)
        additional_messages = list(closed_batch.additional_messages)
        messages.extend(tool_results)
        messages.extend(additional_messages)
        validate_tool_transcript(messages)

        protocol_failed = not closed_batch.is_valid and not batch_cancelled
        requested_continue = (
            tool_results_result.should_continue
            if tool_results_result is not None
            else True
        )
        should_continue = requested_continue and not protocol_failed and not batch_cancelled
        updates = tool_results_result.updates if tool_results_result is not None else {}
        transition = QueryTransition(reason="tool_results", turn=turn)
        yield QueryEvent(
            "query.tool_result",
            {
                "turn": turn,
                "transition": transition.__dict__,
                "messages": tool_results,
                "additional_messages": additional_messages,
                "updates": updates,
                "should_continue": should_continue,
                "protocol_diagnostics": [
                    diagnostic.to_dict() for diagnostic in closed_batch.diagnostics
                ],
            },
        )
        if not should_continue:
            completion_reason = (
                "cancelled"
                if batch_cancelled
                else "tool_protocol_violation"
                if protocol_failed
                else tool_results_result.reason
                if tool_results_result is not None and tool_results_result.reason is not None
                else "completed"
            )
            transition = QueryTransition(reason=completion_reason, turn=turn)
            yield QueryEvent(
                "query.completed",
                {
                    "transition": transition.__dict__,
                    "messages": messages,
                },
            )
            return
