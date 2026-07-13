"""Inner agent loop: model calls, tool execution, and continuation decisions."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig

from open_deep_research.configuration import Configuration
from open_deep_research.observability import (
    apply_helicone_config,
    get_trace_recorder,
    invoke_model_with_retry_observability,
    observe_tool_call,
)
from open_deep_research.runtime import normalize_messages
from open_deep_research.tools.base import (
    Tool,
    build_tool_registry,
    tools_to_model_definitions,
)
from open_deep_research.tools.governance import (
    AgentRole,
    GovernedToolCallResult,
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
            while cursor < len(projected) and isinstance(projected[cursor], ToolMessage):
                pending_ids.add(projected[cursor].tool_call_id)
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
        model = model.bind_tools(
            await tools_to_model_definitions(
                list(params.tools),
                max_description_chars=params.max_tool_description_chars,
            )
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
    transition = QueryTransition(reason="start", turn=params.initial_turn)
    yield QueryEvent("query.started", {"transition": transition.__dict__})

    turn = params.initial_turn
    while True:
        if params.max_turns is not None and turn >= params.max_turns:
            transition = QueryTransition(reason="max_turns", turn=turn)
            yield QueryEvent("query.completed", {"transition": transition.__dict__, "messages": messages})
            return

        for before_turn_hook in params.before_turn_hooks:
            before_turn_result = await before_turn_hook(messages, turn + 1, params.config)
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

        if params.call_model:
            with recorder.start_span(
                name=params.model_span_name,
                kind="llm",
                agent_role=params.role.value,
                attributes={"custom_call_model": True},
                input_payload=request_messages,
            ):
                response = await params.call_model(request_messages)
        else:
            response = await _default_call_model(params, request_messages)
        messages.append(response)
        yield QueryEvent("query.model_event", {"turn": turn, "message": response})

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

        tools_by_name = build_tool_registry(list(params.tools))
        allowed = resolve_allowed_tools(params.role, params.config, set(tools_by_name))
        yield QueryEvent("query.tool_call", {"turn": turn, "tool_calls": tool_calls})

        tool_results_result = None
        if params.tool_batch_hook is not None:
            tool_results_result = await params.tool_batch_hook(
                messages,
                tool_calls,
                tools_by_name,
                turn,
                params.config,
            )
            outcomes: list[GovernedToolCallResult] = []
        else:
            configurable = Configuration.from_runnable_config(params.config)

            async def _execute_tool(tool_call: dict[str, Any]) -> GovernedToolCallResult:
                return await observe_tool_call(
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

            tool_tasks = [_execute_tool(tool_call) for tool_call in tool_calls]
            outcomes = await asyncio.gather(*tool_tasks)
            if params.tool_results_hook is not None:
                tool_results_result = await params.tool_results_hook(
                    messages,
                    tool_calls,
                    outcomes,
                    tools_by_name,
                    turn,
                    params.config,
                )
        tool_results = (
            list(tool_results_result.messages)
            if tool_results_result is not None and tool_results_result.messages is not None
            else [outcome.message for outcome in outcomes]
        )
        messages.extend(tool_results)
        if tool_results_result is not None:
            messages.extend(tool_results_result.additional_messages)
        transition = QueryTransition(reason="tool_results", turn=turn)
        yield QueryEvent(
            "query.tool_result",
            {
                "turn": turn,
                "transition": transition.__dict__,
                "messages": tool_results,
                "additional_messages": (
                    tool_results_result.additional_messages
                    if tool_results_result is not None
                    else []
                ),
                "updates": (
                    tool_results_result.updates if tool_results_result is not None else {}
                ),
                "should_continue": (
                    tool_results_result.should_continue
                    if tool_results_result is not None
                    else True
                ),
            },
        )
        if tool_results_result is not None and not tool_results_result.should_continue:
            transition = QueryTransition(reason="completed", turn=turn)
            yield QueryEvent(
                "query.completed",
                {
                    "transition": transition.__dict__,
                    "messages": messages,
                    "updates": tool_results_result.updates,
                },
            )
            return
