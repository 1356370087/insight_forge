"""Inner agent loop: model calls, tool execution, and continuation decisions."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from open_deep_research.observability import (
    apply_helicone_config,
    get_trace_recorder,
    invoke_model_with_observability,
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
    execute_governed_tool_call,
    resolve_allowed_tools,
)

TransitionReason = Literal[
    "start",
    "tool_results",
    "stop_hook_blocked",
    "token_budget_continue",
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
    reason: TransitionReason = "stop_hook_blocked"


StopHook = Callable[[list[BaseMessage], RunnableConfig], Awaitable[StopHookResult | None]]
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
    model_config: dict[str, Any] = field(default_factory=dict)
    max_turns: int | None = None
    context_policy: ContextPolicy = field(default_factory=ContextPolicy)
    stop_hooks: Sequence[StopHook] = field(default_factory=list)
    call_model: CallModel | None = None


def prepare_messages_for_query(
    messages: list[BaseMessage],
    policy: ContextPolicy,
) -> list[BaseMessage]:
    """Create the compact request view from the full inner-loop history."""
    projected = list(messages)
    if policy.keep_last_messages is not None and len(projected) > policy.keep_last_messages:
        projected = projected[-policy.keep_last_messages:]

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
            await tools_to_model_definitions(list(params.tools))
        )
    model_config = apply_helicone_config(
        params.model_config,
        params.config,
        span_name="query.model",
        agent_role=params.role.value,
    )
    if model_config and hasattr(model, "with_config"):
        model = model.with_config(model_config)
    return await invoke_model_with_observability(
        model,
        messages,
        params.config,
        span_name="query.model",
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
    transition = QueryTransition(reason="start", turn=0)
    yield QueryEvent("query.started", {"transition": transition.__dict__})

    turn = 0
    while True:
        if params.max_turns is not None and turn >= params.max_turns:
            transition = QueryTransition(reason="max_turns", turn=turn)
            yield QueryEvent("query.completed", {"transition": transition.__dict__, "messages": messages})
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
                name="query.model",
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
            for hook in params.stop_hooks:
                hook_result = await hook(messages, params.config)
                if hook_result and hook_result.should_continue:
                    messages.extend(hook_result.messages)
                    transition = QueryTransition(reason=hook_result.reason, turn=turn)
                    yield QueryEvent(
                        "query.transition",
                        {"transition": transition.__dict__, "messages": hook_result.messages},
                    )
                    break
            else:
                transition = QueryTransition(reason="completed", turn=turn)
                yield QueryEvent("query.completed", {"transition": transition.__dict__, "messages": messages})
                return
            continue

        tools_by_name = build_tool_registry(list(params.tools))
        allowed = resolve_allowed_tools(params.role, params.config, set(tools_by_name))
        yield QueryEvent("query.tool_call", {"turn": turn, "tool_calls": tool_calls})

        async def _execute_tool(tool_call: dict[str, Any]) -> BaseMessage:
            outcome = await observe_tool_call(
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
                ),
            )
            return outcome.message

        tool_tasks = [_execute_tool(tool_call) for tool_call in tool_calls]
        tool_results = await asyncio.gather(*tool_tasks)
        messages.extend(tool_results)
        transition = QueryTransition(reason="tool_results", turn=turn)
        yield QueryEvent(
            "query.tool_result",
            {
                "turn": turn,
                "transition": transition.__dict__,
                "messages": tool_results,
            },
        )
