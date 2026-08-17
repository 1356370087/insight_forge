"""Provider-neutral model failure and output recovery classification."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from langchain_core.messages import BaseMessage, HumanMessage

from open_deep_research.agents.research_context import response_was_truncated
from open_deep_research.model_fallback import (
    ModelCandidate as ModelCandidate,
)
from open_deep_research.model_fallback import (
    ModelErrorKind,
)
from open_deep_research.model_fallback import (
    build_model_candidate_chain as build_model_candidate_chain,
)
from open_deep_research.model_fallback import (
    classify_model_error as classify_model_error,
)
from open_deep_research.model_fallback import (
    invoke_with_model_fallback as invoke_with_model_fallback,
)
from open_deep_research.model_fallback import (
    sanitize_messages_for_model_fallback as sanitize_messages_for_model_fallback,
)


@dataclass(frozen=True, slots=True)
class ModelResponseStatus:
    """Normalized completion metadata for one successful model response."""

    finish_reason: str
    error_kind: ModelErrorKind | None = None
    recoverable: bool = False


class OutputRecoveryExhausted(RuntimeError):
    """Raised when a non-agent model output remains truncated after recovery."""


def _message_text(message: BaseMessage) -> str:
    if isinstance(message.content, str):
        return message.content
    return "\n".join(
        str(item.get("text", item)) if isinstance(item, dict) else str(item)
        for item in message.content
    )


async def invoke_with_output_recovery(
    call_model: Callable[
        [list[BaseMessage], int | None],
        Awaitable[BaseMessage],
    ],
    messages: list[BaseMessage],
    *,
    requested_output_tokens: int,
    maximum_output_tokens: int,
    escalation_enabled: bool = True,
    continuation_max_attempts: int = 3,
) -> BaseMessage:
    """Invoke a non-agent model with bounded escalation and continuation."""
    canonical_messages = list(messages)
    request_messages = list(messages)
    fragments: list[str] = []
    recovery_messages: list[BaseMessage] = []
    escalation_attempted = False
    continuation_count = 0
    output_override: int | None = None

    while True:
        response = await call_model(request_messages, output_override)
        if not classify_model_response(response).recoverable:
            if not fragments:
                return response
            consolidated = response.model_copy(deep=True)
            consolidated.content = "".join(
                [*fragments, _message_text(response)]
            )
            return consolidated

        can_escalate = (
            escalation_enabled
            and not escalation_attempted
            and requested_output_tokens > 0
            and maximum_output_tokens > requested_output_tokens
        )
        if can_escalate:
            escalation_attempted = True
            output_override = min(
                maximum_output_tokens,
                max(
                    requested_output_tokens * 2,
                    requested_output_tokens + 4_096,
                ),
            )
            request_messages = list(canonical_messages)
            recovery_messages = []
            continue

        if continuation_count >= max(0, continuation_max_attempts):
            raise OutputRecoveryExhausted(
                "output_recovery_exhausted"
            )
        continuation_count += 1
        fragments.append(_message_text(response))
        fragment = response.model_copy(deep=True)
        fragment.additional_kwargs = {
            **(fragment.additional_kwargs or {}),
            "query_internal": "output_fragment",
        }
        recovery_messages.extend(
            [
                fragment,
                HumanMessage(
                content=(
                    "Output token limit hit. Resume directly from the "
                    "previous fragment without repeating completed text."
                ),
                additional_kwargs={
                    "query_internal": "output_continuation"
                },
            ),
            ]
        )
        request_messages = [*canonical_messages, *recovery_messages]


def classify_model_response(response: BaseMessage) -> ModelResponseStatus:
    """Classify provider completion metadata without trusting one key shape."""
    metadata = getattr(response, "response_metadata", {}) or {}
    finish_reason = str(
        metadata.get("finish_reason")
        or metadata.get("stop_reason")
        or metadata.get("stop_sequence")
        or ""
    ).lower()
    if response_was_truncated(response):
        return ModelResponseStatus(
            finish_reason=finish_reason,
            error_kind=ModelErrorKind.OUTPUT_TRUNCATED,
            recoverable=True,
        )
    return ModelResponseStatus(finish_reason=finish_reason)


def resolve_model_context_window(
    model_name: str,
    *,
    overrides: dict[str, int] | None = None,
    unknown_default: int = 32_768,
) -> int:
    """Resolve a model window without an unsafe unknown-model 200K default."""
    from open_deep_research.tools.model_limits import get_model_token_limit

    if overrides and model_name in overrides:
        return max(1, int(overrides[model_name]))
    known = get_model_token_limit(model_name)
    return max(1, int(known if known is not None else unknown_default))


def resolve_model_max_output_tokens(
    model_name: str,
    *,
    requested: int,
    overrides: dict[str, int] | None = None,
) -> int:
    """Resolve the highest safe output cap known for one candidate."""
    if overrides and model_name in overrides:
        return max(requested, int(overrides[model_name]))
    return max(1, requested)
