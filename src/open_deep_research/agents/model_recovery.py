"""Provider-neutral model failure and output recovery classification."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from open_deep_research.agents.research_context import response_was_truncated
from open_deep_research.tools.utils import (
    get_model_token_limit,
    is_token_limit_exceeded,
)


class ModelErrorKind(str, Enum):
    """Normalized model error categories used by the Query recovery policy."""

    PROMPT_TOO_LONG = "prompt_too_long"
    OUTPUT_TRUNCATED = "output_truncated"
    TRANSIENT = "transient"
    RATE_LIMITED = "rate_limited"
    MODEL_UNAVAILABLE = "model_unavailable"
    AUTH = "auth"
    INVALID_REQUEST = "invalid_request"
    INVALID_MEDIA = "invalid_media"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class ModelResponseStatus:
    """Normalized completion metadata for one successful model response."""

    finish_reason: str
    error_kind: ModelErrorKind | None = None
    recoverable: bool = False


@dataclass(frozen=True, slots=True)
class ModelCandidate:
    """One model and its frozen request configuration."""

    model_id: str
    model: Any
    model_config: dict[str, Any] = field(default_factory=dict)


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


def classify_model_error(
    error: BaseException,
    model_name: str | None = None,
) -> ModelErrorKind:
    """Map provider exceptions to the bounded Query recovery taxonomy."""
    if isinstance(error, TimeoutError):
        return ModelErrorKind.MODEL_UNAVAILABLE
    if isinstance(error, KeyboardInterrupt | SystemExit):
        return ModelErrorKind.CANCELLED
    if isinstance(error, Exception) and is_token_limit_exceeded(
        error,
        model_name or "",
    ):
        return ModelErrorKind.PROMPT_TOO_LONG

    text = str(error).lower()
    class_name = type(error).__name__.lower()
    status_code = getattr(error, "status_code", None)
    code = str(getattr(error, "code", "") or "").lower()
    error_type = str(getattr(error, "type", "") or "").lower()

    if status_code in {401, 403} or any(
        marker in text
        for marker in (
            "authentication",
            "unauthorized",
            "invalid api key",
            "permission denied",
        )
    ):
        return ModelErrorKind.AUTH
    if status_code == 429 or "rate limit" in text or "ratelimit" in class_name:
        return ModelErrorKind.RATE_LIMITED
    if status_code in {502, 503, 504} or any(
        marker in text
        for marker in (
            "model unavailable",
            "overloaded",
            "high demand",
            "service unavailable",
        )
    ):
        return ModelErrorKind.MODEL_UNAVAILABLE
    if status_code in {408, 425, 500} or any(
        marker in class_name
        for marker in ("connection", "transport", "timeout")
    ):
        return ModelErrorKind.TRANSIENT
    if any(
        marker in text
        for marker in (
            "image size",
            "invalid image",
            "unsupported image",
            "media type",
        )
    ):
        return ModelErrorKind.INVALID_MEDIA
    if status_code in {400, 404, 409, 413, 422} or code in {
        "invalid_request_error",
        "invalid_request",
    } or error_type == "invalid_request_error":
        return ModelErrorKind.INVALID_REQUEST
    return ModelErrorKind.UNKNOWN


_PROVIDER_BOUND_KEYS = {
    "signature",
    "thinking",
    "reasoning",
    "reasoning_content",
    "cache_control",
    "prompt_cache",
    "server_tool_use",
}


def sanitize_messages_for_model_fallback(
    messages: list[BaseMessage],
) -> list[BaseMessage]:
    """Remove provider-bound metadata while preserving the standard transcript."""
    sanitized: list[BaseMessage] = []
    for message in messages:
        copied = message.model_copy(deep=True)
        additional_kwargs = dict(
            getattr(copied, "additional_kwargs", {}) or {}
        )
        for key in tuple(additional_kwargs):
            if key.lower() in _PROVIDER_BOUND_KEYS:
                additional_kwargs.pop(key, None)
        copied.additional_kwargs = additional_kwargs
        if isinstance(copied, AIMessage):
            response_metadata = dict(copied.response_metadata or {})
            for key in tuple(response_metadata):
                if key.lower() in _PROVIDER_BOUND_KEYS:
                    response_metadata.pop(key, None)
            copied.response_metadata = response_metadata
        sanitized.append(copied)
    return sanitized


def resolve_model_context_window(
    model_name: str,
    *,
    overrides: dict[str, int] | None = None,
    unknown_default: int = 32_768,
) -> int:
    """Resolve a model window without an unsafe unknown-model 200K default."""
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
