"""Low-level, provider-neutral model fallback routing.

This module deliberately lives below ``agents`` and ``tools`` so both layers can
reuse the same routing policy without reversing their dependency direction.
"""

from __future__ import annotations

import inspect
import logging
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, TypeVar

from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.runnables import RunnableConfig

from open_deep_research.configuration import Configuration
from open_deep_research.model_circuit import (
    CircuitOpenError,
    get_model_circuit_registry,
    model_circuit_policy_from_configuration,
)
from open_deep_research.model_errors import is_token_limit_exceeded
from open_deep_research.model_resolution import build_model_config
from open_deep_research.observability import (
    get_trace_recorder,
    observe_model_circuit_transition,
)
from open_deep_research.public_events import event_publisher_from_config

logger = logging.getLogger(__name__)
_ModelResult = TypeVar("_ModelResult")


class ModelErrorKind(str, Enum):
    """Normalized model error categories used by recovery policies."""

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
class ModelCandidate:
    """One model and its frozen request configuration."""

    model_id: str
    model: Any
    model_config: dict[str, Any] = field(default_factory=dict)


_FALLBACK_ERROR_KINDS = frozenset(
    {
        ModelErrorKind.MODEL_UNAVAILABLE,
        ModelErrorKind.RATE_LIMITED,
        ModelErrorKind.TRANSIENT,
    }
)
_PROVIDER_BOUND_KEYS = {
    "signature",
    "thinking",
    "reasoning",
    "reasoning_content",
    "cache_control",
    "prompt_cache",
    "server_tool_use",
}
_ROLE_STAGE = {
    "supervisor": "planning",
    "researcher": "researching",
    "summarization": "researching",
    "message_summary": "researching",
    "compression": "synthesizing",
    "final_report": "writing",
    "quality_evaluation": "finalizing",
}


def build_model_candidate_chain(
    primary_model: str,
    fallback_models: Sequence[str],
    *,
    max_tokens: int,
    config: Mapping[str, Any] | None,
    role: str,
    model: Any,
) -> list[ModelCandidate]:
    """Build a stable, de-duplicated candidate chain for one model role."""
    candidates: list[ModelCandidate] = []
    seen: set[str] = set()
    for model_id in (primary_model, *fallback_models):
        normalized = str(model_id).strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        candidates.append(
            ModelCandidate(
                model_id=normalized,
                model=model,
                model_config=build_model_config(
                    normalized,
                    max_tokens,
                    config,
                    role=role,
                ),
            )
        )
    return candidates


async def invoke_with_model_fallback(
    invoke: Callable[[str, list[BaseMessage]], Awaitable[_ModelResult]],
    messages: Sequence[BaseMessage],
    *,
    primary_model: str,
    model_fallbacks: Mapping[str, Sequence[str]] | None,
    role: str,
    config: RunnableConfig | None = None,
    on_fallback: Callable[[dict[str, Any]], Awaitable[None] | None] | None = None,
) -> _ModelResult:
    """Invoke one role's model chain for bounded availability errors."""
    configured = (model_fallbacks or {}).get(role, ())
    chain: list[str] = []
    for model_id in (primary_model, *configured):
        normalized = str(model_id).strip()
        if normalized and normalized not in chain:
            chain.append(normalized)

    if config is not None:
        try:
            configuration = Configuration.from_runnable_config(config)
            if configuration.model_circuit_breaker_enabled and chain:
                selected, transition = (
                    await get_model_circuit_registry().select_candidate_index(
                        chain,
                        model_circuit_policy_from_configuration(configuration),
                    )
                )
                if transition is not None and selected:
                    chain = [chain[selected], *chain[:selected], *chain[selected + 1 :]]
                await observe_model_circuit_transition(
                    transition,
                    config,
                    agent_role=role,
                )
        except Exception:
            logger.debug("Model circuit fallback preflight failed open", exc_info=True)

    request_messages = list(messages)
    for index, model_id in enumerate(chain):
        try:
            return await invoke(model_id, request_messages)
        except Exception as error:
            error_kind = classify_model_error(error, model_id)
            if error_kind not in _FALLBACK_ERROR_KINDS or index + 1 >= len(chain):
                raise
            next_model = chain[index + 1]
            request_messages = sanitize_messages_for_model_fallback(request_messages)
            event = {
                "turn": _query_turn(config),
                "from_model": model_id,
                "to_model": next_model,
                "reason": error_kind.value,
            }
            await _record_fallback_event(
                event,
                role=role,
                attempt=index + 1,
                config=config,
            )
            if on_fallback is not None:
                callback_result = on_fallback(event)
                if inspect.isawaitable(callback_result):
                    await callback_result

    raise RuntimeError("model_fallback_chain_empty")


def _query_turn(config: RunnableConfig | None) -> int:
    metadata = (config or {}).get("metadata") or {}
    value = metadata.get("query_turn", metadata.get("turn", 0))
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


async def _record_fallback_event(
    event: dict[str, Any],
    *,
    role: str,
    attempt: int,
    config: RunnableConfig | None,
) -> None:
    logger.info(
        "query.model_fallback",
        extra={"model_fallback": {**event, "role": role}},
    )
    if config is None:
        return
    get_trace_recorder(config).active_span().record_retry(
        attempt=attempt,
        error_type=f"model_fallback:{event['reason']}",
        retryable=True,
        message=f"{event['from_model']} -> {event['to_model']}",
    )
    metadata = config.get("metadata") or {}
    if not metadata.get("run_id"):
        return
    try:
        await event_publisher_from_config(config).publish(
            "query.model_fallback",
            stage=_ROLE_STAGE.get(role, "researching"),
            payload=event,
            dedupe_key=(
                f"model-fallback:{role}:{event['turn']}:"
                f"{event['from_model']}:{event['to_model']}"
            ),
        )
    except Exception:
        logger.warning(
            "query.model_fallback publication failed",
            exc_info=True,
        )


def classify_model_error(
    error: BaseException,
    model_name: str | None = None,
) -> ModelErrorKind:
    """Map provider exceptions to the bounded Query recovery taxonomy."""
    if isinstance(error, CircuitOpenError):
        return ModelErrorKind.MODEL_UNAVAILABLE
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
        marker in class_name for marker in ("connection", "transport", "timeout")
    ):
        return ModelErrorKind.TRANSIENT
    if any(
        marker in text
        for marker in ("image size", "invalid image", "unsupported image", "media type")
    ):
        return ModelErrorKind.INVALID_MEDIA
    if (
        status_code in {400, 404, 409, 413, 422}
        or code in {"invalid_request_error", "invalid_request"}
        or error_type == "invalid_request_error"
    ):
        return ModelErrorKind.INVALID_REQUEST
    return ModelErrorKind.UNKNOWN


def sanitize_messages_for_model_fallback(
    messages: list[BaseMessage],
) -> list[BaseMessage]:
    """Remove provider-bound metadata while preserving the standard transcript."""
    sanitized: list[BaseMessage] = []
    for message in messages:
        copied = message.model_copy(deep=True)
        additional_kwargs = dict(getattr(copied, "additional_kwargs", {}) or {})
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
