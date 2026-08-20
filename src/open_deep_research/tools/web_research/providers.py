"""Shared native-provider search SDK helpers."""

from __future__ import annotations

import asyncio
import contextlib
import os
import random
from collections.abc import Awaitable, Callable
from typing import Any

from anthropic import AsyncAnthropic
from langchain_core.messages import HumanMessage
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.runnables import RunnableConfig
from openai import AsyncOpenAI

from open_deep_research.budgets import BudgetGate
from open_deep_research.configuration import Configuration
from open_deep_research.events.public import event_publisher_from_config
from open_deep_research.observability import TokenUsage, get_trace_recorder
from open_deep_research.tools.governance import classify_llm_retryable_error
from open_deep_research.tools.legacy_shims import get_api_key_for_model
from open_deep_research.tools.tavily_search import summarization


def strip_provider_prefix(model_name: str, provider: str) -> str:
    """Return a model id without its matching provider prefix."""
    if model_name and ":" in model_name and model_name.split(":", 1)[0] == provider:
        return model_name.split(":", 1)[1]
    return model_name


def _to_int(value: Any) -> int:
    try:
        return int(value) if value is not None else 0
    except (TypeError, ValueError):
        return 0


def _exception_status(exc: BaseException) -> int | None:
    for attribute in ("status_code", "status"):
        status = getattr(exc, attribute, None)
        if isinstance(status, int):
            return status
    return None


def _safe_exception_message(exc: BaseException) -> str:
    try:
        text = str(exc)
    except Exception:  # noqa: BLE001 - observability must never mask the SDK error
        text = ""
    return text or type(exc).__name__


def _is_uncertain_failure(exc: BaseException) -> bool:
    if isinstance(exc, TimeoutError | ConnectionError | asyncio.CancelledError):
        return True
    text = f"{type(exc).__name__} {_safe_exception_message(exc)}".lower()
    return any(
        marker in text
        for marker in (
            "timeout",
            "connection",
            "disconnect",
            "network",
            "transport",
            "brokenpipe",
            "incomplete read",
        )
    )


def _approximate_text_tokens(value: Any) -> int:
    text = str(value or "")
    if not text:
        return 0
    try:
        return max(
            1,
            int(count_tokens_approximately([HumanMessage(content=text)])),
        )
    except Exception:  # noqa: BLE001 - estimation remains fail-open
        return max(1, len(text) // 4)


def sdk_usage(response: Any) -> TokenUsage:
    """Normalize a native SDK response's token usage."""
    usage_obj = getattr(response, "usage", None)
    if usage_obj is None:
        return TokenUsage()
    input_tokens = 0
    output_tokens = 0
    for attribute in ("input_tokens", "prompt_tokens", "input_token_count"):
        input_tokens = _to_int(getattr(usage_obj, attribute, None))
        if input_tokens:
            break
    for attribute in ("output_tokens", "completion_tokens", "output_token_count"):
        output_tokens = _to_int(getattr(usage_obj, attribute, None))
        if output_tokens:
            break
    total_tokens = _to_int(getattr(usage_obj, "total_tokens", None)) or (
        input_tokens + output_tokens
    )
    input_details = getattr(usage_obj, "input_tokens_details", None) or getattr(
        usage_obj, "prompt_tokens_details", None
    )
    output_details = getattr(usage_obj, "output_tokens_details", None) or getattr(
        usage_obj, "completion_tokens_details", None
    )
    cached_detail = (
        input_details.get("cached_tokens")
        if isinstance(input_details, dict)
        else getattr(input_details, "cached_tokens", None)
    )
    reasoning_detail = (
        output_details.get("reasoning_tokens")
        if isinstance(output_details, dict)
        else getattr(output_details, "reasoning_tokens", None)
    )
    cached_input_tokens = _to_int(
        getattr(usage_obj, "cache_read_input_tokens", None) or cached_detail
    )
    cache_creation_input_tokens = _to_int(
        getattr(usage_obj, "cache_creation_input_tokens", None)
    )
    reasoning_tokens = _to_int(reasoning_detail)
    return TokenUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        cached_input_tokens=cached_input_tokens,
        cache_creation_input_tokens=cache_creation_input_tokens,
        reasoning_tokens=reasoning_tokens,
        raw_usage={
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "cached_input_tokens": cached_input_tokens,
            "cache_creation_input_tokens": cache_creation_input_tokens,
            "reasoning_tokens": reasoning_tokens,
        },
    )


def _response_text(response: Any) -> str:
    text = getattr(response, "output_text", None)
    if text:
        return str(text)
    return "\n".join(
        str(getattr(block, "text", "") or "")
        for block in getattr(response, "content", None) or []
        if getattr(block, "type", None) == "text"
    )


async def sdk_call_with_observability(
    call: Callable[[], Awaitable[Any]],
    *,
    span_name: str,
    provider: str,
    model: str,
    config: RunnableConfig,
    input_preview: Any = None,
) -> Any:
    """Run a native provider call with standard tracing and retry semantics."""
    recorder = get_trace_recorder(config)
    configurable = recorder.configuration
    with recorder.start_span(
        name=span_name,
        kind="llm",
        agent_role="researcher",
        attributes={"provider": provider, "model": str(model)},
        input_payload=input_preview,
        provider=provider,
        model=str(model),
    ) as span:
        metadata = config.get("metadata") or {}
        budget_gate = BudgetGate.from_config(
            configurable,
            str(metadata.get("run_id") or "default"),
            started_at=(
                float(stored["started_at"])
                if recorder.store is not None
                and (stored := recorder._safe(
                    recorder.store.get_run,
                    str(metadata.get("run_id") or "default"),
                ))
                and stored.get("started_at")
                else None
            ),
        )
        attempt = 0
        while True:
            budget_operation_key = f"usage:{span.span_id}:{attempt + 1}:native"
            budget_gate.reserve_model_call(
                budget_operation_key,
                estimated_input_tokens=max(1, _approximate_text_tokens(input_preview)),
                estimated_output_tokens=max(1, configurable.research_model_max_tokens),
                model_name=f"{provider}:{model}",
            )
            try:
                response = await call()
            except Exception as exc:  # noqa: BLE001 - classified before retry
                uncertain = _is_uncertain_failure(exc)
                budget_gate.fail_model_call(
                    budget_operation_key,
                    uncertain=uncertain,
                )
                failed_revision = span.add_usage(
                    TokenUsage(
                        usage_source="missing",
                        response_status=(
                            "unknown_failed" if uncertain else "rejected"
                        ),
                    ),
                    provider,
                    str(model),
                    event_key=f"{span.span_id}:{attempt + 1}:failed",
                    attempt_index=attempt + 1,
                    stage="researching",
                    task_id=str(config.get("metadata", {}).get("task_id") or "") or None,
                    operation=span_name,
                    response_status="unknown_failed" if uncertain else "rejected",
                )
                if failed_revision:
                    with contextlib.suppress(Exception):
                        await event_publisher_from_config(config).publish(
                            "run.usage.updated",
                            payload={"revision": failed_revision, "accounting_status": "partial"},
                            dedupe_key=f"run-usage:{failed_revision}",
                        )
                error_type, retryable = classify_llm_retryable_error(exc)
                attempts_made = attempt + 1
                if (
                    not retryable
                    or attempts_made >= configurable.max_structured_output_retries
                ):
                    span.record_outcome(
                        error_type=error_type.value,
                        http_status=_exception_status(exc),
                    )
                    raise
                delay = min(
                    configurable.tool_retry_max_delay,
                    configurable.tool_retry_base_delay * (2**attempt),
                ) + random.uniform(0, configurable.tool_retry_base_delay)
                span.record_retry(
                    attempt=attempts_made,
                    error_type=error_type.value,
                    http_status=_exception_status(exc),
                    retryable=True,
                    delay_s=delay,
                    message=_safe_exception_message(exc),
                )
                await asyncio.sleep(delay)
                attempt += 1
                continue
            usage = sdk_usage(response)
            if usage.has_reported_tokens:
                usage.usage_source = (
                    "provider_reported"
                    if usage.input_tokens > 0 and usage.output_tokens > 0
                    else "provider_partial"
                )
            elif configurable.token_usage_estimation_enabled:
                input_text = str(input_preview or "")
                output_text = _response_text(response)
                estimated_input = _approximate_text_tokens(input_text)
                estimated_output = _approximate_text_tokens(output_text)
                usage = TokenUsage(
                    estimated_input_tokens=estimated_input,
                    estimated_output_tokens=estimated_output,
                    estimated_total_tokens=estimated_input + estimated_output,
                    usage_source="tokenizer_estimated",
                )
            else:
                usage = TokenUsage(usage_source="missing")
            revision = span.add_usage(
                usage,
                provider,
                str(model),
                event_key=f"{span.span_id}:{attempt + 1}:1",
                attempt_index=attempt + 1,
                stage="researching",
                task_id=str(config.get("metadata", {}).get("task_id") or "") or None,
                operation=span_name,
            )
            budget_gate.settle_model_call(
                budget_operation_key,
                input_tokens=usage.input_tokens or usage.estimated_input_tokens,
                output_tokens=usage.output_tokens or usage.estimated_output_tokens,
                model_name=f"{provider}:{model}",
            )
            if revision:
                with contextlib.suppress(Exception):
                    await event_publisher_from_config(config).publish(
                        "run.usage.updated",
                        payload={
                            "revision": revision,
                            "accounting_status": (
                                "complete" if usage.usage_source == "provider_reported" else "partial"
                            ),
                        },
                        dedupe_key=f"run-usage:{revision}",
                    )
            if configurable.observability_enabled and configurable.trace_payload_mode != "none":
                span.set_output(_response_text(response))
            return response


def build_openai_client(config: RunnableConfig) -> AsyncOpenAI:
    """Build an OpenAI search client from the configured credentials."""
    kwargs: dict[str, Any] = {
        "api_key": get_api_key_for_model("openai:gpt-4.1", config),
        "timeout": 60.0,
    }
    if base_url := os.getenv("OPENAI_BASE_URL"):
        kwargs["base_url"] = base_url
    return AsyncOpenAI(**kwargs)


def build_anthropic_client(config: RunnableConfig) -> AsyncAnthropic:
    """Build an Anthropic search client from the configured credentials."""
    kwargs: dict[str, Any] = {
        "api_key": get_api_key_for_model("anthropic:claude-sonnet-4", config),
        "timeout": 60.0,
    }
    if base_url := os.getenv("ANTHROPIC_BASE_URL"):
        kwargs["base_url"] = base_url
    return AsyncAnthropic(**kwargs)


def parse_openai_search(response: Any) -> tuple[str, list[dict[str, str]]]:
    """Extract synthesized text and URL citations from OpenAI Responses."""
    text = str(getattr(response, "output_text", "") or "")
    sources: list[dict[str, str]] = []
    for item in getattr(response, "output", None) or []:
        if getattr(item, "type", None) != "message":
            continue
        for part in getattr(item, "content", None) or []:
            for annotation in getattr(part, "annotations", None) or []:
                url = getattr(annotation, "url", None)
                if url:
                    sources.append(
                        {
                            "url": str(url),
                            "title": str(getattr(annotation, "title", None) or url),
                        }
                    )
    return text, sources


def parse_anthropic_search(response: Any) -> tuple[str, list[dict[str, str]]]:
    """Extract synthesized text and sources from Anthropic web-search blocks."""
    text_parts: list[str] = []
    sources: list[dict[str, str]] = []
    for block in getattr(response, "content", None) or []:
        block_type = getattr(block, "type", None)
        if block_type == "text":
            text_parts.append(str(getattr(block, "text", "") or ""))
        elif block_type == "web_search_tool_result":
            for result in getattr(block, "content", None) or []:
                url = getattr(result, "url", None)
                if url:
                    sources.append(
                        {
                            "url": str(url),
                            "title": str(getattr(result, "title", None) or url),
                        }
                    )
    return "\n".join(text_parts), sources


def deduplicate_sources(sources: list[dict[str, str]]) -> list[dict[str, str]]:
    """Deduplicate sources by URL while preserving discovery order."""
    seen: set[str] = set()
    unique: list[dict[str, str]] = []
    for source in sources:
        if source["url"] not in seen:
            seen.add(source["url"])
            unique.append(source)
    return unique


async def format_synthesized_search(
    synthesized_text: str,
    sources: list[dict[str, str]],
    config: RunnableConfig,
) -> str:
    """Summarize a provider answer and format its cited source list."""
    if not sources and not synthesized_text.strip():
        return (
            "No valid search results found. Please try different search queries "
            "or use a different search API."
        )
    configurable = Configuration.from_runnable_config(config)
    summary = await summarization.summarize_webpage(
        summarization.build_summarization_model(config),
        synthesized_text[: configurable.max_content_length]
        if synthesized_text
        else "",
        config=config,
        model_name=configurable.summarization_model,
    )
    output = "Search results: \n"
    for index, source in enumerate(sources, 1):
        output += (
            f"\n\n--- SOURCE {index}: {source['title']} ---\n"
            f"URL: {source['url']}\n"
        )
    return output + f"\n\nSUMMARY:\n{summary}\n\n" + ("-" * 80) + "\n"


__all__ = [
    "build_anthropic_client",
    "build_openai_client",
    "deduplicate_sources",
    "format_synthesized_search",
    "parse_anthropic_search",
    "parse_openai_search",
    "sdk_call_with_observability",
    "strip_provider_prefix",
]
