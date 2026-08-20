"""Quarantining webpage summarization shared by search and fetch tools."""

from __future__ import annotations

import asyncio
import logging

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

from open_deep_research.configuration import Configuration
from open_deep_research.models.fallback import invoke_with_model_fallback
from open_deep_research.models.resolution import build_model_config
from open_deep_research.observability import invoke_model_with_retry_observability
from open_deep_research.prompts import summarize_webpage_prompt
from open_deep_research.state import Summary
from open_deep_research.tools.legacy_shims import get_today_str


def build_summarization_model(config: RunnableConfig) -> BaseChatModel:
    """Build the structured-output summarization model used by search tools."""
    configurable = Configuration.from_runnable_config(config)
    return init_chat_model(
        **build_model_config(
            configurable.summarization_model,
            configurable.summarization_model_max_tokens,
            config,
            role="summarization",
        )
    ).with_structured_output(Summary, method="function_calling")


async def summarize_webpage(
    model: BaseChatModel,
    webpage_content: str,
    *,
    config: RunnableConfig = None,
    model_name: str | None = None,
) -> str:
    """Summarize untrusted webpage content without ever failing open."""
    try:
        prompt_content = summarize_webpage_prompt.format(
            webpage_content=webpage_content,
            date=get_today_str(),
        )
        messages: list[BaseMessage] = [HumanMessage(content=prompt_content)]
        if model_name:
            configurable = Configuration.from_runnable_config(config)

            async def invoke_candidate(
                candidate_model: str,
                request_messages: list[BaseMessage],
            ):
                candidate = model
                if candidate_model != model_name:
                    candidate = init_chat_model(
                        **build_model_config(
                            candidate_model,
                            configurable.summarization_model_max_tokens,
                            config,
                            role="summarization",
                        )
                    ).with_structured_output(Summary, method="function_calling")
                return await invoke_model_with_retry_observability(
                    candidate,
                    request_messages,
                    config,
                    span_name="tool.tavily.summarize_webpage",
                    agent_role="researcher",
                    model_name=candidate_model,
                    stage="researching",
                )

            invocation = invoke_with_model_fallback(
                invoke_candidate,
                messages,
                primary_model=model_name,
                model_fallbacks=configurable.model_fallbacks,
                role="summarization",
                config=config,
            )
        else:
            invocation = invoke_model_with_retry_observability(
                model,
                messages,
                config,
                span_name="tool.tavily.summarize_webpage",
                agent_role="researcher",
                model_name=model_name,
                stage="researching",
            )
        summary = await asyncio.wait_for(invocation, timeout=120.0)
        return (
            f"<summary>\n{summary.summary}\n</summary>\n\n"
            f"<key_excerpts>\n{summary.key_excerpts}\n</key_excerpts>"
        )
    except TimeoutError:
        logging.warning(
            "Summarization timed out after 120 seconds; content quarantined"
        )
        return '<external_content_quarantined reason="summarization_timeout"/>'
    except Exception as exc:  # noqa: BLE001 - external content remains quarantined
        logging.warning(
            "Summarization failed; external content quarantined: %s",
            str(exc)[:200],
        )
        return '<external_content_quarantined reason="summarization_failed"/>'


__all__ = ["build_summarization_model", "summarize_webpage"]
