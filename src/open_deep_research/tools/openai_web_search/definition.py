"""Definition and implementation of the OpenAI native web search tool."""

from __future__ import annotations

import asyncio
from typing import Annotated, Any

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import InjectedToolArg, tool

from open_deep_research.configuration import Configuration, SearchAPI
from open_deep_research.tools.adapters import adapt_langchain_tool
from open_deep_research.tools.availability import provider_search_enabled
from open_deep_research.tools.base import ToolOrigin
from open_deep_research.tools.openai_web_search.prompt import (
    DESCRIPTION,
    render_prompt,
)
from open_deep_research.tools.web_research import pipeline, providers


@tool("openai_web_search", description=DESCRIPTION)
async def _openai_web_search_call(
    queries: list[str],
    max_results: Annotated[int, InjectedToolArg] = 5,
    config: RunnableConfig = None,
) -> str:
    """Run OpenAI server-side web search and summarize its cited digest."""
    configurable = Configuration.from_runnable_config(config)
    client = providers.build_openai_client(config)
    model = providers.strip_provider_prefix(configurable.research_model, "openai")

    async def run_one(query: str) -> Any:
        async def call():
            return await client.responses.create(
                model=model,
                input=query,
                tools=[{"type": "web_search_preview"}],
            )

        return await providers.sdk_call_with_observability(
            call,
            span_name="tool.openai.web_search",
            provider="openai",
            model=model,
            config=config,
            input_preview=query,
        )

    responses = await asyncio.gather(*[run_one(query) for query in queries])
    text_parts: list[str] = []
    all_sources: list[dict[str, str]] = []
    for response in responses:
        text, sources = providers.parse_openai_search(response)
        if text:
            text_parts.append(text)
        all_sources.extend(sources)
    synthesized = "\n\n".join(text_parts)
    capped_sources = providers.deduplicate_sources(all_sources)[
        : max_results * max(1, len(queries))
    ]
    await pipeline._record_shadow_candidates(
        [
            candidate
            for rank, source in enumerate(capped_sources, 1)
            if (
                candidate := pipeline._candidate(
                    "openai",
                    source["url"],
                    source["title"],
                    "",
                    rank,
                    "shadow",
                )
            )
        ],
        config,
    )
    return await providers.format_synthesized_search(
        synthesized, capped_sources, config
    )


openai_web_search = adapt_langchain_tool(
    _openai_web_search_call,
    origin=ToolOrigin.SEARCH,
    retryable=True,
    concurrency_safe=True,
    prompt=render_prompt,
    is_enabled=lambda config: provider_search_enabled(config, SearchAPI.OPENAI),
)
