"""Definition and implementation of the Tavily search tool."""

from __future__ import annotations

import asyncio
from typing import Annotated, Literal

from langchain.chat_models import init_chat_model
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import InjectedToolArg, tool

from open_deep_research.configuration import Configuration, SearchAPI
from open_deep_research.model_resolution import build_model_config
from open_deep_research.state import Summary
from open_deep_research.tools.adapters import adapt_langchain_tool
from open_deep_research.tools.availability import provider_search_enabled
from open_deep_research.tools.base import ToolOrigin
from open_deep_research.tools.tavily_search import summarization
from open_deep_research.tools.tavily_search.client import tavily_search_async
from open_deep_research.tools.tavily_search.prompt import DESCRIPTION, render_prompt
from open_deep_research.tools.web_research import pipeline


@tool("tavily_search", description=DESCRIPTION)
async def _tavily_search_call(
    queries: list[str],
    config: RunnableConfig,
    max_results: Annotated[int, InjectedToolArg] = 5,
    topic: Annotated[
        Literal["general", "news", "finance"], InjectedToolArg
    ] = "general",
) -> str:
    """Fetch and summarize search results from Tavily search API.

    Args:
        queries: Short, search-engine-ready queries. Begin broad on the first call; on
            later calls, narrow each query only as needed to address evidence-backed gaps.
        max_results: Maximum number of results to return per query.
        topic: Topic filter for search results (general, news, or finance).
        config: Runtime configuration for API keys and model settings.
    """
    responses = await tavily_search_async(
        queries,
        max_results=max_results,
        topic=topic,
        include_raw_content=True,
        config=config,
    )
    unique_results: dict[str, dict] = {}
    for response in responses:
        for result in response["results"]:
            unique_results.setdefault(
                result["url"],
                {**result, "query": response["query"]},
            )

    configurable = Configuration.from_runnable_config(config)
    model = init_chat_model(
        **build_model_config(
            configurable.summarization_model,
            configurable.summarization_model_max_tokens,
            config,
            role="summarization",
        )
    ).with_structured_output(Summary, method="function_calling")

    async def no_summary() -> None:
        return None

    summaries = await asyncio.gather(
        *[
            no_summary()
            if not result.get("raw_content")
            else summarization.summarize_webpage(
                model,
                result["raw_content"][: configurable.max_content_length],
                config=config,
                model_name=configurable.summarization_model,
            )
            for result in unique_results.values()
        ]
    )
    summarized_results = {
        url: {
            "title": result["title"],
            "content": result.get("content", "") if summary is None else summary,
        }
        for (url, result), summary in zip(unique_results.items(), summaries)
    }
    shadow_candidates = [
        candidate
        for rank, (url, result) in enumerate(unique_results.items(), 1)
        if (
            candidate := pipeline._candidate(
                "tavily",
                url,
                result.get("title", ""),
                result.get("content", ""),
                rank,
                result.get("query", ""),
            )
        )
    ]
    await pipeline._record_shadow_candidates(shadow_candidates, config)
    if not summarized_results:
        return (
            "No valid search results found. Please try different search queries "
            "or use a different search API."
        )
    output = "Search results: \n\n"
    for index, (url, result) in enumerate(summarized_results.items(), 1):
        output += f"\n\n--- SOURCE {index}: {result['title']} ---\n"
        output += f"URL: {url}\n\nSUMMARY:\n{result['content']}\n\n"
        output += "\n\n" + "-" * 80 + "\n"
    return output


tavily_search = adapt_langchain_tool(
    _tavily_search_call,
    origin=ToolOrigin.SEARCH,
    retryable=True,
    concurrency_safe=True,
    prompt=render_prompt,
    is_enabled=lambda config: provider_search_enabled(config, SearchAPI.TAVILY),
)
