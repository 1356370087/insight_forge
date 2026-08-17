"""Tavily API client helpers."""

from __future__ import annotations

import asyncio
import os
from typing import Literal

from langchain_core.runnables import RunnableConfig
from tavily import AsyncTavilyClient  # type: ignore[import-untyped]


def get_tavily_api_key(config: RunnableConfig) -> str | None:
    """Resolve the Tavily key from runnable config or the environment."""
    if os.getenv("GET_API_KEYS_FROM_CONFIG", "false").lower() == "true":
        api_keys = config.get("configurable", {}).get("apiKeys", {})
        return api_keys.get("TAVILY_API_KEY") if api_keys else None
    return os.getenv("TAVILY_API_KEY")


async def tavily_search_async(
    search_queries: list[str],
    max_results: int = 5,
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = True,
    config: RunnableConfig = None,
) -> list[dict]:
    """Execute multiple Tavily queries concurrently."""
    client = AsyncTavilyClient(api_key=get_tavily_api_key(config))
    return await asyncio.gather(
        *[
            client.search(
                query,
                max_results=max_results,
                include_raw_content=include_raw_content,
                topic=topic,
            )
            for query in search_queries
        ]
    )


__all__ = ["get_tavily_api_key", "tavily_search_async"]
