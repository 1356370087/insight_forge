"""Definition and implementation of the enforced web research tool."""

from __future__ import annotations

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from open_deep_research.configuration import Configuration
from open_deep_research.tools.adapters import adapt_langchain_tool
from open_deep_research.tools.availability import enforced_pipeline_enabled
from open_deep_research.tools.base import ToolOrigin
from open_deep_research.tools.web_research import pipeline
from open_deep_research.tools.web_research.prompt import DESCRIPTION, render_prompt
from open_deep_research.web.models import SearchRequest
from open_deep_research.web.pipeline import WebResearchPipeline


@tool("web_research", description=DESCRIPTION)
async def _web_research_call(
    objective: str,
    queries: list[str],
    iteration: int = 1,
    config: RunnableConfig = None,
) -> str:
    """Run bounded Search -> Top-K Fetch -> Evidence stages."""
    configurable = Configuration.from_runnable_config(config)
    settings = pipeline._web_pipeline_settings(configurable)
    settings.cache_namespace = str(
        config.get("metadata", {}).get("run_id", "default")
    )
    request = SearchRequest(
        objective=objective,
        queries=queries[:3],
        candidate_limit=configurable.search_candidate_limit,
        iteration=iteration,
    )
    runner = WebResearchPipeline(
        search=lambda req: pipeline._discover_web_candidates(req, config),
        settings=settings,
        reranker=lambda obj, items: pipeline._rerank_web_candidates(
            obj, items, config
        ),
        approve=lambda items, index: pipeline._approve_candidate_batch(
            items, index, config
        ),
        render_dynamic=(
            (lambda url: pipeline._render_with_browser_mcp(url, config))
            if "playwright" in configurable.fetch_backend_order
            else None
        ),
        external_extractors=pipeline._configured_external_extractors(
            configurable, config
        ),
        evidence_extractor=lambda obj, docs, chunks: pipeline._extract_web_evidence(
            obj, docs, chunks, config
        ),
    )
    reserved, release_budget = await pipeline._reserve_fetch_budget(
        config, configurable.fetch_top_k
    )
    result = await runner.run(request, remaining_fetches=reserved)
    consumed = sum(fetch.adapter != "run_cache" for fetch in result.fetches)
    await release_budget(reserved - consumed)
    pipeline._record_web_pipeline_metrics(result, config)
    return pipeline._compact_web_result(result, config)


web_research = adapt_langchain_tool(
    _web_research_call,
    origin=ToolOrigin.SEARCH,
    retryable=True,
    concurrency_safe=True,
    prompt=render_prompt,
    is_enabled=enforced_pipeline_enabled,
)
