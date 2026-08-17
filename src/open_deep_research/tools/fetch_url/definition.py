"""Definition and implementation of the governed exact-URL fetch tool."""

from __future__ import annotations

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import ToolException, tool

from open_deep_research.configuration import Configuration
from open_deep_research.tools.adapters import adapt_langchain_tool
from open_deep_research.tools.availability import enforced_pipeline_enabled
from open_deep_research.tools.base import ToolOrigin
from open_deep_research.tools.fetch_url.prompt import DESCRIPTION, render_prompt
from open_deep_research.tools.web_research import pipeline
from open_deep_research.web.models import SearchBatch, SearchRequest
from open_deep_research.web.pipeline import WebResearchPipeline


def _egress_urls(args: dict) -> list[str]:
    url = args.get("url")
    return [url] if isinstance(url, str) else []


@tool("fetch_url", description=DESCRIPTION)
async def _fetch_url_call(
    url: str,
    objective: str,
    config: RunnableConfig = None,
) -> str:
    """Fetch one known URL without running discovery or unrelated reranking."""
    configurable = Configuration.from_runnable_config(config)
    candidate = pipeline._candidate("direct", url, "", "", 1, "direct-url")
    if candidate is None:
        raise ToolException("Invalid public HTTP(S) URL")

    async def direct_search(_request: SearchRequest) -> SearchBatch:
        return SearchBatch(candidates=[candidate])

    request = SearchRequest(
        objective=objective,
        queries=[url],
        candidate_limit=1,
    )
    settings = pipeline._web_pipeline_settings(configurable)
    settings.fetch_top_k = 1
    settings.min_source_authority = 0.0
    settings.cache_namespace = str(
        config.get("metadata", {}).get("run_id", "default")
    )
    runner = WebResearchPipeline(
        search=direct_search,
        settings=settings,
        approve=lambda items, index: pipeline._approve_candidate_batch(
            items, index, config
        ),
        render_dynamic=(
            (lambda target: pipeline._render_with_browser_mcp(target, config))
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
    reserved, release_budget = await pipeline._reserve_fetch_budget(config, 1)
    result = await runner.run(request, remaining_fetches=reserved)
    consumed = sum(fetch.adapter != "run_cache" for fetch in result.fetches)
    await release_budget(reserved - consumed)
    pipeline._record_web_pipeline_metrics(result, config)
    return pipeline._compact_web_result(result)


fetch_url = adapt_langchain_tool(
    _fetch_url_call,
    origin=ToolOrigin.SEARCH,
    retryable=True,
    concurrency_safe=True,
    prompt=render_prompt,
    is_enabled=enforced_pipeline_enabled,
    egress_urls=_egress_urls,
)
