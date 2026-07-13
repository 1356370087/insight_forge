"""Tests for the deterministic Search -> Top-K Fetch -> Evidence pipeline."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

import pymupdf
import pytest

from open_deep_research.quality import deterministic_tool_checks
from open_deep_research.security.network import validate_connected_peer
from open_deep_research.tools import utils
from open_deep_research.web.models import (
    CandidateSource,
    DomainApprovalBatch,
    FetchResult,
    ProviderSynthesis,
    SearchBatch,
    SearchRequest,
)
from open_deep_research.web.pipeline import (
    RawFetch,
    WebPipelineSettings,
    WebResearchPipeline,
    _heuristic_authority,
    canonicalize_url,
    extract_document,
    fetch_local,
    normalize_candidates,
    rank_candidates,
)


def candidate(url: str, rank: int = 1, snippet: str = "research evidence") -> CandidateSource:
    canonical = canonicalize_url(url)
    return CandidateSource(
        candidate_id=hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16],
        provider="test",
        query_ids=["q1"],
        provider_rank=rank,
        original_url=url,
        canonical_url=canonical,
        domain=canonical.split("/", 3)[2],
        title="Research source",
        snippet=snippet,
    )


def test_canonicalize_url_removes_tracking_and_fragment() -> None:
    assert canonicalize_url(
        "HTTPS://Example.COM:443/a?utm_source=x&b=2&a=1#section"
    ) == "https://example.com/a?a=1&b=2"


def test_normalize_candidates_merges_queries_for_same_canonical_url() -> None:
    first = candidate("https://example.com/a?utm_source=x")
    second = candidate("https://example.com/a")
    second.query_ids = ["q2"]

    normalized = normalize_candidates([first, second], 20)

    assert len(normalized) == 1
    assert normalized[0].query_ids == ["q1", "q2"]


@pytest.mark.asyncio
async def test_ranking_enforces_top_k_and_domain_diversity() -> None:
    candidates = [
        candidate(f"https://same.example/{index}", rank=index)
        for index in range(1, 5)
    ] + [candidate("https://other.example/source", rank=5)]

    ranked = await rank_candidates("research evidence", candidates, top_k=3, max_per_domain=2)

    selected = [item for item in ranked if item.selected]
    assert len(selected) == 3
    assert sum(item.candidate.domain == "same.example" for item in selected) == 2


@pytest.mark.asyncio
async def test_ranking_rejects_candidates_below_authority_threshold() -> None:
    candidates = [
        candidate("https://official.example/standard", rank=1),
        candidate("https://blog.example/opinion", rank=2),
    ]

    async def rerank(_objective, items):
        return {
            items[0].candidate_id: (0.9, 0.95, 0.8),
            items[1].candidate_id: (0.9, 0.40, 0.8),
        }

    ranked = await rank_candidates(
        "research evidence",
        candidates,
        reranker=rerank,
        top_k=2,
        min_authority=0.65,
    )

    assert [item.candidate.domain for item in ranked if item.selected] == [
        "official.example"
    ]
    assert next(
        item.reason for item in ranked if item.candidate.domain == "blog.example"
    ) == "below_authority_threshold"


@pytest.mark.asyncio
async def test_authority_fallback_uses_exact_trusted_domain_rules() -> None:
    trusted = candidate("https://www.who.int/publication")
    spoofed = candidate("https://who.int.attacker.example/payload")
    europa = candidate("https://digital-strategy.ec.europa.eu/policy")

    assert _heuristic_authority(trusted) == 0.9
    assert _heuristic_authority(spoofed) == 0.55
    assert _heuristic_authority(europa) == 0.9

    async def failed_reranker(_objective, _items):
        raise TimeoutError("reranker unavailable")

    ranked = await rank_candidates(
        "official policy",
        [trusted, spoofed, europa],
        reranker=failed_reranker,
        top_k=3,
        min_authority=0.65,
    )

    assert all(item.authority_method == "heuristic" for item in ranked)
    assert next(item for item in ranked if item.candidate is spoofed).reason == (
        "below_authority_threshold"
    )


@pytest.mark.asyncio
async def test_allow_search_only_does_not_require_unreachable_sync_approval(
    monkeypatch,
) -> None:
    monkeypatch.setenv("SANDBOX_NETWORK_MODE", "allow-search-only")
    config = {
        "configurable": {"sandbox_network_mode": "allow-search-only"},
        "metadata": {"run_id": "sync-search"},
    }

    batch = await utils._approve_candidate_batch(
        [candidate("https://example.org/research")],
        1,
        config,
    )

    assert batch.pending_domains == []
    assert batch.denied_domains == []


@pytest.mark.asyncio
async def test_allowlist_domain_keeps_explicit_domain_approval(monkeypatch) -> None:
    monkeypatch.setenv("SANDBOX_NETWORK_MODE", "allowlist-domain")
    utils.get_domain_approval_registry().clear_run("approval-required")
    config = {
        "configurable": {
            "sandbox_network_mode": "allowlist-domain",
            "sandbox_allowed_domains": [],
        },
        "metadata": {"run_id": "approval-required"},
    }

    batch = await utils._approve_candidate_batch(
        [candidate("https://example.org/research")],
        1,
        config,
    )

    assert batch.pending_domains == ["example.org"]


def test_html_extraction_removes_navigation_and_preserves_metadata() -> None:
    item = candidate("https://example.com/article")
    html = b"""
    <html lang="en"><head><title>Fallback</title>
    <meta property="og:title" content="Main title">
    <meta name="author" content="A. Writer"></head>
    <body><nav>Navigation noise</nav><article><h1>Main title</h1>
    <p>This is a sufficiently long factual paragraph about research evidence. """ + b"x" * 700 + b"""</p>
    </article><footer>Footer noise</footer></body></html>
    """
    raw = RawFetch(
        result=FetchResult(
            candidate_id=item.candidate_id,
            requested_url=item.canonical_url,
            final_url=item.canonical_url,
            content_type="text/html",
            success=True,
        ),
        body=html,
    )

    document = extract_document(item, raw, WebPipelineSettings())

    assert document.title == "Main title"
    assert document.author == "A. Writer"
    assert "Navigation noise" not in document.markdown
    assert "Footer noise" not in document.markdown
    assert "dynamic_render_recommended" not in document.quality_flags


def test_pdf_extraction_preserves_page_locator() -> None:
    pdf = pymupdf.open()
    page = pdf.new_page()
    page.insert_text((72, 72), "A factual PDF paragraph " * 20)
    body = pdf.tobytes()
    item = candidate("https://example.com/paper.pdf")
    raw = RawFetch(
        result=FetchResult(
            candidate_id=item.candidate_id,
            requested_url=item.canonical_url,
            final_url=item.canonical_url,
            content_type="application/pdf",
            success=True,
        ),
        body=body,
    )

    document = extract_document(item, raw, WebPipelineSettings())

    assert document.page_count == 1
    assert "<!-- page:1 -->" in document.markdown
    assert document.extractor == "pymupdf"


@pytest.mark.asyncio
async def test_pipeline_fetches_only_top_k_and_synthesis_is_not_evidence(monkeypatch) -> None:
    candidates = [candidate(f"https://source{index}.example/article", rank=index) for index in range(1, 8)]
    fetched: list[str] = []

    async def search(_request: SearchRequest) -> SearchBatch:
        return SearchBatch(
            candidates=candidates,
            syntheses=[ProviderSynthesis(provider="test", text="Unverified provider answer")],
        )

    async def fake_fetch(item, settings, *, redirect_allowed=None):
        fetched.append(item.canonical_url)
        body = ("<article><h1>Evidence</h1><p>" + "Relevant factual evidence. " * 50 + "</p></article>").encode()
        return RawFetch(
            result=FetchResult(
                candidate_id=item.candidate_id,
                requested_url=item.canonical_url,
                final_url=item.canonical_url,
                content_type="text/html",
                byte_count=len(body),
                content_hash=hashlib.sha256(body).hexdigest(),
                fetched_at=datetime.now(timezone.utc),
                success=True,
            ),
            body=body,
        )

    async def approve(items, iteration):
        return DomainApprovalBatch(
            run_id="run",
            iteration=iteration,
            domains=[item.domain for item in items],
            urls=[item.canonical_url for item in items],
        )

    monkeypatch.setattr("open_deep_research.web.pipeline.fetch_local", fake_fetch)
    pipeline = WebResearchPipeline(
        search=search,
        settings=WebPipelineSettings(fetch_top_k=3),
        approve=approve,
    )

    result = await pipeline.run(SearchRequest(objective="factual evidence", queries=["q"]))

    assert len(fetched) == 3
    assert result.provider_syntheses[0].evidence_eligible is False
    assert all("Unverified provider answer" not in record.claim for record in result.evidence)
    assert {record.source_url for record in result.evidence} <= {
        document.final_url for document in result.documents
    }


@pytest.mark.asyncio
async def test_enforced_mode_exposes_pipeline_tools_not_provider_search() -> None:
    tools = await utils.get_all_tools(
        {
            "configurable": {
                "web_pipeline_mode": "enforced",
                "search_api": "tavily",
                "browser_mcp_enabled": False,
            },
            "metadata": {"run_id": "test"},
        }
    )
    names = {tool.name for tool in tools}
    assert {"web_research", "fetch_url"} <= names
    assert "tavily_search" not in names
    assert "fetch_webpage" not in names


def test_connected_peer_rejects_private_address() -> None:
    with pytest.raises(ValueError, match="Connected peer"):
        validate_connected_peer(("127.0.0.1", 443))


@pytest.mark.asyncio
async def test_fetch_local_retries_transient_failures(monkeypatch) -> None:
    calls = 0

    async def fake_once(item, settings, *, redirect_allowed=None):
        nonlocal calls
        calls += 1
        return RawFetch(
            result=FetchResult(
                candidate_id=item.candidate_id,
                requested_url=item.canonical_url,
                success=calls == 3,
                failure_class=None if calls == 3 else "http_503",
            )
        )

    async def no_sleep(_delay):
        return None

    monkeypatch.setattr("open_deep_research.web.pipeline._fetch_local_once", fake_once)
    monkeypatch.setattr("open_deep_research.web.pipeline.asyncio.sleep", no_sleep)

    result = await fetch_local(candidate("https://retry.example/a"), WebPipelineSettings())

    assert result.result.success is True
    assert result.result.attempts == 3
    assert calls == 3


def test_quality_counts_only_fetched_structured_evidence() -> None:
    payload = {
        "documents": [{"final_url": "https://read.example/a"}],
        "evidence": [
            {"source_url": "https://read.example/a"},
            {"source_url": "https://candidate-only.example/b"},
        ],
        "provider_syntheses": [
            {"text": "answer", "cited_candidate_ids": ["candidate-only"]}
        ],
    }

    checks = deterministic_tool_checks(
        [{"name": "web_research", "content": json.dumps(payload), "error": False}],
        min_sources=1,
    )

    assert checks["source_count"] == 1
    assert checks["structured_evidence_count"] == 1
