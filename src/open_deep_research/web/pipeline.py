"""Deterministic Search -> Top-K Fetch -> Evidence pipeline."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import urllib.robotparser
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Awaitable, Callable, Protocol
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit

import aiohttp
import pymupdf  # type: ignore[import-untyped]
from bs4 import BeautifulSoup
from markdownify import markdownify  # type: ignore[import-untyped]

from open_deep_research.security.content import inspect_untrusted_content
from open_deep_research.security.network import (
    validate_public_http_url,
    validate_response_peer,
)
from open_deep_research.web.models import (
    BudgetSnapshot,
    CandidateSource,
    DocumentChunk,
    DomainApprovalBatch,
    EvidenceRecord,
    ExtractedDocument,
    FetchResult,
    GapAnalysis,
    RankedCandidate,
    SearchBatch,
    SearchRequest,
    WebResearchResult,
)

TRACKING_PARAMS = {"gclid", "fbclid", "dclid", "msclkid", "mc_cid", "mc_eid"}
WORD_RE = re.compile(r"[\w\u3400-\u9fff]{2,}", re.UNICODE)
SENTENCE_RE = re.compile(r"(?<=[。！？.!?])\s+|\n+")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
JS_SHELL_MARKERS = (
    "enable javascript",
    "javascript is required",
    "please turn on javascript",
    "__next_data__",
    "id=\"root\"",
    "id=\"app\"",
)
_DOCUMENT_CACHE: dict[tuple[str, str], ExtractedDocument] = {}
_FETCH_LOCKS: dict[tuple[str, str], asyncio.Lock] = {}
_ROBOTS_CACHE: dict[tuple[str, str], bool] = {}


def clear_run_web_cache(run_id: str) -> None:
    """Remove run-scoped extracted documents, locks, and robots decisions."""
    for mapping in (_DOCUMENT_CACHE, _FETCH_LOCKS, _ROBOTS_CACHE):
        for key in [key for key in mapping if key[0] == run_id]:
            mapping.pop(key, None)


class SearchAdapter(Protocol):
    """Provider-neutral candidate discovery contract."""

    async def __call__(self, request: SearchRequest) -> SearchBatch:
        """Return normalized candidates and optional non-evidence synthesis."""
        ...


class RerankAdapter(Protocol):
    """Optional semantic scorer used after deterministic filtering."""

    async def __call__(
        self, objective: str, candidates: list[CandidateSource]
    ) -> dict[str, tuple[float, float, float]]:
        """Return candidate_id -> relevance, authority, information gain."""
        ...


ApprovalAdapter = Callable[[list[CandidateSource], int], Awaitable[DomainApprovalBatch]]
DynamicRenderAdapter = Callable[[str], Awaitable[str | None]]
ExternalExtractAdapter = Callable[[str], Awaitable[ExtractedDocument | None]]
EvidenceAdapter = Callable[
    [str, dict[str, ExtractedDocument], list[DocumentChunk]],
    Awaitable[list[EvidenceRecord]],
]


@dataclass(slots=True)
class WebPipelineSettings:
    """Bounded settings passed from the runtime configuration."""

    fetch_top_k: int = 5
    min_source_authority: float = 0.0
    max_fetches: int = 12
    global_concurrency: int = 4
    per_host_concurrency: int = 2
    timeout_seconds: float = 30.0
    max_redirects: int = 5
    html_max_bytes: int = 2 * 1024 * 1024
    pdf_max_bytes: int = 20 * 1024 * 1024
    pdf_max_pages: int = 100
    max_chunks_per_document: int = 3
    max_chunks_per_iteration: int = 20
    chunk_chars: int = 4000
    chunk_overlap_chars: int = 600
    respect_robots_txt: bool = True
    user_agent: str = "OpenDeepResearchBot/0.0.16"
    cache_namespace: str = "default"


@dataclass(slots=True)
class RawFetch:
    """Private in-memory response body; never emitted to model-facing state."""

    result: FetchResult
    body: bytes = b""


def canonicalize_url(url: str) -> str:
    """Normalize a public HTTP URL without changing content-bearing parameters."""
    parsed = urlsplit(str(url).strip())
    scheme = parsed.scheme.lower()
    host = (parsed.hostname or "").rstrip(".").lower()
    if scheme not in {"http", "https"} or not host:
        raise ValueError("Candidate URL must be absolute HTTP(S)")
    port = parsed.port
    netloc = host
    if port and not ((scheme == "http" and port == 80) or (scheme == "https" and port == 443)):
        netloc = f"{host}:{port}"
    params = [
        (key, value)
        for key, value in parse_qsl(parsed.query, keep_blank_values=True)
        if not key.lower().startswith("utm_") and key.lower() not in TRACKING_PARAMS
    ]
    query = urlencode(sorted(params), doseq=True)
    path = parsed.path or "/"
    return urlunsplit((scheme, netloc, path, query, ""))


def stable_id(prefix: str, value: str) -> str:
    """Return a compact stable identifier for a normalized value."""
    return f"{prefix}_{hashlib.sha256(value.encode('utf-8')).hexdigest()[:20]}"


def normalize_candidates(candidates: list[CandidateSource], limit: int) -> list[CandidateSource]:
    """Canonicalize and merge candidates discovered by multiple queries."""
    merged: dict[str, CandidateSource] = {}
    for candidate in candidates:
        try:
            canonical = canonicalize_url(candidate.original_url or candidate.canonical_url)
        except (TypeError, ValueError):
            continue
        domain = urlsplit(canonical).hostname or ""
        existing = merged.get(canonical)
        if existing is not None:
            existing.query_ids = list(dict.fromkeys(existing.query_ids + candidate.query_ids))
            if not existing.snippet and candidate.snippet:
                existing.snippet = candidate.snippet
            existing.provider_rank = min(existing.provider_rank, candidate.provider_rank)
            continue
        data = candidate.model_copy(
            update={
                "candidate_id": stable_id("src", canonical),
                "canonical_url": canonical,
                "domain": domain,
            }
        )
        merged[canonical] = data
    return list(merged.values())[:limit]


def _terms(text: str) -> set[str]:
    return {match.group(0).lower() for match in WORD_RE.finditer(text or "")}


def _heuristic_authority(candidate: CandidateSource) -> float:
    domain = candidate.domain.lower()
    if domain.endswith((".gov", ".edu", ".ac.uk")):
        return 1.0
    trusted_hosts = ("doi.org", "arxiv.org", "who.int", "un.org", "europa.eu")
    if any(domain == host or domain.endswith(f".{host}") for host in trusted_hosts):
        return 0.9
    return 0.55


async def rank_candidates(
    objective: str,
    candidates: list[CandidateSource],
    *,
    reranker: RerankAdapter | None = None,
    top_k: int = 5,
    max_per_domain: int = 2,
    min_authority: float = 0.0,
) -> list[RankedCandidate]:
    """Apply stable rules, authority admission, and domain diversity."""
    semantic: dict[str, tuple[float, float, float]] = {}
    if reranker and candidates:
        try:
            semantic = await reranker(objective, candidates)
        except Exception:  # noqa: BLE001 - deterministic fallback is intentional
            semantic = {}
    objective_terms = _terms(objective)
    ranked: list[RankedCandidate] = []
    total = max(1, len(candidates))
    for position, candidate in enumerate(candidates):
        overlap = len(objective_terms & _terms(f"{candidate.title} {candidate.snippet}"))
        lexical = min(1.0, overlap / max(3, len(objective_terms) * 0.25))
        semantic_scores = semantic.get(candidate.candidate_id)
        relevance, authority, information_gain = semantic_scores or (
            lexical,
            _heuristic_authority(candidate),
            min(1.0, len(candidate.snippet) / 500),
        )
        rank_score = max(0.0, 1.0 - (position / total))
        freshness = 0.5
        score = (
            0.45 * relevance
            + 0.20 * authority
            + 0.15 * information_gain
            + 0.10 * freshness
            + 0.10 * rank_score
        )
        ranked.append(
            RankedCandidate(
                candidate=candidate,
                relevance=relevance,
                authority=authority,
                authority_method=("reranker" if semantic_scores is not None else "heuristic"),
                information_gain=information_gain,
                freshness=freshness,
                provider_rank_score=rank_score,
                final_score=min(1.0, max(0.0, score)),
            )
        )
    ranked.sort(key=lambda item: (-item.final_score, item.candidate.provider_rank, item.candidate.canonical_url))
    domain_counts: Counter[str] = Counter()
    selected = 0
    for item in ranked:
        domain = item.candidate.domain
        if item.authority < min_authority:
            item.reason = "below_authority_threshold"
        elif selected >= top_k:
            item.reason = "below_top_k"
        elif domain_counts[domain] >= max_per_domain:
            item.reason = "domain_diversity_limit"
        else:
            item.selected = True
            item.reason = "selected_top_k"
            domain_counts[domain] += 1
            selected += 1
    return ranked


def _decode_body(body: bytes, charset: str | None) -> str:
    for encoding in (charset, "utf-8", "gb18030", "latin-1"):
        if not encoding:
            continue
        try:
            return body.decode(encoding)
        except (LookupError, UnicodeDecodeError):
            continue
    return body.decode("utf-8", errors="replace")


async def _robots_allowed(
    session: aiohttp.ClientSession,
    url: str,
    settings: WebPipelineSettings,
) -> bool:
    """Check and cache robots.txt for a target host without persisting content."""
    if not settings.respect_robots_txt:
        return True
    parsed = urlsplit(url)
    # robots rules are path-sensitive; caching one boolean for an entire host
    # can allow a disallowed path after an allowed path was visited.
    key = (settings.cache_namespace, canonicalize_url(url))
    if key in _ROBOTS_CACHE:
        return _ROBOTS_CACHE[key]
    robots_url = urlunsplit((parsed.scheme, parsed.netloc, "/robots.txt", "", ""))
    try:
        await validate_public_http_url(robots_url)
        async with session.get(robots_url, allow_redirects=False) as response:
            validate_response_peer(response)
            if response.status >= 400:
                allowed = True
            else:
                body = await response.content.read(256 * 1024 + 1)
                if len(body) > 256 * 1024:
                    allowed = False
                else:
                    parser = urllib.robotparser.RobotFileParser()
                    parser.set_url(robots_url)
                    parser.parse(_decode_body(body, response.charset).splitlines())
                    allowed = parser.can_fetch(settings.user_agent, url)
    except (aiohttp.ClientError, asyncio.TimeoutError, ValueError):
        # A missing/unavailable robots file is not interpreted as a disallow.
        allowed = True
    _ROBOTS_CACHE[key] = allowed
    return allowed


async def _fetch_local_once(
    candidate: CandidateSource,
    settings: WebPipelineSettings,
    *,
    redirect_allowed: Callable[[str], Awaitable[bool]] | None = None,
) -> RawFetch:
    """Fetch one URL with bounded redirects, decompressed-size limits, and SSRF checks."""
    requested = candidate.canonical_url
    current = requested
    redirects: list[str] = []
    timeout = aiohttp.ClientTimeout(total=settings.timeout_seconds)
    headers = {"User-Agent": settings.user_agent, "Accept": "text/html,application/pdf,text/plain;q=0.9,*/*;q=0.1"}
    result = FetchResult(candidate_id=candidate.candidate_id, requested_url=requested)
    try:
        async with aiohttp.ClientSession(timeout=timeout, headers=headers, auto_decompress=True) as session:
            for redirect_index in range(settings.max_redirects + 1):
                if not await _robots_allowed(session, current, settings):
                    raise PermissionError("robots_disallowed")
                await validate_public_http_url(current)
                async with session.get(current, allow_redirects=False) as response:
                    validate_response_peer(response)
                    if response.status in {301, 302, 303, 307, 308}:
                        location = response.headers.get("Location")
                        if not location or redirect_index >= settings.max_redirects:
                            raise RuntimeError("redirect_limit")
                        target = canonicalize_url(urljoin(current, location))
                        if urlsplit(target).hostname != urlsplit(current).hostname:
                            if redirect_allowed is None or not await redirect_allowed(target):
                                raise PermissionError("cross_domain_redirect_not_approved")
                        redirects.append(target)
                        current = target
                        continue
                    if response.status >= 400:
                        raise RuntimeError(f"http_{response.status}")
                    content_type = response.headers.get("Content-Type", "").lower()
                    max_bytes = settings.pdf_max_bytes if "application/pdf" in content_type else settings.html_max_bytes
                    chunks: list[bytes] = []
                    size = 0
                    async for chunk in response.content.iter_chunked(16_384):
                        size += len(chunk)
                        if size > max_bytes:
                            raise RuntimeError("response_too_large")
                        chunks.append(chunk)
                    body = b"".join(chunks)
                    digest = hashlib.sha256(body).hexdigest()
                    result = FetchResult(
                        candidate_id=candidate.candidate_id,
                        requested_url=requested,
                        final_url=current,
                        redirect_chain=redirects,
                        status_code=response.status,
                        content_type=content_type.split(";", 1)[0],
                        byte_count=len(body),
                        content_hash=digest,
                        fetched_at=datetime.now(timezone.utc),
                        adapter="local",
                        success=True,
                    )
                    return RawFetch(result=result, body=body)
        raise RuntimeError("redirect_limit")
    except PermissionError as exc:
        result.failure_class = (
            "robots_disallowed" if "robots_disallowed" in str(exc) else "approval_required"
        )
        result.failure_message = str(exc)
    except asyncio.TimeoutError as exc:
        result.failure_class = "timeout"
        result.failure_message = str(exc)
    except (aiohttp.ClientError, RuntimeError, ValueError) as exc:
        result.failure_class = str(exc) if isinstance(exc, RuntimeError) else "network_error"
        result.failure_message = str(exc)[:500]
    return RawFetch(result=result)


async def fetch_local(
    candidate: CandidateSource,
    settings: WebPipelineSettings,
    *,
    redirect_allowed: Callable[[str], Awaitable[bool]] | None = None,
) -> RawFetch:
    """Fetch with two bounded retries for transient network/429/5xx failures."""
    last: RawFetch | None = None
    for attempt in range(1, 4):
        last = await _fetch_local_once(
            candidate, settings, redirect_allowed=redirect_allowed
        )
        last.result.attempts = attempt
        if last.result.success:
            return last
        failure = last.result.failure_class or ""
        retryable = failure in {"timeout", "network_error"} or failure == "http_429"
        if failure.startswith("http_5"):
            retryable = True
        if not retryable or attempt >= 3:
            return last
        await asyncio.sleep(min(2.0, 0.25 * (2 ** (attempt - 1))))
    return last or RawFetch(
        result=FetchResult(
            candidate_id=candidate.candidate_id,
            requested_url=candidate.canonical_url,
            failure_class="network_error",
            failure_message="fetch did not run",
        )
    )


def _metadata_from_html(soup: BeautifulSoup) -> dict[str, str | None]:
    def meta(*names: str) -> str | None:
        for name in names:
            tag = soup.find("meta", attrs={"property": name}) or soup.find("meta", attrs={"name": name})
            if tag and tag.get("content"):
                return str(tag.get("content")).strip()
        return None

    author = meta("author", "article:author")
    published = meta("article:published_time", "datePublished", "date")
    for tag in soup.find_all("script", attrs={"type": "application/ld+json"}):
        try:
            payload = json.loads(tag.string or "null")
        except (TypeError, json.JSONDecodeError):
            continue
        nodes = payload if isinstance(payload, list) else [payload]
        for node in nodes:
            if not isinstance(node, dict):
                continue
            author_obj = node.get("author")
            if not author and isinstance(author_obj, dict):
                author = str(author_obj.get("name") or "") or None
            published = published or node.get("datePublished")
    html = soup.find("html")
    return {
        "title": meta("og:title", "twitter:title") or (soup.title.string.strip() if soup.title and soup.title.string else ""),
        "author": author,
        "published_at": str(published) if published else None,
        "language": str(html.get("lang")) if html and html.get("lang") else None,
    }


def extract_html(candidate: CandidateSource, raw: RawFetch) -> ExtractedDocument:
    """Extract main HTML content, metadata, and Markdown."""
    content_type = raw.result.content_type or "text/html"
    text = _decode_body(raw.body, None)
    soup = BeautifulSoup(text, "html.parser")
    metadata = _metadata_from_html(soup)
    for selector in ("script", "style", "noscript", "nav", "footer", "aside", "form", "iframe", "object", "embed"):
        for node in soup.select(selector):
            node.decompose()
    for node in soup.select("[class*='advert'],[class*='recommend'],[id*='advert'],[id*='recommend']"):
        node.decompose()
    main = soup.find("article") or soup.find("main") or soup.find(attrs={"role": "main"}) or soup.body or soup
    markdown = markdownify(str(main), heading_style="ATX").strip()
    visible_chars = len(re.sub(r"\s+", "", markdown))
    flags: list[str] = []
    lowered = text.lower()
    if visible_chars < 600 or any(marker in lowered for marker in JS_SHELL_MARKERS) and visible_chars < 1500:
        flags.append("dynamic_render_recommended")
    canonical_tag = soup.find("link", attrs={"rel": lambda value: value and "canonical" in value})
    canonical = candidate.canonical_url
    if canonical_tag and canonical_tag.get("href"):
        try:
            canonical = canonicalize_url(urljoin(raw.result.final_url or canonical, str(canonical_tag.get("href"))))
        except ValueError:
            pass
    digest = hashlib.sha256(markdown.encode("utf-8")).hexdigest()
    return ExtractedDocument(
        document_id=stable_id("doc", f"{canonical}:{digest}"),
        candidate_id=candidate.candidate_id,
        requested_url=raw.result.requested_url,
        final_url=raw.result.final_url or candidate.canonical_url,
        canonical_url=canonical,
        title=str(metadata["title"] or candidate.title),
        author=metadata["author"],
        published_at=metadata["published_at"],
        language=metadata["language"],
        content_type=content_type,
        markdown=markdown,
        extractor="beautifulsoup+markdownify",
        content_hash=digest,
        quality_flags=flags,
    )


def extract_pdf(candidate: CandidateSource, raw: RawFetch, max_pages: int) -> ExtractedDocument:
    """Extract text and page locators from a non-scanned PDF."""
    flags: list[str] = []
    try:
        pdf = pymupdf.open(stream=raw.body, filetype="pdf")
        if pdf.needs_pass:
            raise ValueError("encrypted_pdf")
        page_count = min(len(pdf), max_pages)
        parts: list[str] = []
        for index in range(page_count):
            page_text = pdf[index].get_text("text").strip()
            parts.append(f"<!-- page:{index + 1} -->\n\n{page_text}")
        markdown = "\n\n".join(parts).strip()
        metadata = pdf.metadata or {}
    except Exception as exc:  # noqa: BLE001 - normalized as extraction quality
        raise ValueError(f"pdf_extract_failed:{exc}") from exc
    if page_count and len(re.sub(r"\s+", "", markdown)) / page_count < 100:
        flags.append("needs_external_extraction")
    digest = hashlib.sha256(markdown.encode("utf-8")).hexdigest()
    return ExtractedDocument(
        document_id=stable_id("doc", f"{candidate.canonical_url}:{digest}"),
        candidate_id=candidate.candidate_id,
        requested_url=raw.result.requested_url,
        final_url=raw.result.final_url or candidate.canonical_url,
        canonical_url=candidate.canonical_url,
        title=str(metadata.get("title") or candidate.title),
        author=str(metadata.get("author") or "") or None,
        published_at=str(metadata.get("creationDate") or "") or None,
        content_type="application/pdf",
        markdown=markdown,
        page_count=page_count,
        extractor="pymupdf",
        content_hash=digest,
        quality_flags=flags,
    )


def extract_document(candidate: CandidateSource, raw: RawFetch, settings: WebPipelineSettings) -> ExtractedDocument:
    """Route a successful response to the appropriate local extractor."""
    content_type = raw.result.content_type or ""
    if content_type == "application/pdf" or raw.body.startswith(b"%PDF"):
        return extract_pdf(candidate, raw, settings.pdf_max_pages)
    if content_type.startswith("text/") or content_type in {"application/xhtml+xml", ""}:
        return extract_html(candidate, raw)
    raise ValueError(f"unsupported_content_type:{content_type}")


def chunk_document(document: ExtractedDocument, settings: WebPipelineSettings) -> list[DocumentChunk]:
    """Split Markdown into overlapping, stable chunks with page/heading locators."""
    text = document.markdown
    chunks: list[DocumentChunk] = []
    start = 0
    heading: str | None = None
    while start < len(text):
        end = min(len(text), start + settings.chunk_chars)
        if end < len(text):
            boundary = max(text.rfind("\n\n", start, end), text.rfind("。", start, end), text.rfind(". ", start, end))
            if boundary > start + settings.chunk_chars // 2:
                end = boundary + 1
        segment = text[start:end].strip()
        heading_matches = list(HEADING_RE.finditer(text, 0, start + 1))
        if heading_matches:
            heading = heading_matches[-1].group(2).strip()
        page_matches = list(re.finditer(r"<!-- page:(\d+) -->", text[: start + 1]))
        page = int(page_matches[-1].group(1)) if page_matches else None
        if segment:
            digest = hashlib.sha256(segment.encode("utf-8")).hexdigest()
            chunks.append(
                DocumentChunk(
                    chunk_id=stable_id("chk", f"{document.document_id}:{start}:{digest}"),
                    document_id=document.document_id,
                    heading=heading,
                    page=page,
                    start_offset=start,
                    end_offset=end,
                    text=segment,
                    content_hash=digest,
                )
            )
        if end >= len(text):
            break
        start = max(start + 1, end - settings.chunk_overlap_chars)
    return chunks


def select_chunks(objective: str, chunks: list[DocumentChunk], limit: int) -> list[DocumentChunk]:
    """Select relevant chunks without sending full documents to an LLM."""
    terms = _terms(objective)
    scored = [
        (len(terms & _terms(f"{chunk.heading or ''} {chunk.text}")), -chunk.start_offset, chunk)
        for chunk in chunks
    ]
    scored.sort(key=lambda item: (-item[0], -item[1], item[2].chunk_id))
    return [item[2] for item in scored[:limit]]


def evidence_from_chunks(
    objective: str,
    document_by_id: dict[str, ExtractedDocument],
    chunks: list[DocumentChunk],
) -> list[EvidenceRecord]:
    """Create bounded claim evidence while quarantining instruction-shaped chunks."""
    objective_terms = _terms(objective)
    evidence: list[EvidenceRecord] = []
    for chunk in chunks:
        if inspect_untrusted_content(chunk.text):
            continue
        sentences = [line.strip(" -*\t") for line in SENTENCE_RE.split(chunk.text) if len(line.strip()) >= 40]
        sentences.sort(key=lambda line: len(objective_terms & _terms(line)), reverse=True)
        excerpt = (sentences[0] if sentences else chunk.text[:800]).strip()[:1000]
        if not excerpt:
            continue
        document = document_by_id[chunk.document_id]
        locator = f"page {chunk.page}" if chunk.page else f"chars {chunk.start_offset}-{chunk.end_offset}"
        evidence.append(
            EvidenceRecord(
                evidence_id=stable_id("ev", f"{chunk.chunk_id}:{excerpt}"),
                claim=excerpt,
                supporting_excerpt=excerpt,
                document_id=document.document_id,
                chunk_id=chunk.chunk_id,
                locator=locator,
                source_url=document.final_url,
                source_title=document.title,
                confidence=0.7,
            )
        )
    return evidence


def analyze_gaps(
    request: SearchRequest,
    evidence: list[EvidenceRecord],
    documents: list[ExtractedDocument],
    budget: BudgetSnapshot,
    *,
    pending_domains: list[str] | None = None,
) -> GapAnalysis:
    """Produce a deterministic baseline gap decision for the agent quality loop."""
    if pending_domains:
        return GapAnalysis(
            missing_dimensions=["domain approval"],
            next_queries=[],
            decision="approval_required",
            reason="Top-K candidate domains require approval before fetching.",
            budget=budget,
        )
    if budget.exhausted:
        return GapAnalysis(
            covered_dimensions=[record.claim for record in evidence[:5]],
            missing_dimensions=[] if evidence else [request.objective],
            decision="budget_exhausted",
            reason="The configured successful-document budget has been exhausted.",
            budget=budget,
        )
    independent_domains = {urlsplit(item.source_url).hostname for item in evidence}
    if len(documents) >= 3 and len(independent_domains) >= 2 and evidence:
        return GapAnalysis(
            covered_dimensions=[record.claim for record in evidence[:5]],
            decision="complete",
            reason="At least three fetched documents and two independent domains produced evidence.",
            budget=budget,
        )
    return GapAnalysis(
        covered_dimensions=[record.claim for record in evidence[:3]],
        missing_dimensions=[request.objective],
        next_queries=request.queries[:2],
        decision="continue",
        reason="More successfully fetched, independently sourced evidence is required.",
        budget=budget,
    )


class WebResearchPipeline:
    """Coordinate one deterministic web research iteration."""

    def __init__(
        self,
        *,
        search: SearchAdapter,
        settings: WebPipelineSettings,
        reranker: RerankAdapter | None = None,
        approve: ApprovalAdapter | None = None,
        render_dynamic: DynamicRenderAdapter | None = None,
        external_extractors: list[ExternalExtractAdapter] | None = None,
        evidence_extractor: EvidenceAdapter | None = None,
    ) -> None:
        """Store provider adapters and bounded runtime settings."""
        self.search = search
        self.settings = settings
        self.reranker = reranker
        self.approve = approve
        self.render_dynamic = render_dynamic
        self.external_extractors = external_extractors or []
        self.evidence_extractor = evidence_extractor

    async def run(self, request: SearchRequest, *, remaining_fetches: int | None = None) -> WebResearchResult:
        """Run Search, select Top K, fetch, extract, and create citable evidence."""
        batch = await self.search(request)
        candidates = normalize_candidates(batch.candidates, request.candidate_limit)
        allowed_fetches = min(
            self.settings.fetch_top_k,
            self.settings.max_fetches if remaining_fetches is None else max(0, remaining_fetches),
        )
        ranked = await rank_candidates(
            request.objective,
            candidates,
            reranker=self.reranker,
            top_k=allowed_fetches,
            min_authority=self.settings.min_source_authority,
        )
        selected = [item.candidate for item in ranked if item.selected]
        authority_rejected_all = bool(
            candidates
            and not selected
            and any(item.reason == "below_authority_threshold" for item in ranked)
        )
        run_id = "default"
        approval = None
        if self.approve and selected:
            approval = await self.approve(selected, request.iteration)
            run_id = approval.run_id
        pending = set(approval.pending_domains if approval else [])
        denied = set(approval.denied_domains if approval else [])
        fetchable = [c for c in selected if c.domain not in pending | denied]
        for item in ranked:
            if item.selected and item.candidate.domain in pending | denied:
                item.selected = False
                item.reason = (
                    "domain_approval_pending"
                    if item.candidate.domain in pending
                    else "domain_denied"
                )
        # Fill denied slots from the ranked pool when the replacement's domain is
        # already allowed. New undecided domains are collected into the same
        # logical approval batch but are never fetched prematurely.
        if self.approve and len(fetchable) < allowed_fetches:
            replacement_domain_counts = Counter(item.domain for item in fetchable)
            for replacement in [
                item
                for item in ranked
                if not item.selected and item.authority >= self.settings.min_source_authority
            ]:
                if len(fetchable) >= allowed_fetches:
                    break
                candidate = replacement.candidate
                if (
                    candidate.domain in denied
                    or candidate in fetchable
                    or replacement_domain_counts[candidate.domain] >= 2
                ):
                    continue
                replacement_approval = await self.approve([candidate], request.iteration)
                if approval:
                    approval.domains = list(dict.fromkeys(approval.domains + replacement_approval.domains))
                    approval.urls = list(dict.fromkeys(approval.urls + replacement_approval.urls))
                    approval.pending_domains = list(
                        dict.fromkeys(approval.pending_domains + replacement_approval.pending_domains)
                    )
                    approval.denied_domains = list(
                        dict.fromkeys(approval.denied_domains + replacement_approval.denied_domains)
                    )
                pending.update(replacement_approval.pending_domains)
                denied.update(replacement_approval.denied_domains)
                if candidate.domain not in pending | denied:
                    replacement.selected = True
                    replacement.reason = "selected_replacement"
                    fetchable.append(candidate)
                    replacement_domain_counts[candidate.domain] += 1
        if pending and not fetchable:
            budget = BudgetSnapshot(
                search_calls=len(request.queries),
                candidates=len(candidates),
                max_fetches=self.settings.max_fetches,
            )
            gap = analyze_gaps(request, [], [], budget, pending_domains=sorted(pending))
            return WebResearchResult(
                request=request,
                candidates=candidates,
                ranked_candidates=ranked,
                provider_syntheses=batch.syntheses,
                approval_batch=approval,
                gap_analysis=gap,
                errors=batch.errors,
            )

        semaphore = asyncio.Semaphore(self.settings.global_concurrency)
        host_semaphores: defaultdict[str, asyncio.Semaphore] = defaultdict(
            lambda: asyncio.Semaphore(self.settings.per_host_concurrency)
        )

        async def redirect_allowed(target: str) -> bool:
            host = urlsplit(target).hostname or ""
            if approval and host in approval.domains and host not in pending | denied:
                return True
            return False

        async def fetch_one(candidate: CandidateSource) -> RawFetch:
            cache_key = (self.settings.cache_namespace, candidate.canonical_url)
            lock = _FETCH_LOCKS.setdefault(cache_key, asyncio.Lock())
            async with lock:
                cached = _DOCUMENT_CACHE.get(cache_key)
                if cached is not None:
                    return RawFetch(
                        result=FetchResult(
                            candidate_id=candidate.candidate_id,
                            requested_url=candidate.canonical_url,
                            final_url=cached.final_url,
                            content_type=cached.content_type,
                            content_hash=cached.content_hash,
                            fetched_at=datetime.now(timezone.utc),
                            adapter="run_cache",
                            success=True,
                        )
                    )
                async with semaphore, host_semaphores[candidate.domain]:
                    return await fetch_local(candidate, self.settings, redirect_allowed=redirect_allowed)

        raws = await asyncio.gather(*(fetch_one(candidate) for candidate in fetchable))
        documents: list[ExtractedDocument] = []
        errors = list(batch.errors)
        if authority_rejected_all:
            errors.append("no_candidates_met_source_authority_threshold")
        for candidate, raw in zip(fetchable, raws):
            if not raw.result.success:
                errors.append(f"{candidate.canonical_url}: {raw.result.failure_class}")
                continue
            cache_key = (self.settings.cache_namespace, candidate.canonical_url)
            document = _DOCUMENT_CACHE.get(cache_key) if raw.result.adapter == "run_cache" else None
            if document is None:
                try:
                    document = extract_document(candidate, raw, self.settings)
                except ValueError as exc:
                    document = None
                    errors.append(f"{candidate.canonical_url}: {exc}")
            needs_external = document is None or "needs_external_extraction" in document.quality_flags
            needs_dynamic = bool(document and "dynamic_render_recommended" in document.quality_flags)
            if needs_dynamic and self.render_dynamic:
                try:
                    rendered = await self.render_dynamic(candidate.canonical_url)
                    if rendered:
                        rendered_raw = RawFetch(
                            result=raw.result.model_copy(update={"adapter": "playwright"}),
                            body=rendered.encode("utf-8"),
                        )
                        document = extract_html(candidate, rendered_raw)
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"{candidate.canonical_url}: playwright:{exc}")
            if needs_dynamic and (
                document is None or "dynamic_render_recommended" in document.quality_flags
            ):
                needs_external = True
            if needs_external:
                external_document = None
                for extractor in self.external_extractors:
                    try:
                        external = await extractor(candidate.canonical_url)
                    except Exception as exc:  # noqa: BLE001
                        errors.append(f"{candidate.canonical_url}: external_extract:{exc}")
                        continue
                    if external is not None:
                        external_document = external
                        break
                document = external_document
                if document is None:
                    errors.append(
                        f"{candidate.canonical_url}: no_extractor_produced_usable_content"
                    )
            if document and document.markdown.strip():
                documents.append(document)
                _DOCUMENT_CACHE[cache_key] = document

        # Content-hash dedupe after extraction.
        unique_documents = list({doc.content_hash: doc for doc in documents}.values())
        all_chunks: list[DocumentChunk] = []
        for document in unique_documents:
            document_chunks = chunk_document(document, self.settings)
            all_chunks.extend(
                select_chunks(request.objective, document_chunks, self.settings.max_chunks_per_document)
            )
        selected_chunks = select_chunks(
            request.objective, all_chunks, self.settings.max_chunks_per_iteration
        )
        document_by_id = {document.document_id: document for document in unique_documents}
        if self.evidence_extractor:
            try:
                evidence = await self.evidence_extractor(
                    request.objective, document_by_id, selected_chunks
                )
            except Exception:  # noqa: BLE001 - deterministic evidence is the safe fallback
                evidence = evidence_from_chunks(request.objective, document_by_id, selected_chunks)
        else:
            evidence = evidence_from_chunks(request.objective, document_by_id, selected_chunks)
        authority_by_candidate = {
            item.candidate.candidate_id: item.authority for item in ranked
        }
        candidate_by_document = {
            document.document_id: document.candidate_id for document in unique_documents
        }
        evidence = [
            record.model_copy(
                update={
                    "source_authority": authority_by_candidate.get(
                        candidate_by_document.get(record.document_id, ""),
                        0.0,
                    )
                }
            )
            for record in evidence
        ]
        budget = BudgetSnapshot(
            search_calls=len(request.queries),
            candidates=len(candidates),
            fetch_attempts=len(raws),
            fetched_documents=len(unique_documents),
            max_fetches=self.settings.max_fetches,
            exhausted=len(unique_documents) >= self.settings.max_fetches or allowed_fetches == 0,
        )
        gap = analyze_gaps(request, evidence, unique_documents, budget, pending_domains=sorted(pending) or None)
        return WebResearchResult(
            request=request,
            candidates=candidates,
            ranked_candidates=ranked,
            provider_syntheses=batch.syntheses,
            approval_batch=approval
            or DomainApprovalBatch(
                run_id=run_id,
                iteration=request.iteration,
                domains=sorted({candidate.domain for candidate in selected}),
                urls=[candidate.canonical_url for candidate in selected],
            ),
            fetches=[raw.result for raw in raws],
            documents=unique_documents,
            chunks=selected_chunks,
            evidence=evidence,
            gap_analysis=gap,
            errors=errors,
        )
