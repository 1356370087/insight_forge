"""Structured contracts for the web research pipeline."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    """One normalized candidate-discovery request."""

    objective: str
    queries: list[str] = Field(min_length=1, max_length=3)
    topic: Literal["general", "news", "finance"] = "general"
    locale: str | None = None
    published_after: datetime | None = None
    candidate_limit: int = Field(default=20, ge=1, le=100)
    iteration: int = Field(default=1, ge=1)


class CandidateSource(BaseModel):
    """A source discovered by Search but not yet accepted as evidence."""

    candidate_id: str
    provider: str
    query_ids: list[str] = Field(default_factory=list)
    provider_rank: int = Field(default=0, ge=0)
    provider_score: float | None = None
    original_url: str
    canonical_url: str
    domain: str
    title: str = ""
    snippet: str = ""
    author: str | None = None
    published_at: datetime | None = None
    content_hint: str | None = None


class ProviderSynthesis(BaseModel):
    """Provider-produced search synthesis; never eligible as evidence."""

    provider: str
    text: str = ""
    cited_candidate_ids: list[str] = Field(default_factory=list)
    evidence_eligible: Literal[False] = False


class SearchBatch(BaseModel):
    """Normalized output shared by all Search providers."""

    candidates: list[CandidateSource] = Field(default_factory=list)
    syntheses: list[ProviderSynthesis] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class RankedCandidate(BaseModel):
    """Candidate after deterministic filtering and reranking."""

    candidate: CandidateSource
    relevance: float = Field(default=0.0, ge=0.0, le=1.0)
    authority: float = Field(default=0.0, ge=0.0, le=1.0)
    authority_method: Literal["reranker", "heuristic"] = "heuristic"
    information_gain: float = Field(default=0.0, ge=0.0, le=1.0)
    freshness: float = Field(default=0.5, ge=0.0, le=1.0)
    provider_rank_score: float = Field(default=0.0, ge=0.0, le=1.0)
    final_score: float = Field(default=0.0, ge=0.0, le=1.0)
    selected: bool = False
    reason: str = ""


class FetchResult(BaseModel):
    """Auditable result of fetching one candidate URL."""

    candidate_id: str
    requested_url: str
    final_url: str | None = None
    redirect_chain: list[str] = Field(default_factory=list)
    status_code: int | None = None
    content_type: str | None = None
    byte_count: int = 0
    content_hash: str | None = None
    fetched_at: datetime | None = None
    adapter: str = "local"
    attempts: int = 1
    success: bool = False
    failure_class: str | None = None
    failure_message: str | None = None


class ExtractedDocument(BaseModel):
    """Clean document produced from a successful fetch."""

    document_id: str
    candidate_id: str
    requested_url: str
    final_url: str
    canonical_url: str
    title: str = ""
    author: str | None = None
    published_at: str | None = None
    language: str | None = None
    content_type: str
    markdown: str
    page_count: int | None = None
    extractor: str
    extractor_version: str = "1"
    content_hash: str
    quality_flags: list[str] = Field(default_factory=list)


class DocumentChunk(BaseModel):
    """Stable, source-located unit used by evidence extraction."""

    chunk_id: str
    document_id: str
    heading: str | None = None
    page: int | None = None
    start_offset: int = 0
    end_offset: int = 0
    text: str
    content_hash: str


class EvidenceRecord(BaseModel):
    """Claim-level evidence that can be cited by the report writer."""

    evidence_id: str
    claim: str
    supporting_excerpt: str
    document_id: str
    chunk_id: str
    locator: str
    source_url: str
    source_title: str = ""
    source_authority: float = Field(default=0.0, ge=0.0, le=1.0)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    conflict_group: str | None = None
    security_status: Literal["accepted", "quarantined"] = "accepted"


class BudgetSnapshot(BaseModel):
    """Budget state emitted after every deterministic web iteration."""

    search_calls: int = 0
    candidates: int = 0
    fetch_attempts: int = 0
    fetched_documents: int = 0
    max_fetches: int = 0
    exhausted: bool = False


class DomainApprovalBatch(BaseModel):
    """Target domains selected for a single Top-K fetch batch."""

    run_id: str
    iteration: int
    domains: list[str] = Field(default_factory=list)
    urls: list[str] = Field(default_factory=list)
    pending_domains: list[str] = Field(default_factory=list)
    denied_domains: list[str] = Field(default_factory=list)


class GapAnalysis(BaseModel):
    """Structured decision about whether another web iteration is required."""

    covered_dimensions: list[str] = Field(default_factory=list)
    missing_dimensions: list[str] = Field(default_factory=list)
    conflicts: list[str] = Field(default_factory=list)
    next_queries: list[str] = Field(default_factory=list)
    decision: Literal["continue", "complete", "approval_required", "budget_exhausted"]
    reason: str
    budget: BudgetSnapshot


class WebResearchResult(BaseModel):
    """Complete structured output of one web research iteration."""

    request: SearchRequest
    candidates: list[CandidateSource] = Field(default_factory=list)
    ranked_candidates: list[RankedCandidate] = Field(default_factory=list)
    provider_syntheses: list[ProviderSynthesis] = Field(default_factory=list)
    approval_batch: DomainApprovalBatch | None = None
    fetches: list[FetchResult] = Field(default_factory=list)
    documents: list[ExtractedDocument] = Field(default_factory=list)
    chunks: list[DocumentChunk] = Field(default_factory=list)
    evidence: list[EvidenceRecord] = Field(default_factory=list)
    gap_analysis: GapAnalysis
    errors: list[str] = Field(default_factory=list)
