"""Shared evidence eligibility rules used across the research lifecycle."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, cast
from urllib.parse import urlsplit

SOURCE_SCOPE_POLICY_VERSION = "evidence-source-scope-v1"

_EXCLUSIVE_OFFICIAL_SOURCE_RE = re.compile(
    r"(?:"
    r"based\s+(?:solely|only|exclusively)\s+on|"
    r"(?:solely|only|exclusively)\s+(?:use|using|from)|"
    r"(?:use|using)\s+only|"
    r"仅(?:基于|使用)|只(?:基于|使用)"
    r").{0,240}(?:official|first[- ]party|官方)",
    flags=re.IGNORECASE | re.DOTALL,
)
_TEMPORARY_HOST_SUFFIXES = (
    ".mintlify.app",
    ".netlify.app",
    ".vercel.app",
)
_COMMUNITY_GITHUB_PATH_PARTS = {
    "issues",
    "pull",
    "pulls",
    "discussions",
}
_QUARANTINED_CONTENT_PLACEHOLDER_RE = re.compile(
    r"\[\s*quarantined\s+external\s+content\s*\]",
    flags=re.IGNORECASE,
)


class SourceKind(str, Enum):
    """Deterministic provenance class used by source-scope admission."""

    FIRST_PARTY_DOCS = "first_party_docs"
    OFFICIAL_REPO_SOURCE = "official_repo_source"
    COMMUNITY_ISSUE = "community_issue"
    OUT_OF_SCOPE = "out_of_scope"


class SourceScopeStatus(str, Enum):
    """Whether one source satisfies an explicit source-scope contract."""

    IN_SCOPE = "in_scope"
    OUT_OF_SCOPE = "out_of_scope"
    UNVERIFIED = "unverified"
    NOT_CONSTRAINED = "not_constrained"


@dataclass(frozen=True, slots=True)
class SourceScopeDecision:
    """Auditable deterministic source-scope classification."""

    source_kind: SourceKind
    source_scope_status: SourceScopeStatus
    reason: str
    policy_version: str = SOURCE_SCOPE_POLICY_VERSION


@dataclass(frozen=True, slots=True)
class _OfficialSourceProfile:
    aliases: tuple[str, ...]
    documentation_paths: tuple[tuple[str, str], ...]
    repositories: tuple[tuple[str, str], ...]


_OFFICIAL_SOURCE_PROFILES = (
    _OfficialSourceProfile(
        aliases=("langgraph",),
        documentation_paths=(
            ("docs.langchain.com", "/langgraph"),
            ("reference.langchain.com", "/langgraph"),
            ("api.python.langchain.com", "/langgraph"),
            ("langchain-ai.github.io", "/langgraph"),
        ),
        repositories=(("langchain-ai", "langgraph"),),
    ),
)


def is_evidence_eligible(record: object) -> bool:
    """Return whether one evidence record passed the security admission gate."""
    return (
        isinstance(record, dict)
        and record.get("security_status", "accepted") == "accepted"
        and not any(
            _QUARANTINED_CONTENT_PLACEHOLDER_RE.search(
                str(record.get(field_name, ""))
            )
            for field_name in ("claim", "supporting_excerpt")
        )
    )


def eligible_evidence_records(records: Iterable[object]) -> list[dict[str, Any]]:
    """Return admitted evidence records while preserving their original order."""
    return [
        cast(dict[str, Any], record)
        for record in records
        if is_evidence_eligible(record)
    ]


def _contract_requirement_text(coverage_contract: object) -> str:
    if isinstance(coverage_contract, dict):
        requirements = coverage_contract.get("requirements", ())
    else:
        requirements = getattr(coverage_contract, "requirements", ())
    texts: list[str] = []
    for requirement in requirements or ():
        if isinstance(requirement, dict):
            text = requirement.get("text")
        else:
            text = getattr(requirement, "text", None)
        if text:
            texts.append(str(text))
    return "\n".join(texts)


def contract_requires_official_sources(coverage_contract: object) -> bool:
    """Return whether the user explicitly required exclusive official sources."""
    return bool(
        _EXCLUSIVE_OFFICIAL_SOURCE_RE.search(
            _contract_requirement_text(coverage_contract)
        )
    )


def _matching_official_profiles(
    coverage_contract: object,
) -> tuple[_OfficialSourceProfile, ...]:
    normalized = re.sub(
        r"[^a-z0-9]+",
        "",
        _contract_requirement_text(coverage_contract).casefold(),
    )
    return tuple(
        profile
        for profile in _OFFICIAL_SOURCE_PROFILES
        if any(
            re.sub(r"[^a-z0-9]+", "", alias.casefold()) in normalized
            for alias in profile.aliases
        )
    )


def _github_path_parts(url: str) -> tuple[str, ...]:
    parsed = urlsplit(url)
    if parsed.hostname and parsed.hostname.casefold() == "github.com":
        return tuple(
            part.casefold()
            for part in parsed.path.split("/")
            if part
        )
    return ()


def _is_community_github_url(url: str) -> bool:
    parts = _github_path_parts(url)
    return len(parts) >= 3 and parts[2] in _COMMUNITY_GITHUB_PATH_PARTS


def _matches_profile_docs(
    *,
    host: str,
    path: str,
    profile: _OfficialSourceProfile,
) -> bool:
    return any(
        host == expected_host and expected_path in path
        for expected_host, expected_path in profile.documentation_paths
    )


def _matches_profile_repository(
    *,
    host: str,
    path_parts: tuple[str, ...],
    profile: _OfficialSourceProfile,
) -> bool:
    if host == "github.com" and len(path_parts) >= 2:
        return (path_parts[0], path_parts[1]) in profile.repositories
    if host == "raw.githubusercontent.com" and len(path_parts) >= 2:
        return (path_parts[0], path_parts[1]) in profile.repositories
    return False


def classify_evidence_source(
    record: dict[str, Any],
    coverage_contract: object,
) -> SourceScopeDecision:
    """Classify source provenance without treating authority as ownership."""
    url = str(record.get("source_url") or "").strip()
    parsed = urlsplit(url)
    host = (parsed.hostname or "").casefold()
    path = parsed.path.casefold()
    path_parts = tuple(part for part in path.split("/") if part)
    constrained = contract_requires_official_sources(coverage_contract)

    if _is_community_github_url(url):
        return SourceScopeDecision(
            source_kind=SourceKind.COMMUNITY_ISSUE,
            source_scope_status=(
                SourceScopeStatus.OUT_OF_SCOPE
                if constrained
                else SourceScopeStatus.NOT_CONSTRAINED
            ),
            reason="community_github_surface",
        )
    if any(host.endswith(suffix) for suffix in _TEMPORARY_HOST_SUFFIXES):
        return SourceScopeDecision(
            source_kind=SourceKind.OUT_OF_SCOPE,
            source_scope_status=(
                SourceScopeStatus.OUT_OF_SCOPE
                if constrained
                else SourceScopeStatus.NOT_CONSTRAINED
            ),
            reason="temporary_host_not_first_party_verified",
        )
    for profile in _matching_official_profiles(coverage_contract):
        if _matches_profile_docs(host=host, path=path, profile=profile):
            return SourceScopeDecision(
                source_kind=SourceKind.FIRST_PARTY_DOCS,
                source_scope_status=(
                    SourceScopeStatus.IN_SCOPE
                    if constrained
                    else SourceScopeStatus.NOT_CONSTRAINED
                ),
                reason="matched_versioned_official_docs_profile",
            )
        if _matches_profile_repository(
            host=host,
            path_parts=path_parts,
            profile=profile,
        ):
            return SourceScopeDecision(
                source_kind=SourceKind.OFFICIAL_REPO_SOURCE,
                source_scope_status=(
                    SourceScopeStatus.IN_SCOPE
                    if constrained
                    else SourceScopeStatus.NOT_CONSTRAINED
                ),
                reason="matched_versioned_official_repository_profile",
            )
    return SourceScopeDecision(
        source_kind=SourceKind.OUT_OF_SCOPE,
        source_scope_status=(
            SourceScopeStatus.UNVERIFIED
            if constrained
            else SourceScopeStatus.NOT_CONSTRAINED
        ),
        reason=(
            "official_ownership_not_verified"
            if constrained
            else "source_scope_not_constrained"
        ),
    )


def source_scoped_evidence_records(
    records: Iterable[object],
    coverage_contract: object,
) -> list[dict[str, Any]]:
    """Annotate evidence and fail closed under exclusive source constraints."""
    constrained = contract_requires_official_sources(coverage_contract)
    scoped: list[dict[str, Any]] = []
    for record in eligible_evidence_records(records):
        decision = classify_evidence_source(record, coverage_contract)
        if (
            constrained
            and decision.source_scope_status is not SourceScopeStatus.IN_SCOPE
        ):
            continue
        annotated = dict(record)
        annotated.update(
            {
                "source_kind": decision.source_kind.value,
                "source_scope_status": decision.source_scope_status.value,
                "source_scope_reason": decision.reason,
                "source_scope_policy_version": decision.policy_version,
            }
        )
        scoped.append(annotated)
    return scoped
