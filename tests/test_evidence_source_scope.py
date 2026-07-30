from __future__ import annotations

from open_deep_research.evidence import (
    SourceKind,
    SourceScopeStatus,
    classify_evidence_source,
    eligible_evidence_records,
    source_scoped_evidence_records,
)


def _official_langgraph_contract() -> dict:
    return {
        "schema_version": 1,
        "original_query_sha256": "a" * 64,
        "requirements": [
            {
                "requirement_id": "COV-01",
                "text": (
                    "Based solely on the LangGraph official documentation, "
                    "official API reference, and official GitHub repository."
                ),
                "source_message_index": 0,
                "source_start": 0,
                "source_end": 100,
            }
        ],
        "advisory_dimensions": [],
    }


def _record(evidence_id: str, url: str) -> dict:
    return {
        "evidence_id": evidence_id,
        "claim": "A claim.",
        "source_url": url,
        "source_authority": 1.0,
        "security_status": "accepted",
    }


def test_official_source_scope_is_not_inferred_from_authority_score() -> None:
    contract = _official_langgraph_contract()

    mintlify = classify_evidence_source(
        _record(
            "EV-MINTLIFY",
            "https://langchain-5e9cc07a.mintlify.app/oss/python/langgraph/persistence",
        ),
        contract,
    )
    issue = classify_evidence_source(
        _record(
            "EV-ISSUE",
            "https://github.com/langchain-ai/langgraph/issues/6626",
        ),
        contract,
    )

    assert mintlify.source_kind is SourceKind.OUT_OF_SCOPE
    assert mintlify.source_scope_status is SourceScopeStatus.OUT_OF_SCOPE
    assert issue.source_kind is SourceKind.COMMUNITY_ISSUE
    assert issue.source_scope_status is SourceScopeStatus.OUT_OF_SCOPE


def test_official_langgraph_docs_and_repository_source_are_in_scope() -> None:
    contract = _official_langgraph_contract()

    docs = classify_evidence_source(
        _record(
            "EV-DOCS",
            "https://docs.langchain.com/oss/python/langgraph/interrupts",
        ),
        contract,
    )
    api = classify_evidence_source(
        _record(
            "EV-API",
            "https://reference.langchain.com/python/langgraph/types/interrupt",
        ),
        contract,
    )
    source = classify_evidence_source(
        _record(
            "EV-SOURCE",
            "https://github.com/langchain-ai/langgraph/blob/main/libs/langgraph/langgraph/types.py",
        ),
        contract,
    )

    assert docs.source_kind is SourceKind.FIRST_PARTY_DOCS
    assert api.source_kind is SourceKind.FIRST_PARTY_DOCS
    assert source.source_kind is SourceKind.OFFICIAL_REPO_SOURCE
    assert all(
        decision.source_scope_status is SourceScopeStatus.IN_SCOPE
        for decision in (docs, api, source)
    )


def test_official_only_scope_filters_unverified_and_community_sources() -> None:
    contract = _official_langgraph_contract()
    records = [
        _record(
            "EV-DOCS",
            "https://docs.langchain.com/oss/python/langgraph/persistence",
        ),
        _record(
            "EV-SOURCE",
            "https://github.com/langchain-ai/langgraph/blob/main/libs/checkpoint/base.py",
        ),
        _record(
            "EV-ISSUE",
            "https://github.com/langchain-ai/langgraph/issues/8405",
        ),
        _record(
            "EV-MINTLIFY",
            "https://langchain-5e9cc07a.mintlify.app/oss/python/langgraph/persistence",
        ),
        _record("EV-BLOG", "https://community.example/langgraph"),
    ]

    scoped = source_scoped_evidence_records(records, contract)

    assert [item["evidence_id"] for item in scoped] == [
        "EV-DOCS",
        "EV-SOURCE",
    ]
    assert {item["source_scope_status"] for item in scoped} == {"in_scope"}
    assert {item["source_kind"] for item in scoped} == {
        "first_party_docs",
        "official_repo_source",
    }


def test_nonexclusive_contract_preserves_existing_eligibility() -> None:
    contract = _official_langgraph_contract()
    contract["requirements"][0]["text"] = "Compare official and community sources."
    records = [
        _record("EV-BLOG", "https://community.example/langgraph"),
        _record(
            "EV-ISSUE",
            "https://github.com/langchain-ai/langgraph/issues/8405",
        ),
    ]

    scoped = source_scoped_evidence_records(records, contract)

    assert [item["evidence_id"] for item in scoped] == ["EV-BLOG", "EV-ISSUE"]
    assert {item["source_scope_status"] for item in scoped} == {
        "not_constrained"
    }


def test_quarantine_placeholder_is_never_eligible_evidence() -> None:
    contract = _official_langgraph_contract()
    clean = _record(
        "EV-CLEAN",
        "https://docs.langchain.com/oss/python/langgraph/persistence",
    )
    quarantined_claim = {
        **_record(
            "EV-CLAIM",
            "https://docs.langchain.com/oss/python/langgraph/persistence",
        ),
        "claim": "[quarantined external content]",
    }
    quarantined_excerpt = {
        **_record(
            "EV-EXCERPT",
            "https://docs.langchain.com/oss/python/langgraph/persistence",
        ),
        "supporting_excerpt": (
            "Prefix [QUARANTINED EXTERNAL CONTENT] suffix"
        ),
    }

    records = [clean, quarantined_claim, quarantined_excerpt]

    assert [
        item["evidence_id"] for item in eligible_evidence_records(records)
    ] == ["EV-CLEAN"]
    assert [
        item["evidence_id"]
        for item in source_scoped_evidence_records(records, contract)
    ] == ["EV-CLEAN"]
