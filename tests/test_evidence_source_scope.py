from __future__ import annotations

from open_deep_research.evidence import (
    SourceKind,
    SourceScopeStatus,
    classify_evidence_source,
    compile_source_scope,
    contract_requires_official_sources,
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


def _official_postgresql_contract() -> dict:
    contract = _official_langgraph_contract()
    contract["requirements"][0]["text"] = (
        "请仅依据 PostgreSQL 官方文档和 postgresql.org 官方发布说明。"
    )
    return contract


def _official_python_contract() -> dict:
    contract = _official_langgraph_contract()
    contract["requirements"][0]["text"] = (
        "只使用一个 Python 官方来源确认 Python 3.13 的发布日期。"
    )
    return contract


def _record(evidence_id: str, url: str) -> dict:
    return {
        "evidence_id": evidence_id,
        "claim": "A claim.",
        "source_url": url,
        "source_authority": 1.0,
        "security_status": "accepted",
    }


def test_chinese_only_allowed_source_phrase_enforces_scope() -> None:
    for wording in (
        "请只允许使用这 3 个 PostgreSQL 官方 URL 作为证据。",
        "请仅允许使用 PostgreSQL 官方文档。",
    ):
        contract = _official_postgresql_contract()
        contract["requirements"][0]["text"] = wording

        assert contract_requires_official_sources(contract) is True


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


def test_official_postgresql_docs_and_newsroom_are_in_scope() -> None:
    contract = _official_postgresql_contract()

    release_notes = classify_evidence_source(
        _record(
            "EV-PG-RELEASE",
            "https://www.postgresql.org/docs/release/18.0/",
        ),
        contract,
    )
    newsroom = classify_evidence_source(
        _record(
            "EV-PG-NEWS",
            "https://www.postgresql.org/about/news/postgresql-18-released-3142/",
        ),
        contract,
    )
    third_party = classify_evidence_source(
        _record(
            "EV-PG-BLOG",
            "https://example.com/postgresql-18-skip-scan",
        ),
        contract,
    )

    assert release_notes.source_kind is SourceKind.FIRST_PARTY_DOCS
    assert newsroom.source_kind is SourceKind.FIRST_PARTY_DOCS
    assert release_notes.source_scope_status is SourceScopeStatus.IN_SCOPE
    assert newsroom.source_scope_status is SourceScopeStatus.IN_SCOPE
    assert third_party.source_scope_status is SourceScopeStatus.UNVERIFIED


def test_official_python_sites_and_repository_are_in_scope() -> None:
    contract = _official_python_contract()
    decisions = [
        classify_evidence_source(
            _record("EV-PYTHON-RELEASE", "https://www.python.org/downloads/release/python-3130/"),
            contract,
        ),
        classify_evidence_source(
            _record("EV-PYTHON-DOCS", "https://docs.python.org/3/whatsnew/3.13.html"),
            contract,
        ),
        classify_evidence_source(
            _record("EV-PYTHON-PEP", "https://peps.python.org/pep-0719/"),
            contract,
        ),
        classify_evidence_source(
            _record("EV-CPYTHON", "https://github.com/python/cpython/blob/main/README.rst"),
            contract,
        ),
    ]
    third_party = classify_evidence_source(
        _record("EV-PYTHON-BLOG", "https://example.com/python-313"),
        contract,
    )

    assert all(
        decision.source_scope_status is SourceScopeStatus.IN_SCOPE
        for decision in decisions
    )
    assert third_party.source_scope_status is SourceScopeStatus.UNVERIFIED


def test_explicit_url_allowlist_does_not_expand_to_entire_official_site() -> None:
    contract = _official_postgresql_contract()
    contract["requirements"] = [
        {
            "requirement_id": "COV-01",
            "text": (
                "只允许使用这 3 个 PostgreSQL 官方 URL 作为证据："
                "https://www.postgresql.org/docs/release/18.0/"
            ),
            "source_message_index": 0,
            "source_start": 0,
            "source_end": 100,
        },
        {
            "requirement_id": "COV-02",
            "text": (
                "https://www.postgresql.org/docs/18/"
                "indexes-multicolumn.html"
            ),
            "source_message_index": 0,
            "source_start": 101,
            "source_end": 160,
        },
        {
            "requirement_id": "COV-03",
            "text": (
                "https://www.postgresql.org/about/news/"
                "postgresql-18-released-3142/"
            ),
            "source_message_index": 0,
            "source_start": 161,
            "source_end": 230,
        },
    ]

    allowed = classify_evidence_source(
        _record(
            "EV-ALLOWED",
            "https://www.postgresql.org/docs/18/indexes-multicolumn.html",
        ),
        contract,
    )
    extra_official_page = classify_evidence_source(
        _record(
            "EV-CURRENT",
            "https://www.postgresql.org/docs/current/indexes-multicolumn.html",
        ),
        contract,
    )

    assert allowed.source_scope_status is SourceScopeStatus.IN_SCOPE
    assert extra_official_page.source_kind is SourceKind.OUT_OF_SCOPE
    assert (
        extra_official_page.source_scope_status
        is SourceScopeStatus.OUT_OF_SCOPE
    )
    assert extra_official_page.reason == "outside_explicit_url_allowlist"


def test_explicit_url_allowlist_is_enforced_without_official_wording() -> None:
    contract = _official_langgraph_contract()
    contract["requirements"] = [{
        "requirement_id": "COV-01",
        "text": (
            "只允许使用以下 URL 作为证据：https://allowed.example/research 。"
            "不得使用 https://blocked.example/private 。"
        ),
        "source_message_index": 0,
        "source_start": 0,
        "source_end": 120,
    }]

    allowed = classify_evidence_source(
        _record("EV-ALLOWED", "https://allowed.example/research"),
        contract,
    )
    blocked = classify_evidence_source(
        _record("EV-BLOCKED", "https://blocked.example/private"),
        contract,
    )
    unlisted = classify_evidence_source(
        _record("EV-UNLISTED", "https://unlisted.example/research"),
        contract,
    )

    assert allowed.source_scope_status is SourceScopeStatus.IN_SCOPE
    assert blocked.source_scope_status is SourceScopeStatus.OUT_OF_SCOPE
    assert blocked.reason == "inside_explicit_url_denylist"
    assert unlisted.source_scope_status is SourceScopeStatus.OUT_OF_SCOPE
    assert unlisted.reason == "outside_explicit_url_allowlist"
    assert [
        item["evidence_id"]
        for item in source_scoped_evidence_records(
            [
                _record("EV-ALLOWED", "https://allowed.example/research"),
                _record("EV-BLOCKED", "https://blocked.example/private"),
                _record("EV-UNLISTED", "https://unlisted.example/research"),
            ],
            contract,
        )
    ] == ["EV-ALLOWED"]


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


def test_explicit_url_allowlist_stops_at_json_escaped_newlines() -> None:
    contract = {
        "requirements": [
            {
                "requirement_id": "COV-01",
                "text": (
                    "严格仅使用以下 URL：\\n"
                    "1. https://peps.python.org/pep-0703/\\n"
                    "2. https://docs.python.org/3.13/whatsnew/3.13.html\\n"
                    "完成对比"
                ),
            }
        ]
    }

    scope = compile_source_scope(contract)

    assert scope.allowed_urls == frozenset(
        {
            "https://peps.python.org/pep-0703",
            "https://docs.python.org/3.13/whatsnew/3.13.html",
        }
    )
    assert (
        classify_evidence_source(
            {"source_url": "https://peps.python.org/pep-0703/"},
            contract,
        ).source_scope_status
        is SourceScopeStatus.IN_SCOPE
    )


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
