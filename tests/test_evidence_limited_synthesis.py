from __future__ import annotations

import json

import pytest
from langchain_core.messages import AIMessage

from open_deep_research.report import evidence_synthesis
from open_deep_research.report.evidence_synthesis import (
    EvidenceBoundClaim,
    EvidenceSynthesisDraft,
    EvidenceSynthesisGroundingAssessment,
    EvidenceSynthesisSection,
    build_evidence_limited_report,
)


def _contract() -> dict:
    return {
        "schema_version": 1,
        "original_query_sha256": "a" * 64,
        "requirements": [
            {
                "requirement_id": "COV-01",
                "text": "说明检查点如何恢复",
                "source_message_index": 0,
                "source_start": 0,
                "source_end": 9,
            }
        ],
        "advisory_dimensions": [],
    }


def _evidence() -> list[dict]:
    return [
        {
            "evidence_id": "EV-01",
            "claim": "检查点可保存待恢复状态。",
            "supporting_excerpt": "Checkpoint saves pending state.",
            "source_url": "https://example.com/checkpoint",
            "source_title": "Checkpoint docs",
            "security_status": "accepted",
        }
    ]


@pytest.mark.asyncio
async def test_draft_invocation_includes_complete_json_schema(
    monkeypatch,
) -> None:
    from open_deep_research.agents import deep_researcher

    captured: dict = {}

    class FakeConfigurableModel:
        def with_config(self, model_config):
            captured["model_config"] = model_config
            return object()

    async def fake_invoke(
        _model,
        messages,
        _config,
        **_kwargs,
    ):
        captured["messages"] = messages
        return AIMessage(
            content=json.dumps(
                {
                    "title": "部分研究报告",
                    "summary": "检查点可保存待恢复状态。",
                    "summary_evidence_ids": ["EV-01"],
                    "sections": [],
                    "unresolved_requirements": [],
                },
                ensure_ascii=False,
            )
        )

    monkeypatch.setattr(
        deep_researcher,
        "configurable_model",
        FakeConfigurableModel(),
    )
    monkeypatch.setattr(
        evidence_synthesis,
        "invoke_model_with_retry_observability",
        fake_invoke,
    )

    draft = await evidence_synthesis._invoke_draft(
        {
            "eligible_evidence": _evidence(),
            "allowed_source_urls": ["https://example.com/checkpoint"],
        },
        {"configurable": {}, "metadata": {}},
    )

    system_prompt = str(captured["messages"][0].content)
    expected_schema = json.dumps(
        EvidenceSynthesisDraft.model_json_schema(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    assert draft.summary_evidence_ids == ["EV-01"]
    assert expected_schema in system_prompt


@pytest.mark.asyncio
async def test_evidence_limited_writer_sees_only_allowlisted_inputs(
    monkeypatch,
) -> None:
    captured: dict = {}

    async def fake_draft(payload, _config):
        captured.update(payload)
        return EvidenceSynthesisDraft(
            title="部分研究报告",
            summary="检查点可保存待恢复状态。",
            summary_evidence_ids=["EV-01"],
            sections=[
                EvidenceSynthesisSection(
                    heading="恢复机制",
                    claims=[
                        EvidenceBoundClaim(
                            text="检查点可保存待恢复状态。",
                            evidence_ids=["EV-01"],
                        )
                    ],
                )
            ],
            unresolved_requirements=[],
        )

    async def fake_grounding(_payload, _config):
        return EvidenceSynthesisGroundingAssessment(
            supported=True,
            reason="supported",
        )

    monkeypatch.setattr(evidence_synthesis, "_invoke_draft", fake_draft)
    monkeypatch.setattr(
        evidence_synthesis,
        "_invoke_grounding_judge",
        fake_grounding,
    )

    report = await build_evidence_limited_report(
        _evidence(),
        coverage_contract=_contract(),
        coverage_ledger={
            "COV-01": {
                "status": "supported",
                "evidence_ids": ["EV-01"],
            }
        },
        caveats=["未确认精确版本"],
        uncovered_requirement_ids=[],
        rejection_reasons=["REJECTED_SUMMARY_SENTINEL"],
        artifact_refs=[],
        config={"configurable": {}, "metadata": {}},
    )

    assert "部分研究报告" in report
    assert "EV-01" in report
    assert "REJECTED_SUMMARY_SENTINEL" not in json.dumps(
        captured,
        ensure_ascii=False,
    )
    assert set(captured) == {
        "coverage_contract",
        "eligible_evidence",
        "requirement_to_evidence",
        "caveats",
        "uncovered_requirement_ids",
        "allowed_source_urls",
    }


@pytest.mark.asyncio
async def test_large_evidence_set_is_budgeted_and_source_diverse(
    monkeypatch,
) -> None:
    captured: dict = {}
    evidence = [
        {
            "evidence_id": f"EV-{index:03d}",
            "claim": f"Claim {index} " + ("x" * 300),
            "supporting_excerpt": f"Excerpt {index} " + ("y" * 500),
            "source_url": f"https://source-{index % 8}.example/doc/{index}",
            "source_title": f"Source {index % 8}",
            "source_authority": 0.9,
            "confidence": 0.8,
            "security_status": "accepted",
        }
        for index in range(400)
    ]

    async def fake_draft(payload, _config):
        captured.update(payload)
        selected_id = payload["eligible_evidence"][0]["evidence_id"]
        return EvidenceSynthesisDraft(
            title="部分研究报告",
            summary="预算内证据综合。",
            summary_evidence_ids=[selected_id],
            sections=[],
            unresolved_requirements=["COV-01"],
        )

    async def fake_grounding(_payload, _config):
        return EvidenceSynthesisGroundingAssessment(
            supported=True,
            reason="supported",
        )

    monkeypatch.setattr(evidence_synthesis, "_invoke_draft", fake_draft)
    monkeypatch.setattr(
        evidence_synthesis,
        "_invoke_grounding_judge",
        fake_grounding,
    )

    report = await build_evidence_limited_report(
        evidence,
        coverage_contract=_contract(),
        coverage_ledger={},
        caveats=[],
        uncovered_requirement_ids=["COV-01"],
        rejection_reasons=[],
        artifact_refs=[],
        config={
            "configurable": {
                "unknown_model_context_window_tokens": 32768,
                "final_report_model_max_tokens": 10000,
            },
            "metadata": {},
        },
    )

    selected = captured["eligible_evidence"]
    assert report.startswith("# 部分研究报告")
    assert 1 < len(selected) < len(evidence)
    assert len(json.dumps(captured, ensure_ascii=False)) < 70_000
    assert len({item["source_url"].split("/", 3)[2] for item in selected}) >= 4


@pytest.mark.asyncio
async def test_official_only_contract_filters_partial_synthesis_sources(
    monkeypatch,
) -> None:
    captured: dict = {}
    contract = _contract()
    contract["requirements"][0]["text"] = (
        "Based solely on LangGraph official documentation and the official "
        "repository."
    )
    evidence = [
        {
            **_evidence()[0],
            "source_url": (
                "https://docs.langchain.com/oss/python/langgraph/persistence"
            ),
            "source_authority": 0.95,
        },
        {
            "evidence_id": "EV-02",
            "claim": "Third-party interpretation.",
            "supporting_excerpt": "Unofficial commentary.",
            "source_url": "https://blog.example.net/checkpoint",
            "source_title": "Community blog",
            "source_authority": 0.7,
            "security_status": "accepted",
        },
        {
            "evidence_id": "EV-03",
            "claim": "Issue discussion.",
            "supporting_excerpt": "A community member reported a behavior.",
            "source_url": (
                "https://github.com/langchain-ai/langgraph/issues/6626"
            ),
            "source_title": "GitHub issue",
            "source_authority": 1.0,
            "security_status": "accepted",
        },
        {
            "evidence_id": "EV-04",
            "claim": "Temporary preview documentation.",
            "supporting_excerpt": "Preview content.",
            "source_url": (
                "https://langchain-5e9cc07a.mintlify.app/"
                "oss/python/langgraph/persistence"
            ),
            "source_title": "Temporary Mintlify preview",
            "source_authority": 1.0,
            "security_status": "accepted",
        },
    ]

    async def fake_draft(payload, _config):
        captured.update(payload)
        return EvidenceSynthesisDraft(
            title="部分研究报告",
            summary="仅使用官方证据。",
            summary_evidence_ids=["EV-01"],
            sections=[],
            unresolved_requirements=[],
        )

    async def fake_grounding(_payload, _config):
        return EvidenceSynthesisGroundingAssessment(
            supported=True,
            reason="supported",
        )

    monkeypatch.setattr(evidence_synthesis, "_invoke_draft", fake_draft)
    monkeypatch.setattr(
        evidence_synthesis,
        "_invoke_grounding_judge",
        fake_grounding,
    )

    report = await build_evidence_limited_report(
        evidence,
        coverage_contract=contract,
        coverage_ledger={},
        caveats=[],
        uncovered_requirement_ids=[],
        rejection_reasons=[],
        artifact_refs=[],
        config={"configurable": {}, "metadata": {}},
    )

    assert report.startswith("# 部分研究报告")
    assert [item["evidence_id"] for item in captured["eligible_evidence"]] == [
        "EV-01"
    ]
    assert captured["allowed_source_urls"] == [
        "https://docs.langchain.com/oss/python/langgraph/persistence"
    ]
    assert captured["eligible_evidence"][0]["source_kind"] == (
        "first_party_docs"
    )
    assert captured["eligible_evidence"][0]["source_scope_status"] == "in_scope"


@pytest.mark.asyncio
async def test_unknown_evidence_id_forces_deterministic_fallback(
    monkeypatch,
) -> None:
    async def fake_draft(_payload, _config):
        return EvidenceSynthesisDraft(
            title="越权草稿",
            summary="未知事实",
            summary_evidence_ids=["EV-UNKNOWN"],
            sections=[],
            unresolved_requirements=["COV-01"],
        )

    async def grounding_must_not_run(*_args):
        raise AssertionError("invalid draft reached grounding judge")

    monkeypatch.setattr(evidence_synthesis, "_invoke_draft", fake_draft)
    monkeypatch.setattr(
        evidence_synthesis,
        "_invoke_grounding_judge",
        grounding_must_not_run,
    )

    report = await build_evidence_limited_report(
        _evidence(),
        coverage_contract=_contract(),
        coverage_ledger={},
        caveats=[],
        uncovered_requirement_ids=["COV-01"],
        rejection_reasons=["quality gate rejected"],
        artifact_refs=[],
        config={"configurable": {}, "metadata": {}},
    )

    assert report.startswith("# 质量门禁未通过的证据恢复报告")
    assert "EV-UNKNOWN" not in report


@pytest.mark.asyncio
async def test_claim_without_evidence_forces_deterministic_fallback(
    monkeypatch,
) -> None:
    from open_deep_research.agents import deep_researcher

    class FakeConfigurableModel:
        def with_config(self, _model_config):
            return object()

    async def fake_invoke(
        _model,
        _messages,
        _config,
        **_kwargs,
    ):
        return AIMessage(
            content=json.dumps(
                {
                    "title": "无证据草稿",
                    "summary": "检查点可保存待恢复状态。",
                    "summary_evidence_ids": ["EV-01"],
                    "sections": [
                        {
                            "heading": "恢复机制",
                            "claims": [
                                {
                                    "text": "没有证据绑定的事实主张。",
                                    "evidence_ids": [],
                                    "qualification": None,
                                }
                            ],
                        }
                    ],
                    "unresolved_requirements": [],
                },
                ensure_ascii=False,
            )
        )

    async def grounding_must_not_run(*_args):
        raise AssertionError("invalid draft reached grounding judge")

    monkeypatch.setattr(
        deep_researcher,
        "configurable_model",
        FakeConfigurableModel(),
    )
    monkeypatch.setattr(
        evidence_synthesis,
        "invoke_model_with_retry_observability",
        fake_invoke,
    )
    monkeypatch.setattr(
        evidence_synthesis,
        "_invoke_grounding_judge",
        grounding_must_not_run,
    )

    report = await build_evidence_limited_report(
        _evidence(),
        coverage_contract=_contract(),
        coverage_ledger={},
        caveats=[],
        uncovered_requirement_ids=[],
        rejection_reasons=["quality gate rejected"],
        artifact_refs=[],
        config={"configurable": {}, "metadata": {}},
    )

    assert report.startswith("# 质量门禁未通过的证据恢复报告")
    assert "没有证据绑定的事实主张" not in report


@pytest.mark.asyncio
async def test_source_title_cannot_inject_non_allowlisted_markdown_url(
    monkeypatch,
) -> None:
    async def fake_draft(_payload, _config):
        return EvidenceSynthesisDraft(
            title="部分研究报告",
            summary="检查点可保存待恢复状态。",
            summary_evidence_ids=["EV-01"],
            sections=[],
            unresolved_requirements=[],
        )

    async def fake_grounding(_payload, _config):
        return EvidenceSynthesisGroundingAssessment(
            supported=True,
            reason="supported",
        )

    monkeypatch.setattr(evidence_synthesis, "_invoke_draft", fake_draft)
    monkeypatch.setattr(
        evidence_synthesis,
        "_invoke_grounding_judge",
        fake_grounding,
    )
    evidence = _evidence()
    evidence[0]["source_title"] = "safe](https://evil.example)[x"

    report = await build_evidence_limited_report(
        evidence,
        coverage_contract=_contract(),
        coverage_ledger={
            "COV-01": {
                "status": "supported",
                "evidence_ids": ["EV-01"],
            }
        },
        caveats=[],
        uncovered_requirement_ids=[],
        rejection_reasons=[],
        artifact_refs=[],
        config={"configurable": {}, "metadata": {}},
    )

    assert report.startswith("# 部分研究报告")
    assert "https://evil.example" not in report
    assert "(https://example.com/checkpoint)" in report


@pytest.mark.asyncio
async def test_rendered_report_url_is_rechecked_against_allowlist(
    monkeypatch,
) -> None:
    async def fake_draft(_payload, _config):
        return EvidenceSynthesisDraft(
            title="部分研究报告",
            summary="检查点可保存待恢复状态。",
            summary_evidence_ids=["EV-01"],
            sections=[],
            unresolved_requirements=[],
        )

    async def fake_grounding(_payload, _config):
        return EvidenceSynthesisGroundingAssessment(
            supported=True,
            reason="supported",
        )

    monkeypatch.setattr(evidence_synthesis, "_invoke_draft", fake_draft)
    monkeypatch.setattr(
        evidence_synthesis,
        "_invoke_grounding_judge",
        fake_grounding,
    )
    monkeypatch.setattr(
        evidence_synthesis,
        "_render_draft",
        lambda *_args: "# 部分研究报告\n\nhttps://evil.example",
    )
    evidence = _evidence()
    evidence[0]["source_title"] = "safe](https://evil.example)[x"

    report = await build_evidence_limited_report(
        evidence,
        coverage_contract=_contract(),
        coverage_ledger={},
        caveats=[],
        uncovered_requirement_ids=[],
        rejection_reasons=[],
        artifact_refs=[],
        config={"configurable": {}, "metadata": {}},
    )

    assert report.startswith("# 质量门禁未通过的证据恢复报告")
    assert "https://evil.example" not in report
    assert "(https://example.com/checkpoint)" in report
