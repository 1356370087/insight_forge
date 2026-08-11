"""Tests for Qwen JSON runtime quality gates."""

import json
from types import SimpleNamespace

import pytest
from langchain_core.messages import ToolMessage

from open_deep_research.agents import deep_researcher
from open_deep_research.configuration import Configuration
from open_deep_research.quality import (
    TOOL_RESULT_EVALUATION_PROMPT,
    ToolResultAssessment,
    _bounded_evidence_records,
    _build_quality_model,
    _evaluate_json,
    _evidence_optional_requirement_ids,
    _normalize_quality_payload,
    _unwrap_single_key_schema_payload,
    deterministic_handoff_checks,
    deterministic_tool_checks,
    evaluate_subagent_handoff,
    evaluate_tool_results,
)
from open_deep_research.quality_contract import ResearchCoverageContract
from open_deep_research.tools.utils import get_notes_from_tool_calls


def test_quality_model_uses_json_mode_and_disables_thinking(monkeypatch) -> None:
    captured: dict = {}

    class FakeModel:
        def bind(self, **kwargs):
            captured["bind"] = kwargs
            return self

    def fake_init_chat_model(**kwargs):
        captured["init"] = kwargs
        return FakeModel()

    monkeypatch.setattr("open_deep_research.quality.init_chat_model", fake_init_chat_model)
    configurable = Configuration(
        quality_evaluation_model="openai:qwen3.7-plus",
        quality_evaluation_base_url="https://example.test/v1",
    )

    _build_quality_model(configurable, {"configurable": {}})

    assert captured["init"]["extra_body"] == {"enable_thinking": False}
    assert captured["bind"]["response_format"] == {"type": "json_object"}
    assert "JSON" in TOOL_RESULT_EVALUATION_PROMPT


def test_quality_model_enables_thinking_for_qwen_max_series(monkeypatch) -> None:
    captured: dict = {}

    class FakeModel:
        def bind(self, **kwargs):
            captured["bind"] = kwargs
            return self

    def fake_init_chat_model(**kwargs):
        captured["init"] = kwargs
        return FakeModel()

    monkeypatch.setattr("open_deep_research.quality.init_chat_model", fake_init_chat_model)
    configurable = Configuration(
        quality_evaluation_model="openai:qwen3.7-max-2026-05-17",
        quality_evaluation_base_url=(
            "https://workspace.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
        ),
    )

    _build_quality_model(configurable, {"configurable": {}})

    assert captured["init"]["extra_body"] == {
        "enable_thinking": True,
        "thinking_budget": configurable.quality_evaluation_model_max_tokens,
    }
    assert "max_tokens" not in captured["init"]
    assert captured["bind"]["response_format"] == {"type": "json_object"}


def test_deterministic_tool_checks_require_traceable_search_sources() -> None:
    passed = deterministic_tool_checks(
        [{
            "name": "tavily_search",
            "content": "Evidence https://a.example/x and https://b.example/y",
            "error": False,
        }],
        min_sources=2,
    )
    failed = deterministic_tool_checks(
        [{"name": "web_search", "content": "One source https://a.example/x", "error": False}],
        min_sources=2,
    )

    assert passed["passed"] is True
    assert failed["failures"] == ["insufficient_traceable_sources"]


def test_deterministic_tool_checks_use_cumulative_evidence_sources() -> None:
    checks = deterministic_tool_checks(
        [{
            "name": "web_research",
            "content": (
                '{"documents":[{"final_url":"https://a.example/paper"}],'
                '"evidence":[{"source_url":"https://a.example/paper"}]}'
            ),
            "error": False,
        }],
        min_sources=2,
        evidence_registry=[
            {
                "source_url": "https://a.example/paper",
                "security_status": "accepted",
            },
            {
                "source_url": "https://b.example/repository",
                "security_status": "accepted",
            },
        ],
    )

    assert checks["passed"] is True
    assert checks["source_count"] == 2


def test_deterministic_handoff_checks_reject_short_unsourced_output() -> None:
    checks = deterministic_handoff_checks(
        {"compressed_research": "too short", "raw_notes": []},
        min_sources=2,
    )

    assert checks["passed"] is False
    assert checks["failures"] == ["handoff_too_short", "insufficient_traceable_sources"]


def test_bounded_evidence_preserves_late_source_host_diversity() -> None:
    records = [
        {
            "evidence_id": f"docs-{index:03d}",
            "claim": f"Documentation claim {index}. " + ("d" * 420),
            "supporting_excerpt": "Official documentation excerpt. " + ("x" * 420),
            "source_url": f"https://docs.example.test/topic/{index}",
            "source_authority": 0.95,
            "confidence": 0.8,
            "security_status": "accepted",
        }
        for index in range(40)
    ]
    records.extend(
        {
            "evidence_id": f"reference-{index:03d}",
            "claim": f"API reference claim {index}. " + ("r" * 420),
            "supporting_excerpt": "Official API reference excerpt. " + ("y" * 420),
            "source_url": f"https://reference.example.test/api/{index}",
            "source_authority": 0.95,
            "confidence": 0.8,
            "security_status": "accepted",
        }
        for index in range(40)
    )
    records.append({
        "evidence_id": "github-late",
        "claim": "The official repository defines the checkpoint schema.",
        "supporting_excerpt": "class Checkpoint(TypedDict): ...",
        "source_url": (
            "https://github.com/example/project/blob/main/checkpoint.py"
        ),
        "source_authority": 0.9,
        "confidence": 0.8,
        "security_status": "accepted",
    })

    selected, stats = _bounded_evidence_records(
        records,
        max_chars=15_000,
    )
    selected_reversed, reversed_stats = _bounded_evidence_records(
        list(reversed(records)),
        max_chars=15_000,
    )
    selected_again, repeated_stats = _bounded_evidence_records(
        records,
        max_chars=15_000,
    )

    assert "github-late" in {
        str(record.get("evidence_id")) for record in selected
    }
    assert {
        record["source_url"].split("/", 3)[2] for record in selected
    } >= {
        "docs.example.test",
        "reference.example.test",
        "github.com",
    }
    assert selected_again == selected
    assert repeated_stats == stats
    assert "github-late" in {
        str(record.get("evidence_id")) for record in selected_reversed
    }
    assert {
        record["source_url"].split("/", 3)[2]
        for record in selected_reversed
    } >= {
        "docs.example.test",
        "reference.example.test",
        "github.com",
    }
    assert reversed_stats == stats
    assert len(json.dumps(selected, ensure_ascii=False, default=str)) <= 15_000
    assert stats["accepted_count"] == 81
    assert stats["unique_count"] == 81
    assert stats["included_count"] == len(selected)
    assert stats["truncated"] is True


def test_bounded_evidence_prioritizes_strong_records_and_filters_unsafe() -> None:
    records = [
        {
            "evidence_id": "weak-first",
            "claim": "Weak evidence. " + ("w" * 1_000),
            "supporting_excerpt": "Weak excerpt. " + ("w" * 1_000),
            "source_url": "https://same.example.test/weak",
            "source_authority": 0.2,
            "confidence": 0.2,
            "security_status": "accepted",
        },
        {
            "evidence_id": "strong-late",
            "claim": "Strong evidence. " + ("s" * 1_000),
            "supporting_excerpt": "Strong excerpt. " + ("s" * 1_000),
            "source_url": "https://same.example.test/strong",
            "source_authority": 0.99,
            "confidence": 0.99,
            "security_status": "accepted",
        },
        {
            "evidence_id": "unsafe",
            "claim": "This must never reach the evaluator.",
            "source_url": "https://unsafe.example.test/rejected",
            "source_authority": 1.0,
            "confidence": 1.0,
            "security_status": "quarantined",
        },
    ]

    selected, stats = _bounded_evidence_records(records, max_chars=2_500)

    assert [record["evidence_id"] for record in selected] == ["strong-late"]
    assert stats == {
        "accepted_count": 2,
        "unique_count": 2,
        "included_count": 1,
        "truncated": True,
    }


def test_bounded_evidence_prioritizes_handoff_citations() -> None:
    cited_quote = (
        "Skip scan works by generating a dynamic equality constraint "
        "internally for a leading index column."
    )
    records = [
        {
            "evidence_id": f"irrelevant-{index}",
            "claim": f"Unrelated high-authority claim {index}. " + ("x" * 300),
            "supporting_excerpt": "Unrelated excerpt. " + ("y" * 300),
            "source_url": "https://www.postgresql.org/docs/18/other.html",
            "source_authority": 1.0,
            "confidence": 1.0,
            "security_status": "accepted",
        }
        for index in range(20)
    ]
    records.append({
        "evidence_id": "cited-late",
        "claim": "The planner can generate an equality constraint.",
        "supporting_excerpt": cited_quote,
        "source_url": (
            "https://www.postgresql.org/docs/18/indexes-multicolumn.html"
        ),
        "source_authority": 0.5,
        "confidence": 0.7,
        "security_status": "accepted",
    })

    selected, stats = _bounded_evidence_records(
        records,
        max_chars=1_100,
        priority_text=f"The report quotes: {cited_quote}",
    )

    assert selected[0]["evidence_id"] == "cited-late"
    assert stats["priority_matched_count"] == 1
    assert stats["priority_included_count"] == 1


def test_bounded_evidence_prioritizes_explicit_ids_over_fuzzy_matches() -> None:
    cited = {
        "evidence_id": "ev_explicit_citation",
        "claim": "A short cited claim.",
        "supporting_excerpt": "A short cited excerpt.",
        "source_url": "https://example.com/cited",
        "source_authority": 0.5,
        "confidence": 0.7,
        "security_status": "accepted",
    }
    fuzzy = {
        "evidence_id": "ev_fuzzy_match",
        "claim": "Skip scan dynamic equality constraint planner conditions.",
        "supporting_excerpt": "Skip scan dynamic equality constraint.",
        "source_url": "https://example.com/fuzzy",
        "source_authority": 1.0,
        "confidence": 1.0,
        "security_status": "accepted",
    }
    cited_size = len(json.dumps(cited, ensure_ascii=False)) + 2

    selected, stats = _bounded_evidence_records(
        [fuzzy, cited],
        max_chars=cited_size,
        priority_text=(
            "Skip scan dynamic equality constraint "
            "[ev_explicit_citation]."
        ),
    )

    assert [record["evidence_id"] for record in selected] == [
        "ev_explicit_citation"
    ]
    assert stats["explicit_citation_count"] == 1
    assert stats["explicit_citation_included_count"] == 1


def test_bounded_evidence_fits_every_explicit_citation_in_tight_budget() -> None:
    records = [
        {
            "evidence_id": f"ev_cited_{index:02d}",
            "claim": f"Cited claim {index}. " + ("c" * 1_000),
            "supporting_excerpt": f"Quoted evidence {index}. " + ("e" * 2_000),
            "source_url": f"https://example.com/source-{index}",
            "source_title": "Official source " + ("t" * 400),
            "source_authority": 1.0,
            "confidence": 0.9,
            "security_status": "accepted",
        }
        for index in range(12)
    ]
    priority_text = " ".join(record["evidence_id"] for record in records)

    selected, stats = _bounded_evidence_records(
        records,
        max_chars=12_000,
        priority_text=priority_text,
    )

    assert {record["evidence_id"] for record in selected} == {
        record["evidence_id"] for record in records
    }
    assert stats["explicit_citation_count"] == 12
    assert stats["explicit_citation_included_count"] == 12
    assert len(json.dumps(selected, ensure_ascii=False)) <= 12_000


def test_parallel_and_source_contract_requirements_are_evidence_optional() -> None:
    contract = ResearchCoverageContract.model_validate({
        "schema_version": 1,
        "original_query_sha256": "a" * 64,
        "requirements": [
            {
                "requirement_id": "COV-01",
                "text": (
                    "严格仅使用以下 URL 作为证据：https://example.com/a；"
                    "至少拆分为两个并行 Subagent"
                ),
                "source_message_index": 0,
                "source_start": 0,
                "source_end": 80,
            },
            {
                "requirement_id": "COV-02",
                "text": "每个 Subagent 都必须交叉读取并引用全部 3 个 URL",
                "source_message_index": 0,
                "source_start": 81,
                "source_end": 130,
            },
            {
                "requirement_id": "COV-03",
                "text": "研究迁移风险",
                "source_message_index": 0,
                "source_start": 131,
                "source_end": 140,
            },
        ],
        "advisory_dimensions": [],
    })

    assert _evidence_optional_requirement_ids(contract) == (
        "COV-01",
        "COV-02",
    )


def test_bounded_evidence_omits_claim_duplicated_by_excerpt() -> None:
    selected, _stats = _bounded_evidence_records(
        [
            {
                "evidence_id": "ev_duplicate",
                "claim": "Skip scan uses a dynamic equality constraint.",
                "supporting_excerpt": (
                    "Skip scan uses a dynamic equality constraint. "
                    "The planner repeats the index search."
                ),
                "source_url": "https://example.com/source",
                "security_status": "accepted",
            }
        ],
        max_chars=2_000,
    )

    assert selected[0]["evidence_id"] == "ev_duplicate"
    assert "claim" not in selected[0]
    assert "dynamic equality constraint" in selected[0][
        "supporting_excerpt"
    ]


def test_compact_handoff_can_report_source_count_without_raw_notes() -> None:
    checks = deterministic_handoff_checks(
        {
            "compressed_research": "A detailed compressed finding. " * 10,
            "artifact_ref": {"path": "context/artifact.json", "sha256": "a" * 64},
            "metrics": {"sources_read": 3},
        },
        min_sources=2,
    )

    assert checks["passed"] is True
    assert checks["source_count"] == 3


def test_quality_payload_normalizes_cross_provider_score_variations() -> None:
    normalized = _normalize_quality_payload({
        "decision": " Continue ",
        "relevance": "5",
        "source_quality": 6,
        "evidence_coverage": 0,
        "corroboration": 3.6,
        "unresolved_conflicts": None,
        "missing_information": [],
        "suggested_queries": None,
        "reason": "Useful but incomplete.",
    })

    assessment = ToolResultAssessment.model_validate(normalized)

    assert assessment.decision == "continue"
    assert assessment.relevance == 5
    assert assessment.source_quality == 5
    assert assessment.evidence_coverage == 1
    assert assessment.corroboration == 4
    assert assessment.unresolved_conflicts == []
    assert assessment.suggested_queries == []


def test_quality_payload_conservatively_collapses_nested_corroboration() -> None:
    """Qwen-style score dimensions become one non-inflated scalar score."""
    normalized = _normalize_quality_payload({
        "decision": "continue",
        "relevance": 4,
        "source_quality": 4,
        "evidence_coverage": 3,
        "corroboration": {
            "rigor": 4,
            "independence": 3,
            "traceability": 4,
        },
        "unresolved_conflicts": [],
        "missing_information": ["One official source is still missing."],
        "suggested_queries": [],
        "reason": "Useful but incomplete.",
    })

    assessment = ToolResultAssessment.model_validate(normalized)

    assert assessment.corroboration == 3


def test_quality_payload_collapses_provider_named_score_dimensions() -> None:
    """Captured DeepSeek/Qwen-compatible dimensions remain a scalar score."""
    normalized = _normalize_quality_payload({
        "decision": "complete",
        "relevance": 5,
        "source_quality": 5,
        "evidence_coverage": 4,
        "corroboration": {
            "core_claims": 5,
            "syntax_details": 4,
            "limitations": 4,
            "comparison": 5,
        },
        "unresolved_conflicts": 0,
        "missing_information": [],
        "suggested_queries": [],
        "reason": "The evidence is complete.",
    })

    assessment = ToolResultAssessment.model_validate(normalized)

    assert assessment.corroboration == 4
    assert assessment.unresolved_conflicts == []


def test_quality_payload_preserves_provider_reported_gap_counts() -> None:
    """Captured DeepSeek count-only gaps remain conservative quality signals."""
    normalized = _normalize_quality_payload({
        "decision": "continue",
        "relevance": 4,
        "source_quality": 2,
        "evidence_coverage": 2,
        "corroboration": 3,
        "unresolved_conflicts": 3,
        "missing_information": 2,
        "suggested_queries": ["PostgreSQL 18 official asynchronous I/O docs"],
        "reason": "More official evidence is required.",
    })

    assessment = ToolResultAssessment.model_validate(normalized)

    assert assessment.unresolved_conflicts == [
        "Provider reported 3 unresolved conflicts without itemized details."
    ]
    assert assessment.missing_information == [
        "Provider reported 2 missing information items without itemized details."
    ]


def test_quality_payload_preserves_numeric_gaps_despite_no_gap_reason() -> None:
    """Contradictory provider fields must retain the conservative gap signal."""
    normalized = _normalize_quality_payload({
        "decision": "complete",
        "relevance": 5,
        "source_quality": 5,
        "evidence_coverage": 5,
        "corroboration": 4,
        "unresolved_conflicts": 5,
        "missing_information": 5,
        "suggested_queries": [],
        "reason": (
            "All owned requirements are fully satisfied, with no unresolved "
            "gaps or conflicts."
        ),
    })

    assessment = ToolResultAssessment.model_validate(normalized)

    assert assessment.decision == "complete"
    assert assessment.unresolved_conflicts == [
        "Provider reported 5 unresolved conflicts without itemized details."
    ]
    assert assessment.missing_information == [
        "Provider reported 5 missing information items without itemized details."
    ]


def test_quality_payload_wraps_provider_single_gap_strings() -> None:
    """Captured DeepSeek singleton gap strings satisfy the requested list shape."""
    normalized = _normalize_quality_payload({
        "decision": "continue",
        "relevance": 4,
        "source_quality": 3,
        "evidence_coverage": 2,
        "corroboration": 4,
        "unresolved_conflicts": "Official-source admissibility remains unresolved.",
        "missing_information": "A PostgreSQL 17 baseline is still missing.",
        "suggested_queries": "site:postgresql.org/docs/17 generated columns",
        "reason": "More official evidence is required.",
    })

    assessment = ToolResultAssessment.model_validate(normalized)

    assert assessment.unresolved_conflicts == [
        "Official-source admissibility remains unresolved."
    ]
    assert assessment.missing_information == [
        "A PostgreSQL 17 baseline is still missing."
    ]
    assert assessment.suggested_queries == [
        "site:postgresql.org/docs/17 generated columns"
    ]


@pytest.mark.asyncio
async def test_runtime_judge_repairs_strict_single_key_schema_wrapper(
    monkeypatch,
) -> None:
    wrapped = {
        "assessment": {
            "decision": "complete",
            "relevance": 5,
            "source_quality": 5,
            "evidence_coverage": 5,
            "corroboration": 5,
            "unresolved_conflicts": [],
            "missing_information": [],
            "suggested_queries": [],
            "reason": "The evidence is complete.",
        }
    }

    async def fake_invoke(*_args, **_kwargs):
        return SimpleNamespace(content=json.dumps(wrapped))

    monkeypatch.setattr(
        "open_deep_research.quality._build_quality_model",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        "open_deep_research.quality.invoke_model_with_retry_observability",
        fake_invoke,
    )

    result = await _evaluate_json(
        ToolResultAssessment,
        (
            "Return decision, relevance, source_quality, evidence_coverage, "
            "corroboration, unresolved_conflicts, missing_information, "
            "suggested_queries, and reason as JSON."
        ),
        {"payload": "value"},
        {"configurable": {}, "metadata": {"run_id": "wrapped-runtime"}},
        span_name="quality.wrapper_test",
    )

    assert isinstance(result, ToolResultAssessment)
    assert result.decision == "complete"
    assert result.reason == "The evidence is complete."


def test_runtime_judge_does_not_unwrap_incomplete_or_ambiguous_payload() -> None:
    incomplete = {
        "assessment": {
            "decision": "complete",
            "relevance": 5,
            "source_quality": 5,
            "evidence_coverage": 5,
            "corroboration": 5,
            "reason": "Complete.",
        }
    }
    ambiguous = {
        "assessment": {
            "decision": "complete",
            "relevance": 5,
            "source_quality": 5,
            "evidence_coverage": 5,
            "corroboration": 5,
            "reason": "Complete.",
        },
        "other": "value",
    }

    assert (
        _unwrap_single_key_schema_payload(ToolResultAssessment, incomplete)
        is incomplete
    )
    assert (
        _unwrap_single_key_schema_payload(
            ToolResultAssessment,
            incomplete,
            expected_fields={
                "decision",
                "relevance",
                "source_quality",
                "evidence_coverage",
                "corroboration",
                "unresolved_conflicts",
                "missing_information",
                "suggested_queries",
                "reason",
            },
        )
        is incomplete
    )
    assert (
        _unwrap_single_key_schema_payload(ToolResultAssessment, ambiguous)
        is ambiguous
    )


def test_final_notes_include_only_accepted_research_handoffs() -> None:
    messages = [
        ToolMessage(content="planning", name="think_tool", tool_call_id="think-1"),
        ToolMessage(
            content='{"status":"rejected_by_supervisor_quality_gate"}',
            name="ConductResearch",
            tool_call_id="research-1",
        ),
        ToolMessage(
            content="accepted evidence https://primary.example/paper",
            name="ConductResearch",
            tool_call_id="research-2",
        ),
        ToolMessage(
            content='{"content":"selected evidence https://primary.example/detail"}',
            name="ReadResearchArtifact",
            tool_call_id="artifact-1",
        ),
        ToolMessage(
            content='{"error_type":"validation_error"}',
            name="ReadResearchArtifact",
            tool_call_id="artifact-2",
        ),
    ]

    assert get_notes_from_tool_calls(messages) == [
        "accepted evidence https://primary.example/paper",
        '{"content":"selected evidence https://primary.example/detail"}',
    ]


@pytest.mark.asyncio
async def test_assessment_node_routes_retry_back_to_researcher(monkeypatch) -> None:
    captured: dict = {}

    async def fake_evaluate(*_args, **_kwargs):
        captured["evidence_registry"] = _kwargs["evidence_registry"]
        return ToolResultAssessment(
            decision="retry",
            relevance=4,
            source_quality=4,
            evidence_coverage=2,
            corroboration=2,
            missing_information=["independent confirmation"],
            suggested_queries=["official confirmation"],
            reason="More evidence is needed.",
        )

    monkeypatch.setattr(deep_researcher, "evaluate_tool_results", fake_evaluate)
    state = {
        "research_topic": "topic",
        "tool_call_iterations": 1,
        "pending_tool_results": [{
            "name": "web_search",
            "content": "Evidence https://a.example and https://b.example",
            "error": False,
        }],
        "evidence_registry": [{
            "claim": "Cumulative evidence",
            "source_url": "https://a.example",
        }],
    }

    command = await deep_researcher.assess_research_results(
        state,
        {"configurable": {"max_react_tool_calls": 10}},
    )

    assert command.goto == "researcher"
    assert command.update["result_assessment"]["decision"] == "retry"
    assert "assessment JSON" in command.update["researcher_messages"][0].content
    assert captured["evidence_registry"] == state["evidence_registry"]


@pytest.mark.asyncio
async def test_assessment_node_routes_complete_to_compression(monkeypatch) -> None:
    async def fake_evaluate(*_args, **_kwargs):
        return ToolResultAssessment(
            decision="complete",
            relevance=5,
            source_quality=4,
            evidence_coverage=4,
            corroboration=4,
            reason="Evidence is sufficient.",
        )

    monkeypatch.setattr(deep_researcher, "evaluate_tool_results", fake_evaluate)
    state = {
        "research_topic": "topic",
        "tool_call_iterations": 1,
        "pending_tool_results": [{
            "name": "web_search",
            "content": "Evidence https://a.example and https://b.example",
            "error": False,
        }],
    }

    command = await deep_researcher.assess_research_results(
        state,
        {"configurable": {"max_react_tool_calls": 10}},
    )

    assert command.goto == "compress_research"


def _protocol_config(*, fail_open: bool) -> dict:
    return {
        "configurable": {
            "quality_evaluation_model": "openai:qwen3.7-max",
            "quality_evaluation_fail_open": fail_open,
            "quality_evaluation_min_score": 3,
            "quality_evaluation_min_sources": 2,
        },
        "metadata": {
            "runtime_config_frozen": True,
            "run_id": "quality-protocol-test",
            "quality_policy_version": "quality-gate-v2",
            "quality_evaluation_epoch": "epoch-17",
        },
    }


def _traceable_results() -> list[dict]:
    return [{
        "name": "tavily_search",
        "content": "https://a.example/source https://b.example/source",
        "error": False,
    }]


def _contradictory_retry() -> dict:
    return {
        "decision": "retry",
        "relevance": 5,
        "source_quality": 5,
        "evidence_coverage": 5,
        "corroboration": 5,
        "unresolved_conflicts": [],
        "missing_information": [],
        "suggested_queries": [],
        "reason": "Everything is fully complete; no more work is needed.",
    }


@pytest.mark.asyncio
async def test_contradictory_retry_is_repaired_once(monkeypatch) -> None:
    responses = [
        _contradictory_retry(),
        {
            **_contradictory_retry(),
            "decision": "complete",
            "reason": "All requirements are satisfied.",
        },
    ]
    calls: list[list] = []

    async def fake_invoke(_model, messages, *_args, **_kwargs):
        calls.append(messages)
        return SimpleNamespace(content=json.dumps(responses.pop(0)))

    monkeypatch.setattr(
        "open_deep_research.quality._build_quality_model",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        "open_deep_research.quality.invoke_model_with_retry_observability",
        fake_invoke,
    )

    result = await evaluate_tool_results(
        "Synthetic topic",
        _traceable_results(),
        _protocol_config(fail_open=False),
    )

    assert result.decision == "complete"
    assert result.protocol_repair_count == 1
    assert len(calls) == 2
    assert "retry_or_continue_requires_gap_or_action" in calls[1][-1].content
    assert result.evaluator_model == "openai:qwen3.7-max"
    assert result.policy_version == "quality-gate-v2"
    assert result.evaluation_epoch == "epoch-17"


@pytest.mark.asyncio
async def test_second_protocol_contradiction_uses_fail_open(monkeypatch) -> None:
    async def fake_invoke(*_args, **_kwargs):
        return SimpleNamespace(content=json.dumps(_contradictory_retry()))

    monkeypatch.setattr(
        "open_deep_research.quality._build_quality_model",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        "open_deep_research.quality.invoke_model_with_retry_observability",
        fake_invoke,
    )

    result = await evaluate_tool_results(
        "Synthetic topic",
        _traceable_results(),
        _protocol_config(fail_open=True),
    )

    assert result.decision == "continue"
    assert result.evaluator_error.startswith("quality_protocol_error:")
    assert result.protocol_errors == [
        "retry_or_continue_requires_gap_or_action"
    ]


@pytest.mark.asyncio
async def test_second_protocol_contradiction_fail_closed_stops_spending(
    monkeypatch,
) -> None:
    async def fake_invoke(*_args, **_kwargs):
        return SimpleNamespace(content=json.dumps(_contradictory_retry()))

    monkeypatch.setattr(
        "open_deep_research.quality._build_quality_model",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        "open_deep_research.quality.invoke_model_with_retry_observability",
        fake_invoke,
    )

    result = await evaluate_tool_results(
        "Synthetic topic",
        _traceable_results(),
        _protocol_config(fail_open=False),
    )

    assert result.decision == "complete"
    assert result.missing_information == ["quality_evaluator_unavailable"]
    assert result.evaluator_error.startswith("quality_protocol_error:")


@pytest.mark.asyncio
async def test_handoff_timeout_fail_closed_returns_structured_rejection(
    monkeypatch,
) -> None:
    async def fake_invoke(*_args, **_kwargs):
        raise TimeoutError("judge timed out")

    monkeypatch.setattr(
        "open_deep_research.quality._build_quality_model",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        "open_deep_research.quality.invoke_model_with_retry_observability",
        fake_invoke,
    )
    handoff = {
        "compressed_research": (
            "Detailed evidence from https://a.example/source and "
            "https://b.example/source. "
        )
        * 6,
        "raw_notes": [],
    }

    result = await evaluate_subagent_handoff(
        "Synthetic topic",
        handoff,
        _protocol_config(fail_open=False),
    )

    assert result.accepted is False
    assert result.missing_information == ["quality_evaluator_unavailable"]
    assert result.follow_up_tasks == ["reassess_sha_verified_artifact"]
    assert result.evaluator_error == "judge timed out"
    assert result.evaluator_model == "openai:qwen3.7-max"
