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
    _build_quality_model,
    _normalize_quality_payload,
    deterministic_handoff_checks,
    deterministic_tool_checks,
    evaluate_subagent_handoff,
    evaluate_tool_results,
)
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
