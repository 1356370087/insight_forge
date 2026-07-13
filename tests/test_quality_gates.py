"""Tests for Qwen JSON runtime quality gates."""

import pytest
from langchain_core.messages import ToolMessage

from open_deep_research.agents import deep_researcher
from open_deep_research.configuration import Configuration
from open_deep_research.quality import (
    TOOL_RESULT_EVALUATION_PROMPT,
    ToolResultAssessment,
    _build_quality_model,
    deterministic_handoff_checks,
    deterministic_tool_checks,
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


def test_deterministic_handoff_checks_reject_short_unsourced_output() -> None:
    checks = deterministic_handoff_checks(
        {"compressed_research": "too short", "raw_notes": []},
        min_sources=2,
    )

    assert checks["passed"] is False
    assert checks["failures"] == ["handoff_too_short", "insufficient_traceable_sources"]


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
    ]

    assert get_notes_from_tool_calls(messages) == [
        "accepted evidence https://primary.example/paper"
    ]


@pytest.mark.asyncio
async def test_assessment_node_routes_retry_back_to_researcher(monkeypatch) -> None:
    async def fake_evaluate(*_args, **_kwargs):
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
    }

    command = await deep_researcher.assess_research_results(
        state,
        {"configurable": {"max_react_tool_calls": 10}},
    )

    assert command.goto == "researcher"
    assert command.update["result_assessment"]["decision"] == "retry"
    assert "assessment JSON" in command.update["researcher_messages"][0].content


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
