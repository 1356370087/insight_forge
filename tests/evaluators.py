"""LangSmith evaluators for the current QueryEngine state contract."""

from __future__ import annotations

import json
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any, cast

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from tests.prompts import (
    CITATION_ACCURACY_PROMPT,
    COMPLETENESS_PROMPT,
    CORRECTNESS_PROMPT,
    GROUNDEDNESS_PROMPT,
    OVERALL_QUALITY_PROMPT,
    RELEVANCE_PROMPT,
    STRUCTURE_PROMPT,
    TOOL_EFFICIENCY_PROMPT,
)

load_dotenv(Path(__file__).resolve().parents[1] / ".env")


@lru_cache(maxsize=1)
def _get_eval_model() -> ChatOpenAI:
    """Build the judge lazily so importing tests does not require credentials."""
    return ChatOpenAI(model="gpt-4.1")


def _today() -> str:
    return date.today().isoformat()


def _format_input_query(inputs: dict[str, Any]) -> str:
    messages = inputs.get("messages", [])
    if len(messages) == 1:
        return str(messages[0].get("content", ""))

    role_tags = {
        "user": "user_input",
        "assistant": "assistant_follow_up",
    }
    formatted = []
    for message in messages:
        role = str(message.get("role", "user"))
        tag = role_tags.get(role, role)
        formatted.append(f"<{tag}>\n{message.get('content', '')}\n</{tag}>")
    return "\n\n".join(formatted)


def _failure_reason(outputs: dict[str, Any]) -> str | None:
    result = outputs.get("result")
    if isinstance(result, dict) and result.get("status") in {"error", "cancelled"}:
        return str(result.get("error") or f"run status: {result.get('status')}")
    report = outputs.get("final_report")
    if not isinstance(report, str) or not report.strip():
        return "run produced no final_report"
    if report.startswith("Error generating final report:"):
        return report
    return None


def _zero(key: str, reason: str) -> dict[str, Any]:
    return {"key": key, "score": 0.0, "comment": f"Not scored: {reason}"}


def _judge_input(content: str) -> str | list[dict[str, Any]]:
    model = _get_eval_model()
    if isinstance(model, ChatAnthropic):
        return [{"type": "text", "text": content, "cache_control": {"type": "ephemeral", "ttl": "1h"}}]
    return content


class OverallQualityScore(BaseModel):
    research_depth: int = Field(ge=1, le=5)
    source_quality: int = Field(ge=1, le=5)
    analytical_rigor: int = Field(ge=1, le=5)
    practical_value: int = Field(ge=1, le=5)
    balance_and_objectivity: int = Field(ge=1, le=5)
    writing_quality: int = Field(ge=1, le=5)


def eval_overall_quality(inputs: dict, outputs: dict) -> list[dict[str, Any]]:
    keys = [
        "research_depth_score",
        "source_quality_score",
        "analytical_rigor_score",
        "practical_value_score",
        "balance_and_objectivity_score",
        "writing_quality_score",
    ]
    if reason := _failure_reason(outputs):
        return [_zero(key, reason) for key in keys]

    content = (
        f"User input: {_format_input_query(inputs)}\n\n"
        f"Report:\n\n{outputs['final_report']}\n\n"
        "Evaluate whether the report meets the criteria."
    )
    model = _get_eval_model()
    result = cast(
        OverallQualityScore,
        model.with_structured_output(OverallQualityScore).invoke(
            [
                {"role": "system", "content": OVERALL_QUALITY_PROMPT.format(today=_today())},
                {"role": "user", "content": _judge_input(content)},
            ]
        ),
    )
    values = [
        result.research_depth,
        result.source_quality,
        result.analytical_rigor,
        result.practical_value,
        result.balance_and_objectivity,
        result.writing_quality,
    ]
    return [{"key": key, "score": value / 5} for key, value in zip(keys, values, strict=True)]


class RelevanceScore(BaseModel):
    reasoning: str
    score: int = Field(ge=1, le=5)


def eval_relevance(inputs: dict, outputs: dict) -> dict[str, Any]:
    if reason := _failure_reason(outputs):
        return _zero("relevance_score", reason)
    content = f"User input: {_format_input_query(inputs)}\n\nReport:\n\n{outputs['final_report']}"
    model = _get_eval_model()
    result = cast(
        RelevanceScore,
        model.with_structured_output(RelevanceScore).invoke(
            [
                {"role": "system", "content": RELEVANCE_PROMPT.format(today=_today())},
                {"role": "user", "content": _judge_input(content)},
            ]
        ),
    )
    return {"key": "relevance_score", "score": result.score / 5, "comment": result.reasoning}


class StructureScore(BaseModel):
    reasoning: str
    score: int = Field(ge=1, le=5)


def eval_structure(inputs: dict, outputs: dict) -> dict[str, Any]:
    if reason := _failure_reason(outputs):
        return _zero("structure_and_cohesiveness_score", reason)
    content = STRUCTURE_PROMPT.format(
        user_question=_format_input_query(inputs),
        report=outputs["final_report"],
        today=_today(),
    )
    result = cast(
        StructureScore,
        _get_eval_model().with_structured_output(StructureScore).invoke(
            [{"role": "user", "content": _judge_input(content)}]
        ),
    )
    return {
        "key": "structure_and_cohesiveness_score",
        "score": result.score / 5,
        "comment": result.reasoning,
    }


class CorrectnessScore(BaseModel):
    reasoning: str
    score: int = Field(ge=1, le=5)


def eval_correctness(inputs: dict, outputs: dict, reference_outputs: dict) -> dict[str, Any]:
    if reason := _failure_reason(outputs):
        return _zero("correctness_score", reason)
    answer = reference_outputs.get("answer")
    if not answer:
        return _zero("correctness_score", "reference output has no answer")
    content = CORRECTNESS_PROMPT.format(
        user_question=_format_input_query(inputs),
        report=outputs["final_report"],
        answer=answer,
        today=_today(),
    )
    result = cast(
        CorrectnessScore,
        _get_eval_model().with_structured_output(CorrectnessScore).invoke(
            [{"role": "user", "content": _judge_input(content)}]
        ),
    )
    return {"key": "correctness_score", "score": result.score / 5, "comment": result.reasoning}


class GroundednessClaim(BaseModel):
    claim: str
    grounded: bool


class GroundednessScore(BaseModel):
    claims: list[GroundednessClaim]


def eval_groundedness(inputs: dict, outputs: dict) -> list[dict[str, Any]]:
    del inputs
    keys = ["groundedness_score", "factual_accuracy_score"]
    if reason := _failure_reason(outputs):
        return [_zero(key, reason) for key in keys]
    context = "\n\n".join(str(note) for note in outputs.get("raw_notes", []))
    if not context.strip():
        return [_zero(key, "run produced no retrieved evidence") for key in keys]
    content = GROUNDEDNESS_PROMPT.format(
        context=context,
        report=outputs["final_report"],
        today=_today(),
    )
    result = cast(
        GroundednessScore,
        _get_eval_model()
        .with_structured_output(GroundednessScore)
        .with_retry(stop_after_attempt=3)
        .invoke([{"role": "user", "content": _judge_input(content)}]),
    )
    score = sum(claim.grounded for claim in result.claims) / len(result.claims) if result.claims else 0.0
    comment = json.dumps([claim.model_dump() for claim in result.claims], ensure_ascii=False)
    return [{"key": key, "score": score, "comment": comment} for key in keys]


class CompletenessScore(BaseModel):
    reasoning: str
    score: int = Field(ge=1, le=5)


def eval_completeness(inputs: dict, outputs: dict) -> dict[str, Any]:
    if reason := _failure_reason(outputs):
        return _zero("completeness_score", reason)
    brief = outputs.get("research_brief")
    if not brief:
        return _zero("completeness_score", "run produced no research_brief")
    content = COMPLETENESS_PROMPT.format(
        user_question=_format_input_query(inputs),
        research_brief=brief,
        report=outputs["final_report"],
        today=_today(),
    )
    result = cast(
        CompletenessScore,
        _get_eval_model().with_structured_output(CompletenessScore).invoke(
            [{"role": "user", "content": _judge_input(content)}]
        ),
    )
    return {"key": "completeness_score", "score": result.score / 5, "comment": result.reasoning}


class CitationAssessment(BaseModel):
    claim: str
    citation: str
    supported: bool


class CitationAccuracyScore(BaseModel):
    citations: list[CitationAssessment]
    reasoning: str


def eval_citation_accuracy(inputs: dict, outputs: dict) -> dict[str, Any]:
    if reason := _failure_reason(outputs):
        return _zero("citation_accuracy_score", reason)
    context = "\n\n".join(str(note) for note in outputs.get("raw_notes", []))
    if not context.strip():
        return _zero("citation_accuracy_score", "run produced no retrieved evidence")
    content = CITATION_ACCURACY_PROMPT.format(
        user_question=_format_input_query(inputs),
        context=context,
        report=outputs["final_report"],
        today=_today(),
    )
    result = cast(
        CitationAccuracyScore,
        _get_eval_model()
        .with_structured_output(CitationAccuracyScore)
        .with_retry(stop_after_attempt=3)
        .invoke([{"role": "user", "content": _judge_input(content)}]),
    )
    score = sum(item.supported for item in result.citations) / len(result.citations) if result.citations else 0.0
    details = json.dumps([item.model_dump() for item in result.citations], ensure_ascii=False)
    return {
        "key": "citation_accuracy_score",
        "score": score,
        "comment": f"{result.reasoning}\n{details}",
    }


class ToolEfficiencyScore(BaseModel):
    tool_selection_score: int = Field(ge=1, le=5)
    call_efficiency_score: int = Field(ge=1, le=5)
    reasoning: str


def _extract_tool_trace(outputs: dict[str, Any]) -> dict[str, Any]:
    calls: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    for message in outputs.get("supervisor_messages", []):
        tool_calls = message.get("tool_calls", []) if isinstance(message, dict) else getattr(message, "tool_calls", [])
        for call in tool_calls or []:
            calls.append({
                "name": call.get("name"),
                "args": call.get("args", {}),
                "id": call.get("id"),
            })
        message_type = message.get("type") if isinstance(message, dict) else getattr(message, "type", None)
        if message_type == "tool":
            results.append({
                "name": message.get("name") if isinstance(message, dict) else getattr(message, "name", None),
                "content": str(message.get("content", "") if isinstance(message, dict) else getattr(message, "content", ""))[:1000],
            })
    result = outputs.get("result", {})
    return {
        "supervisor_tool_calls": calls,
        "supervisor_tool_results": results,
        "completed_task_metrics": [
            {
                key: task.get(key)
                for key in ("research_topic", "query_count", "source_count", "citation_count", "elapsed_seconds")
            }
            for task in outputs.get("completed_task_outputs", [])
            if isinstance(task, dict)
        ],
        "run_metrics": result.get("metrics", {}) if isinstance(result, dict) else {},
        "limits": outputs.get("evaluation_metadata", {}),
        "scope_note": "Researcher-level tool names are not retained in the final state; score only observable evidence.",
    }


def eval_tool_efficiency(inputs: dict, outputs: dict) -> dict[str, Any]:
    if reason := _failure_reason(outputs):
        return _zero("tool_efficiency_score", reason)
    trace = _extract_tool_trace(outputs)
    content = TOOL_EFFICIENCY_PROMPT.format(
        user_question=_format_input_query(inputs),
        tool_trace=json.dumps(trace, ensure_ascii=False, default=str),
        today=_today(),
    )
    result = cast(
        ToolEfficiencyScore,
        _get_eval_model().with_structured_output(ToolEfficiencyScore).invoke(
            [{"role": "user", "content": _judge_input(content)}]
        ),
    )
    return {
        "key": "tool_efficiency_score",
        "score": (result.tool_selection_score + result.call_efficiency_score) / 10,
        "comment": result.reasoning,
    }
