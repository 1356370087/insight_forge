"""LangSmith evaluators for the current QueryEngine state contract."""

from __future__ import annotations

import json
import os
from datetime import date
from functools import lru_cache
from typing import Any, Literal, cast

from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from open_deep_research.report.coverage import derive_coverage_checklist
from tests.prompts import (
    CITATION_ACCURACY_PROMPT,
    COMPLETENESS_PROMPT,
    CORRECTNESS_PROMPT,
    EVIDENCE_INTEGRITY_PROMPT,
    GROUNDEDNESS_PROMPT,
    OVERALL_QUALITY_PROMPT,
    RELEVANCE_PROMPT,
    STRUCTURE_PROMPT,
    TOOL_EFFICIENCY_PROMPT,
)


@lru_cache(maxsize=1)
def _get_eval_model() -> ChatOpenAI:
    """Build the judge lazily so importing tests does not require credentials."""
    return ChatOpenAI(
        model=os.getenv("EVALUATION_MODEL", "deepseek-v4-flash[1m]"),
        api_key=os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("EVALUATION_BASE_URL") or os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com"),
        max_tokens=int(os.getenv("EVALUATION_MODEL_MAX_TOKENS", "8192")),
        extra_body={"thinking": {"type": "disabled"}},
    )


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


def _not_scored(key: str, reason: str) -> dict[str, Any]:
    return {"key": key, "score": None, "comment": f"Not scored: {reason}"}


def _judge_input(content: str) -> str | list[dict[str, Any]]:
    model = _get_eval_model()
    if isinstance(model, ChatAnthropic):
        return [{"type": "text", "text": content, "cache_control": {"type": "ephemeral", "ttl": "1h"}}]
    return content


def _structured_output(schema: type[BaseModel]):
    """Use tool calling because DeepSeek does not accept OpenAI json_schema output."""
    return _get_eval_model().with_structured_output(schema, method="function_calling")


def _evidence_context(outputs: dict[str, Any]) -> str:
    """Prefer bounded, accepted canonical evidence over legacy raw notes."""
    max_chars = max(256, int(os.getenv("EVALUATION_EVIDENCE_MAX_CHARS", "120000")))
    registry = outputs.get("evidence_registry") or []
    if isinstance(registry, dict) and registry.get("type") == "override":
        registry = registry.get("value") or []
    if not isinstance(registry, list):
        registry = []
    accepted: list[str] = []
    used_chars = 0
    for record in registry:
        if not isinstance(record, dict):
            continue
        if record.get("security_status", "accepted") != "accepted":
            continue
        projected = {
            key: record.get(key)
            for key in (
                "evidence_id",
                "claim",
                "supporting_excerpt",
                "source_url",
                "source_title",
                "source_authority",
                "locator",
                "confidence",
                "conflict_group",
            )
            if record.get(key) not in {None, ""}
        }
        line = json.dumps(projected, ensure_ascii=False, sort_keys=True)
        separator_chars = 1 if accepted else 0
        if used_chars + separator_chars + len(line) > max_chars:
            if not accepted:
                accepted.append(line[:max_chars])
            break
        accepted.append(line)
        used_chars += separator_chars + len(line)
    if registry:
        return "\n".join(accepted)[:max_chars]

    # Compatibility fallback for old persisted runs without evidence_registry.
    notes = outputs.get("raw_notes") or outputs.get("notes") or []
    if notes:
        return "\n\n".join(str(note) for note in notes)[:max_chars]
    task_notes: list[str] = []
    for task in outputs.get("completed_task_outputs", []):
        if not isinstance(task, dict):
            continue
        task_notes.extend(str(note) for note in task.get("raw_notes", []))
        if task.get("compressed_research"):
            task_notes.append(str(task["compressed_research"]))
    return "\n\n".join(task_notes)[:max_chars]


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
    result = cast(
        OverallQualityScore,
        _structured_output(OverallQualityScore).invoke(
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
    result = cast(
        RelevanceScore,
        _structured_output(RelevanceScore).invoke(
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
        _structured_output(StructureScore).invoke(
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
        _structured_output(CorrectnessScore).invoke(
            [{"role": "user", "content": _judge_input(content)}]
        ),
    )
    return {"key": "correctness_score", "score": result.score / 5, "comment": result.reasoning}


class GroundednessClaim(BaseModel):
    claim: str
    grounded: bool


class GroundednessScore(BaseModel):
    claims: list[GroundednessClaim]


class EvidenceIntegrityClaim(BaseModel):
    """One canonical claim verdict shared by all evidence metrics."""

    claim: str
    citation: str = ""
    has_citation: bool
    entailed_by_evidence: bool
    cited_source_entails_claim: bool
    source_authority: Literal[
        "primary", "high_quality_secondary", "low_quality", "unknown"
    ]
    reasoning: str


class EvidenceIntegrityScore(BaseModel):
    """One internally consistent claim inventory for evidence-related scores."""

    claims: list[EvidenceIntegrityClaim]
    reasoning: str


def eval_evidence_integrity(inputs: dict, outputs: dict) -> list[dict[str, Any]]:
    """Derive all evidence scores from one canonical Judge decision."""
    keys = [
        "groundedness_score",
        "factual_accuracy_score",
        "citation_accuracy_score",
        "source_authority_score",
    ]
    if reason := _failure_reason(outputs):
        return [_zero(key, reason) for key in keys]
    context = _evidence_context(outputs)
    if not context.strip():
        return [_zero(key, "run produced no retrieved evidence") for key in keys]
    content = EVIDENCE_INTEGRITY_PROMPT.format(
        user_question=_format_input_query(inputs),
        context=context,
        report=outputs["final_report"],
        today=_today(),
    )
    result = cast(
        EvidenceIntegrityScore,
        _structured_output(EvidenceIntegrityScore)
        .with_retry(stop_after_attempt=3)
        .invoke([{"role": "user", "content": _judge_input(content)}]),
    )
    claims = result.claims
    grounded = (
        sum(item.entailed_by_evidence for item in claims) / len(claims)
        if claims
        else 0.0
    )
    cited = [item for item in claims if item.has_citation]
    citation_accuracy = (
        sum(item.cited_source_entails_claim for item in cited) / len(cited)
        if cited
        else 0.0
    )
    authority_weights = {
        "primary": 1.0,
        "high_quality_secondary": 0.75,
        "low_quality": 0.25,
        "unknown": 0.0,
    }
    authority = (
        sum(authority_weights.get(item.source_authority, 0.0) for item in cited)
        / len(cited)
        if cited
        else 0.0
    )
    details = json.dumps([item.model_dump() for item in claims], ensure_ascii=False)
    comment = f"{result.reasoning}\nCanonical claim inventory:\n{details}"
    return [
        {"key": "groundedness_score", "score": grounded, "comment": comment},
        _not_scored(
            "factual_accuracy_score",
            "independent truth verification requires a separate reference or fact-checking source",
        ),
        {
            "key": "citation_accuracy_score",
            "score": citation_accuracy,
            "comment": comment,
        },
        {"key": "source_authority_score", "score": authority, "comment": comment},
    ]


def eval_groundedness(inputs: dict, outputs: dict) -> list[dict[str, Any]]:
    del inputs
    keys = ["groundedness_score", "factual_accuracy_score"]
    if reason := _failure_reason(outputs):
        return [_zero(key, reason) for key in keys]
    context = _evidence_context(outputs)
    if not context.strip():
        return [_zero(key, "run produced no retrieved evidence") for key in keys]
    content = GROUNDEDNESS_PROMPT.format(
        context=context,
        report=outputs["final_report"],
        today=_today(),
    )
    result = cast(
        GroundednessScore,
        _structured_output(GroundednessScore)
        .with_retry(stop_after_attempt=3)
        .invoke([{"role": "user", "content": _judge_input(content)}]),
    )
    score = sum(claim.grounded for claim in result.claims) / len(result.claims) if result.claims else 0.0
    comment = json.dumps([claim.model_dump() for claim in result.claims], ensure_ascii=False)
    return [
        {"key": "groundedness_score", "score": score, "comment": comment},
        _not_scored(
            "factual_accuracy_score",
            "independent truth verification requires a separate reference or fact-checking source",
        ),
    ]


class CoverageAssessment(BaseModel):
    requirement_id: str
    status: Literal["covered", "partial", "missing"]
    explanation: str


class CompletenessScore(BaseModel):
    reasoning: str
    score: int = Field(ge=1, le=5)
    checklist: list[CoverageAssessment] = Field(default_factory=list)


def eval_completeness(inputs: dict, outputs: dict) -> dict[str, Any]:
    if reason := _failure_reason(outputs):
        return _zero("completeness_score", reason)
    brief = outputs.get("research_brief")
    if not brief:
        return _zero("completeness_score", "run produced no research_brief")
    user_question = _format_input_query(inputs)
    coverage_checklist = list(
        dict.fromkeys(
            [
                *derive_coverage_checklist(user_question),
                *[str(item) for item in outputs.get("coverage_checklist", [])],
            ]
        )
    )[:20]
    indexed_requirements = [
        {"requirement_id": f"COV-{index:02d}", "requirement": requirement}
        for index, requirement in enumerate(coverage_checklist, 1)
    ]
    content = COMPLETENESS_PROMPT.format(
        user_question=user_question,
        research_brief=brief,
        coverage_checklist=json.dumps(
            indexed_requirements,
            ensure_ascii=False,
        ),
        report=outputs["final_report"],
        today=_today(),
    )
    result = cast(
        CompletenessScore,
        _structured_output(CompletenessScore).invoke(
            [{"role": "user", "content": _judge_input(content)}]
        ),
    )
    expected = {
        item["requirement_id"]: item["requirement"] for item in indexed_requirements
    }
    by_id: dict[str, CoverageAssessment] = {}
    for item in result.checklist:
        if item.requirement_id in expected and item.requirement_id not in by_id:
            by_id[item.requirement_id] = item
    assessed = [
        {
            "requirement_id": requirement_id,
            "requirement": requirement,
            **by_id[requirement_id].model_dump(exclude={"requirement_id"}),
        }
        for requirement_id, requirement in expected.items()
        if requirement_id in by_id
    ]
    missing_assessments = len(expected) - len(by_id)
    missing = sum(item.status == "missing" for item in by_id.values()) + missing_assessments
    partial = sum(item.status == "partial" for item in by_id.values())
    score = result.score / 5
    if missing >= 2:
        score = min(score, 0.6)
    elif missing == 1 or partial:
        score = min(score, 0.8)
    checklist = json.dumps(assessed, ensure_ascii=False)
    return {
        "key": "completeness_score",
        "score": score,
        "comment": (
            f"{result.reasoning}\nCoverage checklist ({len(assessed)}/{len(coverage_checklist)} "
            f"assessed; {missing} missing; {partial} partial):\n{checklist}"
        ),
    }


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
    context = _evidence_context(outputs)
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
        _structured_output(CitationAccuracyScore)
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
        _structured_output(ToolEfficiencyScore).invoke(
            [{"role": "user", "content": _judge_input(content)}]
        ),
    )
    return {
        "key": "tool_efficiency_score",
        "score": (result.tool_selection_score + result.call_efficiency_score) / 10,
        "comment": result.reasoning,
    }
