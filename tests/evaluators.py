"""LangSmith evaluators for the current QueryEngine state contract."""

from __future__ import annotations

import json
import os
from datetime import date
from functools import lru_cache
from typing import Any, Literal

from pydantic import BaseModel, Field

from open_deep_research.evaluation import (
    JUDGE_SECURITY_PROTOCOL,
    JudgeConfig,
    MetricStatus,
    build_judge_model,
    langsmith_metric,
)
from open_deep_research.evidence import eligible_evidence_records
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
def _get_eval_model() -> Any:
    """Build the judge lazily so importing tests does not require credentials."""
    return build_judge_model(JudgeConfig.from_env())


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


def _scored(
    key: str,
    score: bool | int | float,
    comment: str = "",
) -> dict[str, Any]:
    return langsmith_metric(
        key,
        status=MetricStatus.SCORED,
        score=score,
        comment=comment,
    )


def _not_scored(key: str, reason: str) -> dict[str, Any]:
    return langsmith_metric(
        key,
        status=MetricStatus.NOT_SCORED,
        comment=f"Not scored: {reason}",
    )


def _run_failed(key: str, reason: str) -> dict[str, Any]:
    return langsmith_metric(
        key,
        status=MetricStatus.RUN_FAILED,
        comment=f"Run failed: {reason}",
    )


def _judge_input(content: str) -> str | list[dict[str, Any]]:
    if JudgeConfig.from_env().provider == "anthropic":
        return [{"type": "text", "text": content, "cache_control": {"type": "ephemeral", "ttl": "1h"}}]
    return content


def _structured_output(schema: type[BaseModel]):
    """Use tool calling because DeepSeek does not accept OpenAI json_schema output."""
    return _get_eval_model().with_structured_output(schema, method="function_calling")


def _invoke_structured_output(
    schema: type[BaseModel],
    messages: list[dict[str, Any]],
    *,
    attempts: int = 3,
) -> Any:
    """Retry empty, invalid, and failed structured Judge responses uniformly."""
    runner = _structured_output(schema)
    last_error: Exception | None = None
    for _attempt in range(max(1, attempts)):
        try:
            result = runner.invoke(messages)
            if result is None:
                raise ValueError(
                    f"{schema.__name__} Judge returned no structured output"
                )
            if isinstance(result, schema):
                return result
            return schema.model_validate(result)
        except Exception as exc:  # noqa: BLE001 - bounded Judge retry boundary
            last_error = exc
    raise RuntimeError(
        f"{schema.__name__} Judge failed after {max(1, attempts)} attempts"
    ) from last_error


def _judge_messages(
    content: str,
    *,
    rubric: str | None = None,
) -> list[dict[str, Any]]:
    """Place all run-derived text in an explicitly untrusted envelope."""
    system_content = JUDGE_SECURITY_PROTOCOL
    if rubric:
        system_content = f"{system_content}\n\nEvaluation rubric:\n{rubric}"
    payload = (
        "<untrusted_evaluation_payload>\n"
        f"{content}\n"
        "</untrusted_evaluation_payload>"
    )
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": _judge_input(payload)},
    ]


def _evaluation_snapshot(outputs: dict[str, Any]) -> dict[str, Any] | None:
    """Return a supported persisted snapshot, or ``None`` for legacy runs."""
    snapshot = outputs.get("evaluation_snapshot")
    if not isinstance(snapshot, dict) or snapshot.get("schema_version") != "1.0":
        return None
    return snapshot


def _evidence_context(outputs: dict[str, Any]) -> str:
    """Prefer bounded, accepted canonical evidence over legacy raw notes."""
    max_chars = max(256, int(os.getenv("EVALUATION_EVIDENCE_MAX_CHARS", "120000")))
    snapshot = _evaluation_snapshot(outputs)
    registry = (
        snapshot.get("evidence_registry", [])
        if snapshot is not None
        else outputs.get("evidence_registry") or []
    )
    if isinstance(registry, dict) and registry.get("type") == "override":
        registry = registry.get("value") or []
    if not isinstance(registry, list):
        registry = []
    accepted: list[str] = []
    used_chars = 0
    for record in eligible_evidence_records(registry):
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
            if record.get(key) is not None and record.get(key) != ""
        }
        line = json.dumps(projected, ensure_ascii=False, sort_keys=True)
        separator_chars = 1 if accepted else 0
        if used_chars + separator_chars + len(line) > max_chars:
            if not accepted:
                accepted.append(line[:max_chars])
            break
        accepted.append(line)
        used_chars += separator_chars + len(line)
    if snapshot is not None or registry:
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
        return [_run_failed(key, reason) for key in keys]

    content = json.dumps(
        {
            "user_input": _format_input_query(inputs),
            "report": outputs["final_report"],
        },
        ensure_ascii=False,
    )
    result = _invoke_structured_output(
        OverallQualityScore,
        _judge_messages(
            content,
            rubric=OVERALL_QUALITY_PROMPT.format(today=_today()),
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
    return [
        _scored(key, value / 5)
        for key, value in zip(keys, values, strict=True)
    ]


class RelevanceScore(BaseModel):
    reasoning: str
    score: int = Field(ge=1, le=5)


def eval_relevance(inputs: dict, outputs: dict) -> dict[str, Any]:
    if reason := _failure_reason(outputs):
        return _run_failed("relevance_score", reason)
    content = json.dumps(
        {
            "user_input": _format_input_query(inputs),
            "report": outputs["final_report"],
        },
        ensure_ascii=False,
    )
    result = _invoke_structured_output(
        RelevanceScore,
        _judge_messages(
            content,
            rubric=RELEVANCE_PROMPT.format(today=_today()),
        ),
    )
    return _scored("relevance_score", result.score / 5, result.reasoning)


class StructureScore(BaseModel):
    reasoning: str
    score: int = Field(ge=1, le=5)


def eval_structure(inputs: dict, outputs: dict) -> dict[str, Any]:
    if reason := _failure_reason(outputs):
        return _run_failed("structure_and_cohesiveness_score", reason)
    rubric = STRUCTURE_PROMPT.format(
        user_question="[provided in the untrusted payload]",
        report="[provided in the untrusted payload]",
        today=_today(),
    )
    content = json.dumps(
        {
            "user_question": _format_input_query(inputs),
            "report": outputs["final_report"],
        },
        ensure_ascii=False,
    )
    result = _invoke_structured_output(
        StructureScore,
        _judge_messages(content, rubric=rubric),
    )
    return _scored(
        "structure_and_cohesiveness_score",
        result.score / 5,
        result.reasoning,
    )


class CorrectnessScore(BaseModel):
    reasoning: str
    score: int = Field(ge=1, le=5)


def eval_correctness(inputs: dict, outputs: dict, reference_outputs: dict) -> dict[str, Any]:
    if reason := _failure_reason(outputs):
        return _run_failed("correctness_score", reason)
    answer = reference_outputs.get("answer")
    if not answer:
        return _not_scored("correctness_score", "reference output has no answer")
    rubric = CORRECTNESS_PROMPT.format(
        user_question="[provided in the untrusted payload]",
        report="[provided in the untrusted payload]",
        answer="[provided in the untrusted payload]",
        today=_today(),
    )
    content = json.dumps(
        {
            "user_question": _format_input_query(inputs),
            "authority_answer": answer,
            "report": outputs["final_report"],
        },
        ensure_ascii=False,
    )
    result = _invoke_structured_output(
        CorrectnessScore,
        _judge_messages(content, rubric=rubric),
    )
    return _scored("correctness_score", result.score / 5, result.reasoning)


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
        return [_run_failed(key, reason) for key in keys]
    context = _evidence_context(outputs)
    if not context.strip():
        return [
            _scored(
                "groundedness_score",
                0.0,
                "No eligible retrieved evidence was available.",
            ),
            _not_scored(
                "factual_accuracy_score",
                "independent truth verification requires a separate reference or fact-checking source",
            ),
            _scored(
                "citation_accuracy_score",
                0.0,
                "No eligible retrieved evidence was available.",
            ),
            _scored(
                "source_authority_score",
                0.0,
                "No eligible retrieved evidence was available.",
            ),
        ]
    rubric = EVIDENCE_INTEGRITY_PROMPT.format(
        user_question="[provided in the untrusted payload]",
        context="[provided in the untrusted payload]",
        report="[provided in the untrusted payload]",
        today=_today(),
    )
    content = json.dumps(
        {
            "user_question": _format_input_query(inputs),
            "retrieved_evidence": context,
            "report": outputs["final_report"],
        },
        ensure_ascii=False,
    )
    result = _invoke_structured_output(
        EvidenceIntegrityScore,
        _judge_messages(content, rubric=rubric),
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
        _scored("groundedness_score", grounded, comment),
        _not_scored(
            "factual_accuracy_score",
            "independent truth verification requires a separate reference or fact-checking source",
        ),
        _scored("citation_accuracy_score", citation_accuracy, comment),
        _scored("source_authority_score", authority, comment),
    ]


def eval_groundedness(inputs: dict, outputs: dict) -> list[dict[str, Any]]:
    del inputs
    keys = ["groundedness_score", "factual_accuracy_score"]
    if reason := _failure_reason(outputs):
        return [_run_failed(key, reason) for key in keys]
    context = _evidence_context(outputs)
    if not context.strip():
        return [
            _scored(
                "groundedness_score",
                0.0,
                "No eligible retrieved evidence was available.",
            ),
            _not_scored(
                "factual_accuracy_score",
                "independent truth verification requires a separate reference or fact-checking source",
            ),
        ]
    rubric = GROUNDEDNESS_PROMPT.format(
        context="[provided in the untrusted payload]",
        report="[provided in the untrusted payload]",
    )
    content = json.dumps(
        {
            "retrieved_context": context,
            "report": outputs["final_report"],
        },
        ensure_ascii=False,
    )
    result = _invoke_structured_output(
        GroundednessScore,
        _judge_messages(content, rubric=rubric),
    )
    score = sum(claim.grounded for claim in result.claims) / len(result.claims) if result.claims else 0.0
    comment = json.dumps([claim.model_dump() for claim in result.claims], ensure_ascii=False)
    return [
        _scored("groundedness_score", score, comment),
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
        return _run_failed("completeness_score", reason)
    snapshot = _evaluation_snapshot(outputs)
    brief = (
        snapshot.get("research_brief")
        if snapshot is not None
        else outputs.get("research_brief")
    )
    if not brief:
        return _not_scored("completeness_score", "run produced no research_brief")
    user_question = _format_input_query(inputs)
    persisted_checklist = (
        snapshot.get("coverage_checklist", [])
        if snapshot is not None
        else outputs.get("coverage_checklist", [])
    )
    if not isinstance(persisted_checklist, list):
        persisted_checklist = []
    coverage_checklist = list(
        dict.fromkeys(
            [
                *derive_coverage_checklist(user_question),
                *[str(item) for item in persisted_checklist],
            ]
        )
    )[:20]
    indexed_requirements = [
        {"requirement_id": f"COV-{index:02d}", "requirement": requirement}
        for index, requirement in enumerate(coverage_checklist, 1)
    ]
    rubric = COMPLETENESS_PROMPT.format(
        user_question="[provided in the untrusted payload]",
        research_brief="[provided in the untrusted payload]",
        coverage_checklist="[provided in the untrusted payload]",
        report="[provided in the untrusted payload]",
        today=_today(),
    )
    content = json.dumps(
        {
            "user_question": user_question,
            "research_brief": brief,
            "coverage_checklist": indexed_requirements,
            "report": outputs["final_report"],
        },
        ensure_ascii=False,
    )
    result = _invoke_structured_output(
        CompletenessScore,
        _judge_messages(content, rubric=rubric),
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
    return _scored(
        "completeness_score",
        score,
        (
            f"{result.reasoning}\nCoverage checklist ({len(assessed)}/{len(coverage_checklist)} "
            f"assessed; {missing} missing; {partial} partial):\n{checklist}"
        ),
    )


class CitationAssessment(BaseModel):
    claim: str
    citation: str
    supported: bool


class CitationAccuracyScore(BaseModel):
    citations: list[CitationAssessment]
    reasoning: str


def eval_citation_accuracy(inputs: dict, outputs: dict) -> dict[str, Any]:
    if reason := _failure_reason(outputs):
        return _run_failed("citation_accuracy_score", reason)
    context = _evidence_context(outputs)
    if not context.strip():
        return _scored(
            "citation_accuracy_score",
            0.0,
            "No eligible retrieved evidence was available.",
        )
    rubric = CITATION_ACCURACY_PROMPT.format(
        user_question="[provided in the untrusted payload]",
        context="[provided in the untrusted payload]",
        report="[provided in the untrusted payload]",
        today=_today(),
    )
    content = json.dumps(
        {
            "user_question": _format_input_query(inputs),
            "retrieved_context": context,
            "report": outputs["final_report"],
        },
        ensure_ascii=False,
    )
    result = _invoke_structured_output(
        CitationAccuracyScore,
        _judge_messages(content, rubric=rubric),
    )
    score = sum(item.supported for item in result.citations) / len(result.citations) if result.citations else 0.0
    details = json.dumps([item.model_dump() for item in result.citations], ensure_ascii=False)
    return _scored(
        "citation_accuracy_score",
        score,
        f"{result.reasoning}\n{details}",
    )


class ToolEfficiencyScore(BaseModel):
    tool_selection_score: int = Field(ge=1, le=5)
    call_efficiency_score: int = Field(ge=1, le=5)
    reasoning: str


def _extract_tool_trace(outputs: dict[str, Any]) -> dict[str, Any]:
    snapshot = _evaluation_snapshot(outputs)
    if snapshot is not None and isinstance(snapshot.get("tool_trace"), dict):
        trace = dict(snapshot["tool_trace"])
        result = outputs.get("result", {})
        if not trace.get("run_metrics") and isinstance(result, dict):
            run_metrics = result.get("metrics", {})
            if run_metrics:
                trace["run_metrics"] = run_metrics
        if not trace.get("limits") and outputs.get("evaluation_metadata"):
            trace["limits"] = outputs["evaluation_metadata"]
        return trace

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
        "researcher_tool_calls": [],
        "researcher_tool_results": [],
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
        "availability": {
            "supervisor_messages_present": bool(
                outputs.get("supervisor_messages")
            ),
            "completed_task_outputs_present": bool(
                outputs.get("completed_task_outputs")
            ),
            "researcher_tool_names_retained": False,
        },
        "scope_note": "Researcher-level tool names are not retained in the final state; score only observable evidence.",
    }


def eval_tool_efficiency(inputs: dict, outputs: dict) -> dict[str, Any]:
    if reason := _failure_reason(outputs):
        return _run_failed("tool_efficiency_score", reason)
    trace = _extract_tool_trace(outputs)
    supervisor_calls = trace.get("supervisor_tool_calls", [])
    delegated_research = any(
        isinstance(call, dict)
        and call.get("name") in {"ConductResearch", "StartResearchTask"}
        for call in supervisor_calls
    )
    availability = trace.get("availability", {})
    researcher_trace_available = (
        isinstance(availability, dict)
        and availability.get("researcher_tool_names_retained") is True
    )
    if delegated_research and not researcher_trace_available:
        return _not_scored(
            "tool_efficiency_score",
            "researcher tool trace is unavailable for delegated research",
        )
    rubric = TOOL_EFFICIENCY_PROMPT.format(
        user_question="[provided in the untrusted payload]",
        tool_trace="[provided in the untrusted payload]",
        today=_today(),
    )
    content = json.dumps(
        {
            "user_question": _format_input_query(inputs),
            "observable_tool_trace": trace,
        },
        ensure_ascii=False,
        default=str,
    )
    result = _invoke_structured_output(
        ToolEfficiencyScore,
        _judge_messages(content, rubric=rubric),
    )
    return _scored(
        "tool_efficiency_score",
        (result.tool_selection_score + result.call_efficiency_score) / 10,
        result.reasoning,
    )
