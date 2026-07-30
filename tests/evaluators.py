"""LangSmith evaluators for the current QueryEngine state contract."""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import date
from functools import lru_cache
from typing import Any, Literal, get_args, get_origin

from langchain_core.messages import AIMessage
from pydantic import BaseModel, Field

from open_deep_research import quality as quality_evaluation
from open_deep_research.evaluation import (
    JUDGE_SECURITY_PROTOCOL,
    JudgeConfig,
    MetricStatus,
    build_judge_model,
    langsmith_metric,
)
from open_deep_research.evaluation.execution import (
    content_coverage_requirements,
    evaluate_execution_compliance,
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

logger = logging.getLogger(__name__)
_SAFE_ERROR_TOKEN = re.compile(r"^[A-Za-z0-9_.:-]{1,80}$")


class JudgeOutputError(ValueError):
    """Classified local Judge protocol failure safe for persisted metadata."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


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


def _evaluator_error(key: str, reason: str) -> dict[str, Any]:
    return langsmith_metric(
        key,
        status=MetricStatus.EVALUATOR_ERROR,
        comment=f"Evaluator error: {reason}",
    )


def _judge_input(content: str) -> str | list[dict[str, Any]]:
    if JudgeConfig.from_env().provider == "anthropic":
        return [{"type": "text", "text": content, "cache_control": {"type": "ephemeral", "ttl": "1h"}}]
    return content


def _structured_output(schema: type[BaseModel]):
    """Use tool calling because DeepSeek does not accept OpenAI json_schema output."""
    return _get_eval_model().with_structured_output(
        schema,
        method="function_calling",
        include_raw=True,
    )


def _unwrap_single_key_schema_payload(
    schema: type[BaseModel],
    payload: Any,
) -> Any:
    """Repair only an unambiguous provider wrapper around a schema payload."""
    if not isinstance(payload, dict) or len(payload) != 1:
        return payload

    nested = next(iter(payload.values()))
    if isinstance(nested, str):
        try:
            nested = json.loads(nested)
        except json.JSONDecodeError:
            return payload
    if not isinstance(nested, dict):
        return payload

    schema_fields = set(schema.model_fields)
    if not schema_fields.issubset(nested):
        return payload
    return nested


def _decode_json_list_schema_fields(
    schema: type[BaseModel],
    payload: Any,
) -> Any:
    """Decode only JSON strings whose declared schema field explicitly expects a list."""
    if not isinstance(payload, dict):
        return payload

    decoded_payload: dict[str, Any] | None = None
    for field_name, field in schema.model_fields.items():
        value = payload.get(field_name)
        if not isinstance(value, str) or get_origin(field.annotation) is not list:
            continue
        try:
            decoded_value = json.loads(value)
        except json.JSONDecodeError:
            continue
        if not isinstance(decoded_value, list):
            continue
        if decoded_payload is None:
            decoded_payload = dict(payload)
        decoded_payload[field_name] = decoded_value
    return decoded_payload if decoded_payload is not None else payload


def _payload_from_structured_result(result: Any) -> Any:
    """Select parsed output or one normalized tool-call args payload."""
    envelope_fields = {"raw", "parsed", "parsing_error"}
    if not isinstance(result, dict) or not envelope_fields.issubset(result):
        return result

    parsed = result.get("parsed")
    if parsed is not None:
        return parsed

    raw = result.get("raw")
    if not isinstance(raw, AIMessage):
        raise JudgeOutputError(
            "invalid_raw_message",
            "Judge raw response was not a normalized AIMessage",
        )
    tool_calls = raw.tool_calls
    if not isinstance(tool_calls, list) or len(tool_calls) != 1:
        raise JudgeOutputError(
            "ambiguous_tool_calls",
            "Judge raw response did not contain exactly one normalized tool call",
        )
    tool_call = tool_calls[0]
    if not isinstance(tool_call, dict):
        raise JudgeOutputError(
            "invalid_tool_call",
            "Judge normalized tool call was not a mapping",
        )
    arguments = tool_call.get("args")
    if not isinstance(arguments, dict):
        raise JudgeOutputError(
            "invalid_tool_call_args",
            "Judge normalized tool-call arguments were not a mapping",
        )
    return arguments


def _invoke_structured_output(
    schema: type[BaseModel],
    messages: list[dict[str, Any]],
    *,
    attempts: int = 3,
) -> Any:
    """Retry empty, invalid, and failed structured Judge responses uniformly."""
    runner = _structured_output(schema)
    last_error: Exception | None = None
    last_failure_summary: str | None = None
    attempt_count = max(1, attempts)
    for attempt_index in range(attempt_count):
        payload_shape: str | None = None
        try:
            result = runner.invoke(messages)
            if result is None:
                raise JudgeOutputError(
                    "no_structured_output",
                    f"{schema.__name__} Judge returned no structured output",
                )
            if isinstance(result, schema):
                return result
            payload = _payload_from_structured_result(result)
            if isinstance(payload, schema):
                return payload
            payload_shape = _safe_payload_shape(payload, schema)
            unwrapped_payload = _unwrap_single_key_schema_payload(schema, payload)
            return schema.model_validate(
                _decode_json_list_schema_fields(schema, unwrapped_payload)
            )
        except Exception as exc:  # noqa: BLE001 - bounded Judge retry boundary
            last_error = exc
            last_failure_summary = _judge_failure_summary(
                exc,
                payload_shape=payload_shape,
            )
            logger.warning(
                "%s Judge attempt %d/%d failed (%s)",
                schema.__name__,
                attempt_index + 1,
                attempt_count,
                last_failure_summary,
            )
    assert last_error is not None
    assert last_failure_summary is not None
    raise RuntimeError(
        f"{schema.__name__} Judge failed after {attempt_count} attempts; "
        f"last error: {last_failure_summary}"
    ) from last_error


def _judge_failure_summary(
    exc: BaseException,
    *,
    payload_shape: str | None = None,
) -> str:
    """Return useful provider metadata without persisting exception text."""
    parts = [type(exc).__name__]
    status = next(
        (
            value
            for value in (
                getattr(exc, "status_code", None),
                getattr(exc, "status", None),
                getattr(getattr(exc, "response", None), "status_code", None),
            )
            if isinstance(value, int)
        ),
        None,
    )
    if status is not None:
        parts.append(f"status={status}")
    body = getattr(exc, "body", None)
    code = getattr(exc, "code", None)
    if code is None and isinstance(body, dict):
        code = body.get("code")
        if code is None and isinstance(body.get("error"), dict):
            code = body["error"].get("code")
    code_text = str(code or "")
    if _SAFE_ERROR_TOKEN.fullmatch(code_text):
        parts.append(f"code={code_text}")
    validation_signatures = _validation_error_signatures(exc)
    if validation_signatures:
        parts.append(f"validation={';'.join(validation_signatures)}")
        if payload_shape:
            parts.append(f"payload_shape={payload_shape}")
    return ", ".join(parts)


def _validation_error_signatures(
    exc: BaseException,
    *,
    max_items: int = 12,
) -> list[str]:
    """Extract bounded validation paths and types without input values or messages."""
    errors_method = getattr(exc, "errors", None)
    if not callable(errors_method):
        return []
    try:
        errors = errors_method(
            include_url=False,
            include_context=False,
            include_input=False,
        )
    except TypeError:
        errors = errors_method()
    except Exception:  # noqa: BLE001 - diagnostics must never replace root failure
        return []
    if not isinstance(errors, list):
        return []

    signatures: list[str] = []
    for item in errors[:max_items]:
        if not isinstance(item, dict):
            continue
        location = item.get("loc", ())
        if not isinstance(location, (list, tuple)):
            location = (location,)
        safe_location = ".".join(
            _safe_validation_token(part)
            for part in location
        ) or "<root>"
        error_type = _safe_validation_token(item.get("type", "unknown"))
        signatures.append(f"{safe_location}:{error_type}")
    if len(errors) > max_items:
        signatures.append(f"<truncated>:{len(errors) - max_items}")
    return signatures


def _safe_validation_token(value: Any) -> str:
    if isinstance(value, int):
        return str(value)
    text = str(value)
    if _SAFE_ERROR_TOKEN.fullmatch(text):
        return text
    return "<redacted>"


def _safe_payload_shape(
    payload: Any,
    schema: type[BaseModel],
    *,
    max_depth: int = 2,
    max_keys: int = 12,
) -> str:
    """Describe payload containers without serializing any scalar values."""
    remaining_keys = [max(0, max_keys)]
    allowed_fields = _schema_payload_field_names(schema)

    def render(value: Any, depth: int) -> str:
        if isinstance(value, dict):
            if depth >= max_depth:
                return "dict"
            fields: list[str] = []
            for key in sorted(value, key=lambda item: str(item)):
                if remaining_keys[0] <= 0:
                    fields.append("<truncated>")
                    break
                remaining_keys[0] -= 1
                key_text = str(key)
                safe_key = (
                    _safe_validation_token(key_text)
                    if key_text in allowed_fields
                    else "<redacted>"
                )
                fields.append(
                    f"{safe_key}:"
                    f"{render(value[key], depth + 1)}"
                )
            return f"dict{{{','.join(fields)}}}"
        if isinstance(value, list):
            return f"list(len={len(value)})"
        if isinstance(value, tuple):
            return f"tuple(len={len(value)})"
        if value is None:
            return "none"
        if isinstance(value, bool):
            return "bool"
        if isinstance(value, int):
            return "int"
        if isinstance(value, float):
            return "float"
        if isinstance(value, str):
            return "str"
        if isinstance(value, bytes):
            return "bytes"
        return _safe_validation_token(type(value).__name__)

    shape = render(payload, 0)
    if shape.startswith("dict{"):
        return shape.removeprefix("dict")
    return shape


@lru_cache(maxsize=32)
def _schema_payload_field_names(schema: type[BaseModel]) -> frozenset[str]:
    """Return code-defined field names and aliases for a schema model graph."""
    allowed: set[str] = set()
    pending = [schema]
    visited: set[type[BaseModel]] = set()
    while pending:
        model = pending.pop()
        if model in visited:
            continue
        visited.add(model)
        for field_name, field in model.model_fields.items():
            allowed.add(field_name)
            for alias in (
                field.alias,
                field.serialization_alias,
                field.validation_alias,
            ):
                if isinstance(alias, str):
                    allowed.add(alias)
            pending.extend(_nested_schema_models(field.annotation))
    return frozenset(allowed)


def _nested_schema_models(annotation: Any) -> list[type[BaseModel]]:
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return [annotation]
    nested: list[type[BaseModel]] = []
    for argument in get_args(annotation):
        nested.extend(_nested_schema_models(argument))
    return nested


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
    selected, _stats = quality_evaluation._bounded_evidence_records(
        eligible_evidence_records(registry),
        max_chars=max_chars,
    )
    accepted: list[str] = []
    for record in selected:
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
        accepted.append(line)
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
    original_requirements = derive_coverage_checklist(user_question)
    if not original_requirements:
        original_requirements = [str(item) for item in persisted_checklist]
    coverage_checklist = content_coverage_requirements(
        list(dict.fromkeys(original_requirements))
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


def eval_execution_compliance(
    inputs: dict,
    outputs: dict,
) -> dict[str, Any] | list[dict[str, Any]]:
    """Evaluate explicit workflow constraints from the trusted execution trace."""
    snapshot = _evaluation_snapshot(outputs)
    trace = _extract_tool_trace(outputs)
    evidence_registry = (
        snapshot.get("evidence_registry", [])
        if snapshot is not None
        else outputs.get("evidence_registry", [])
    )
    result = evaluate_execution_compliance(
        _format_input_query(inputs),
        trace,
        evidence_registry if isinstance(evidence_registry, list) else [],
    )
    if not result.applicable:
        return []
    comment = json.dumps(result.model_dump(mode="json"), ensure_ascii=False)
    if result.status == "evaluator_error":
        return _evaluator_error("execution_compliance_score", comment)
    return _scored(
        "execution_compliance_score",
        result.score or 0.0,
        comment,
    )
