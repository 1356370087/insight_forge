"""Build the immutable evaluation view before transient state is cleared."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from open_deep_research.evidence import eligible_evidence_records

from .models import (
    CompletedTaskMetric,
    EvaluationEvidence,
    EvaluationSnapshot,
    EvaluationToolTrace,
    ResearcherToolCall,
    ResearcherToolResult,
    SupervisorToolCall,
    SupervisorToolResult,
    ToolTraceAvailability,
)

EVALUATION_SNAPSHOT_VERSION = "1.0"
_MAX_PREVIEW_CHARS = 1_000
_MAX_COLLECTION_ITEMS = 50
_SENSITIVE_KEY_PARTS = (
    "api_key",
    "apikey",
    "authorization",
    "cookie",
    "credential",
    "password",
    "secret",
    "token",
)
_EVIDENCE_FIELDS = (
    "evidence_id",
    "claim",
    "supporting_excerpt",
    "source_url",
    "source_title",
    "source_authority",
    "locator",
    "confidence",
    "conflict_group",
    "security_status",
)
_TASK_METRIC_FIELDS = (
    "task_id",
    "research_topic",
    "query_count",
    "source_count",
    "citation_count",
    "elapsed_seconds",
    "admission_status",
)
_AUTHORIZATION_VALUE = re.compile(
    r"(?i)(\bauthorization\b\s*[:=]\s*)(?:bearer\s+)?[^\s,;]+"
)
_NAMED_SECRET_VALUE = re.compile(
    r"""(?ix)
    (["']?(?:
        api[_-]?key
        |access[_-]?token
        |refresh[_-]?token
        |password
        |secret
        |cookie
    )["']?\s*[:=]\s*)
    ("[^"]*"|'[^']*'|[^\s,;]+)
    """
)


def _has_projectable_value(value: Any) -> bool:
    return value is not None and value != ""


def _unwrap_override(value: Any) -> Any:
    if isinstance(value, dict) and value.get("type") == "override":
        return value.get("value")
    return value


def _message_value(message: object, key: str, default: Any = None) -> Any:
    if isinstance(message, Mapping):
        return message.get(key, default)
    return getattr(message, key, default)


def _is_sensitive_key(key: object) -> bool:
    normalized = str(key).lower()
    return any(part in normalized for part in _SENSITIVE_KEY_PARTS)


def _redact_secret_text(value: object) -> str:
    """Redact common credential assignments in unstructured tool output."""
    text = str(value)
    text = _AUTHORIZATION_VALUE.sub(r"\1[REDACTED]", text)
    return _NAMED_SECRET_VALUE.sub(r"\1[REDACTED]", text)


def _bounded_json_value(value: Any, *, key: object = "", depth: int = 0) -> Any:
    """Return a JSON-safe, bounded value while redacting likely credentials."""
    if _is_sensitive_key(key):
        return "[REDACTED]"
    if depth >= 4:
        return "[TRUNCATED]"
    if value is None or isinstance(value, bool | int | float):
        return value
    if isinstance(value, str):
        return _redact_secret_text(value)[:_MAX_PREVIEW_CHARS]
    if isinstance(value, Mapping):
        return {
            str(item_key): _bounded_json_value(
                item_value,
                key=item_key,
                depth=depth + 1,
            )
            for item_key, item_value in list(value.items())[:_MAX_COLLECTION_ITEMS]
        }
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [
            _bounded_json_value(item, depth=depth + 1)
            for item in list(value)[:_MAX_COLLECTION_ITEMS]
        ]
    return str(value)[:_MAX_PREVIEW_CHARS]


def _project_evidence(state: Mapping[str, Any]) -> list[EvaluationEvidence]:
    registry = _unwrap_override(state.get("evidence_registry", []))
    if not isinstance(registry, list):
        return []
    return [
        EvaluationEvidence.model_validate(
            {
                key: _bounded_json_value(record.get(key), key=key)
                for key in _EVIDENCE_FIELDS
                if _has_projectable_value(record.get(key))
            }
        )
        for record in eligible_evidence_records(registry)
    ]


def _project_task_metrics(state: Mapping[str, Any]) -> list[CompletedTaskMetric]:
    task_outputs = _unwrap_override(state.get("completed_task_outputs", []))
    if not isinstance(task_outputs, list):
        return []
    return [
        CompletedTaskMetric.model_validate(
            {
                key: _bounded_json_value(task.get(key), key=key)
                for key in _TASK_METRIC_FIELDS
                if task.get(key) is not None
            }
        )
        for task in task_outputs
        if isinstance(task, dict)
    ]


def _project_supervisor_trace(
    state: Mapping[str, Any],
) -> tuple[list[SupervisorToolCall], list[SupervisorToolResult]]:
    messages = _unwrap_override(state.get("supervisor_messages", []))
    if not isinstance(messages, list):
        return [], []
    calls: list[SupervisorToolCall] = []
    results: list[SupervisorToolResult] = []
    for message in messages:
        raw_calls = _message_value(message, "tool_calls", [])
        if isinstance(raw_calls, list):
            for call in raw_calls:
                if not isinstance(call, Mapping):
                    continue
                raw_args = call.get("args", {})
                args = (
                    _bounded_json_value(raw_args)
                    if isinstance(raw_args, Mapping)
                    else {}
                )
                calls.append(
                    SupervisorToolCall(
                        name=str(call["name"]) if call.get("name") is not None else None,
                        args=args,
                        id=str(call["id"]) if call.get("id") is not None else None,
                    )
                )
        if _message_value(message, "type") != "tool":
            continue
        content = _message_value(message, "content", "")
        results.append(
            SupervisorToolResult(
                name=(
                    str(_message_value(message, "name"))
                    if _message_value(message, "name") is not None
                    else None
                ),
                tool_call_id=(
                    str(_message_value(message, "tool_call_id"))
                    if _message_value(message, "tool_call_id") is not None
                    else None
                ),
                content_preview=_redact_secret_text(content)[
                    :_MAX_PREVIEW_CHARS
                ],
            )
        )
    return calls, results


def _artifact_message_data(message: object) -> object:
    """Return the LangChain message payload stored in a task artifact."""
    if not isinstance(message, Mapping):
        return message
    data = message.get("data")
    return data if isinstance(data, Mapping) else message


def _project_researcher_trace(
    artifacts: Sequence[Mapping[str, Any]] | None,
) -> tuple[list[ResearcherToolCall], list[ResearcherToolResult]]:
    """Project bounded researcher calls/results from integrity-checked artifacts."""
    calls: list[ResearcherToolCall] = []
    results: list[ResearcherToolResult] = []
    for artifact in list(artifacts or [])[:_MAX_COLLECTION_ITEMS]:
        task_id = (
            str(artifact["task_id"])
            if artifact.get("task_id") is not None
            else None
        )
        messages = artifact.get("researcher_messages", [])
        if not isinstance(messages, list):
            continue
        for raw_message in messages:
            message = _artifact_message_data(raw_message)
            raw_calls = _message_value(message, "tool_calls", [])
            if isinstance(raw_calls, list):
                for call in raw_calls:
                    if (
                        len(calls) >= _MAX_COLLECTION_ITEMS
                        or not isinstance(call, Mapping)
                    ):
                        continue
                    raw_args = call.get("args", {})
                    calls.append(ResearcherToolCall(
                        task_id=task_id,
                        name=(
                            str(call["name"])
                            if call.get("name") is not None
                            else None
                        ),
                        args=(
                            _bounded_json_value(raw_args)
                            if isinstance(raw_args, Mapping)
                            else {}
                        ),
                        id=(
                            str(call["id"])
                            if call.get("id") is not None
                            else None
                        ),
                    ))
            if (
                len(results) >= _MAX_COLLECTION_ITEMS
                or _message_value(message, "type") != "tool"
            ):
                continue
            results.append(ResearcherToolResult(
                task_id=task_id,
                name=(
                    str(_message_value(message, "name"))
                    if _message_value(message, "name") is not None
                    else None
                ),
                tool_call_id=(
                    str(_message_value(message, "tool_call_id"))
                    if _message_value(message, "tool_call_id") is not None
                    else None
                ),
                status=(
                    str(_message_value(message, "status"))
                    if _message_value(message, "status") is not None
                    else None
                ),
            ))
    return calls, results


def build_evaluation_snapshot(
    state: Mapping[str, Any],
    *,
    coverage_checklist: Sequence[str] | None = None,
    researcher_task_artifacts: Sequence[Mapping[str, Any]] | None = None,
) -> EvaluationSnapshot:
    """Project mutable runtime state into the versioned evaluation contract."""
    calls, results = _project_supervisor_trace(state)
    researcher_calls, researcher_results = _project_researcher_trace(
        researcher_task_artifacts
    )
    task_outputs = _unwrap_override(state.get("completed_task_outputs", []))
    messages = _unwrap_override(state.get("supervisor_messages", []))
    result = state.get("result", {})
    run_metrics = result.get("metrics", {}) if isinstance(result, Mapping) else {}
    limits = state.get("evaluation_metadata", {})
    return EvaluationSnapshot(
        research_brief=(
            str(state["research_brief"])
            if state.get("research_brief") is not None
            else None
        ),
        coverage_checklist=[
            str(item)
            for item in (
                coverage_checklist
                if coverage_checklist is not None
                else _unwrap_override(state.get("coverage_checklist", [])) or []
            )
        ],
        evidence_registry=_project_evidence(state),
        tool_trace=EvaluationToolTrace(
            supervisor_tool_calls=calls,
            supervisor_tool_results=results,
            researcher_tool_calls=researcher_calls,
            researcher_tool_results=researcher_results,
            completed_task_metrics=_project_task_metrics(state),
            run_metrics=(
                _bounded_json_value(run_metrics)
                if isinstance(run_metrics, Mapping)
                else {}
            ),
            limits=(
                _bounded_json_value(limits)
                if isinstance(limits, Mapping)
                else {}
            ),
            availability=ToolTraceAvailability(
                supervisor_messages_present=bool(messages),
                completed_task_outputs_present=bool(task_outputs),
                researcher_tool_names_retained=bool(researcher_task_artifacts),
            ),
            scope_note=(
                "Researcher tool names, bounded redacted arguments, and outcome "
                "statuses were recovered from integrity-checked task artifacts."
                if researcher_task_artifacts
                else (
                    "Researcher-level tool names are not retained; score only "
                    "observable evidence."
                )
            ),
        ),
    )
