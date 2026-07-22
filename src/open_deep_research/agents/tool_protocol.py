"""Tool-call transcript normalization and closure validation."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

from open_deep_research.tools.governance import ToolError, ToolErrorType


@dataclass(frozen=True, slots=True)
class ToolProtocolDiagnostic:
    """One deterministic protocol defect found in a tool batch."""

    code: str
    tool_call_id: str | None = None
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible diagnostic."""
        return {
            "code": self.code,
            "tool_call_id": self.tool_call_id,
            "detail": dict(self.detail),
        }


@dataclass(frozen=True, slots=True)
class ClosedToolBatch:
    """A canonical one-result-per-call tool batch."""

    messages: tuple[ToolMessage, ...]
    additional_messages: tuple[BaseMessage, ...]
    diagnostics: tuple[ToolProtocolDiagnostic, ...]

    @property
    def is_valid(self) -> bool:
        """Return whether reconciliation found no protocol defects."""
        return not self.diagnostics


def _canonical_tool_call_id(
    tool_call: dict[str, Any],
    *,
    run_id: str,
    role: str,
    turn: int,
    index: int,
) -> str:
    payload = json.dumps(
        {
            "run_id": run_id,
            "role": role,
            "turn": turn,
            "index": index,
            "name": str(tool_call.get("name", "")),
            "args": tool_call.get("args", {}) or {},
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return f"runtime-{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:24]}"


def canonicalize_ai_tool_calls(
    message: BaseMessage,
    *,
    run_id: str,
    role: str,
    turn: int,
) -> tuple[BaseMessage, tuple[ToolProtocolDiagnostic, ...]]:
    """Return an AI message whose tool-call IDs are non-empty and unique."""
    if not isinstance(message, AIMessage) or not message.tool_calls:
        return message, ()

    seen: set[str] = set()
    canonical_calls: list[dict[str, Any]] = []
    diagnostics: list[ToolProtocolDiagnostic] = []
    for index, original_call in enumerate(message.tool_calls):
        call = dict(original_call)
        original_id = str(call.get("id", "")).strip()
        call_id = original_id
        diagnostic_code: str | None = None
        if not call_id:
            diagnostic_code = "missing_tool_call_id"
        elif call_id in seen:
            diagnostic_code = "duplicate_tool_call_id"
        if diagnostic_code is not None:
            call_id = _canonical_tool_call_id(
                call,
                run_id=run_id,
                role=role,
                turn=turn,
                index=index,
            )
            suffix = 1
            base = call_id
            while call_id in seen:
                call_id = f"{base}-{suffix}"
                suffix += 1
            diagnostics.append(ToolProtocolDiagnostic(
                code=diagnostic_code,
                tool_call_id=call_id,
                detail={"original_id": original_id or None, "index": index},
            ))
        call["id"] = call_id
        seen.add(call_id)
        canonical_calls.append(call)

    copied = message.model_copy(deep=True)
    copied.tool_calls = canonical_calls
    raw_calls = copied.additional_kwargs.get("tool_calls")
    if isinstance(raw_calls, list) and len(raw_calls) == len(canonical_calls):
        copied.additional_kwargs["tool_calls"] = [
            {**dict(raw_call), "id": canonical_call["id"]}
            if isinstance(raw_call, dict)
            else raw_call
            for raw_call, canonical_call in zip(raw_calls, canonical_calls)
        ]
    return copied, tuple(diagnostics)


def _protocol_error_message(
    call: dict[str, Any],
    error_type: ToolErrorType,
    message: str,
    *,
    detail: dict[str, Any] | None = None,
) -> ToolMessage:
    return ToolError(
        error_type=error_type,
        tool_name=str(call.get("name", "unknown_tool")),
        message=message,
        detail=detail or {},
    ).to_tool_message(str(call["id"]))


def close_tool_batch(
    tool_calls: list[dict[str, Any]],
    candidate_messages: list[BaseMessage],
    additional_messages: list[BaseMessage] | None = None,
    *,
    missing_error_type: ToolErrorType = ToolErrorType.runtime_missing_result,
    missing_message: str = "The runtime did not produce a result for this tool call.",
    initial_diagnostics: tuple[ToolProtocolDiagnostic, ...] = (),
) -> ClosedToolBatch:
    """Reconcile arbitrary hook output into exactly one result per call."""
    expected = {str(call["id"]): call for call in tool_calls}
    grouped: dict[str, list[ToolMessage]] = defaultdict(list)
    diagnostics = list(initial_diagnostics)

    for message in candidate_messages:
        if not isinstance(message, ToolMessage):
            diagnostics.append(ToolProtocolDiagnostic(
                code="non_tool_batch_result",
                detail={"message_type": getattr(message, "type", type(message).__name__)},
            ))
            continue
        call_id = str(message.tool_call_id)
        if call_id not in expected:
            diagnostics.append(ToolProtocolDiagnostic(
                code="unknown_tool_result",
                tool_call_id=call_id,
            ))
            continue
        grouped[call_id].append(message)

    clean_additional: list[BaseMessage] = []
    for message in additional_messages or []:
        if isinstance(message, ToolMessage):
            diagnostics.append(ToolProtocolDiagnostic(
                code="tool_message_in_additional_messages",
                tool_call_id=str(message.tool_call_id),
            ))
        else:
            clean_additional.append(message)

    closed: list[ToolMessage] = []
    for call in tool_calls:
        call_id = str(call["id"])
        matches = grouped.get(call_id, [])
        if not matches:
            diagnostics.append(ToolProtocolDiagnostic(
                code="missing_tool_result",
                tool_call_id=call_id,
            ))
            closed.append(_protocol_error_message(
                call,
                missing_error_type,
                missing_message,
            ))
            continue
        if len(matches) > 1:
            diagnostics.append(ToolProtocolDiagnostic(
                code="duplicate_tool_result",
                tool_call_id=call_id,
                detail={"count": len(matches)},
            ))
            closed.append(_protocol_error_message(
                call,
                ToolErrorType.runtime_duplicate_result,
                "The runtime produced multiple results for one tool call.",
                detail={"result_count": len(matches)},
            ))
            continue
        copied = matches[0].model_copy(deep=True)
        copied.tool_call_id = call_id
        copied.name = str(call.get("name", copied.name or "unknown_tool"))
        closed.append(copied)

    return ClosedToolBatch(
        messages=tuple(closed),
        additional_messages=tuple(clean_additional),
        diagnostics=tuple(diagnostics),
    )


def validate_tool_transcript(
    messages: list[BaseMessage],
    *,
    allow_pending_tail: bool = False,
) -> None:
    """Raise when any AI tool call lacks one ordered following ToolMessage."""
    result_indices: set[int] = set()
    for index, message in enumerate(messages):
        if not isinstance(message, AIMessage) or not message.tool_calls:
            continue
        expected_ids = [str(call.get("id", "")) for call in message.tool_calls]
        if not all(expected_ids) or len(set(expected_ids)) != len(expected_ids):
            raise ValueError("invalid_tool_call_ids")
        result_ids: list[str] = []
        cursor = index + 1
        while cursor < len(messages) and isinstance(messages[cursor], ToolMessage):
            result_indices.add(cursor)
            result_ids.append(str(messages[cursor].tool_call_id))
            cursor += 1
        if allow_pending_tail and index == len(messages) - 1 and not result_ids:
            continue
        if not result_ids or len(result_ids) != len(expected_ids):
            raise ValueError("tool_batch_not_closed")
        if result_ids != expected_ids:
            raise ValueError("tool_result_order_mismatch")

    if any(
        isinstance(message, ToolMessage) and index not in result_indices
        for index, message in enumerate(messages)
    ):
        raise ValueError("orphan_tool_result")
