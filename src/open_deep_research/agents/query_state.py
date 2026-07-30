"""Serializable immutable state for the inner Query agent loop."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Protocol

from langchain_core.messages import BaseMessage, messages_from_dict, messages_to_dict

from open_deep_research.tools.governance import AgentRole

QUERY_STATE_SCHEMA_VERSION = 2
SUPPORTED_QUERY_STATE_SCHEMA_VERSIONS = frozenset({1, 2})


class QueryPhase(str, Enum):
    """Stable phases crossed by one inner Query loop."""

    PREPARING = "preparing"
    CALLING_MODEL = "calling_model"
    EXECUTING_TOOLS = "executing_tools"
    STOP_GOVERNANCE = "stop_governance"
    TERMINAL = "terminal"


class ContinueReason(str, Enum):
    """Reasons that create another inner-loop iteration."""

    NEXT_TURN = "next_turn"
    EXTERNAL_UPDATE = "external_update"
    STOP_HOOK_BLOCKING = "stop_hook_blocking"
    CONTEXT_REPROJECT_RETRY = "context_reproject_retry"
    REACTIVE_COMPACT_RETRY = "reactive_compact_retry"
    OUTPUT_TOKEN_ESCALATE = "output_token_escalate"
    OUTPUT_CONTINUATION = "output_continuation"
    MODEL_FALLBACK = "model_fallback"


class TerminalReason(str, Enum):
    """Reasons that permanently end an inner Query loop."""

    COMPLETED = "completed"
    MAX_TURNS = "max_turns"
    CANCELLED = "cancelled"
    BUDGET_EXHAUSTED = "budget_exhausted"
    DEADLINE_EXCEEDED = "deadline_exceeded"
    MODEL_TIMEOUT = "model_timeout"
    PROMPT_TOO_LONG = "prompt_too_long"
    OUTPUT_RECOVERY_EXHAUSTED = "output_recovery_exhausted"
    MODEL_ERROR = "model_error"
    TOOL_PROTOCOL_VIOLATION = "tool_protocol_violation"
    HOOK_STOPPED = "hook_stopped"


class StopAction(str, Enum):
    """Explicit decision returned by a stop hook."""

    CONTINUE = "continue"
    COMPLETE = "complete"
    HALT = "halt"


@dataclass(frozen=True, slots=True)
class ContextRecoveryState:
    """Bounded prompt-size recovery bookkeeping."""

    attempts: int = 0
    reactive_compact_attempted: bool = False
    compaction_generation: int = 0
    target_ratio: float = 1.0


@dataclass(frozen=True, slots=True)
class OutputRecoveryState:
    """Bounded output-truncation recovery bookkeeping."""

    continuation_count: int = 0
    escalation_attempted: bool = False
    max_output_tokens_override: int | None = None
    pending_fragments: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ModelRouteState:
    """Current position in the frozen model candidate chain."""

    active_candidate_index: int = 0


@dataclass(frozen=True, slots=True)
class QualityRecoveryState:
    """Bounded, crash-recoverable Researcher gap-closure bookkeeping."""

    attempts: int = 0
    active: bool = False
    target_requirement_ids: tuple[str, ...] = ()
    triggering_assessment_revision: int | None = None


@dataclass(frozen=True, slots=True)
class PendingToolBatch:
    """Write-ahead description of a model-emitted tool batch."""

    batch_id: str
    tool_calls: tuple[dict[str, Any], ...]
    committed_tool_call_ids: tuple[str, ...] = ()
    committed_results: tuple[BaseMessage, ...] = ()
    result_refs: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class PendingQueryEvent:
    """Transactional outbox entry for an outer-state domain transition."""

    event_id: str
    event_type: str
    transition_reason: str
    turn: int
    messages: tuple[BaseMessage, ...] = ()
    additional_messages: tuple[BaseMessage, ...] = ()
    updates: Mapping[str, Any] = field(default_factory=dict)
    should_continue: bool | None = None
    protocol_diagnostics: tuple[Mapping[str, Any], ...] = ()


@dataclass(frozen=True, slots=True)
class QueryTerminal:
    """Terminal outcome stored in the Query state."""

    reason: TerminalReason
    detail: str | None = None


@dataclass(frozen=True, slots=True)
class QueryLoopState:
    """Complete recoverable control state for a Supervisor or Researcher loop."""

    state_key: str
    role: AgentRole
    messages: tuple[BaseMessage, ...]
    schema_version: int = QUERY_STATE_SCHEMA_VERSION
    phase: QueryPhase = QueryPhase.PREPARING
    turn: int = 0
    revision: int = 0
    transition_reason: str | None = None
    context_recovery: ContextRecoveryState = field(
        default_factory=ContextRecoveryState
    )
    output_recovery: OutputRecoveryState = field(
        default_factory=OutputRecoveryState
    )
    model_route: ModelRouteState = field(default_factory=ModelRouteState)
    quality_recovery: QualityRecoveryState = field(
        default_factory=QualityRecoveryState
    )
    pending_tool_batch: PendingToolBatch | None = None
    pending_query_event: PendingQueryEvent | None = None
    stop_hook_active: bool = False
    terminal: QueryTerminal | None = None

    def to_snapshot(self) -> dict[str, Any]:
        """Return a JSON-compatible checkpoint payload."""
        return {
            "schema_version": self.schema_version,
            "state_key": self.state_key,
            "role": self.role.value,
            "phase": self.phase.value,
            "messages": messages_to_dict(list(self.messages)),
            "turn": self.turn,
            "revision": self.revision,
            "transition_reason": self.transition_reason,
            "context_recovery": {
                "attempts": self.context_recovery.attempts,
                "reactive_compact_attempted": (
                    self.context_recovery.reactive_compact_attempted
                ),
                "compaction_generation": (
                    self.context_recovery.compaction_generation
                ),
                "target_ratio": self.context_recovery.target_ratio,
            },
            "output_recovery": {
                "continuation_count": self.output_recovery.continuation_count,
                "escalation_attempted": (
                    self.output_recovery.escalation_attempted
                ),
                "max_output_tokens_override": (
                    self.output_recovery.max_output_tokens_override
                ),
                "pending_fragments": list(
                    self.output_recovery.pending_fragments
                ),
            },
            "model_route": {
                "active_candidate_index": (
                    self.model_route.active_candidate_index
                ),
            },
            "quality_recovery": {
                "attempts": self.quality_recovery.attempts,
                "active": self.quality_recovery.active,
                "target_requirement_ids": list(
                    self.quality_recovery.target_requirement_ids
                ),
                "triggering_assessment_revision": (
                    self.quality_recovery.triggering_assessment_revision
                ),
            },
            "pending_tool_batch": (
                {
                    "batch_id": self.pending_tool_batch.batch_id,
                    "tool_calls": list(self.pending_tool_batch.tool_calls),
                    "committed_tool_call_ids": list(
                        self.pending_tool_batch.committed_tool_call_ids
                    ),
                    "committed_results": messages_to_dict(
                        list(self.pending_tool_batch.committed_results)
                    ),
                    "result_refs": list(self.pending_tool_batch.result_refs),
                }
                if self.pending_tool_batch is not None
                else None
            ),
            "pending_query_event": (
                {
                    "event_id": self.pending_query_event.event_id,
                    "event_type": self.pending_query_event.event_type,
                    "transition_reason": (
                        self.pending_query_event.transition_reason
                    ),
                    "turn": self.pending_query_event.turn,
                    "messages": messages_to_dict(
                        list(self.pending_query_event.messages)
                    ),
                    "additional_messages": messages_to_dict(
                        list(self.pending_query_event.additional_messages)
                    ),
                    "updates": deepcopy(dict(self.pending_query_event.updates)),
                    "should_continue": (
                        self.pending_query_event.should_continue
                    ),
                    "protocol_diagnostics": [
                        deepcopy(dict(item))
                        for item in (
                            self.pending_query_event.protocol_diagnostics
                        )
                    ],
                }
                if self.pending_query_event is not None
                else None
            ),
            "stop_hook_active": self.stop_hook_active,
            "terminal": (
                {
                    "reason": self.terminal.reason.value,
                    "detail": self.terminal.detail,
                }
                if self.terminal is not None
                else None
            ),
        }

    @classmethod
    def from_snapshot(cls, payload: Mapping[str, Any]) -> QueryLoopState:
        """Restore and validate a checkpoint payload."""
        schema_version = int(payload.get("schema_version", 0))
        if schema_version not in SUPPORTED_QUERY_STATE_SCHEMA_VERSIONS:
            raise ValueError(
                f"unsupported_query_state_schema:{schema_version}"
            )
        pending_payload = payload.get("pending_tool_batch")
        pending_result_payload = (
            list(pending_payload.get("committed_results", []))
            if isinstance(pending_payload, Mapping)
            else []
        )
        pending_results = (
            pending_result_payload
            if all(
                isinstance(item, BaseMessage)
                for item in pending_result_payload
            )
            else messages_from_dict(pending_result_payload)
        )
        pending = (
            PendingToolBatch(
                batch_id=str(pending_payload["batch_id"]),
                tool_calls=tuple(
                    dict(item) for item in pending_payload.get("tool_calls", [])
                ),
                committed_tool_call_ids=tuple(
                    str(item)
                    for item in pending_payload.get(
                        "committed_tool_call_ids", []
                    )
                ),
                committed_results=tuple(pending_results),
                result_refs=tuple(
                    str(item)
                    for item in pending_payload.get("result_refs", [])
                ),
            )
            if isinstance(pending_payload, Mapping)
            else None
        )
        pending_event_payload = payload.get("pending_query_event")
        pending_event_messages_payload = (
            list(pending_event_payload.get("messages", []))
            if isinstance(pending_event_payload, Mapping)
            else []
        )
        pending_event_messages = (
            pending_event_messages_payload
            if all(
                isinstance(item, BaseMessage)
                for item in pending_event_messages_payload
            )
            else messages_from_dict(pending_event_messages_payload)
        )
        pending_event_additional_payload = (
            list(pending_event_payload.get("additional_messages", []))
            if isinstance(pending_event_payload, Mapping)
            else []
        )
        pending_event_additional = (
            pending_event_additional_payload
            if all(
                isinstance(item, BaseMessage)
                for item in pending_event_additional_payload
            )
            else messages_from_dict(pending_event_additional_payload)
        )
        pending_event = (
            PendingQueryEvent(
                event_id=str(pending_event_payload["event_id"]),
                event_type=str(pending_event_payload["event_type"]),
                transition_reason=str(
                    pending_event_payload["transition_reason"]
                ),
                turn=int(pending_event_payload.get("turn", 0)),
                messages=tuple(pending_event_messages),
                additional_messages=tuple(pending_event_additional),
                updates=deepcopy(
                    dict(pending_event_payload.get("updates", {}))
                ),
                should_continue=(
                    bool(pending_event_payload["should_continue"])
                    if pending_event_payload.get("should_continue")
                    is not None
                    else None
                ),
                protocol_diagnostics=tuple(
                    deepcopy(dict(item))
                    for item in pending_event_payload.get(
                        "protocol_diagnostics",
                        [],
                    )
                    if isinstance(item, Mapping)
                ),
            )
            if isinstance(pending_event_payload, Mapping)
            else None
        )
        context_payload = payload.get("context_recovery") or {}
        output_payload = payload.get("output_recovery") or {}
        route_payload = payload.get("model_route") or {}
        quality_payload = payload.get("quality_recovery") or {}
        terminal_payload = payload.get("terminal")
        terminal = (
            QueryTerminal(
                reason=TerminalReason(str(terminal_payload["reason"])),
                detail=(
                    str(terminal_payload["detail"])
                    if terminal_payload.get("detail") is not None
                    else None
                ),
            )
            if isinstance(terminal_payload, Mapping)
            else None
        )
        raw_messages = list(payload.get("messages", []))
        restored_messages = (
            raw_messages
            if all(isinstance(item, BaseMessage) for item in raw_messages)
            else messages_from_dict(raw_messages)
        )
        return cls(
            schema_version=schema_version,
            state_key=str(payload["state_key"]),
            role=AgentRole(str(payload["role"])),
            phase=QueryPhase(str(payload["phase"])),
            messages=tuple(restored_messages),
            turn=int(payload.get("turn", 0)),
            revision=int(payload.get("revision", 0)),
            transition_reason=(
                str(payload["transition_reason"])
                if payload.get("transition_reason") is not None
                else None
            ),
            context_recovery=ContextRecoveryState(
                attempts=int(context_payload.get("attempts", 0)),
                reactive_compact_attempted=bool(
                    context_payload.get(
                        "reactive_compact_attempted", False
                    )
                ),
                compaction_generation=int(
                    context_payload.get("compaction_generation", 0)
                ),
                target_ratio=float(
                    context_payload.get("target_ratio", 1.0)
                ),
            ),
            output_recovery=OutputRecoveryState(
                continuation_count=int(
                    output_payload.get("continuation_count", 0)
                ),
                escalation_attempted=bool(
                    output_payload.get("escalation_attempted", False)
                ),
                max_output_tokens_override=(
                    int(output_payload["max_output_tokens_override"])
                    if output_payload.get("max_output_tokens_override")
                    is not None
                    else None
                ),
                pending_fragments=tuple(
                    str(item)
                    for item in output_payload.get("pending_fragments", [])
                ),
            ),
            model_route=ModelRouteState(
                active_candidate_index=int(
                    route_payload.get("active_candidate_index", 0)
                ),
            ),
            quality_recovery=QualityRecoveryState(
                attempts=int(quality_payload.get("attempts", 0)),
                active=bool(quality_payload.get("active", False)),
                target_requirement_ids=tuple(
                    str(item)
                    for item in quality_payload.get(
                        "target_requirement_ids",
                        [],
                    )
                ),
                triggering_assessment_revision=(
                    int(
                        quality_payload[
                            "triggering_assessment_revision"
                        ]
                    )
                    if quality_payload.get(
                        "triggering_assessment_revision"
                    )
                    is not None
                    else None
                ),
            ),
            pending_tool_batch=pending,
            pending_query_event=pending_event,
            stop_hook_active=bool(payload.get("stop_hook_active", False)),
            terminal=terminal,
        )


@dataclass(frozen=True, slots=True)
class QueryStateAction:
    """One validated immutable Query state transition."""

    phase: QueryPhase
    reason: str
    changes: Mapping[str, Any] = field(default_factory=dict)


_LEGAL_PHASE_TRANSITIONS: dict[QueryPhase, frozenset[QueryPhase]] = {
    QueryPhase.PREPARING: frozenset(
        {
            QueryPhase.PREPARING,
            QueryPhase.CALLING_MODEL,
            QueryPhase.TERMINAL,
        }
    ),
    QueryPhase.CALLING_MODEL: frozenset(
        {
            QueryPhase.PREPARING,
            QueryPhase.CALLING_MODEL,
            QueryPhase.EXECUTING_TOOLS,
            QueryPhase.STOP_GOVERNANCE,
            QueryPhase.TERMINAL,
        }
    ),
    QueryPhase.EXECUTING_TOOLS: frozenset(
        {
            QueryPhase.PREPARING,
            QueryPhase.EXECUTING_TOOLS,
            QueryPhase.CALLING_MODEL,
            QueryPhase.TERMINAL,
        }
    ),
    QueryPhase.STOP_GOVERNANCE: frozenset(
        {
            QueryPhase.PREPARING,
            QueryPhase.STOP_GOVERNANCE,
            QueryPhase.CALLING_MODEL,
            QueryPhase.TERMINAL,
        }
    ),
    QueryPhase.TERMINAL: frozenset({QueryPhase.TERMINAL}),
}


def advance(
    state: QueryLoopState,
    action: QueryStateAction,
) -> QueryLoopState:
    """Apply one legal transition without mutating the previous state."""
    if action.phase not in _LEGAL_PHASE_TRANSITIONS[state.phase]:
        raise ValueError(
            "illegal_query_phase_transition:"
            f"{state.phase.value}->{action.phase.value}"
        )
    changes = dict(action.changes)
    changes["phase"] = action.phase
    changes["transition_reason"] = action.reason
    changes["revision"] = state.revision + 1
    next_state = replace(state, **changes)
    if action.phase is QueryPhase.TERMINAL and next_state.terminal is None:
        raise ValueError("terminal_query_state_requires_outcome")
    if action.phase is not QueryPhase.TERMINAL and next_state.terminal is not None:
        raise ValueError("non_terminal_query_state_has_outcome")
    return next_state


class QueryCheckpointSink(Protocol):
    """Persistence port invoked before the loop crosses a stable boundary."""

    async def save(self, state: QueryLoopState) -> None:
        """Durably save the latest Query loop state."""
        ...


@dataclass(slots=True)
class InMemoryQueryCheckpointSink:
    """Test and embedding checkpoint sink."""

    states: list[QueryLoopState] = field(default_factory=list)

    async def save(self, state: QueryLoopState) -> None:
        """Append an immutable state snapshot."""
        self.states.append(state)
