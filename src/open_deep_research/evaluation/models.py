"""Versioned, data-minimized models persisted for offline evaluation."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class EvaluationEvidence(BaseModel):
    """The evidence fields required by grounding and citation evaluators."""

    model_config = ConfigDict(extra="forbid")

    evidence_id: str | None = None
    claim: str | None = None
    supporting_excerpt: str | None = None
    source_url: str | None = None
    source_title: str | None = None
    source_authority: float | str | None = None
    locator: str | None = None
    confidence: Any = None
    conflict_group: str | None = None
    security_status: Literal["accepted"] = "accepted"


class CompletedTaskMetric(BaseModel):
    """Bounded metrics retained after full task outputs are released."""

    model_config = ConfigDict(extra="forbid")

    task_id: str | None = None
    research_topic: str | None = None
    query_count: int | float | None = None
    source_count: int | float | None = None
    citation_count: int | float | None = None
    elapsed_seconds: int | float | None = None
    admission_status: str | None = None


class SupervisorToolCall(BaseModel):
    """One supervisor tool request with bounded, secret-redacted arguments."""

    model_config = ConfigDict(extra="forbid")

    name: str | None = None
    args: dict[str, Any] = Field(default_factory=dict)
    id: str | None = None


class SupervisorToolResult(BaseModel):
    """A bounded preview of one supervisor tool result."""

    model_config = ConfigDict(extra="forbid")

    name: str | None = None
    tool_call_id: str | None = None
    content_preview: str = ""


class ResearcherToolCall(BaseModel):
    """One bounded tool request recovered from a durable researcher artifact."""

    model_config = ConfigDict(extra="forbid")

    task_id: str | None = None
    name: str | None = None
    args: dict[str, Any] = Field(default_factory=dict)
    id: str | None = None


class ResearcherToolResult(BaseModel):
    """One researcher tool outcome without retaining its potentially large content."""

    model_config = ConfigDict(extra="forbid")

    task_id: str | None = None
    name: str | None = None
    tool_call_id: str | None = None
    status: str | None = None


class ToolTraceAvailability(BaseModel):
    """Explicitly records which parts of the tool trace were observable."""

    model_config = ConfigDict(extra="forbid")

    supervisor_messages_present: bool
    completed_task_outputs_present: bool
    researcher_tool_names_retained: bool = False


class EvaluationToolTrace(BaseModel):
    """Stable tool-efficiency input independent of mutable runtime state."""

    model_config = ConfigDict(extra="forbid")

    supervisor_tool_calls: list[SupervisorToolCall] = Field(default_factory=list)
    supervisor_tool_results: list[SupervisorToolResult] = Field(default_factory=list)
    researcher_tool_calls: list[ResearcherToolCall] = Field(default_factory=list)
    researcher_tool_results: list[ResearcherToolResult] = Field(default_factory=list)
    completed_task_metrics: list[CompletedTaskMetric] = Field(default_factory=list)
    run_metrics: dict[str, Any] = Field(default_factory=dict)
    limits: dict[str, Any] = Field(default_factory=dict)
    availability: ToolTraceAvailability
    scope_note: str = (
        "Researcher-level tool names are not retained; score only observable evidence."
    )


class EvaluationSnapshot(BaseModel):
    """Versioned inputs needed to evaluate one completed research run."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    research_brief: str | None = None
    coverage_checklist: list[str] = Field(default_factory=list)
    evidence_registry: list[EvaluationEvidence] = Field(default_factory=list)
    tool_trace: EvaluationToolTrace
