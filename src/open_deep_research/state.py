"""Graph state definitions and data structures for the Deep Research agent."""

import operator
from typing import Annotated, Optional

from langchain_core.messages import MessageLikeRepresentation
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


###################
# Structured Outputs
###################
class Summary(BaseModel):
    """Research summary with key findings."""
    
    summary: str
    key_excerpts: str

class ClarifyWithUser(BaseModel):
    """Model for user clarification requests."""
    
    need_clarification: bool = Field(
        description="Whether the user needs to be asked a clarifying question.",
    )
    question: str = Field(
        description="A question to ask the user to clarify the report scope",
    )
    verification: str = Field(
        description="Verify message that we will start research after the user has provided the necessary information.",
    )

class ResearchQuestion(BaseModel):
    """Research question and brief for guiding research."""
    
    research_brief: str = Field(
        description="A research question that will be used to guide the research.",
    )

###################
# State Definitions
###################

def override_reducer(current_value, new_value):
    """Reducer function that allows overriding values in state."""
    if isinstance(new_value, dict) and new_value.get("type") == "override":
        return new_value.get("value", new_value)
    else:
        return operator.add(current_value, new_value)
    
class AgentInputState(TypedDict):
    """InputState is only messages."""

    messages: list[MessageLikeRepresentation]

class AgentState(AgentInputState, total=False):
    """Main agent state containing messages and research data."""

    supervisor_messages: Annotated[list[MessageLikeRepresentation], override_reducer]
    research_brief: Optional[str]
    raw_notes: Annotated[list[str], override_reducer]
    notes: Annotated[list[str], override_reducer]
    final_report: str
    research_plan: Optional[str]
    approved_research_plan: Optional[str]
    report_outline: Optional[str]
    human_feedback: Annotated[list[dict], override_reducer]
    pending_human_action: Optional[dict]
    # Async SubAgent: collected outputs from completed background tasks.
    completed_task_outputs: Annotated[list[dict], override_reducer]
    processed_mailbox_message_ids: Annotated[list[str], override_reducer]
    # Mem0 long-term memory
    memory_context: Optional[str]
    memory_candidates: Annotated[list[dict], override_reducer]
    # Running short-term summary used after compacting long message histories.
    conversation_summary: Optional[str]
    # Report product system: non-markdown output artifacts (structured_json,
    # slides, one_pager, ...). Absent for the default markdown profile.
    report_artifacts: Optional[dict]
    # Structured sources (title+url) recovered from research findings, used to
    # render the References section in the selected style. Populated by the
    # report orchestrator only when reference handling runs (non-default style).
    sources: Annotated[list[dict], override_reducer]
    # Deterministically derived report requirements used by both writer and Judge.
    coverage_checklist: Annotated[list[str], override_reducer]
    coverage_contract: dict
    coverage_ledger: dict
    research_risk_profile: dict
    # Versioned, data-minimized inputs retained for offline evaluation after
    # transient notes and full task outputs are cleared.
    evaluation_snapshot: dict
    candidate_registry: Annotated[list[dict], override_reducer]
    document_registry: Annotated[list[dict], override_reducer]
    evidence_registry: Annotated[list[dict], override_reducer]
    web_research_iterations: Annotated[list[dict], override_reducer]
    completion_decision: dict
    permission_denials: Annotated[list[dict], override_reducer]
    # Quality-gate state promoted from the supervisor so recovery and resume
    # paths use the same persisted assessment history and artifact references.
    handoff_assessments: Annotated[list[dict], override_reducer]
    research_artifact_refs: dict
    quality_gate: dict
    recoverable_artifacts: Annotated[list[dict], override_reducer]
    applied_query_event_ids: Annotated[list[str], override_reducer]

class SupervisorState(TypedDict, total=False):
    """State for the supervisor that manages research tasks."""

    supervisor_messages: Annotated[list[MessageLikeRepresentation], override_reducer]
    research_brief: str
    coverage_contract: dict
    coverage_ledger: dict
    research_risk_profile: dict
    notes: Annotated[list[str], override_reducer]
    research_iterations: int
    raw_notes: Annotated[list[str], override_reducer]
    enable_async_research: bool
    memory_context: Optional[str]
    approved_research_plan: Optional[str]
    human_feedback: Annotated[list[dict], override_reducer]
    handoff_assessments: Annotated[list[dict], override_reducer]
    pending_mailbox_acks: Annotated[list[dict], override_reducer]
    processed_mailbox_message_ids: Annotated[list[str], override_reducer]
    research_artifact_refs: dict
    candidate_registry: Annotated[list[dict], override_reducer]
    document_registry: Annotated[list[dict], override_reducer]
    evidence_registry: Annotated[list[dict], override_reducer]
    web_research_iterations: Annotated[list[dict], override_reducer]
    completion_decision: dict
    permission_denials: Annotated[list[dict], override_reducer]
    applied_query_event_ids: Annotated[list[str], override_reducer]

class ResearcherState(TypedDict, total=False):
    """State for individual researchers conducting research."""
    
    researcher_messages: Annotated[list[MessageLikeRepresentation], operator.add]
    tool_call_iterations: int
    research_topic: str
    requirement_ids: list[str]
    coverage_contract: dict
    research_risk_profile: dict
    compressed_research: str
    raw_notes: Annotated[list[str], override_reducer]
    memory_context: Optional[str]
    pending_tool_results: list[dict]
    research_complete_requested: bool
    research_complete_succeeded: bool
    result_assessment: dict
    completion_decision: dict
    permission_denials: Annotated[list[dict], override_reducer]
    candidate_registry: Annotated[list[dict], override_reducer]
    document_registry: Annotated[list[dict], override_reducer]
    evidence_registry: Annotated[list[dict], override_reducer]
    web_research_iterations: Annotated[list[dict], override_reducer]
    applied_query_event_ids: Annotated[list[str], override_reducer]

class ResearcherOutputState(BaseModel):
    """Output state from individual researchers."""
    
    compressed_research: str
    raw_notes: Annotated[list[str], override_reducer]
