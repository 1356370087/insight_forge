"""Synchronous Supervisor delegation tool."""

from __future__ import annotations

from functools import partial
from typing import Any, cast

from langchain_core.messages import BaseMessage, HumanMessage, message_to_dict
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from open_deep_research.configuration import Configuration
from open_deep_research.run_context import RunContextStore
from open_deep_research.tools.base import (
    ProgressCallback,
    Tool,
    ToolContext,
    ToolOrigin,
    ToolResult,
    build_tool,
)
from open_deep_research.tools.supervisor.common import (
    bind_run_context_fence,
    coverage_bound_input_schema,
    validate_requirement_ids,
)
from open_deep_research.tools.supervisor.conduct_research.prompt import (
    DESCRIPTION,
    TOOL_NAME,
    render_prompt,
)
from open_deep_research.tools.supervisor.deps import SupervisorToolDeps


class ConductResearch(BaseModel):
    """Delegate one complete and independent research objective."""

    research_topic: str = Field(
        description=(
            "A complete, self-contained research objective focused on one independent "
            "direction. Describe what the sub-agent needs to learn and any essential "
            "context. This is a research objective, not a search-engine query to copy "
            "verbatim; the sub-agent will begin with short, broad queries and narrow "
            "them based on evidence."
        )
    )
    display_title: str | None = Field(
        default=None,
        max_length=160,
        description="Short user-visible label for this delegated research task.",
    )
    requirement_ids: list[str] = Field(
        default_factory=list,
        description=(
            "Coverage requirement IDs owned by this task. Quality-gate v4 requires "
            "at least one ID from the original user coverage contract."
        ),
    )


async def _call(
    deps: SupervisorToolDeps,
    input: ConductResearch,
    context: ToolContext,
    on_progress: ProgressCallback | None = None,
) -> ToolResult[Any]:
    """Execute one delegated Researcher without capturing assembly locals."""
    del on_progress
    configurable = Configuration.from_runnable_config(context.config)
    run_id = str(context.config.get("metadata", {}).get("run_id", "default"))
    task_id = context.tool_call_id
    quality_v4 = (
        str(context.config.get("metadata", {}).get("quality_policy_version", ""))
        == "quality-gate-v4"
    )
    requirement_ids = validate_requirement_ids(
        input.requirement_ids,
        deps.coverage_contract,
        required=quality_v4,
    )
    researcher_config = cast(
        RunnableConfig,
        {
        **context.config,
        "metadata": {
            **context.config.get("metadata", {}),
            "task_id": task_id,
        },
        },
    )
    context_store = bind_run_context_fence(
        RunContextStore(
            run_id,
            runs_dir=configurable.runs_dir,
            inline_content_max_chars=configurable.query_journal_inline_content_max_chars,
        ),
        context.config,
    )
    existing_ref = deps.research_artifact_refs.get(task_id, {})
    if existing_ref.get("sha256"):
        try:
            existing = context_store.load_task_result(
                task_id,
                expected_sha256=str(existing_ref["sha256"]),
            )
            return ToolResult(
                output={
                    "task_id": task_id,
                    "research_topic": input.research_topic,
                    "requirement_ids": requirement_ids,
                    "compressed_research": str(
                        existing.get("compressed_research", "")
                    ),
                    "artifact_ref": dict(existing_ref),
                    "metrics": dict(existing.get("metrics", {})),
                }
            )
        except (FileNotFoundError, ValueError):
            pass

    coverage_payload = (
        deps.coverage_contract.model_dump(mode="json")
        if deps.coverage_contract is not None
        else {}
    )
    risk_payload = deps.risk_profile.model_dump(mode="json")
    observation = await deps.researcher_ainvoke(
        {
            "researcher_messages": [HumanMessage(content=input.research_topic)],
            "research_topic": input.research_topic,
            "requirement_ids": requirement_ids,
            "coverage_contract": coverage_payload,
            "research_risk_profile": risk_payload,
            "memory_context": deps.memory_context,
        },
        researcher_config,
    )
    artifact = {
        "schema_version": 2,
        "task_id": task_id,
        "research_topic": input.research_topic,
        "requirement_ids": requirement_ids,
        "coverage_contract": coverage_payload,
        "research_risk_profile": risk_payload,
        "compressed_research": str(observation.get("compressed_research", "")),
        "researcher_messages": [
            message_to_dict(message) if isinstance(message, BaseMessage) else message
            for message in observation.get("researcher_messages", [])
        ],
        "raw_notes": list(observation.get("raw_notes", [])),
        "candidate_registry": list(observation.get("candidate_registry", [])),
        "document_registry": list(observation.get("document_registry", [])),
        "evidence_registry": list(observation.get("evidence_registry", [])),
        "web_research_iterations": list(
            observation.get("web_research_iterations", [])
        ),
        "result_assessment": observation.get("result_assessment", {}),
        "metrics": dict(observation.get("metrics", {})),
    }
    digest = context_store.persist_task_result(task_id, artifact)
    relative_path = f"context/artifacts/research_tasks/{task_id}.json"
    artifact_path = context_store.run_dir / relative_path
    available_sections = [
        key
        for key in (
            "researcher_messages",
            "raw_notes",
            "candidate_registry",
            "document_registry",
            "evidence_registry",
            "web_research_iterations",
            "result_assessment",
        )
        if artifact.get(key)
    ]
    return ToolResult(
        output={
            "task_id": task_id,
            "research_topic": input.research_topic,
            "requirement_ids": requirement_ids,
            "compressed_research": artifact["compressed_research"],
            "artifact_ref": {
                "path": relative_path,
                "sha256": digest,
                "content_bytes": artifact_path.stat().st_size,
                "available_sections": available_sections,
            },
            "metrics": artifact["metrics"],
        }
    )


def build_conduct_research(deps: SupervisorToolDeps) -> Tool:
    """Build the synchronous delegation tool from explicit dependencies."""
    return build_tool(
        name=TOOL_NAME,
        description=DESCRIPTION,
        input_schema=coverage_bound_input_schema(
            ConductResearch,
            deps.coverage_contract,
        ),
        call=partial(_call, deps),
        origin=ToolOrigin.SYSTEM,
        prompt=render_prompt,
    )


__all__ = ["ConductResearch", "build_conduct_research"]
