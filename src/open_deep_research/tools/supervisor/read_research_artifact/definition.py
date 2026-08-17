"""Persisted Researcher artifact reader tool."""

from __future__ import annotations

import json
from functools import partial
from typing import Any, Literal

from pydantic import BaseModel, Field

from open_deep_research.configuration import Configuration
from open_deep_research.quality_contract import ResearchRiskProfile
from open_deep_research.run_context import RunContextStore
from open_deep_research.tools.base import (
    ProgressCallback,
    Tool,
    ToolContext,
    ToolOrigin,
    ToolResult,
    build_tool,
)
from open_deep_research.tools.supervisor.deps import SupervisorToolDeps
from open_deep_research.tools.supervisor.read_research_artifact.prompt import (
    DESCRIPTION,
    TOOL_NAME,
    render_prompt,
)


class ReadResearchArtifact(BaseModel):
    """Read one bounded section from a persisted Researcher artifact."""

    task_id: str = Field(
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$",
        description="Task id returned by ConductResearch.",
    )
    artifact_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$",
        description="Artifact digest returned by ConductResearch.",
    )
    section: Literal[
        "researcher_messages",
        "raw_notes",
        "candidate_registry",
        "document_registry",
        "evidence_registry",
        "web_research_iterations",
        "result_assessment",
    ] = Field(description="Artifact section to inspect.")
    offset: int = Field(default=0, ge=0, description="Character offset for pagination.")
    max_chars: int = Field(
        default=12_000,
        ge=1,
        le=30_000,
        description="Maximum number of characters returned in this call.",
    )


async def _call(
    deps: SupervisorToolDeps,
    input: ReadResearchArtifact,
    context: ToolContext,
    on_progress: ProgressCallback | None = None,
) -> ToolResult[Any]:
    """Read and optionally reassess an artifact without closure state."""
    del on_progress
    configurable = Configuration.from_runnable_config(context.config)
    run_id = str(context.config.get("metadata", {}).get("run_id", "default"))
    context_store = RunContextStore(
        run_id,
        runs_dir=configurable.runs_dir,
        inline_content_max_chars=configurable.query_journal_inline_content_max_chars,
    )
    artifact = context_store.load_task_result(
        input.task_id,
        expected_sha256=input.artifact_sha256,
    )
    if input.section not in artifact:
        raise ValueError(
            f"Section {input.section!r} is unavailable for task {input.task_id}"
        )
    section = artifact[input.section]
    content = (
        section
        if isinstance(section, str)
        else json.dumps(section, ensure_ascii=False, default=str)
    )
    start = min(input.offset, len(content))
    end = min(start + input.max_chars, len(content))
    truncated = end < len(content)
    output: dict[str, Any] = {
        "task_id": input.task_id,
        "section": input.section,
        "offset": start,
        "content": content[start:end],
        "truncated": truncated,
        "next_offset": end if truncated else None,
        "total_chars": len(content),
        "artifact_ref": {
            "path": f"context/artifacts/research_tasks/{input.task_id}.json",
            "sha256": input.artifact_sha256,
        },
    }
    assessment_history = [
        item
        for item in deps.handoff_assessments
        if str(item.get("tool_call_id", "")) == input.task_id
    ]
    latest_assessment = assessment_history[-1] if assessment_history else None
    if (
        configurable.quality_evaluation_enabled
        and latest_assessment is not None
        and latest_assessment.get("accepted") is False
    ):
        quality_handoff = dict(artifact)
        selected_excerpt = json.dumps(
            {
                "section": input.section,
                "offset": start,
                "content": content[start:end],
            },
            ensure_ascii=False,
            default=str,
        )
        quality_handoff["raw_notes"] = [
            "Supervisor-selected excerpt from the SHA-verified research artifact:\n"
            + selected_excerpt,
            *[str(note) for note in artifact.get("raw_notes", [])],
        ]
        kwargs: dict[str, Any] = {}
        if artifact.get("coverage_contract"):
            kwargs = {
                "coverage_contract": artifact["coverage_contract"],
                "requirement_ids": list(artifact.get("requirement_ids", [])),
                "risk_profile": ResearchRiskProfile.model_validate(
                    artifact.get("research_risk_profile", {})
                    or {"level": "standard"}
                ),
            }
        reassessment = await deps.evaluate_handoff(
            str(artifact.get("research_topic", "")),
            quality_handoff,
            context.config,
            **kwargs,
        )
        output.update(
            {
                "status": (
                    "accepted_after_artifact_reassessment"
                    if reassessment.accepted
                    else "rejected_after_artifact_reassessment"
                ),
                "admission_status": (
                    reassessment.admission_status.value
                    if reassessment.admission_status is not None
                    else "accepted"
                    if reassessment.accepted
                    else "rejected"
                ),
                "reassessment": reassessment.model_dump(),
            }
        )
    return ToolResult(output=output)


def build_read_research_artifact(deps: SupervisorToolDeps) -> Tool:
    """Build the artifact reader from explicit dependencies."""
    return build_tool(
        name=TOOL_NAME,
        description=DESCRIPTION,
        input_schema=ReadResearchArtifact,
        call=partial(_call, deps),
        origin=ToolOrigin.SYSTEM,
        prompt=render_prompt,
    )


__all__ = ["ReadResearchArtifact", "build_read_research_artifact"]
