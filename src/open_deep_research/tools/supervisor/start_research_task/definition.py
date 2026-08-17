"""Asynchronous Researcher task launcher."""

from __future__ import annotations

from functools import partial
from typing import Any

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from open_deep_research.configuration import Configuration
from open_deep_research.tasks.async_tools import handle_start_research_task
from open_deep_research.tasks.registry import TaskRecord, get_task_registry
from open_deep_research.tasks.teammate_pool import TeammatePool, get_teammate_pool
from open_deep_research.tools.base import (
    ProgressCallback,
    Tool,
    ToolContext,
    ToolOrigin,
    ToolResult,
    build_tool,
)
from open_deep_research.tools.supervisor.common import (
    coverage_bound_input_schema,
    event_writer,
    tool_call_payload,
    validate_requirement_ids,
)
from open_deep_research.tools.supervisor.deps import SupervisorToolDeps
from open_deep_research.tools.supervisor.start_research_task.prompt import (
    DESCRIPTION,
    TOOL_NAME,
    render_prompt,
)


class StartResearchTask(BaseModel):
    """Launch an asynchronous background research task."""

    research_topic: str = Field(
        description=(
            "A complete, self-contained research objective focused on one independent "
            "direction. Describe what the sub-agent needs to learn and any essential "
            "context. This is a research objective, not a search-engine query to copy "
            "verbatim; the sub-agent will begin with short, broad queries and narrow "
            "them based on evidence."
        )
    )
    requirement_ids: list[str] = Field(
        default_factory=list,
        description=(
            "Coverage requirement IDs owned by this task. For quality-gate-v4 runs "
            "every ID must exist in the user coverage contract."
        ),
    )
    display_title: str | None = Field(
        default=None,
        max_length=160,
        description="Short user-visible label for this delegated research task.",
    )


def _launch_task(
    pool: TeammatePool,
    record: TaskRecord,
    config: RunnableConfig,
):
    del config
    return pool.submit(record)


async def _call(
    deps: SupervisorToolDeps,
    input: StartResearchTask,
    context: ToolContext,
    on_progress: ProgressCallback | None = None,
) -> ToolResult[Any]:
    del on_progress
    configurable = Configuration.from_runnable_config(context.config)
    requirement_ids = validate_requirement_ids(
        input.requirement_ids,
        deps.coverage_contract,
        required=deps.coverage_contract is not None,
    )
    input = input.model_copy(update={"requirement_ids": requirement_ids})
    registry = get_task_registry()
    run_id = str(context.config.get("metadata", {}).get("run_id", "default"))
    writer = event_writer(configurable, run_id)
    pool = get_teammate_pool(context.config, registry, deps.researcher_ainvoke)
    try:
        message = await handle_start_research_task(
            tool_call_payload(TOOL_NAME, input, context),
            context.config,
            registry,
            launch_task=partial(_launch_task, pool),
            event_writer=writer,
            memory_context=deps.memory_context,
            coverage_contract=(
                deps.coverage_contract.model_dump()
                if deps.coverage_contract is not None
                else None
            ),
            research_risk_profile=deps.risk_profile.model_dump(),
        )
        return ToolResult(output=message.content)
    finally:
        if writer is not None:
            writer.close()


def build_start_research_task(deps: SupervisorToolDeps) -> Tool:
    return build_tool(
        name=TOOL_NAME,
        description=DESCRIPTION,
        input_schema=coverage_bound_input_schema(
            StartResearchTask,
            deps.coverage_contract,
        ),
        call=partial(_call, deps),
        origin=ToolOrigin.SYSTEM,
        prompt=render_prompt,
    )


__all__ = ["StartResearchTask", "build_start_research_task"]
