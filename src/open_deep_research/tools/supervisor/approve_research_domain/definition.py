"""Research domain approval tool."""

from functools import partial
from typing import Any

from pydantic import BaseModel, Field

from open_deep_research.configuration import Configuration
from open_deep_research.tasks.async_tools import handle_approve_research_domain
from open_deep_research.tasks.registry import get_task_registry
from open_deep_research.tasks.state import get_task_state_store
from open_deep_research.tools.base import (
    ProgressCallback,
    Tool,
    ToolContext,
    ToolOrigin,
    ToolResult,
    build_tool,
)
from open_deep_research.tools.supervisor.approve_research_domain.prompt import (
    DESCRIPTION,
    TOOL_NAME,
    render_prompt,
)
from open_deep_research.tools.supervisor.common import event_writer, tool_call_payload
from open_deep_research.tools.supervisor.deps import SupervisorToolDeps


class ApproveResearchDomain(BaseModel):
    """Approve or deny a domain requested by a paused research task."""

    task_id: str = Field(description="Task ID that is waiting for confirmation.")
    domain: str = Field(
        description="Domain to approve or deny (e.g. 'example.com')."
    )
    allow: bool = Field(
        description="True to allow the domain for this run, False to deny."
    )


async def _call(
    deps: SupervisorToolDeps,
    input: ApproveResearchDomain,
    context: ToolContext,
    on_progress: ProgressCallback | None = None,
) -> ToolResult[Any]:
    del deps, on_progress
    configurable = Configuration.from_runnable_config(context.config)
    run_id = str(context.config.get("metadata", {}).get("run_id", "default"))
    writer = event_writer(configurable, run_id)
    try:
        message = await handle_approve_research_domain(
            tool_call_payload(TOOL_NAME, input, context),
            context.config,
            get_task_registry(),
            writer,
            get_task_state_store(configurable),
        )
        return ToolResult(output=message.content)
    finally:
        if writer is not None:
            writer.close()


def build_approve_research_domain(deps: SupervisorToolDeps) -> Tool:
    return build_tool(
        name=TOOL_NAME,
        description=DESCRIPTION,
        input_schema=ApproveResearchDomain,
        call=partial(_call, deps),
        origin=ToolOrigin.SYSTEM,
        prompt=render_prompt,
    )


__all__ = ["ApproveResearchDomain", "build_approve_research_domain"]
