"""Asynchronous task cancellation tool."""

from functools import partial
from typing import Any

from pydantic import BaseModel, Field

from open_deep_research.configuration import Configuration
from open_deep_research.tasks.async_tools import handle_cancel_research_task
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
from open_deep_research.tools.supervisor.cancel_research_task.prompt import (
    DESCRIPTION,
    TOOL_NAME,
    render_prompt,
)
from open_deep_research.tools.supervisor.common import event_writer, tool_call_payload
from open_deep_research.tools.supervisor.deps import SupervisorToolDeps


class CancelResearchTask(BaseModel):
    """Cancel one or more running research tasks."""

    task_ids: list[str] = Field(description="List of task IDs to cancel.")
    reason: str | None = Field(
        default=None,
        description="Optional reason for cancellation.",
    )


async def _call(
    deps: SupervisorToolDeps,
    input: CancelResearchTask,
    context: ToolContext,
    on_progress: ProgressCallback | None = None,
) -> ToolResult[Any]:
    del deps, on_progress
    configurable = Configuration.from_runnable_config(context.config)
    run_id = str(context.config.get("metadata", {}).get("run_id", "default"))
    writer = event_writer(configurable, run_id)
    try:
        message = await handle_cancel_research_task(
            tool_call_payload(TOOL_NAME, input, context),
            get_task_registry(),
            writer,
            get_task_state_store(configurable),
            configurable,
            run_id=run_id,
            fence_token=int(
                context.config.get("metadata", {}).get("run_fence_token", 0) or 0
            ),
        )
        return ToolResult(output=message.content)
    finally:
        if writer is not None:
            writer.close()


def build_cancel_research_task(deps: SupervisorToolDeps) -> Tool:
    return build_tool(
        name=TOOL_NAME,
        description=DESCRIPTION,
        input_schema=CancelResearchTask,
        call=partial(_call, deps),
        origin=ToolOrigin.SYSTEM,
        prompt=render_prompt,
    )


__all__ = ["CancelResearchTask", "build_cancel_research_task"]
