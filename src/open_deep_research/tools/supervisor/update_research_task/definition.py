"""Asynchronous task instruction updater."""

from functools import partial
from typing import Any

from pydantic import BaseModel, Field

from open_deep_research.configuration import Configuration
from open_deep_research.tasks.async_tools import handle_update_research_task
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
from open_deep_research.tools.supervisor.common import event_writer, tool_call_payload
from open_deep_research.tools.supervisor.deps import SupervisorToolDeps
from open_deep_research.tools.supervisor.update_research_task.prompt import (
    DESCRIPTION,
    TOOL_NAME,
    render_prompt,
)


class UpdateResearchTask(BaseModel):
    """Append corrective or additional instructions to a running task."""

    task_id: str = Field(description="Task ID to update.")
    instruction: str = Field(
        description="Additional or corrected instruction for the running task."
    )


async def _call(
    deps: SupervisorToolDeps,
    input: UpdateResearchTask,
    context: ToolContext,
    on_progress: ProgressCallback | None = None,
) -> ToolResult[Any]:
    del deps, on_progress
    configurable = Configuration.from_runnable_config(context.config)
    run_id = str(context.config.get("metadata", {}).get("run_id", "default"))
    writer = event_writer(configurable, run_id)
    try:
        message = await handle_update_research_task(
            tool_call_payload(TOOL_NAME, input, context),
            get_task_registry(),
            writer,
            get_task_state_store(configurable),
            run_id=run_id,
            fence_token=int(
                context.config.get("metadata", {}).get("run_fence_token", 0) or 0
            ),
        )
        return ToolResult(output=message.content)
    finally:
        if writer is not None:
            writer.close()


def build_update_research_task(deps: SupervisorToolDeps) -> Tool:
    return build_tool(
        name=TOOL_NAME,
        description=DESCRIPTION,
        input_schema=UpdateResearchTask,
        call=partial(_call, deps),
        origin=ToolOrigin.SYSTEM,
        prompt=render_prompt,
    )


__all__ = ["UpdateResearchTask", "build_update_research_task"]
