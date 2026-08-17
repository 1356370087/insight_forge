"""Asynchronous task listing tool."""

from functools import partial
from typing import Any

from pydantic import BaseModel, Field

from open_deep_research.configuration import Configuration
from open_deep_research.tasks.async_tools import handle_list_research_tasks
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
from open_deep_research.tools.supervisor.common import tool_call_payload
from open_deep_research.tools.supervisor.deps import SupervisorToolDeps
from open_deep_research.tools.supervisor.list_research_tasks.prompt import (
    DESCRIPTION,
    TOOL_NAME,
    render_prompt,
)


class ListResearchTasks(BaseModel):
    """List tracked research tasks and optionally filter by status."""

    status_filter: str | None = Field(
        default=None,
        description=(
            "Optional: filter by status ('running', 'completed', 'failed', "
            "'cancelled')."
        ),
    )


async def _call(
    deps: SupervisorToolDeps,
    input: ListResearchTasks,
    context: ToolContext,
    on_progress: ProgressCallback | None = None,
) -> ToolResult[Any]:
    del deps, on_progress
    configurable = Configuration.from_runnable_config(context.config)
    run_id = str(context.config.get("metadata", {}).get("run_id", "default"))
    message = await handle_list_research_tasks(
        tool_call_payload(TOOL_NAME, input, context),
        get_task_registry(),
        run_id=run_id,
        state_store=get_task_state_store(configurable),
    )
    return ToolResult(output=message.content)


def build_list_research_tasks(deps: SupervisorToolDeps) -> Tool:
    return build_tool(
        name=TOOL_NAME,
        description=DESCRIPTION,
        input_schema=ListResearchTasks,
        call=partial(_call, deps),
        origin=ToolOrigin.SYSTEM,
        prompt=render_prompt,
    )


__all__ = ["ListResearchTasks", "build_list_research_tasks"]
