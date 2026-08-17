"""Supervisor completion signal."""

from typing import Any

from pydantic import BaseModel

from open_deep_research.tools.base import (
    ProgressCallback,
    Tool,
    ToolContext,
    ToolOrigin,
    ToolResult,
    build_tool,
)
from open_deep_research.tools.supervisor.research_complete.prompt import (
    DESCRIPTION,
    TOOL_NAME,
    render_prompt,
)


class ResearchComplete(BaseModel):
    """Indicate that Supervisor research is complete."""


async def _call(
    input: ResearchComplete,
    context: ToolContext,
    on_progress: ProgressCallback | None = None,
) -> ToolResult[Any]:
    del input, context, on_progress
    return ToolResult(output="Research complete.")


def build_research_complete() -> Tool:
    return build_tool(
        name=TOOL_NAME,
        description=DESCRIPTION,
        input_schema=ResearchComplete,
        call=_call,
        origin=ToolOrigin.SYSTEM,
        prompt=render_prompt,
    )


__all__ = ["ResearchComplete", "build_research_complete"]
