"""Definition of the researcher completion signal."""

from pydantic import BaseModel

from open_deep_research.tools.base import ToolOrigin, ToolResult, build_tool
from open_deep_research.tools.research_complete.prompt import DESCRIPTION, render_prompt


class ResearchComplete(BaseModel):
    """Indicate that the Researcher has finished its assigned objective."""


async def _complete_call(input, context, on_progress=None):
    del input, context, on_progress
    return ToolResult(output="Research complete.")


research_complete = build_tool(
    name="ResearchComplete",
    description=DESCRIPTION,
    input_schema=ResearchComplete,
    call=_complete_call,
    origin=ToolOrigin.SYSTEM,
    prompt=render_prompt,
)
