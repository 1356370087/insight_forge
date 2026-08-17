"""Definition and implementation of the strategic reflection tool."""

from langchain_core.tools import tool

from open_deep_research.tools.adapters import adapt_langchain_tool
from open_deep_research.tools.base import ToolOrigin
from open_deep_research.tools.think_tool.prompt import DESCRIPTION, render_prompt


@tool("think_tool", description=DESCRIPTION)
def _think_call(reflection: str) -> str:
    """Record a strategic reflection on findings, gaps, and next steps."""
    return f"Reflection recorded: {reflection}"


think_tool = adapt_langchain_tool(
    _think_call,
    origin=ToolOrigin.SYSTEM,
    concurrency_safe=True,
    prompt=render_prompt,
)
