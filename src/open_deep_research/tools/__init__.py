"""Project-owned Tool Interface and runtime helpers."""

from open_deep_research.tools.base import (
    ProgressCallback,
    Tool,
    ToolContext,
    ToolOrigin,
    ToolResult,
    build_tool,
    build_tool_registry,
    serialize_tool_output,
    tool_to_model_definition,
    tools_to_model_definitions,
)

_REGISTRY_EXPORTS = {
    "assemble_toolset",
    "bindable_definitions",
    "get_all_tools",
    "prepare_toolset",
    "render_tool_guidance",
}


def __getattr__(name: str):
    """Lazily expose registry helpers without creating import cycles."""
    if name in _REGISTRY_EXPORTS:
        from open_deep_research.tools import registry

        return getattr(registry, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "ProgressCallback",
    "Tool",
    "ToolContext",
    "ToolOrigin",
    "ToolResult",
    "assemble_toolset",
    "bindable_definitions",
    "build_tool",
    "build_tool_registry",
    "get_all_tools",
    "prepare_toolset",
    "render_tool_guidance",
    "serialize_tool_output",
    "tool_to_model_definition",
    "tools_to_model_definitions",
]
