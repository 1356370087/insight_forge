"""Symlink-safe task workspace reader."""

from __future__ import annotations

import stat

from pydantic import BaseModel, Field

from open_deep_research.sandbox.local_provider import (
    developer_tools_enabled,
    safe_workspace_path,
)
from open_deep_research.tools.base import (
    ToolContext,
    ToolEffect,
    ToolExecutionZone,
    ToolOrigin,
    ToolResult,
    build_tool,
)


class ReadFileInput(BaseModel):
    """Bounded regular-file read request."""

    path: str = Field(min_length=1, max_length=1024)
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=64_000, ge=1, le=1_000_000)


async def _call(input: ReadFileInput, context: ToolContext, on_progress=None) -> ToolResult:
    del on_progress
    path = safe_workspace_path(context.config, input.path)
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode):
        raise ValueError("sandbox_read_requires_regular_file")
    with path.open("rb") as handle:
        handle.seek(input.offset)
        data = handle.read(input.limit + 1)
    return ToolResult(
        output={
            "path": input.path,
            "content": data[: input.limit].decode("utf-8", errors="replace"),
            "truncated": len(data) > input.limit,
        }
    )


read_file = build_tool(
    name="ReadFile",
    description="Read a bounded UTF-8 preview from a regular file in the task workspace.",
    input_schema=ReadFileInput,
    call=_call,
    origin=ToolOrigin.SYSTEM,
    effect=ToolEffect.SENSITIVE_READ,
    execution_zone=ToolExecutionZone.SANDBOX_LOCAL,
    is_enabled=lambda config: developer_tools_enabled(config, "research.tool.file.read"),
)
