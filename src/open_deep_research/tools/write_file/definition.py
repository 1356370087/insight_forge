"""Symlink-safe task workspace writer."""

from __future__ import annotations

import os
from typing import Literal

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


class WriteFileInput(BaseModel):
    """Symlink-safe regular-file write request."""

    path: str = Field(min_length=1, max_length=1024)
    content: str = Field(max_length=1_000_000)
    mode: Literal["create", "overwrite", "append"] = "create"


async def _call(input: WriteFileInput, context: ToolContext, on_progress=None) -> ToolResult:
    del on_progress
    path = safe_workspace_path(context.config, input.path)
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT
    flags |= os.O_EXCL if input.mode == "create" else 0
    flags |= os.O_APPEND if input.mode == "append" else os.O_TRUNC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    data = input.content.encode("utf-8")
    try:
        os.write(descriptor, data)
    finally:
        os.close(descriptor)
    return ToolResult(output={"path": input.path, "bytes_written": len(data)})


write_file = build_tool(
    name="WriteFile",
    description="Create, overwrite or append a regular file in the task workspace.",
    input_schema=WriteFileInput,
    call=_call,
    origin=ToolOrigin.SYSTEM,
    effect=ToolEffect.LOCAL_WRITE,
    execution_zone=ToolExecutionZone.SANDBOX_LOCAL,
    is_enabled=lambda config: developer_tools_enabled(config, "research.tool.file.write"),
)
