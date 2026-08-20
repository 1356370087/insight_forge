"""Optional developer shell tool confined by the selected SandboxProvider."""

from __future__ import annotations

from pydantic import BaseModel, Field

from open_deep_research.sandbox.local_provider import (
    BubblewrapSandboxProvider,
    developer_tools_enabled,
)
from open_deep_research.tools.base import (
    ToolContext,
    ToolEffect,
    ToolExecutionZone,
    ToolOrigin,
    ToolResult,
    build_tool,
)


class ShellExecInput(BaseModel):
    """One bounded shell command request."""

    command: str = Field(min_length=1, max_length=20_000)
    cwd: str | None = Field(default=None, max_length=1024)
    timeout_seconds: int = Field(default=120, ge=1, le=900)


async def _call(input: ShellExecInput, context: ToolContext, on_progress=None) -> ToolResult:
    del on_progress
    output = await BubblewrapSandboxProvider().run(
        input.command,
        config=context.config,
        cwd=input.cwd,
        timeout_seconds=input.timeout_seconds,
    )
    return ToolResult(output=output)


shell_exec = build_tool(
    name="ShellExec",
    description="Run one command inside the administrator-approved task workspace sandbox.",
    input_schema=ShellExecInput,
    call=_call,
    origin=ToolOrigin.SYSTEM,
    effect=ToolEffect.LOCAL_WRITE,
    execution_zone=ToolExecutionZone.SANDBOX_LOCAL,
    is_enabled=lambda config: developer_tools_enabled(
        config, "research.tool.shell.execute"
    ),
)
