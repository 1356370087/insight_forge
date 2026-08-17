"""Durable mailbox wait tool."""

import asyncio
from functools import partial
from typing import Any

from pydantic import BaseModel, Field

from open_deep_research.configuration import Configuration
from open_deep_research.tasks.coordination import get_mailbox
from open_deep_research.tasks.teammate_pool import find_active_teammate_pool
from open_deep_research.tools.base import (
    ProgressCallback,
    Tool,
    ToolContext,
    ToolOrigin,
    ToolResult,
    build_tool,
)
from open_deep_research.tools.supervisor.deps import SupervisorToolDeps
from open_deep_research.tools.supervisor.wait_for_research_updates.prompt import (
    DESCRIPTION,
    TOOL_NAME,
    render_prompt,
)


class WaitForResearchUpdates(BaseModel):
    """Wait for durable SubAgent mailbox updates without another model call."""

    timeout_seconds: int = Field(default=15, ge=1, le=60)


async def _call(
    deps: SupervisorToolDeps,
    input: WaitForResearchUpdates,
    context: ToolContext,
    on_progress: ProgressCallback | None = None,
) -> ToolResult[Any]:
    del deps, on_progress
    configurable = Configuration.from_runnable_config(context.config)
    run_id = str(context.config.get("metadata", {}).get("run_id", "default"))
    pool = find_active_teammate_pool(run_id)
    if pool is not None and not await pool.lease.is_owner():
        raise RuntimeError(f"This process does not own the Lead lease for run {run_id}")
    mailbox = get_mailbox(configurable, run_id)
    deadline = asyncio.get_running_loop().time() + input.timeout_seconds
    while asyncio.get_running_loop().time() < deadline:
        stats = await mailbox.stats("lead")
        if stats["available"]:
            return ToolResult(
                output=f"{stats['available']} mailbox update(s) are ready."
            )
        await asyncio.sleep(configurable.mailbox_poll_interval_ms / 1000)
    return ToolResult(output="No new research updates before the timeout.")


def build_wait_for_research_updates(deps: SupervisorToolDeps) -> Tool:
    return build_tool(
        name=TOOL_NAME,
        description=DESCRIPTION,
        input_schema=WaitForResearchUpdates,
        call=partial(_call, deps),
        origin=ToolOrigin.SYSTEM,
        prompt=render_prompt,
    )


__all__ = ["WaitForResearchUpdates", "build_wait_for_research_updates"]
