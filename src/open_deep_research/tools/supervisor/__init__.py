"""Folder-owned Supervisor tool assembly."""

from __future__ import annotations

from open_deep_research.tools.base import Tool
from open_deep_research.tools.supervisor.cancel_research_task.definition import (
    build_cancel_research_task,
)
from open_deep_research.tools.supervisor.check_research_task.definition import (
    build_check_research_task,
)
from open_deep_research.tools.supervisor.conduct_research.definition import (
    build_conduct_research,
)
from open_deep_research.tools.supervisor.deps import SupervisorToolDeps
from open_deep_research.tools.supervisor.list_research_tasks.definition import (
    build_list_research_tasks,
)
from open_deep_research.tools.supervisor.read_research_artifact.definition import (
    build_read_research_artifact,
)
from open_deep_research.tools.supervisor.research_complete.definition import (
    build_research_complete,
)
from open_deep_research.tools.supervisor.start_research_task.definition import (
    build_start_research_task,
)
from open_deep_research.tools.supervisor.update_research_task.definition import (
    build_update_research_task,
)
from open_deep_research.tools.supervisor.wait_for_research_updates.definition import (
    build_wait_for_research_updates,
)
from open_deep_research.tools.think_tool import think_tool


def build_supervisor_tools(deps: SupervisorToolDeps) -> list[Tool]:
    """Build Supervisor tools exclusively from module-level definitions."""
    completion = build_research_complete()
    if deps.sandbox_enabled and not deps.enable_async_research:
        raise ValueError("sandbox_requires_async_research")
    if not deps.enable_async_research:
        return [
            build_conduct_research(deps),
            build_read_research_artifact(deps),
            completion,
            think_tool,
        ]
    return [
        build_start_research_task(deps),
        build_check_research_task(deps),
        build_list_research_tasks(deps),
        build_update_research_task(deps),
        build_cancel_research_task(deps),
        build_wait_for_research_updates(deps),
        completion,
        think_tool,
    ]


__all__ = ["SupervisorToolDeps", "build_supervisor_tools"]
