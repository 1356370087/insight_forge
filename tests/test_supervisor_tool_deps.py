"""Tests for the Supervisor tool dependency seam."""

from dataclasses import FrozenInstanceError

import pytest

from open_deep_research.tools.supervisor import (
    SupervisorToolDeps,
    build_supervisor_tools,
)


def test_supervisor_deps_are_frozen_and_drive_sync_tool_assembly():
    deps = SupervisorToolDeps(enable_async_research=False)

    tools = build_supervisor_tools(deps)

    assert {tool.name for tool in tools} == {
        "ConductResearch",
        "ReadResearchArtifact",
        "ResearchComplete",
        "think_tool",
    }
    with pytest.raises(FrozenInstanceError):
        deps.enable_async_research = True  # type: ignore[misc]


def test_supervisor_tool_guidance_is_attached_by_tool_folder():
    tools = build_supervisor_tools(
        SupervisorToolDeps(enable_async_research=True)
    )

    assert all(tool.prompt({}) for tool in tools if tool.name != "think_tool")
