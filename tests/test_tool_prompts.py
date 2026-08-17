"""Prompt-contract tests for folder-organized researcher tools."""

import ast
from pathlib import Path

import pytest

from open_deep_research.prompts import (
    lead_researcher_async_prompt,
    lead_researcher_prompt,
)
from open_deep_research.tools.governance import AgentRole
from open_deep_research.tools.registry import assemble_toolset, render_tool_guidance
from open_deep_research.tools.supervisor import (
    SupervisorToolDeps,
    build_supervisor_tools,
)

TOOLS_ROOT = Path(__file__).parents[1] / "src" / "open_deep_research" / "tools"


def test_tool_prompt_modules_are_pure_string_renderers() -> None:
    prompt_paths = sorted(TOOLS_ROOT.rglob("prompt.py"))

    assert prompt_paths
    for prompt_path in prompt_paths:
        tree = ast.parse(prompt_path.read_text(encoding="utf-8"))
        imports = [
            node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))
        ]
        assert not imports, f"{prompt_path} must not import tool implementation code"


def test_researcher_definitions_own_their_call_implementations() -> None:
    folder_names = {
        "anthropic_web_search",
        "fetch_url",
        "fetch_webpage",
        "openai_web_search",
        "research_complete",
        "tavily_search",
        "think_tool",
        "web_research",
    }

    for folder_name in folder_names:
        source = (TOOLS_ROOT / folder_name / "definition.py").read_text(
            encoding="utf-8"
        )
        tree = ast.parse(source)
        local_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name.endswith("_call")
        ]
        assert local_calls, f"{folder_name} must own its executable call"
        assert "clone_builtin_tool" not in source
        assert "tools.implementations" not in source


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["legacy", "shadow", "enforced"])
async def test_every_enabled_builtin_has_nonempty_description_and_prompt(mode):
    tools = await assemble_toolset(
        AgentRole.RESEARCHER,
        {
            "configurable": {
                "web_pipeline_mode": mode,
                "search_api": "tavily",
            }
        },
    )

    for tool in tools:
        assert (await tool.description()).strip()
        assert (tool.prompt({}) or "").strip()


@pytest.mark.parametrize(
    ("template", "async_enabled", "expected", "absent"),
    [
        (lead_researcher_prompt, False, "`ConductResearch`", "`StartResearchTask`"),
        (
            lead_researcher_async_prompt,
            True,
            "`StartResearchTask`",
            "`ConductResearch`",
        ),
    ],
)
def test_supervisor_available_tools_are_rendered_from_actual_toolset(
    template,
    async_enabled,
    expected,
    absent,
):
    tools = build_supervisor_tools(
        SupervisorToolDeps(
            enable_async_research=async_enabled,
        )
    )
    rendered = template.format(
        date="2026-08-17",
        tool_guidance=render_tool_guidance(tools, {}),
        max_concurrent_research_units=5,
        max_researcher_iterations=6,
        max_react_tool_calls=10,
    )

    assert expected in rendered
    assert absent not in rendered
