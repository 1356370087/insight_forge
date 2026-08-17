"""Tests for declarative tool availability and unified assembly."""

from __future__ import annotations

import pytest

from open_deep_research.tools.governance import AgentRole
from open_deep_research.tools.registry import assemble_toolset, prepare_toolset


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mode", "search_api", "network_mode", "present", "absent"),
    [
        ("enforced", "tavily", "allow-search-only", {"web_research", "fetch_url"}, {"tavily_search", "fetch_webpage"}),
        ("legacy", "tavily", "allow-search-only", {"tavily_search", "fetch_webpage"}, {"web_research", "fetch_url"}),
        ("shadow", "openai", "allow-search-only", {"openai_web_search", "fetch_webpage"}, {"tavily_search", "web_research"}),
        ("legacy", "anthropic", "no-network", {"fetch_webpage"}, {"anthropic_web_search", "web_research"}),
    ],
)
async def test_researcher_is_enabled_matrix(
    mode,
    search_api,
    network_mode,
    present,
    absent,
):
    tools = await assemble_toolset(
        AgentRole.RESEARCHER,
        {
            "configurable": {
                "web_pipeline_mode": mode,
                "search_api": search_api,
                "sandbox_network_mode": network_mode,
            }
        },
    )
    names = {tool.name for tool in tools}

    assert present <= names
    assert not (absent & names)


@pytest.mark.asyncio
async def test_prepare_toolset_builds_guidance_from_enabled_tools_only():
    assembly = await prepare_toolset(
        AgentRole.RESEARCHER,
        {"configurable": {"web_pipeline_mode": "enforced", "search_api": "tavily"}},
    )

    assert "`web_research`" in assembly.guidance
    assert "`fetch_url`" in assembly.guidance
    assert "`tavily_search`" not in assembly.guidance
    assert {item["name"] for item in assembly.definitions} == {
        tool.name for tool in assembly.tools
    }


@pytest.mark.asyncio
async def test_tool_description_budget_is_not_mcp_description_budget():
    assembly = await prepare_toolset(
        AgentRole.RESEARCHER,
        {
            "configurable": {
                "web_pipeline_mode": "enforced",
                "max_tool_description_chars": 64,
                "max_mcp_description_chars": 2000,
            }
        },
    )

    assert all(len(item["description"]) <= 64 for item in assembly.definitions)
