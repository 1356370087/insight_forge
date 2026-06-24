"""Tests for optional browser-level MCP exploration tools."""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.tools import tool as lc_tool

from open_deep_research.configuration import BrowserMCPConfig, Configuration
from open_deep_research.prompts import research_system_prompt
from open_deep_research.sandbox.policy import allowed_domains
from open_deep_research.tools import utils
from open_deep_research.tools.governance import ToolOrigin, get_tool_origin
from open_deep_research.tools.utils import get_all_tools, load_browser_mcp_tools


@lc_tool
async def browser_navigate(url: str) -> str:
    """Navigate the browser to a URL."""
    return url


@lc_tool
async def browser_snapshot() -> str:
    """Return the current browser accessibility snapshot."""
    return "snapshot"


class FakeMCPClient:
    """Capture MCP connection config and return deterministic browser tools."""

    connections_seen: dict[str, Any] | None = None

    def __init__(self, connections: dict[str, Any]) -> None:
        FakeMCPClient.connections_seen = connections

    async def get_tools(self):
        return [browser_navigate, browser_snapshot]


def test_browser_mcp_config_defaults_to_playwright_stdio() -> None:
    cfg = Configuration(
        browser_mcp_enabled=True,
        browser_mcp_config=BrowserMCPConfig(),
    )

    assert cfg.browser_mcp_enabled is True
    assert cfg.browser_mcp_config.transport == "stdio"
    assert cfg.browser_mcp_config.command == "npx"
    assert cfg.browser_mcp_config.args == ["@playwright/mcp@latest"]


@pytest.mark.asyncio
async def test_load_browser_mcp_tools_uses_stdio_connection_and_tags_origin(monkeypatch) -> None:
    monkeypatch.setattr(utils, "MultiServerMCPClient", FakeMCPClient)

    tools = await load_browser_mcp_tools(
        {
            "configurable": {
                "browser_mcp_enabled": True,
                "browser_mcp_config": {
                    "command": "npx",
                    "args": ["@playwright/mcp@latest", "--headless"],
                    "tools": ["browser_navigate"],
                },
            },
            "metadata": {"run_id": "test"},
        },
        existing_tool_names=set(),
    )

    assert [tool.name for tool in tools] == ["browser_navigate"]
    assert get_tool_origin(tools[0]) == ToolOrigin.MCP
    assert FakeMCPClient.connections_seen == {
        "browser": {
            "transport": "stdio",
            "command": "npx",
            "args": ["@playwright/mcp@latest", "--headless"],
        }
    }


@pytest.mark.asyncio
async def test_get_all_tools_adds_browser_mcp_without_replacing_existing_mcp(monkeypatch) -> None:
    async def fake_load_mcp_tools(_config, _existing_tool_names):
        return [browser_snapshot]

    async def fake_load_browser_mcp_tools(_config, _existing_tool_names):
        return [browser_navigate]

    monkeypatch.setattr(utils, "load_mcp_tools", fake_load_mcp_tools)
    monkeypatch.setattr(utils, "load_browser_mcp_tools", fake_load_browser_mcp_tools)

    tools = await get_all_tools(
        {
            "configurable": {
                "search_api": "none",
                "mcp_config": {"url": "https://mcp.example.com", "tools": ["browser_snapshot"]},
                "browser_mcp_enabled": True,
                "browser_mcp_config": {"tools": ["browser_navigate"]},
            },
            "metadata": {"run_id": "test"},
        }
    )

    tool_names = {tool.name for tool in tools if hasattr(tool, "name")}
    assert "browser_snapshot" in tool_names
    assert "browser_navigate" in tool_names


def test_allowed_domains_includes_browser_mcp_http_host() -> None:
    cfg = Configuration(
        browser_mcp_enabled=True,
        browser_mcp_config=BrowserMCPConfig(
            transport="streamable_http",
            url="https://browser-mcp.example.com/mcp",
        ),
    )

    assert "browser-mcp.example.com" in allowed_domains(cfg)


def test_research_prompt_describes_browser_exploration_fallback() -> None:
    rendered = research_system_prompt.format(mcp_prompt="", date="June 24, 2026")

    assert "browser exploration" in rendered
    assert "dynamic or JavaScript-rendered pages" in rendered
    assert "Do not start with browser exploration" in rendered
