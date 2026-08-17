"""Browser MCP connection construction."""

from __future__ import annotations

from typing import Any

from open_deep_research.configuration import BrowserMCPConfig


def build_browser_mcp_connection(
    browser_config: BrowserMCPConfig,
) -> dict[str, Any] | None:
    """Build a langchain-mcp-adapters connection from validated configuration."""
    if browser_config.transport == "stdio":
        if not browser_config.command:
            return None
        connection: dict[str, Any] = {
            "transport": "stdio",
            "command": browser_config.command,
            "args": browser_config.args or [],
        }
        if browser_config.env:
            connection["env"] = browser_config.env
        return connection
    if not browser_config.url:
        return None
    return {"transport": browser_config.transport, "url": browser_config.url}


_build_browser_mcp_connection = build_browser_mcp_connection

__all__ = ["_build_browser_mcp_connection", "build_browser_mcp_connection"]
