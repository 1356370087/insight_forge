"""Dynamic MCP and browser-tool loading behind the project Tool protocol."""

from __future__ import annotations

import logging
import warnings
from functools import partial
from typing import Any, cast

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import StructuredTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from open_deep_research.configuration import BrowserMCPConfig, Configuration
from open_deep_research.security.content import inspect_untrusted_content
from open_deep_research.tools.adapters import adapt_langchain_tool
from open_deep_research.tools.base import Tool, ToolEffect, ToolOrigin
from open_deep_research.tools.mcp.browser import build_browser_mcp_connection
from open_deep_research.tools.mcp.oauth import (
    fetch_tokens,
    wrap_mcp_authenticate_tool,
)


def _description_is_safe(tool: object, configurable: Configuration) -> bool:
    description = str(getattr(tool, "description", "") or "")
    return not inspect_untrusted_content(
        description[: configurable.max_mcp_description_chars]
    )


def _constant_egress_url(url: str, args: dict[str, Any]) -> list[str]:
    del args
    return [url]


async def load_mcp_tools(
    config: RunnableConfig,
    existing_tool_names: set[str],
) -> list[Tool]:
    """Discover configured MCP tools after validating trust boundaries."""
    configurable = Configuration.from_runnable_config(config)
    mcp_config = configurable.mcp_config
    is_http_surface = config.get("metadata", {}).get("deployment_surface") == "http"
    if not (mcp_config and mcp_config.url and mcp_config.tools):
        return []
    configured_names = set(mcp_config.tools)
    if not configured_names.issubset(mcp_config.tool_effects):
        logging.warning(
            "Blocked MCP discovery because one or more tool effects are undeclared"
        )
        return []
    if is_http_surface:
        allowed_servers = {value.rstrip("/") for value in configurable.allowed_mcp_servers}
        if mcp_config.url.rstrip("/") not in allowed_servers:
            logging.warning("Blocked non-allowlisted MCP server on HTTP surface")
            return []

    tokens = await fetch_tokens(config) if mcp_config.auth_required else None
    if mcp_config.auth_required and not tokens:
        return []
    headers = (
        {"Authorization": f"Bearer {tokens['access_token']}"} if tokens else None
    )
    connection = {
        "server_1": {
            "url": mcp_config.url.rstrip("/") + "/mcp",
            "headers": headers,
            "transport": "streamable_http",
        }
    }
    try:
        available_tools = await MultiServerMCPClient(cast(Any, connection)).get_tools()
    except Exception:
        return []

    loaded: list[Tool] = []
    for external_tool in available_tools:
        if external_tool.name in existing_tool_names:
            warnings.warn(
                f"MCP tool '{external_tool.name}' conflicts with existing tool name - skipping"
            )
            continue
        if external_tool.name not in configured_names:
            continue
        effect_value = mcp_config.tool_effects.get(external_tool.name)
        if effect_value is None:
            warnings.warn(
                f"MCP tool '{external_tool.name}' has no explicit tool_effects entry - skipping"
            )
            continue
        if not _description_is_safe(external_tool, configurable):
            warnings.warn(
                f"MCP tool '{external_tool.name}' has an instruction-shaped description - skipping"
            )
            continue
        enhanced = wrap_mcp_authenticate_tool(cast(StructuredTool, external_tool))
        loaded.append(
            adapt_langchain_tool(
                enhanced,
                origin=ToolOrigin.MCP,
                effect=ToolEffect(effect_value),
                retryable=True,
                auth_satisfied=bool(mcp_config.auth_required and tokens),
                egress_urls=partial(_constant_egress_url, mcp_config.url),
            )
        )
    return loaded


async def load_browser_mcp_tools(
    config: RunnableConfig,
    existing_tool_names: set[str],
) -> list[Tool]:
    """Discover optional browser MCP tools after validating server policy."""
    configurable = Configuration.from_runnable_config(config)
    if not configurable.browser_mcp_enabled:
        return []
    browser_config = configurable.browser_mcp_config or BrowserMCPConfig()
    allowed_names = set(browser_config.tools or [])
    if not allowed_names:
        return []
    if not allowed_names.issubset(browser_config.tool_effects):
        logging.warning(
            "Blocked browser MCP discovery because one or more tool effects are undeclared"
        )
        return []
    is_http_surface = config.get("metadata", {}).get("deployment_surface") == "http"
    if (
        is_http_surface
        and browser_config.transport == "stdio"
        and not configurable.allow_http_stdio_mcp
    ):
        logging.warning("Blocked browser stdio MCP on HTTP surface")
        return []
    if is_http_surface and browser_config.url:
        allowed_servers = {value.rstrip("/") for value in configurable.allowed_mcp_servers}
        if browser_config.url.rstrip("/") not in allowed_servers:
            logging.warning("Blocked non-allowlisted browser MCP server on HTTP surface")
            return []
    connection = build_browser_mcp_connection(browser_config)
    if not connection:
        return []
    try:
        available_tools = await MultiServerMCPClient(
            cast(Any, {"browser": connection})
        ).get_tools()
    except Exception:
        return []

    loaded: list[Tool] = []
    for external_tool in available_tools:
        if external_tool.name in existing_tool_names:
            warnings.warn(
                f"Browser MCP tool '{external_tool.name}' conflicts with existing tool name - skipping"
            )
            continue
        if external_tool.name not in allowed_names:
            continue
        effect_value = browser_config.tool_effects.get(external_tool.name)
        if effect_value is None:
            warnings.warn(
                f"Browser MCP tool '{external_tool.name}' has no explicit tool_effects entry - skipping"
            )
            continue
        if not _description_is_safe(external_tool, configurable):
            warnings.warn(
                f"Browser MCP tool '{external_tool.name}' has an instruction-shaped description - skipping"
            )
            continue
        loaded.append(
            adapt_langchain_tool(
                wrap_mcp_authenticate_tool(cast(StructuredTool, external_tool)),
                origin=ToolOrigin.BROWSER,
                effect=ToolEffect(effect_value),
                retryable=True,
                egress_urls=(
                    partial(_constant_egress_url, browser_config.url)
                    if browser_config.url
                    else None
                ),
            )
        )
    return loaded


__all__ = ["load_browser_mcp_tools", "load_mcp_tools"]
