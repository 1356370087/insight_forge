"""Single assembly and model-binding entry point for project tools."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, cast

from langchain_core.runnables import RunnableConfig

from open_deep_research.configuration import Configuration, SearchAPI
from open_deep_research.skills import load_skill_tools
from open_deep_research.tools.adapters import adapt_langchain_tool
from open_deep_research.tools.anthropic_web_search import anthropic_web_search
from open_deep_research.tools.base import (
    Tool,
    ToolEffect,
    ToolOrigin,
    build_tool_registry,
    tools_to_model_definitions,
)
from open_deep_research.tools.fetch_url import fetch_url
from open_deep_research.tools.fetch_webpage import fetch_webpage
from open_deep_research.tools.governance import AgentRole, filter_tools_by_permission
from open_deep_research.tools.mcp import load_browser_mcp_tools, load_mcp_tools
from open_deep_research.tools.openai_web_search import openai_web_search
from open_deep_research.tools.read_file import read_file
from open_deep_research.tools.research_complete import research_complete
from open_deep_research.tools.shell_exec import shell_exec
from open_deep_research.tools.tavily_search import tavily_search
from open_deep_research.tools.think_tool import think_tool
from open_deep_research.tools.web_research import web_research
from open_deep_research.tools.write_file import write_file

_RESEARCHER_BUILTINS: tuple[Tool, ...] = (
    research_complete,
    think_tool,
    tavily_search,
    openai_web_search,
    anthropic_web_search,
    web_research,
    fetch_url,
    fetch_webpage,
    shell_exec,
    read_file,
    write_file,
)


@dataclass(frozen=True, slots=True)
class ToolAssembly:
    """Enabled, permitted tools and their model-facing projections."""

    tools: list[Tool]
    definitions: list[dict]
    guidance: str


def render_tool_guidance(tools: Iterable[Tool], config: RunnableConfig) -> str:
    """Render detailed guidance for exactly the tools exposed to the model."""
    sections = []
    for tool in tools:
        section = tool.prompt(config)
        if section and section.strip():
            sections.append(section.strip())
    return "\n\n".join(sections)


async def get_search_tool(search_api: SearchAPI) -> list[Tool]:
    """Return the folder-owned provider search tool for compatibility callers."""
    provider_tools: dict[SearchAPI, Tool] = {
        SearchAPI.TAVILY: tavily_search,
        SearchAPI.OPENAI: openai_web_search,
        SearchAPI.ANTHROPIC: anthropic_web_search,
    }
    selected = provider_tools.get(SearchAPI(search_api))
    return [selected] if selected is not None else []


async def assemble_toolset(
    role: AgentRole,
    config: RunnableConfig,
    *,
    supervisor_tools: Iterable[Tool] | None = None,
) -> list[Tool]:
    """Assemble enabled static and dynamic tools, then enforce uniqueness."""
    configurable = Configuration.from_runnable_config(config)
    if role is AgentRole.SUPERVISOR:
        tools = list(supervisor_tools or [])
    else:
        tools = [tool for tool in _RESEARCHER_BUILTINS if tool.is_enabled(config)]
        existing_names = {tool.name for tool in tools}
        if os.getenv("SANDBOX_TASK_TOKEN"):
            from open_deep_research.sandbox.gateway_catalog import (
                load_gateway_catalog_tools,
            )

            tools.extend(
                await load_gateway_catalog_tools(
                    role.value,
                    config,
                    existing_names,
                )
            )
        else:
            loaded_mcp_tools = await load_mcp_tools(config, existing_names)
            mcp_tools = [
                tool
                if isinstance(tool, Tool)
                else adapt_langchain_tool(
                    tool,
                    origin=ToolOrigin.MCP,
                    effect=ToolEffect.DESTRUCTIVE,
                    retryable=True,
                )
                for tool in loaded_mcp_tools
            ]
            tools.extend(mcp_tools)
            existing_names.update(tool.name for tool in mcp_tools)

            loaded_browser_tools = await load_browser_mcp_tools(
                config,
                existing_names,
            )
            browser_effects = (
                configurable.browser_mcp_config.tool_effects
                if configurable.browser_mcp_config is not None
                else {}
            )
            browser_tools = [
                tool
                if isinstance(tool, Tool)
                else adapt_langchain_tool(
                    tool,
                    origin=ToolOrigin.BROWSER,
                    effect=ToolEffect(
                        browser_effects.get(
                            tool.name,
                            ToolEffect.DESTRUCTIVE.value,
                        )
                    ),
                    retryable=True,
                )
                for tool in loaded_browser_tools
            ]
            if configurable.web_pipeline_mode == "enforced":
                browser_tools = [
                    tool
                    for tool in browser_tools
                    if tool.effect is ToolEffect.READ_ONLY
                ]
            tools.extend(browser_tools)
            existing_names.update(tool.name for tool in browser_tools)

            skill_tools = await load_skill_tools(config, existing_names)
            tools.extend(
                tool
                if isinstance(tool, Tool)
                else adapt_langchain_tool(
                    tool,
                    origin=ToolOrigin.SKILL,
                    retryable=True,
                )
                for tool in skill_tools
            )
    build_tool_registry(tools)
    if os.getenv("SANDBOX_TASK_TOKEN"):
        from open_deep_research.sandbox.gateway_tool import proxy_gateway_tools

        tools = proxy_gateway_tools(tools)
    return tools


async def prepare_toolset(
    role: AgentRole,
    config: RunnableConfig,
    *,
    supervisor_tools: Iterable[Tool] | None = None,
) -> ToolAssembly:
    """Assemble, permission-filter, project, and document a role toolset."""
    tools = await assemble_toolset(role, config, supervisor_tools=supervisor_tools)
    return await prepare_existing_toolset(tools, role, config)


async def prepare_existing_toolset(
    tools: Iterable[Tool],
    role: AgentRole,
    config: RunnableConfig,
) -> ToolAssembly:
    """Permission-filter and project an already assembled toolset."""
    candidate_tools = list(tools)
    permitted_tools = cast(
        list[Tool],
        filter_tools_by_permission(candidate_tools, role, config),
    )
    if not permitted_tools:
        raise ValueError(
            f"No tools found for {role.value}: configure an allowed tool source and "
            "ensure role tool policies do not exclude every tool."
        )
    definitions = await tools_to_model_definitions(
        permitted_tools,
        max_description_chars=Configuration.from_runnable_config(
            config
        ).max_tool_description_chars,
    )
    return ToolAssembly(
        tools=permitted_tools,
        definitions=definitions,
        guidance=render_tool_guidance(permitted_tools, config),
    )


async def get_all_tools(config: RunnableConfig) -> list[Tool]:
    """Compatibility name for researcher tool assembly."""
    return await assemble_toolset(AgentRole.RESEARCHER, config)


async def bindable_definitions(
    role: AgentRole,
    config: RunnableConfig,
    *,
    supervisor_tools: Iterable[Tool] | None = None,
) -> list[dict]:
    """Return the permitted definitions bound for a role."""
    return (
        await prepare_toolset(role, config, supervisor_tools=supervisor_tools)
    ).definitions


__all__ = [
    "ToolAssembly",
    "assemble_toolset",
    "bindable_definitions",
    "get_all_tools",
    "get_search_tool",
    "prepare_existing_toolset",
    "prepare_toolset",
    "render_tool_guidance",
]
