"""Deprecated compatibility imports for the pre-folder tool API.

New code must import executable tools from their folders and assembly functions
from :mod:`open_deep_research.tools.registry`.
"""
# ruff: noqa: F401

from langchain.chat_models import init_chat_model

from open_deep_research.tools.anthropic_web_search import anthropic_web_search
from open_deep_research.tools.fetch_url import fetch_url
from open_deep_research.tools.fetch_webpage import fetch_webpage
from open_deep_research.tools.legacy_shims import *  # noqa: F403
from open_deep_research.tools.mcp import load_browser_mcp_tools, load_mcp_tools
from open_deep_research.tools.mcp.loader import MultiServerMCPClient
from open_deep_research.tools.model_limits import *  # noqa: F403
from open_deep_research.tools.openai_web_search import openai_web_search
from open_deep_research.tools.registry import get_all_tools, get_search_tool
from open_deep_research.tools.tavily_search import tavily_search
from open_deep_research.tools.tavily_search.client import (
    get_tavily_api_key,
    tavily_search_async,
)
from open_deep_research.tools.tavily_search.summarization import (
    build_summarization_model as _build_summarization_model,
)
from open_deep_research.tools.tavily_search.summarization import (
    summarize_webpage,
)
from open_deep_research.tools.think_tool import think_tool
from open_deep_research.tools.web_research import web_research
from open_deep_research.tools.web_research.providers import (
    build_anthropic_client as _build_anthropic_client,
)
from open_deep_research.tools.web_research.providers import (
    build_openai_client as _build_openai_client,
)

__all__ = [
    "anthropic_web_search",
    "fetch_url",
    "fetch_webpage",
    "get_all_tools",
    "get_search_tool",
    "load_browser_mcp_tools",
    "load_mcp_tools",
    "openai_web_search",
    "tavily_search",
    "think_tool",
    "web_research",
]
