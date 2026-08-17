"""Tests for the native SDK web search tools (OpenAI / Anthropic).

These tests monkeypatch the SDK client builders and the summarization step so no
real API or LLM calls are made. They verify the tools return the multi-source
format, record observable spans + usage, retry on 429, and are governed as SEARCH
tools.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import aiohttp
import pytest
from langchain_core.runnables import RunnableConfig

from open_deep_research.configuration import SearchAPI
from open_deep_research.observability import SQLiteTraceStore, get_trace_recorder
from open_deep_research.tools.anthropic_web_search import anthropic_web_search
from open_deep_research.tools.base import ToolContext, ToolOrigin
from open_deep_research.tools.governance import get_tool_origin, get_tool_retryable
from open_deep_research.tools.openai_web_search import openai_web_search
from open_deep_research.tools.registry import get_all_tools, get_search_tool
from open_deep_research.tools.tavily_search import summarization
from open_deep_research.tools.web_research import providers


def _config(trace_path, run_id: str = "search-run") -> RunnableConfig:
    return {
        "configurable": {
            "trace_store_path": str(trace_path),
            "event_log_enabled": False,
            "search_api": SearchAPI.OPENAI.value,
            "research_model": "openai:gpt-4.1",
            "summarization_model": "openai:gpt-4.1-mini",
        },
        "metadata": {"run_id": run_id, "user_id": "user-1"},
    }


def _openai_response(text: str, sources: list[dict[str, str]], usage: dict[str, int]) -> Any:
    annotations = [SimpleNamespace(type="url_citation", url=s["url"], title=s["title"]) for s in sources]
    return SimpleNamespace(
        output_text=text,
        output=[SimpleNamespace(type="message", content=[SimpleNamespace(type="output_text", text=text, annotations=annotations)])],
        usage=SimpleNamespace(**usage),
    )


def _anthropic_response(text: str, sources: list[dict[str, str]], usage: dict[str, int]) -> Any:
    blocks = []
    if text:
        blocks.append(SimpleNamespace(type="text", text=text))
    if sources:
        blocks.append(SimpleNamespace(
            type="web_search_tool_result",
            content=[SimpleNamespace(url=s["url"], title=s["title"]) for s in sources],
        ))
    return SimpleNamespace(content=blocks, usage=SimpleNamespace(**usage))


class _FakeResponses:
    def __init__(self, responder):
        self.responder = responder
        self.calls = 0
        self.last_kwargs: dict[str, Any] = {}

    async def create(self, **kwargs):
        self.calls += 1
        self.last_kwargs = kwargs
        return self.responder(self.calls)


class _FakeMessages:
    def __init__(self, responder):
        self.responder = responder
        self.calls = 0
        self.last_kwargs: dict[str, Any] = {}

    async def create(self, **kwargs):
        self.calls += 1
        self.last_kwargs = kwargs
        return self.responder(self.calls)


class _FakeOpenAIClient:
    def __init__(self, responder):
        self.responses = _FakeResponses(responder)


class _FakeAnthropicClient:
    def __init__(self, responder):
        self.messages = _FakeMessages(responder)


async def _fake_summarize(_model, text, *, config=None, model_name=None):
    return f"<summary>{(text or '')[:12]}</summary>\n<key_excerpts>none</key_excerpts>"


def _patch_summarize(monkeypatch):
    """Avoid real LLM calls in the summarization step."""
    monkeypatch.setattr(summarization, "summarize_webpage", _fake_summarize)
    monkeypatch.setattr(
        summarization, "build_summarization_model", lambda config: object()
    )


@pytest.mark.asyncio
async def test_get_search_tool_returns_structured_tools():
    openai_tools = await get_search_tool(SearchAPI.OPENAI)
    anthropic_tools = await get_search_tool(SearchAPI.ANTHROPIC)
    tavily_tools = await get_search_tool(SearchAPI.TAVILY)

    assert openai_tools[0].name == "openai_web_search"
    assert anthropic_tools[0].name == "anthropic_web_search"
    assert tavily_tools[0].name == "tavily_search"
    # No more provider-native dict tools -- all are real BaseTools now.
    assert all(hasattr(t, "name") for t in openai_tools + anthropic_tools + tavily_tools)


@pytest.mark.asyncio
async def test_openai_web_search_formats_sources_and_records_span(tmp_path, monkeypatch):
    trace_path = tmp_path / "trace.sqlite3"
    config = _config(trace_path)
    recorder = get_trace_recorder(config)

    sources = [{"url": "https://example.com/a", "title": "Example A"}]
    fake_client = _FakeOpenAIClient(
        lambda calls: _openai_response("Synthesized answer", sources, {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30})
    )
    monkeypatch.setattr(providers, "build_openai_client", lambda cfg: fake_client)
    _patch_summarize(monkeypatch)

    with recorder.start_run("search-run", user_id="user-1"):
        tool = openai_web_search
        result = (
            await tool.call(
                tool.input_schema.model_validate({"queries": ["test query"]}),
                ToolContext(config=config, role="researcher", tool_call_id="openai-1"),
            )
        ).output
        recorder.finish_run("search-run", "success")

    assert "SOURCE 1: Example A" in result
    assert "https://example.com/a" in result
    assert "SUMMARY:" in result
    # The SDK call was made with the web_search_preview tool.
    assert fake_client.responses.last_kwargs["tools"] == [{"type": "web_search_preview"}]

    store = SQLiteTraceStore(str(trace_path))
    spans = store.list_spans("search-run")
    metrics = store.get_metrics("search-run")
    assert any(s["name"] == "tool.openai.web_search" and s["kind"] == "llm" for s in spans)
    assert any(s["total_tokens"] == 30 for s in spans)
    assert metrics["input_tokens"] == 10
    assert metrics["output_tokens"] == 20


@pytest.mark.asyncio
async def test_anthropic_web_search_formats_sources_and_records_span(tmp_path, monkeypatch):
    trace_path = tmp_path / "trace.sqlite3"
    config = _config(trace_path)
    config["configurable"]["search_api"] = SearchAPI.ANTHROPIC.value
    config["configurable"]["research_model"] = "anthropic:claude-sonnet-4"
    recorder = get_trace_recorder(config)

    sources = [{"url": "https://example.com/b", "title": "Example B"}]
    fake_client = _FakeAnthropicClient(
        lambda calls: _anthropic_response("Synthesized answer", sources, {"input_tokens": 5, "output_tokens": 7, "total_tokens": 12})
    )
    monkeypatch.setattr(providers, "build_anthropic_client", lambda cfg: fake_client)
    _patch_summarize(monkeypatch)

    with recorder.start_run("search-run", user_id="user-1"):
        tool = anthropic_web_search
        result = (
            await tool.call(
                tool.input_schema.model_validate({"queries": ["test query"]}),
                ToolContext(config=config, role="researcher", tool_call_id="anthropic-1"),
            )
        ).output
        recorder.finish_run("search-run", "success")

    assert "SOURCE 1: Example B" in result
    assert "https://example.com/b" in result
    # The SDK call used the Anthropic web_search server tool.
    assert fake_client.messages.last_kwargs["tools"] == [
        {"type": "web_search_20250305", "name": "web_search", "max_uses": 5}
    ]

    store = SQLiteTraceStore(str(trace_path))
    spans = store.list_spans("search-run")
    assert any(s["name"] == "tool.anthropic.web_search" and s["total_tokens"] == 12 for s in spans)


@pytest.mark.asyncio
async def test_openai_web_search_retries_on_429(tmp_path, monkeypatch):
    trace_path = tmp_path / "trace.sqlite3"
    config = _config(trace_path)
    recorder = get_trace_recorder(config)

    def responder(calls):
        if calls == 1:
            raise aiohttp.ClientResponseError(None, (), status=429, message="Too Many Requests")
        return _openai_response("ok", [{"url": "https://example.com/c", "title": "C"}], {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2})

    fake_client = _FakeOpenAIClient(responder)
    monkeypatch.setattr(providers, "build_openai_client", lambda cfg: fake_client)

    async def _noop_sleep(_delay):
        return None

    monkeypatch.setattr(asyncio, "sleep", _noop_sleep)
    _patch_summarize(monkeypatch)

    with recorder.start_run("search-run", user_id="user-1"):
        tool = openai_web_search
        result = (
            await tool.call(
                tool.input_schema.model_validate({"queries": ["q"]}),
                ToolContext(config=config, role="researcher", tool_call_id="openai-2"),
            )
        ).output
        recorder.finish_run("search-run", "success")

    assert "SOURCE 1: C" in result
    assert fake_client.responses.calls == 2  # one failed (429) + one success

    store = SQLiteTraceStore(str(trace_path))
    spans = store.list_spans("search-run")
    metrics = store.get_metrics("search-run")
    assert any(s["retry_count"] == 1 for s in spans)
    assert metrics["retry_count"] == 1
    assert metrics["rate_limited_count"] == 1


@pytest.mark.asyncio
async def test_get_all_tools_tags_sdk_search_as_search(tmp_path, monkeypatch):
    config = _config(tmp_path / "trace.sqlite3")
    config["configurable"]["browser_mcp_enabled"] = False
    config["configurable"]["web_pipeline_mode"] = "legacy"
    tools = await get_all_tools(config)
    openai_tool = next(t for t in tools if getattr(t, "name", None) == "openai_web_search")
    assert get_tool_origin(openai_tool) is ToolOrigin.SEARCH
    assert get_tool_retryable(openai_tool) is True
