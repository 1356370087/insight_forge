"""Tests for durable Lead/Supervisor context compaction."""

from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from open_deep_research.agents import deep_researcher
from open_deep_research.tools.utils import get_model_token_limit


def test_deepseek_v4_models_use_one_million_token_context() -> None:
    assert get_model_token_limit("openai:deepseek-v4-flash") == 1_000_000
    assert get_model_token_limit("openai:deepseek-v4-pro") == 1_000_000
    assert get_model_token_limit("openai:deepseek-v4-pro[1m]") == 1_000_000


@pytest.mark.asyncio
async def test_supervisor_compaction_preserves_brief_and_tool_pair(monkeypatch) -> None:
    brief = "Never summarize this complete research brief."
    messages = [
        SystemMessage(content="system"),
        HumanMessage(content=brief),
        HumanMessage(content="old context " * 40),
        AIMessage(
            content="",
            tool_calls=[{"name": "search", "args": {"q": "x"}, "id": "call-1"}],
        ),
        ToolMessage(content="recent result", name="search", tool_call_id="call-1"),
    ]

    async def fake_invoke(*_args, **_kwargs):
        return SimpleNamespace(content="durable summary")

    monkeypatch.setattr(deep_researcher, "get_model_token_limit", lambda _model: 120)
    monkeypatch.setattr(deep_researcher, "invoke_model_with_retry_observability", fake_invoke)
    config = {
        "configurable": {
            "query_context_compaction_enabled": True,
            "query_context_trigger_ratio": 0.2,
            "query_context_recent_window_ratio": 0.1,
            "query_context_summary_max_tokens": 128,
            "research_model": "openai:test",
            "summarization_model": "openai:test",
        }
    }

    compacted = await deep_researcher.compact_query_context(
        messages,
        research_brief=brief,
        channel="supervisor",
        config=config,
    )

    assert compacted is not None
    rebuilt = compacted["messages"]
    assert sum(message.content == brief for message in rebuilt) == 1
    summary = next(message for message in rebuilt if "PersistentContextSummary" in str(message.content))
    assert "durable summary" in summary.content
    assert isinstance(summary, HumanMessage)
    assert not isinstance(summary, SystemMessage)
    recent = compacted["recent_messages"]
    assert isinstance(recent[0], AIMessage)
    assert isinstance(recent[1], ToolMessage)
    assert recent[0].tool_calls[0]["id"] == recent[1].tool_call_id


@pytest.mark.asyncio
async def test_compaction_disabled_keeps_context_untouched() -> None:
    result = await deep_researcher.compact_query_context(
        [HumanMessage(content="large" * 100)],
        research_brief="brief",
        channel="lead",
        config={"configurable": {"query_context_compaction_enabled": False}},
    )

    assert result is None
