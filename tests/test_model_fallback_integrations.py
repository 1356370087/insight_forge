"""Integration seams proving each non-Query role consumes its fallback chain."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel

from open_deep_research.agents import deep_researcher
from open_deep_research.quality import gate as quality
from open_deep_research.report import assembly
from open_deep_research.report.profiles import get_profile
from open_deep_research.tools.web_research import pipeline as tool_pipeline
from open_deep_research.web.models import CandidateSource


class _QualityDecision(BaseModel):
    accepted: bool


class _StructuredModelStub:
    def with_structured_output(self, *_args, **_kwargs):
        return self


def _fallback_config(role: str, primary: str, fallback: str) -> dict:
    return {
        "configurable": {
            "model_fallbacks": {role: [fallback]},
            "quality_evaluation_model": primary,
            "compression_model": primary,
            "final_report_model": primary,
            "message_summary_model": primary,
            "summarization_model": primary,
            "web_rerank_model": primary,
        }
    }


@pytest.mark.asyncio
async def test_quality_evaluation_entrypoint_uses_fallback(monkeypatch) -> None:
    calls: list[str] = []

    monkeypatch.setattr(quality, "_build_quality_model", lambda *_args: object())

    async def fake_invoke(*_args, model_name=None, **_kwargs):
        calls.append(model_name)
        if model_name == "openai:primary":
            raise RuntimeError("service unavailable")
        return AIMessage(content='{"accepted": true}')

    monkeypatch.setattr(quality, "invoke_model_with_retry_observability", fake_invoke)
    result = await quality._evaluate_json(
        _QualityDecision,
        "Return JSON.",
        {"claim": "x"},
        _fallback_config(
            "quality_evaluation",
            "openai:primary",
            "anthropic:fallback",
        ),
        span_name="test.quality",
    )

    assert result.accepted is True
    assert calls == ["openai:primary", "anthropic:fallback"]


@pytest.mark.asyncio
async def test_compression_entrypoint_uses_fallback(monkeypatch) -> None:
    calls: list[str] = []

    async def fake_invoke(*_args, model_name=None, **_kwargs):
        calls.append(model_name)
        if model_name == "openai:primary":
            raise RuntimeError("service unavailable")
        return AIMessage(content="compressed by fallback")

    monkeypatch.setattr(
        deep_researcher,
        "invoke_model_with_retry_observability",
        fake_invoke,
    )
    update = await deep_researcher.compress_research(
        {"researcher_messages": []},
        _fallback_config("compression", "openai:primary", "anthropic:fallback"),
    )

    assert update["compressed_research"] == "compressed by fallback"
    assert calls == ["openai:primary", "anthropic:fallback"]


@pytest.mark.asyncio
async def test_final_report_entrypoint_uses_fallback(monkeypatch) -> None:
    calls: list[str] = []
    config = _fallback_config(
        "final_report",
        "openai:primary",
        "anthropic:fallback",
    )
    context = assembly.ReportContext.from_state(
        {"notes": ["finding"]},
        config,
        get_profile("default"),
    )

    async def fake_invoke(*_args, model_name=None, **_kwargs):
        calls.append(model_name)
        if model_name == "openai:primary":
            raise RuntimeError("service unavailable")
        return AIMessage(content="fallback report")

    monkeypatch.setattr(
        assembly,
        "invoke_model_with_retry_observability",
        fake_invoke,
    )
    response = await context.invoke_writer_with_output_recovery(
        [HumanMessage(content="write")],
        span_name="test.report",
    )

    assert response.content == "fallback report"
    assert calls == ["openai:primary", "anthropic:fallback"]


@pytest.mark.asyncio
async def test_web_rerank_entrypoint_uses_fallback(monkeypatch) -> None:
    calls: list[str] = []
    candidate = CandidateSource(
        candidate_id="candidate-1",
        provider="test",
        provider_rank=1,
        original_url="https://example.com/a",
        canonical_url="https://example.com/a",
        domain="example.com",
        title="A",
        snippet="Relevant source",
    )
    monkeypatch.setattr(
        tool_pipeline,
        "init_chat_model",
        lambda **_kwargs: _StructuredModelStub(),
    )

    async def fake_invoke(*_args, model_name=None, **_kwargs):
        calls.append(model_name)
        if model_name == "openai:primary":
            raise RuntimeError("service unavailable")
        return SimpleNamespace(
            scores=[
                SimpleNamespace(
                    candidate_id="candidate-1",
                    relevance=0.9,
                    authority=0.8,
                    information_gain=0.7,
                )
            ]
        )

    monkeypatch.setattr(
        tool_pipeline,
        "invoke_model_with_retry_observability",
        fake_invoke,
    )
    result = await tool_pipeline._rerank_web_candidates(
        "objective",
        [candidate],
        _fallback_config("summarization", "openai:primary", "anthropic:fallback"),
    )

    assert result["candidate-1"] == (0.9, 0.8, 0.7)
    assert calls == ["openai:primary", "anthropic:fallback"]


@pytest.mark.asyncio
async def test_message_summary_entrypoint_uses_fallback(monkeypatch) -> None:
    calls: list[str] = []
    messages = [
        SystemMessage(content="system"),
        HumanMessage(content="brief"),
        HumanMessage(content="old context " * 40),
        HumanMessage(content="recent context"),
    ]
    monkeypatch.setattr(deep_researcher, "get_model_token_limit", lambda _model: 120)

    async def fake_invoke(*_args, model_name=None, **_kwargs):
        calls.append(model_name)
        if model_name == "openai:primary":
            raise RuntimeError("service unavailable")
        return AIMessage(content="fallback summary")

    monkeypatch.setattr(
        deep_researcher,
        "invoke_model_with_retry_observability",
        fake_invoke,
    )
    config = _fallback_config(
        "message_summary",
        "openai:primary",
        "anthropic:fallback",
    )
    config["configurable"].update(
        {
            "query_context_compaction_enabled": True,
            "query_context_trigger_ratio": 0.2,
            "query_context_recent_window_ratio": 0.1,
            "query_context_summary_max_tokens": 128,
            "research_model": "openai:primary",
        }
    )
    compacted = await deep_researcher.compact_query_context(
        messages,
        research_brief="brief",
        channel="supervisor",
        config=config,
    )

    assert compacted is not None
    assert any(
        "fallback summary" in str(message.content)
        for message in compacted["messages"]
    )
    assert calls == ["openai:primary", "anthropic:fallback"]
