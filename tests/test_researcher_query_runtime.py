from __future__ import annotations

import asyncio
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool as lc_tool

from open_deep_research.agents import deep_researcher
from open_deep_research.agents.query_engine import ResearcherQueryEngine
from open_deep_research.completion import CompletionDecision
from open_deep_research.tasks import registry as task_registry
from open_deep_research.tools import utils
from open_deep_research.tools.adapters import adapt_langchain_tool
from open_deep_research.tools.base import ToolOrigin
from open_deep_research.tools.governance import (
    GovernedToolCallResult,
    ToolError,
    ToolErrorType,
)


@lc_tool("research_echo")
async def _research_echo_impl(text: str) -> str:
    """Echo research evidence."""
    return f"evidence:{text}"


research_echo = adapt_langchain_tool(
    _research_echo_impl,
    origin=ToolOrigin.SEARCH,
)


@lc_tool("slow_research_echo")
async def _slow_research_echo_impl(text: str) -> str:
    """Return research evidence after a short realistic delay."""
    await asyncio.sleep(0.05)
    return f"slow-evidence:{text}"


slow_research_echo = adapt_langchain_tool(
    _slow_research_echo_impl,
    origin=ToolOrigin.SEARCH,
)


class FakeResearchModel:
    def __init__(self, responses: list[AIMessage]):
        self.responses = list(responses)
        self.calls: list[list[Any]] = []

    def bind_tools(self, _tools):
        return self

    def with_config(self, _config):
        return self

    async def ainvoke(self, messages):
        self.calls.append(list(messages))
        return self.responses.pop(0)


def _config(**overrides: Any) -> dict[str, Any]:
    return {
        "configurable": {
            "search_api": "none",
            "observability_enabled": False,
            "quality_evaluation_enabled": False,
            "max_react_tool_calls": 5,
            **overrides,
        },
        "metadata": {"run_id": "research-query-runtime", "task_id": "researcher-1"},
    }


@pytest.mark.asyncio
async def test_researcher_engine_uses_unified_query_loop(monkeypatch) -> None:
    model = FakeResearchModel([
        AIMessage(
            content="",
            tool_calls=[
                {"name": "research_echo", "args": {"text": "fact"}, "id": "tool-1"},
            ],
        ),
        AIMessage(
            content="",
            tool_calls=[{"name": "ResearchComplete", "args": {}, "id": "done-1"}],
        ),
    ])

    async def fake_get_all_tools(_config):
        return [research_echo, *deep_researcher.build_supervisor_tools({})[-2:-1]]

    async def old_node_must_not_run(*_args, **_kwargs):
        raise AssertionError("legacy researcher node loop was invoked")

    async def fake_compress(state, _config):
        contents = [str(message.content) for message in state["researcher_messages"]]
        return {
            "compressed_research": "compressed",
            "raw_notes": contents,
        }

    monkeypatch.setattr(deep_researcher, "configurable_model", model)
    monkeypatch.setattr(deep_researcher, "get_all_tools", fake_get_all_tools)
    monkeypatch.setattr(deep_researcher, "researcher", old_node_must_not_run)
    monkeypatch.setattr(deep_researcher, "researcher_tools", old_node_must_not_run)
    monkeypatch.setattr(deep_researcher, "compress_research", fake_compress)

    result = await ResearcherQueryEngine(_config()).ainvoke(
        {
            "researcher_messages": [HumanMessage(content="topic")],
            "research_topic": "topic",
            "memory_context": None,
            "evidence_registry": [{"id": "ev-1", "source_url": "https://example.com"}],
        },
    )

    assert result["compressed_research"] == "compressed"
    assert result["tool_call_iterations"] == 2
    assert any("evidence:fact" in note for note in result["raw_notes"])
    assert len(model.calls) == 2


@pytest.mark.asyncio
async def test_researcher_tools_use_dedicated_long_timeout(monkeypatch) -> None:
    model = FakeResearchModel([
        AIMessage(
            content="",
            tool_calls=[{
                "name": "slow_research_echo",
                "args": {"text": "fact"},
                "id": "slow-tool-1",
            }],
        ),
        AIMessage(
            content="",
            tool_calls=[{"name": "ResearchComplete", "args": {}, "id": "done-1"}],
        ),
    ])

    async def fake_get_all_tools(_config):
        return [slow_research_echo, *deep_researcher.build_supervisor_tools({})[-2:-1]]

    async def fake_compress(state, _config):
        contents = [str(message.content) for message in state["researcher_messages"]]
        return {
            "compressed_research": "compressed",
            "raw_notes": contents,
        }

    monkeypatch.setattr(deep_researcher, "configurable_model", model)
    monkeypatch.setattr(deep_researcher, "get_all_tools", fake_get_all_tools)
    monkeypatch.setattr(deep_researcher, "compress_research", fake_compress)

    result = await ResearcherQueryEngine(_config(
        tool_call_timeout_seconds=0.01,
        research_tool_call_timeout_seconds=1,
    )).ainvoke({
        "researcher_messages": [HumanMessage(content="topic")],
        "research_topic": "topic",
        "evidence_registry": [{"id": "ev-1", "source_url": "https://example.com"}],
    })

    assert any("slow-evidence:fact" in note for note in result["raw_notes"])


@pytest.mark.asyncio
async def test_researcher_observes_cancellation_at_terminal_boundary(monkeypatch) -> None:
    registry = task_registry.TaskRegistry()
    record = registry.restore(task_registry.TaskRecord(
        task_id="researcher-1",
        run_id="research-query-runtime",
        research_topic="topic",
    ))

    class TerminalCancellingModel(FakeResearchModel):
        async def ainvoke(self, messages):
            response = await super().ainvoke(messages)
            record.cancelled.set()
            return response

    model = TerminalCancellingModel([AIMessage(content="research complete")])

    async def fake_get_all_tools(_config):
        return [research_echo]

    async def compress_must_not_run(*_args, **_kwargs):
        raise AssertionError("cancelled research must not be compressed")

    monkeypatch.setattr(task_registry, "get_task_registry", lambda: registry)
    monkeypatch.setattr(deep_researcher, "configurable_model", model)
    monkeypatch.setattr(deep_researcher, "get_all_tools", fake_get_all_tools)
    monkeypatch.setattr(deep_researcher, "compress_research", compress_must_not_run)

    result = await ResearcherQueryEngine(_config()).ainvoke(
        {
            "researcher_messages": [HumanMessage(content="topic")],
            "research_topic": "topic",
            "memory_context": None,
        },
    )

    assert result["cancelled"] is True
    assert result["compressed_research"] == ""
    assert result["raw_notes"] == []


@pytest.mark.asyncio
async def test_failed_research_complete_is_not_a_successful_completion():
    call = {"name": "ResearchComplete", "args": {}, "id": "done-1"}
    error = ToolError(
        error_type=ToolErrorType.permission_denied,
        tool_name="ResearchComplete",
        message="not permitted",
    )
    outcome = GovernedToolCallResult(
        message=error.to_tool_message("done-1"),
        error=error,
    )

    _messages, update = await deep_researcher.prepare_researcher_tool_outcomes(
        [call],
        [outcome],
        {},
        _config(),
    )

    assert update["research_complete_requested"] is True
    assert update["research_complete_succeeded"] is False


@pytest.mark.asyncio
async def test_structured_evidence_is_extracted_before_large_result_offload(
    tmp_path,
) -> None:
    payload = {
        "candidates": [{"candidate_id": "candidate-1"}],
        "documents": [{"document_id": "document-1"}],
        "evidence": [
            {
                "evidence_id": "evidence-1",
                "source_url": "https://example.test/source",
                "supporting_excerpt": "x" * 500,
            }
        ],
        "gap_analysis": {"decision": "complete"},
    }
    call = {
        "name": "web_research",
        "args": {"objective": "test", "queries": ["test"]},
        "id": "web-1",
    }
    outcome = GovernedToolCallResult(
        message=ToolMessage(
            content=deep_researcher.json.dumps(payload),
            name="web_research",
            tool_call_id="web-1",
        )
    )

    messages, update = await deep_researcher.prepare_researcher_tool_outcomes(
        [call],
        [outcome],
        {"web_research": utils.web_research},
        _config(
            runs_dir=str(tmp_path),
            max_mcp_output_chars=256,
            prompt_injection_protection_enabled=False,
        ),
    )

    assert "artifact_ref" in str(messages[0].content)
    assert update["candidate_registry"] == payload["candidates"]
    assert update["document_registry"] == payload["documents"]
    assert update["evidence_registry"] == payload["evidence"]


@pytest.mark.asyncio
async def test_no_tool_response_without_evidence_does_not_complete(monkeypatch) -> None:
    model = FakeResearchModel([
        AIMessage(content="I think this is done"),
        AIMessage(
            content="",
            tool_calls=[
                {"name": "research_echo", "args": {"text": "fact"}, "id": "tool-1"},
            ],
        ),
        AIMessage(
            content="",
            tool_calls=[{"name": "ResearchComplete", "args": {}, "id": "done-1"}],
        ),
    ])

    async def fake_get_all_tools(_config):
        return [research_echo, *deep_researcher.build_supervisor_tools({})[-2:-1]]

    async def fake_compress(state, _config):
        return {"compressed_research": "compressed", "raw_notes": ["evidence"]}

    monkeypatch.setattr(deep_researcher, "configurable_model", model)
    monkeypatch.setattr(deep_researcher, "get_all_tools", fake_get_all_tools)
    monkeypatch.setattr(deep_researcher, "compress_research", fake_compress)

    result = await ResearcherQueryEngine(_config()).ainvoke({
        "researcher_messages": [HumanMessage(content="topic")],
        "research_topic": "topic",
        "evidence_registry": [{"id": "ev-1", "source_url": "https://example.com"}],
    })

    assert len(model.calls) == 3
    assert result["completion_decision"]["action"] == CompletionDecision.COMPLETE.value


@pytest.mark.asyncio
async def test_legacy_researcher_uses_compatibility_completion_policy(
    monkeypatch,
) -> None:
    model = FakeResearchModel([
        AIMessage(
            content="",
            tool_calls=[{"name": "ResearchComplete", "args": {}, "id": "done-1"}],
        )
    ])

    async def fake_get_all_tools(_config):
        return [*deep_researcher.build_supervisor_tools({})[-2:-1]]

    async def fake_compress(_state, _config):
        return {"compressed_research": "legacy result", "raw_notes": []}

    monkeypatch.setattr(deep_researcher, "configurable_model", model)
    monkeypatch.setattr(deep_researcher, "get_all_tools", fake_get_all_tools)
    monkeypatch.setattr(deep_researcher, "compress_research", fake_compress)

    result = await ResearcherQueryEngine(
        _config(web_pipeline_mode="legacy")
    ).ainvoke({
        "researcher_messages": [HumanMessage(content="topic")],
        "research_topic": "topic",
    })

    assert len(model.calls) == 1
    assert result["completion_decision"]["action"] == CompletionDecision.COMPLETE.value


@pytest.mark.asyncio
async def test_researcher_collects_permission_denials(monkeypatch) -> None:
    model = FakeResearchModel([
        AIMessage(
            content="",
            tool_calls=[
                {"name": "research_echo", "args": {"text": "fact"}, "id": "denied-1"},
            ],
        ),
    ])

    async def fake_get_all_tools(_config):
        return [research_echo, deep_researcher.think_tool]

    async def fake_compress(state, _config):
        return {"compressed_research": "", "raw_notes": []}

    monkeypatch.setattr(deep_researcher, "configurable_model", model)
    monkeypatch.setattr(deep_researcher, "get_all_tools", fake_get_all_tools)
    monkeypatch.setattr(deep_researcher, "compress_research", fake_compress)

    result = await ResearcherQueryEngine(_config(
        researcher_tool_whitelist=["think_tool"],
        max_react_tool_calls=1,
    )).ainvoke({
        "researcher_messages": [HumanMessage(content="topic")],
        "research_topic": "topic",
    })

    assert result["permission_denials"] == [{
        "tool_call_id": "denied-1",
        "tool_name": "research_echo",
        "role": "researcher",
        "reason_code": "permission_denied",
        "turn": 1,
        "task_id": "researcher-1",
    }]
