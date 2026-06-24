from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage

from open_deep_research.agents.query import QueryParams, query
from open_deep_research.observability import (
    SQLiteTraceStore,
    TokenUsage,
    get_trace_recorder,
    invoke_model_with_observability,
)
from open_deep_research.server import app
from security.auth import get_current_user


def _config(trace_path, run_id: str = "obs-run") -> dict[str, Any]:
    return {
        "configurable": {
            "trace_store_path": str(trace_path),
            "event_log_enabled": False,
        },
        "metadata": {"run_id": run_id, "user_id": "user-1"},
    }


class FakeModel:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def with_config(self, _config):
        return self

    def bind_tools(self, _tools):
        return self

    async def ainvoke(self, messages):
        self.calls.append(messages)
        return self.response


def test_token_usage_extracts_usage_metadata_first():
    response = SimpleNamespace(
        usage_metadata={"input_tokens": 3, "output_tokens": 5, "total_tokens": 8},
        response_metadata={"token_usage": {"prompt_tokens": 99}},
    )

    usage = TokenUsage.from_response(response)

    assert usage.input_tokens == 3
    assert usage.output_tokens == 5
    assert usage.total_tokens == 8


def test_token_usage_extracts_provider_metadata_fallbacks():
    response = SimpleNamespace(
        response_metadata={"token_usage": {"prompt_tokens": 7, "completion_tokens": 11}}
    )

    usage = TokenUsage.from_response(response)

    assert usage.input_tokens == 7
    assert usage.output_tokens == 11
    assert usage.total_tokens == 18


@pytest.mark.asyncio
async def test_model_wrapper_persists_span_and_usage(tmp_path):
    trace_path = tmp_path / "trace.sqlite3"
    config = _config(trace_path)
    recorder = get_trace_recorder(config)
    response = AIMessage(
        content="ok",
        usage_metadata={"input_tokens": 2, "output_tokens": 4, "total_tokens": 6},
    )
    model = FakeModel(response)

    with recorder.start_run("obs-run", user_id="user-1"):
        result = await invoke_model_with_observability(
            model,
            [HumanMessage(content="hello")],
            config,
            span_name="test.model",
            agent_role="lead",
            model_name="openai:gpt-test",
        )
        usage = recorder.finish_run("obs-run", "success")

    store = SQLiteTraceStore(str(trace_path))
    spans = store.list_spans("obs-run")

    assert result.content == "ok"
    assert usage == {"input_tokens": 2, "output_tokens": 4, "total_tokens": 6}
    assert any(span["name"] == "test.model" and span["total_tokens"] == 6 for span in spans)
    assert store.get_run("obs-run")["total_tokens"] == 6


@pytest.mark.asyncio
async def test_query_records_model_usage_to_trace_store(tmp_path):
    trace_path = tmp_path / "trace.sqlite3"
    config = _config(trace_path, run_id="query-obs")
    recorder = get_trace_recorder(config)
    model = FakeModel(
        AIMessage(
            content="done",
            usage_metadata={"input_tokens": 1, "output_tokens": 2, "total_tokens": 3},
        )
    )

    with recorder.start_run("query-obs", user_id="user-1"):
        events = [
            event
            async for event in query(QueryParams(
                messages=[HumanMessage(content="start")],
                system_prompt=None,
                model=model,
                config=config,
                max_turns=2,
            ))
        ]
        usage = recorder.finish_run("query-obs", "success")

    assert events[-1].type == "query.completed"
    assert usage["total_tokens"] == 3
    spans = SQLiteTraceStore(str(trace_path)).list_spans("query-obs")
    assert any(span["name"] == "query.model" for span in spans)


def test_observability_endpoints_return_persisted_trace(tmp_path, monkeypatch):
    trace_path = tmp_path / "trace.sqlite3"
    monkeypatch.setenv("TRACE_STORE_PATH", str(trace_path))
    store = SQLiteTraceStore(str(trace_path))
    store.start_run("api-run", "user-1", {})
    store.start_span(
        span_id="span-1",
        run_id="api-run",
        parent_span_id=None,
        name="lead.run",
        kind="run",
        agent_role="lead",
        attributes={},
        input_preview=None,
        provider=None,
        model=None,
    )
    store.finish_span(span_id="span-1", status="success")
    store.finish_run("api-run", "success")

    app.dependency_overrides[get_current_user] = lambda: {"id": "user-1"}
    try:
        client = TestClient(app)
        spans_resp = client.get("/observability/runs/api-run/spans")
        ui_resp = client.get("/observability/ui?run_id=api-run")
    finally:
        app.dependency_overrides.clear()

    assert spans_resp.status_code == 200
    assert spans_resp.json()["spans"][0]["name"] == "lead.run"
    assert ui_resp.status_code == 200
    assert "Open Deep Research Observability" in ui_resp.text
