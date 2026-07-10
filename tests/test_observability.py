from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage
from prometheus_client import generate_latest

from open_deep_research.agents.query import QueryParams, query
from open_deep_research.observability import (
    SQLiteTraceStore,
    TokenUsage,
    bind_span_context,
    get_trace_recorder,
    invoke_model_with_observability,
    invoke_model_with_retry_observability,
    observe_tool_call,
)
from open_deep_research.observability.telemetry import LangfuseSpanBridge
from open_deep_research.server import app
from open_deep_research.tools.governance import (
    AgentRole,
    ToolErrorType,
    classify_llm_retryable_error,
    execute_governed_tool_call,
    invoke_tool_with_retry,
)
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


def test_token_usage_skips_empty_candidate_and_extracts_details():
    response = SimpleNamespace(
        usage_metadata={},
        response_metadata={
            "token_usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "prompt_tokens_details": {"cached_tokens": 40},
                "completion_tokens_details": {"reasoning_tokens": 20},
            }
        },
    )

    usage = TokenUsage.from_response(response)

    assert usage.input_tokens == 100
    assert usage.output_tokens == 50
    assert usage.cached_input_tokens == 40
    assert usage.reasoning_tokens == 20


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
    assert usage["input_tokens"] == 2
    assert usage["output_tokens"] == 4
    assert usage["total_tokens"] == 6
    assert usage["retry_count"] == 0
    assert usage["rate_429"] == 0.0
    assert any(span["name"] == "test.model" and span["total_tokens"] == 6 for span in spans)
    assert store.get_run("obs-run")["total_tokens"] == 6


@pytest.mark.asyncio
async def test_cost_breakdown_and_redaction_are_persisted(tmp_path):
    trace_path = tmp_path / "cost.sqlite3"
    config = _config(trace_path, run_id="cost-run")
    config["configurable"]["model_costs_per_million"] = {
        "openai:gpt-test": {
            "input": 1.0,
            "cached_input": 0.5,
            "output": 2.0,
            "reasoning": 4.0,
        }
    }
    response = AIMessage(
        content="answer with api_key=sk-super-secret-value",
        usage_metadata={
            "input_tokens": 100,
            "output_tokens": 50,
            "total_tokens": 150,
            "input_token_details": {"cache_read": 40},
            "output_token_details": {"reasoning": 20},
        },
    )
    recorder = get_trace_recorder(config)

    with recorder.start_run("cost-run", user_id="user-1"):
        await invoke_model_with_observability(
            FakeModel(response),
            [HumanMessage(content="Authorization: Bearer secret-token")],
            config,
            span_name="cost.model",
            agent_role="lead",
            model_name="openai:gpt-test",
        )
        recorder.finish_run("cost-run", "success")

    store = SQLiteTraceStore(str(trace_path))
    usage = store.get_usage("cost-run")
    span = next(item for item in store.list_spans("cost-run") if item["name"] == "cost.model")
    assert usage["cached_input_tokens"] == 40
    assert usage["reasoning_tokens"] == 20
    assert usage["estimated_cost_usd"] == pytest.approx(0.00022)
    assert "secret-token" not in span["input_preview"]
    assert "sk-super-secret-value" not in span["output_preview"]


def test_rate_429_counts_distinct_calls_not_retry_events(tmp_path):
    store = SQLiteTraceStore(str(tmp_path / "rate.sqlite3"))
    store.start_run("rate-run", "user-1", {})
    for span_id in ("limited", "ok"):
        store.start_span(
            span_id=span_id,
            run_id="rate-run",
            parent_span_id=None,
            name=f"tool.{span_id}",
            kind="tool",
            agent_role="researcher",
            attributes={},
            input_preview=None,
            provider=None,
            model=None,
        )
        store.finish_span(span_id=span_id, status="success")
    for attempt in (1, 2, 3):
        store.record_retry_event(
            run_id="rate-run",
            span_id="limited",
            attempt=attempt,
            error_type="rate_limited",
        )
    store.finish_run("rate-run", "success")

    metrics = store.get_metrics("rate-run")

    assert metrics["rate_limit_events"] == 3
    assert metrics["rate_limited_count"] == 1
    assert metrics["rate_429"] == 0.5


def test_explicit_background_context_preserves_parent(tmp_path):
    trace_path = tmp_path / "parent.sqlite3"
    config = _config(trace_path, run_id="parent-run")
    recorder = get_trace_recorder(config)
    with recorder.start_run("parent-run", user_id="user-1") as root:
        root_id = root.span_id
    with bind_span_context("parent-run", root_id):
        with recorder.start_span(name="researcher.topic", kind="agent"):
            pass
    recorder.finish_run("parent-run", "success")

    child = next(
        item
        for item in SQLiteTraceStore(str(trace_path)).list_spans("parent-run")
        if item["name"] == "researcher.topic"
    )
    assert child["parent_span_id"] == root_id


@pytest.mark.asyncio
async def test_governance_rejection_marks_tool_span_as_error(tmp_path):
    trace_path = tmp_path / "governance.sqlite3"
    config = _config(trace_path, run_id="governance-run")
    recorder = get_trace_recorder(config)
    tool_call = {"id": "missing-1", "name": "missing_tool", "args": {}}
    with recorder.start_run("governance-run", user_id="user-1"):
        outcome = await observe_tool_call(
            tool_call,
            AgentRole.RESEARCHER.value,
            config,
            lambda: execute_governed_tool_call(
                tool_call,
                {},
                AgentRole.RESEARCHER,
                config,
            ),
        )
        recorder.finish_run("governance-run", "success")

    span = next(
        item
        for item in SQLiteTraceStore(str(trace_path)).list_spans("governance-run")
        if item["name"] == "tool.missing_tool"
    )
    assert outcome.error.error_type is ToolErrorType.tool_not_found
    assert span["status"] == "error"
    assert span["error_type"] == "tool_not_found"


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


def test_observability_endpoints_are_scoped_to_authenticated_owner(tmp_path, monkeypatch):
    trace_path = tmp_path / "owner.sqlite3"
    monkeypatch.setenv("TRACE_STORE_PATH", str(trace_path))
    store = SQLiteTraceStore(str(trace_path))
    for run_id, owner in (("mine", "user-1"), ("theirs", "user-2")):
        store.start_run(run_id, owner, {})
        store.finish_run(run_id, "success")

    app.dependency_overrides[get_current_user] = lambda: {"identity": "user-1"}
    try:
        client = TestClient(app)
        listed = client.get("/observability/runs")
        hidden = client.get("/observability/runs/theirs")
    finally:
        app.dependency_overrides.clear()

    assert listed.status_code == 200
    assert [run["run_id"] for run in listed.json()["runs"]] == ["mine"]
    assert hidden.status_code == 404


def test_metrics_endpoint_returns_aggregated_metrics(tmp_path, monkeypatch):
    trace_path = tmp_path / "trace.sqlite3"
    monkeypatch.setenv("TRACE_STORE_PATH", str(trace_path))
    store = SQLiteTraceStore(str(trace_path))
    store.start_run("metrics-run", "user-1", {})
    store.start_span(
        span_id="span-m",
        run_id="metrics-run",
        parent_span_id=None,
        name="test.model",
        kind="llm",
        agent_role="lead",
        attributes={},
        input_preview=None,
        provider="openai",
        model="gpt-test",
    )
    store.record_retry_event(
        run_id="metrics-run",
        span_id="span-m",
        attempt=1,
        error_type="rate_limited",
        http_status=429,
        retryable=True,
        delay_s=0.5,
        message="429",
    )
    store.finish_span(span_id="span-m", status="success", retry_count=1)
    store.finish_run("metrics-run", "success")

    app.dependency_overrides[get_current_user] = lambda: {"id": "user-1"}
    try:
        client = TestClient(app)
        resp = client.get("/observability/runs/metrics-run/metrics")
    finally:
        app.dependency_overrides.clear()

    assert resp.status_code == 200
    metrics = resp.json()["metrics"]
    assert metrics["retry_count"] == 1
    assert metrics["rate_limited_count"] == 1
    assert metrics["total_llm_tool_calls"] == 1
    assert metrics["rate_429"] == 1.0
    assert metrics["by_span"][0]["retry_count"] == 1


# ---------------------------------------------------------------------------
# Retry observability (Phase 1)
# ---------------------------------------------------------------------------


class FlakyModel:
    """Model that raises a configured sequence of exceptions before succeeding."""

    def __init__(self, failures, response):
        self.failures = list(failures)
        self.response = response
        self.calls = 0

    def with_config(self, _config):
        return self

    def bind_tools(self, _tools):
        return self

    async def ainvoke(self, messages):
        self.calls += 1
        if self.failures:
            raise self.failures.pop(0)
        return self.response


def _fake_openai_sdk():
    """A stand-in for the openai SDK module with the exception classes we inspect."""
    return SimpleNamespace(
        RateLimitError=type("RateLimitError", (Exception,), {}),
        APITimeoutError=type("APITimeoutError", (Exception,), {}),
        APIConnectionError=type("APIConnectionError", (Exception,), {}),
        APIStatusError=type("APIStatusError", (Exception,), {}),
    )


def test_classify_llm_retryable_error_native_429(monkeypatch):
    import open_deep_research.tools.governance as gov

    fake = _fake_openai_sdk()
    monkeypatch.setattr(gov, "_openai_sdk", fake)
    error_type, retryable = classify_llm_retryable_error(fake.RateLimitError())
    assert error_type is ToolErrorType.rate_limited
    assert retryable is True


def test_classify_llm_retryable_error_api_status_5xx(monkeypatch):
    import open_deep_research.tools.governance as gov

    fake = _fake_openai_sdk()
    monkeypatch.setattr(gov, "_openai_sdk", fake)
    err = fake.APIStatusError()
    err.status_code = 503
    error_type, retryable = classify_llm_retryable_error(err)
    assert error_type is ToolErrorType.service_unavailable
    assert retryable is True


def test_classify_llm_retryable_error_parse_failure_not_retryable():
    from langchain_core.exceptions import OutputParserException

    error_type, retryable = classify_llm_retryable_error(OutputParserException("bad json"))
    assert retryable is False


@pytest.mark.asyncio
async def test_llm_retry_loop_records_retry_and_429(tmp_path, monkeypatch):
    import open_deep_research.tools.governance as gov

    fake = _fake_openai_sdk()
    monkeypatch.setattr(gov, "_openai_sdk", fake)

    trace_path = tmp_path / "trace.sqlite3"
    config = _config(trace_path, run_id="retry-run")
    recorder = get_trace_recorder(config)
    response = AIMessage(
        content="recovered",
        usage_metadata={"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
    )
    model = FlakyModel([fake.RateLimitError("429")], response)

    sleeps: list[float] = []

    async def fake_sleep(delay):
        sleeps.append(delay)

    with recorder.start_run("retry-run", user_id="user-1"):
        result = await invoke_model_with_retry_observability(
            model,
            [HumanMessage(content="q")],
            config,
            span_name="test.retry",
            agent_role="lead",
            model_name="openai:gpt-test",
            max_attempts=3,
            base_delay=0.1,
            max_delay=1.0,
            sleeper=fake_sleep,
        )
        usage = recorder.finish_run("retry-run", "success")

    store = SQLiteTraceStore(str(trace_path))
    spans = store.list_spans("retry-run")
    retry_summary = store.get_retry_summary("retry-run")
    metrics = store.get_metrics("retry-run")

    assert result.content == "recovered"
    assert len(sleeps) == 1  # one backoff before the successful attempt
    assert any(span["retry_count"] == 1 for span in spans)
    assert retry_summary["retry_count"] == 1
    assert retry_summary["rate_limited_count"] == 1
    assert metrics["rate_limited_count"] == 1
    assert metrics["total_llm_tool_calls"] == 1
    assert metrics["rate_429"] == 1.0
    assert usage["retry_count"] == 1


@pytest.mark.asyncio
async def test_llm_retry_loop_exhausts_and_records_outcome(tmp_path, monkeypatch):
    import open_deep_research.tools.governance as gov

    fake = _fake_openai_sdk()
    monkeypatch.setattr(gov, "_openai_sdk", fake)

    trace_path = tmp_path / "trace.sqlite3"
    config = _config(trace_path, run_id="exhaust-run")
    recorder = get_trace_recorder(config)
    response = AIMessage(content="never", usage_metadata={"input_tokens": 1, "output_tokens": 1, "total_tokens": 2})
    model = FlakyModel([fake.RateLimitError("429"), fake.RateLimitError("429")], response)

    async def fake_sleep(_delay):
        return None

    with recorder.start_run("exhaust-run", user_id="user-1"):
        with pytest.raises(fake.RateLimitError):
            await invoke_model_with_retry_observability(
                model,
                [HumanMessage(content="q")],
                config,
                span_name="test.exhaust",
                agent_role="lead",
                model_name="openai:gpt-test",
                max_attempts=2,
                base_delay=0.1,
                max_delay=1.0,
                sleeper=fake_sleep,
            )
        usage = recorder.finish_run("exhaust-run", "error", "exhausted")

    store = SQLiteTraceStore(str(trace_path))
    spans = store.list_spans("exhaust-run")
    metrics = store.get_metrics("exhaust-run")

    assert any(span["retry_count"] == 1 and span["error_type"] == "rate_limited" for span in spans)
    assert metrics["retry_count"] == 1
    assert metrics["rate_limited_count"] == 1
    assert usage["retry_count"] == 1


@pytest.mark.asyncio
async def test_llm_parse_failure_is_not_retried(tmp_path):
    from langchain_core.exceptions import OutputParserException

    trace_path = tmp_path / "trace.sqlite3"
    config = _config(trace_path, run_id="parse-run")
    recorder = get_trace_recorder(config)
    response = AIMessage(content="ok")
    model = FlakyModel([OutputParserException("bad")], response)

    sleeps: list[float] = []

    async def fake_sleep(delay):
        sleeps.append(delay)

    with recorder.start_run("parse-run", user_id="user-1"):
        with pytest.raises(OutputParserException):
            await invoke_model_with_retry_observability(
                model,
                [HumanMessage(content="q")],
                config,
                span_name="test.parse",
                agent_role="lead",
                model_name="openai:gpt-test",
                sleeper=fake_sleep,
            )

    assert sleeps == []  # parse failures must not trigger a backoff/retry


@pytest.mark.asyncio
async def test_tool_retry_records_retry_event(tmp_path):
    from langchain_core.tools import ToolException
    from pydantic import BaseModel

    from open_deep_research.tools.base import (
        ToolContext,
        ToolOrigin,
        ToolResult,
    )

    trace_path = tmp_path / "trace.sqlite3"
    config = _config(trace_path, run_id="tool-retry")
    recorder = get_trace_recorder(config)

    calls = {"n": 0}

    class FakeInput(BaseModel):
        pass

    class FakeTool:
        name = "fake_search"
        input_schema = FakeInput
        origin = ToolOrigin.SEARCH
        retryable = True

        async def description(self, input=None):
            return "fake"

        async def call(self, input, context, on_progress=None):
            calls["n"] += 1
            if calls["n"] == 1:
                raise ToolException("429 rate limit exceeded")
            return ToolResult(output="tool-result")

    async def fake_sleep(_delay):
        return None

    with recorder.start_run("tool-retry", user_id="user-1"):
        with recorder.start_span(name="tool.fake_search", kind="tool", agent_role="researcher"):
            result = await invoke_tool_with_retry(
                FakeTool(),
                FakeInput(),
                ToolContext(config=config, role="researcher", tool_call_id="retry"),
                max_retries=3,
                base_delay=0.1,
                max_delay=1.0,
                sleeper=fake_sleep,
            )
        metrics_from_run = recorder.finish_run("tool-retry", "success")

    store = SQLiteTraceStore(str(trace_path))
    retry_summary = store.get_retry_summary("tool-retry")
    metrics = store.get_metrics("tool-retry")

    assert result.output == "tool-result"
    assert calls["n"] == 2  # one failed attempt + one successful
    assert retry_summary["retry_count"] == 1
    assert retry_summary["rate_limited_count"] == 1
    assert metrics["total_llm_tool_calls"] == 1  # one tool span
    assert metrics["rate_429"] == 1.0
    assert metrics_from_run["retry_count"] == 1


def test_get_metrics_empty_run(tmp_path):
    store = SQLiteTraceStore(str(tmp_path / "trace.sqlite3"))
    metrics = store.get_metrics("nonexistent")
    assert metrics["input_tokens"] == 0
    assert metrics["retry_count"] == 0
    assert metrics["rate_429"] == 0.0
    assert metrics["total_llm_tool_calls"] == 0
    assert metrics["by_span"] == []


@pytest.mark.asyncio
async def test_prometheus_records_llm_tokens_and_run_metrics(tmp_path):
    trace_path = tmp_path / "prometheus.sqlite3"
    config = _config(trace_path, run_id="prom-run")
    config["configurable"].update({
        "prometheus_enabled": True,
        "prometheus_namespace": "odr_observability_test",
    })
    recorder = get_trace_recorder(config)
    model = FakeModel(
        AIMessage(
            content="ok",
            usage_metadata={"input_tokens": 3, "output_tokens": 5, "total_tokens": 8},
        )
    )

    with recorder.start_run("prom-run", user_id="user-1"):
        await invoke_model_with_observability(
            model,
            [HumanMessage(content="hello")],
            config,
            span_name="lead.test",
            agent_role="lead",
            model_name="openai:gpt-test",
        )
        recorder.finish_run("prom-run", "success")

    exposition = generate_latest().decode()
    assert 'odr_observability_test_llm_requests_total{agent_role="lead"' in exposition
    assert 'direction="input",model="gpt-test",provider="openai"} 3.0' in exposition
    assert 'direction="output",model="gpt-test",provider="openai"} 5.0' in exposition
    assert 'odr_observability_test_runs_total{status="success"} 1.0' in exposition


def test_langfuse_bridge_maps_generation_usage_and_trace_id():
    class FakeObservation:
        def __init__(self):
            self.updates = []

        def update(self, **kwargs):
            self.updates.append(kwargs)

    class FakeContext:
        def __init__(self, observation):
            self.observation = observation

        def __enter__(self):
            return self.observation

        def __exit__(self, *_args):
            return None

    class FakeClient:
        def __init__(self):
            self.started = []
            self.observation = FakeObservation()

        def create_trace_id(self, *, seed):
            assert seed == "run-1"
            return "a" * 32

        def start_as_current_observation(self, **kwargs):
            self.started.append(kwargs)
            return FakeContext(self.observation)

    class FakeSink:
        def __init__(self):
            self.client = FakeClient()
            self.user_id = "user-1"
            self.session_id = "session-1"

        @staticmethod
        def propagate_attributes(**_kwargs):
            return FakeContext(FakeObservation())

    span = SimpleNamespace(
        kind="llm",
        name="lead.model",
        input_preview="prompt",
        output_preview="answer",
        attributes={},
        run_id="run-1",
        parent_span_id=None,
        agent_role="lead",
        provider="openai",
        model="gpt-test",
        usage=TokenUsage(input_tokens=2, output_tokens=4, total_tokens=6),
        retry_count=0,
        error_type=None,
        http_status=None,
    )
    sink = FakeSink()
    bridge = LangfuseSpanBridge(sink, span)

    bridge.enter()
    bridge.exit(None, None, None)

    started = sink.client.started[0]
    assert started["as_type"] == "generation"
    assert started["trace_context"] == {"trace_id": "a" * 32}
    assert started["model"] == "gpt-test"
    assert sink.client.observation.updates[0]["usage_details"] == {
        "input": 2,
        "output": 4,
        "total": 6,
    }


def test_langfuse_bridge_uses_captured_external_parent_id():
    class FakeObservation:
        def update(self, **_kwargs):
            return None

    class FakeContext:
        def __enter__(self):
            return FakeObservation()

        def __exit__(self, *_args):
            return None

    class FakeClient:
        def __init__(self):
            self.started = []

        @staticmethod
        def create_trace_id(*, seed):
            assert seed == "run-parent"
            return "a" * 32

        def start_as_current_observation(self, **kwargs):
            self.started.append(kwargs)
            return FakeContext()

        @staticmethod
        def get_current_observation_id():
            return "c" * 16

    sink = SimpleNamespace(
        client=FakeClient(),
        user_id="user-1",
        session_id="session-1",
        propagate_attributes=lambda **_kwargs: FakeContext(),
    )
    span = SimpleNamespace(
        kind="agent",
        name="researcher.topic",
        input_preview=None,
        output_preview=None,
        attributes={},
        run_id="run-parent",
        parent_span_id="local-parent",
        langfuse_parent_span_id="b" * 16,
        langfuse_observation_id=None,
        agent_role="researcher",
        provider=None,
        model=None,
        usage=TokenUsage(),
        retry_count=0,
        error_type=None,
        http_status=None,
        final_status=None,
        error_message=None,
    )

    bridge = LangfuseSpanBridge(sink, span)
    bridge.enter()
    bridge.exit(None, None, None)

    assert sink.client.started[0]["trace_context"] == {
        "trace_id": "a" * 32,
        "parent_span_id": "b" * 16,
    }
    assert span.langfuse_observation_id == "c" * 16


@pytest.mark.asyncio
async def test_optional_langchain_callback_is_passed_to_model(tmp_path, monkeypatch):
    import open_deep_research.observability.core as core

    marker = object()

    class FakeLangfuseSink:
        @staticmethod
        def callback_handler():
            return marker

        @staticmethod
        def span(_span):
            return None

    class CallbackAwareModel:
        def __init__(self):
            self.config = None

        async def ainvoke(self, _messages, config=None):
            self.config = config
            return AIMessage(content="ok")

    config = _config(tmp_path / "callback.sqlite3", run_id="callback-run")
    config["configurable"]["langfuse_langchain_callback_enabled"] = True
    recorder = get_trace_recorder(config)
    recorder.langfuse = FakeLangfuseSink()
    monkeypatch.setattr(core, "get_trace_recorder", lambda _config: recorder)
    model = CallbackAwareModel()

    with recorder.start_run("callback-run"):
        await invoke_model_with_observability(
            model,
            [HumanMessage(content="hello")],
            config,
            span_name="callback.model",
            model_name="openai:gpt-test",
        )

    assert model.config is not None
    assert model.config["callbacks"][-1] is marker
