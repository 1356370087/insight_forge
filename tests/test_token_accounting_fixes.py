from __future__ import annotations

import sqlite3
import time
from typing import Any
from zoneinfo import ZoneInfo

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from open_deep_research.budgets import (
    BudgetGate,
    RunBudgetLedger,
    RunBudgetPolicy,
)
from open_deep_research.configuration import Configuration, frozen_run_config_values
from open_deep_research.observability import (
    SQLiteTraceStore,
    TokenUsage,
    get_trace_recorder,
    invoke_model_with_observability,
)
from open_deep_research.server import _build_usage_analytics_response


def _config(tmp_path, run_id: str, **values: Any) -> dict[str, Any]:
    return {
        "configurable": {
            "trace_store_path": str(tmp_path / "trace.sqlite3"),
            "runs_dir": str(tmp_path / "runs"),
            "event_log_enabled": False,
            **values,
        },
        "metadata": {"run_id": run_id, "user_id": "owner-1"},
    }


class CallbacklessModel:
    def __init__(self, response: AIMessage) -> None:
        self.response = response

    async def ainvoke(self, _messages: list, config: dict | None = None) -> AIMessage:
        return self.response


class StreamFallbackModel(CallbacklessModel):
    def __init__(self, response: AIMessage, error: BaseException) -> None:
        super().__init__(response)
        self.error = error

    async def astream(self, _messages: list, config: dict | None = None):
        if False:
            yield AIMessage(content="unreachable")
        raise self.error


@pytest.mark.asyncio
async def test_full_provider_model_price_key_reaches_budget_boundary(tmp_path) -> None:
    run_id = "priced-run"
    config = _config(
        tmp_path,
        run_id,
        max_run_cost_micro_usd=1_000_000,
        model_costs_per_million={
            "openai:gpt-priced": {"input": 2.0, "output": 4.0}
        },
    )
    recorder = get_trace_recorder(config)
    model = CallbacklessModel(
        AIMessage(
            content="ok",
            usage_metadata={"input_tokens": 3, "output_tokens": 2, "total_tokens": 5},
        )
    )

    with recorder.start_run(run_id, user_id="owner-1"):
        await invoke_model_with_observability(
            model,
            [HumanMessage(content="question")],
            config,
            span_name="test.priced",
            agent_role="researcher",
            model_name="openai:gpt-priced",
            stage="researching",
        )

    ledger = RunBudgetLedger(run_id, runs_dir=str(tmp_path / "runs"))
    assert ledger.snapshot().cost_micro_usd == 14


def test_deterministic_failure_releases_token_and_cost_reservations(tmp_path) -> None:
    ledger = RunBudgetLedger(
        "rejected-run",
        runs_dir=str(tmp_path),
        policy=RunBudgetPolicy(
            max_model_calls=10,
            max_input_tokens=1_000,
            max_output_tokens=1_000,
            max_cost_micro_usd=1_000,
        ),
    )
    gate = BudgetGate(
        ledger=ledger,
        cost_pricing={"openai:gpt-test": {"input": 1, "output": 2}},
    )
    gate.reserve_model_call(
        "attempt-1",
        estimated_input_tokens=100,
        estimated_output_tokens=50,
        model_name="openai:gpt-test",
    )

    gate.fail_model_call("attempt-1", uncertain=False)

    snapshot = ledger.snapshot()
    assert snapshot.model_calls == 1
    assert snapshot.input_tokens == 0
    assert snapshot.output_tokens == 0
    assert snapshot.cost_micro_usd == 0
    assert not any(gate.outstanding_reservations().values())


def test_uncertain_failure_keeps_conservative_reservations(tmp_path) -> None:
    ledger = RunBudgetLedger(
        "uncertain-run",
        runs_dir=str(tmp_path),
        policy=RunBudgetPolicy(
            max_model_calls=10,
            max_input_tokens=1_000,
            max_output_tokens=1_000,
            max_cost_micro_usd=1_000,
        ),
    )
    gate = BudgetGate(
        ledger=ledger,
        cost_pricing={"openai:gpt-test": {"input": 1, "output": 2}},
    )
    gate.reserve_model_call(
        "attempt-1",
        estimated_input_tokens=100,
        estimated_output_tokens=50,
        model_name="openai:gpt-test",
    )

    gate.fail_model_call("attempt-1", uncertain=True)

    assert gate.outstanding_reservations()["input_tokens"] == 100
    assert gate.outstanding_reservations()["output_tokens"] == 50
    assert gate.outstanding_reservations()["cost_micro_usd"] == 200


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error", "failed_status", "accounting_status"),
    [
        (NotImplementedError("stream unsupported"), "rejected", "complete"),
        (TimeoutError("stream timed out"), "unknown_failed", "partial"),
    ],
)
async def test_stream_fallback_records_both_physical_attempts(
    tmp_path,
    error: BaseException,
    failed_status: str,
    accounting_status: str,
) -> None:
    run_id = f"stream-{failed_status}"
    config = _config(
        tmp_path,
        run_id,
        model_circuit_breaker_enabled=True,
        model_first_packet_probe="shadow",
    )
    recorder = get_trace_recorder(config)
    model = StreamFallbackModel(
        AIMessage(
            content="fallback",
            usage_metadata={"input_tokens": 4, "output_tokens": 3, "total_tokens": 7},
        ),
        error,
    )

    with recorder.start_run(run_id, user_id="owner-1"):
        await invoke_model_with_observability(
            model,
            [HumanMessage(content="question")],
            config,
            span_name="test.stream_fallback",
            agent_role="researcher",
            model_name="openai:gpt-test",
            stage="researching",
        )

    store = SQLiteTraceStore(config["configurable"]["trace_store_path"])
    rows = store._usage_rows(run_id)  # noqa: SLF001 - verify physical ordinals
    accounting = store.get_usage_accounting(run_id)
    assert [row["attempt_index"] for row in rows] == [1, 2]
    assert [row["response_status"] for row in rows] == [failed_status, "success"]
    assert accounting["totals"]["reported"]["total_tokens"] == 7
    assert accounting["accounting_status"] == accounting_status


def test_provider_filter_aggregates_only_matching_usage(tmp_path) -> None:
    store = SQLiteTraceStore(str(tmp_path / "filter.sqlite3"))
    store.start_run("filter-run", "owner-1", {})
    for index, (provider, total) in enumerate(
        (("openai", 5), ("anthropic", 11)),
        start=1,
    ):
        span_id = f"span-{index}"
        store.start_span(
            span_id=span_id,
            run_id="filter-run",
            parent_span_id=None,
            name="test.model",
            kind="llm",
            agent_role="researcher",
            attributes={},
            input_preview=None,
            provider=provider,
            model="model-test",
        )
        store.add_usage(
            "filter-run",
            span_id,
            provider,
            "model-test",
            TokenUsage(input_tokens=total - 1, output_tokens=1, total_tokens=total),
            event_key=f"event-{index}",
            stage="researching",
        )

    report = store.get_usage_accounting("filter-run", provider="openai")
    batch_report = store.get_usage_accounting_many(
        ["filter-run"], provider="openai"
    )["filter-run"]

    assert report["totals"]["reported"]["total_tokens"] == 5
    assert batch_report["totals"] == report["totals"]
    assert batch_report["operations"]["llm_call_count"] == 1
    assert [bucket["key"] for bucket in report["breakdowns"]["by_model"]] == [
        "openai:model-test"
    ]


def test_legacy_llm_span_without_usage_is_marked_unclassified(tmp_path) -> None:
    path = tmp_path / "legacy-missing.sqlite3"
    with sqlite3.connect(path) as conn:
        conn.execute(
            """CREATE TABLE spans (
                span_id TEXT PRIMARY KEY, run_id TEXT NOT NULL,
                parent_span_id TEXT, name TEXT NOT NULL, kind TEXT NOT NULL,
                agent_role TEXT, status TEXT NOT NULL, started_at REAL NOT NULL,
                ended_at REAL, duration_ms INTEGER, model TEXT, provider TEXT,
                input_tokens INTEGER NOT NULL DEFAULT 0,
                output_tokens INTEGER NOT NULL DEFAULT 0,
                total_tokens INTEGER NOT NULL DEFAULT 0,
                estimated_cost_usd REAL NOT NULL DEFAULT 0,
                attributes_json TEXT NOT NULL DEFAULT '{}', input_preview TEXT,
                output_preview TEXT, error TEXT
            )"""
        )
        conn.execute(
            """CREATE TABLE usage_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL,
                span_id TEXT NOT NULL, provider TEXT, model TEXT,
                input_tokens INTEGER NOT NULL DEFAULT 0,
                output_tokens INTEGER NOT NULL DEFAULT 0,
                total_tokens INTEGER NOT NULL DEFAULT 0,
                raw_usage_json TEXT NOT NULL DEFAULT '{}', created_at REAL NOT NULL
            )"""
        )
        conn.execute(
            """INSERT INTO spans (
                span_id, run_id, name, kind, agent_role, status, started_at,
                ended_at, duration_ms, model, provider
            ) VALUES ('old-span', 'old-run', 'legacy.model', 'llm',
                      'lead', 'success', 1, 2, 1000, 'gpt-old', 'openai')"""
        )

    store = SQLiteTraceStore(str(path))
    accounting = store.get_usage_accounting("old-run")

    assert accounting["totals"]["calls"]["legacy_unclassified"] == 1
    assert accounting["totals"]["reported"]["total_tokens"] == 0
    assert accounting["accounting_status"] == "partial"


def test_zero_retention_means_forever_and_history_filters_usage_rows(
    tmp_path,
    monkeypatch,
) -> None:
    trace_path = tmp_path / "history.sqlite3"
    monkeypatch.setenv("TRACE_STORE_PATH", str(trace_path))
    monkeypatch.setenv("RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("TRACE_RETENTION_DAYS", "0")
    store = SQLiteTraceStore(str(trace_path))
    store.start_run("old-run", "owner-1", {"title": "Old run"})
    with sqlite3.connect(trace_path) as conn:
        conn.execute(
            "UPDATE runs SET started_at = ?, status = 'success' WHERE run_id = ?",
            (time.time() - 100 * 86400, "old-run"),
        )
    for index, (provider, total) in enumerate(
        (("openai", 5), ("anthropic", 11)),
        start=1,
    ):
        span_id = f"history-span-{index}"
        store.start_span(
            span_id=span_id,
            run_id="old-run",
            parent_span_id=None,
            name="history.model",
            kind="llm",
            agent_role="researcher",
            attributes={},
            input_preview=None,
            provider=provider,
            model="history-model",
        )
        store.add_usage(
            "old-run",
            span_id,
            provider,
            "history-model",
            TokenUsage(input_tokens=total - 1, output_tokens=1, total_tokens=total),
            event_key=f"history-event-{index}",
            stage="researching",
        )

    retained = _build_usage_analytics_response(
        range_name="retained",
        status="completed",
        provider="openai",
        model=None,
        query=None,
        timezone_name="UTC",
        timezone_info=ZoneInfo("UTC"),
        limit=50,
        offset=0,
        user_id="owner-1",
    )
    recent = _build_usage_analytics_response(
        range_name="7d",
        status=None,
        provider=None,
        model=None,
        query=None,
        timezone_name="UTC",
        timezone_info=ZoneInfo("UTC"),
        limit=50,
        offset=0,
        user_id="owner-1",
    )

    assert retained["summary"]["run_count"] == 1
    assert retained["summary"]["reported"]["total_tokens"] == 5
    assert retained["actual_range_days"] >= 99
    assert recent["summary"]["run_count"] == 0


def test_usage_projection_exposes_outstanding_budget_and_retry_timeline(tmp_path) -> None:
    store = SQLiteTraceStore(str(tmp_path / "projection.sqlite3"))
    store.start_run("projection-run", "owner-1", {})
    store.start_span(
        span_id="projection-span",
        run_id="projection-run",
        parent_span_id=None,
        name="projection.model",
        kind="llm",
        agent_role="researcher",
        attributes={},
        input_preview=None,
        provider="openai",
        model="gpt-test",
    )
    store.add_usage(
        "projection-run",
        "projection-span",
        "openai",
        "gpt-test",
        TokenUsage(input_tokens=4, output_tokens=2, total_tokens=6),
        event_key="projection-event",
        stage="writing",
    )
    store.record_retry_event(
        run_id="projection-run",
        span_id="projection-span",
        attempt=1,
        error_type="rate_limited",
    )

    report = store.get_usage_accounting(
        "projection-run",
        reserved_budget={
            "input_tokens": 20,
            "output_tokens": 30,
            "model_calls": 1,
            "cost_micro_usd": 40,
        },
    )

    assert report["totals"]["budgets"]["input_tokens"]["reserved"] == 20
    assert report["totals"]["budgets"]["model_calls"]["reserved"] == 1
    assert sum(bucket["retry_count"] for bucket in report["timeline"]) == 1
    assert report["breakdowns"]["by_stage"][0]["key"] == "writing"


def test_no_usage_has_specific_unavailable_reason(tmp_path) -> None:
    store = SQLiteTraceStore(str(tmp_path / "empty.sqlite3"))
    store.start_run("empty-run", "owner-1", {})

    report = store.get_usage_accounting("empty-run")

    assert report["accounting_status"] == "unavailable"
    assert report["unavailable_reason"] == "no_usage_events"


def test_configuration_accepts_unlimited_trace_retention() -> None:
    configuration = Configuration(trace_retention_days=0)
    assert configuration.trace_retention_days == 0


def test_v5_fingerprint_contract_excludes_v6_token_fields() -> None:
    values = {
        "token_usage_accounting_enabled": False,
        "token_usage_estimation_enabled": False,
        "model_costs_per_million": {"openai:gpt-test": {"input": 1}},
    }
    old = frozen_run_config_values(
        {
            "configurable": values,
            "metadata": {"run_config_schema_version": 5},
        }
    )
    current = frozen_run_config_values(
        {
            "configurable": values,
            "metadata": {"run_config_schema_version": 6},
        }
    )

    assert not set(values).intersection(old)
    assert set(values).issubset(current)
