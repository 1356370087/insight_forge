"""Metrics, alert-rule, and dashboard governance tests."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from prometheus_client import generate_latest

from open_deep_research import server
from open_deep_research.observability.telemetry import PrometheusMetrics

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_runs_dir_capacity_metrics_have_no_dynamic_labels() -> None:
    metrics = PrometheusMetrics("odr_governance_capacity_test")

    metrics.set_runs_dir_usage(512, 1024)

    exposition = generate_latest().decode()
    assert "odr_governance_capacity_test_runs_dir_bytes 512.0" in exposition
    assert "odr_governance_capacity_test_runs_dir_quota_bytes 1024.0" in exposition
    assert "odr_governance_capacity_test_runs_dir_quota_ratio 0.5" in exposition
    assert "odr_governance_capacity_test_runs_dir_bytes{" not in exposition


@pytest.mark.asyncio
async def test_metrics_scrape_refreshes_runs_dir_capacity(tmp_path, monkeypatch) -> None:
    (tmp_path / "run-a").mkdir()
    (tmp_path / "run-a" / "manifest.json").write_bytes(b"12345")
    (tmp_path / "traces.sqlite3").write_bytes(b"1234567")
    observed: list[tuple[int, int]] = []
    fake_metrics = SimpleNamespace(
        set_runs_dir_usage=lambda used, quota: observed.append((used, quota)),
        observe_export_error=lambda *_args: None,
    )
    config = SimpleNamespace(runs_dir=str(tmp_path), runs_dir_max_bytes=100)
    monkeypatch.setattr(
        server.Configuration,
        "from_runnable_config",
        lambda _config: config,
    )
    monkeypatch.setattr(server, "get_prometheus_metrics", lambda _config: fake_metrics)

    await server._refresh_operational_metrics()

    assert observed == [(12, 100)]


@pytest.mark.asyncio
async def test_metrics_capacity_refresh_is_fail_open(tmp_path, monkeypatch) -> None:
    errors: list[tuple[str, str]] = []
    fake_metrics = SimpleNamespace(
        set_runs_dir_usage=lambda *_args: None,
        observe_export_error=lambda component, operation: errors.append(
            (component, operation)
        ),
    )
    config = SimpleNamespace(runs_dir=str(tmp_path), runs_dir_max_bytes=100)
    monkeypatch.setattr(
        server.Configuration,
        "from_runnable_config",
        lambda _config: config,
    )
    monkeypatch.setattr(server, "get_prometheus_metrics", lambda _config: fake_metrics)
    monkeypatch.setattr(
        server,
        "_runs_dir_size_bytes",
        lambda _root: (_ for _ in ()).throw(OSError("unavailable")),
    )

    await server._refresh_operational_metrics()

    assert errors == [("prometheus", "runs_dir_usage")]


def test_prometheus_alert_rules_cover_spec_signals() -> None:
    rule_path = REPO_ROOT / "deploy" / "prometheus" / "alerts.yml"
    document = yaml.safe_load(rule_path.read_text(encoding="utf-8"))
    rules = document["groups"][0]["rules"]
    alerts = {rule["alert"]: rule for rule in rules}

    assert set(alerts) == {
        "InsightForgeHighRunFailureRate",
        "InsightForgeTerminalRateLimits",
        "InsightForgeObservabilityExportErrors",
        "InsightForgeResearchQueueWaitHigh",
        "InsightForgeRunsDirQuotaHigh",
    }
    expressions = "\n".join(str(rule["expr"]) for rule in rules)
    for metric in (
        "open_deep_research_runs_total",
        "open_deep_research_terminal_rate_limits_total",
        "open_deep_research_observability_export_errors_total",
        "open_deep_research_research_task_queue_wait_seconds_bucket",
        "open_deep_research_runs_dir_quota_ratio",
    ):
        assert metric in expressions
    assert 'status="failed"' in alerts["InsightForgeHighRunFailureRate"]["expr"]
    assert "> 0.20" in alerts["InsightForgeHighRunFailureRate"]["expr"]
    assert "> 120" in alerts["InsightForgeResearchQueueWaitHigh"]["expr"]
    assert "> 0.80" in alerts["InsightForgeRunsDirQuotaHigh"]["expr"]


def test_grafana_dashboard_is_importable_and_covers_spec_views() -> None:
    dashboard_path = REPO_ROOT / "deploy" / "grafana" / "dashboard.json"
    dashboard = json.loads(dashboard_path.read_text(encoding="utf-8"))

    assert dashboard["uid"] == "insightforge-operations"
    assert dashboard["schemaVersion"] >= 39
    panels = dashboard["panels"]
    assert len({panel["id"] for panel in panels}) == len(panels)
    assert all(panel["targets"] for panel in panels)
    expressions = "\n".join(
        target["expr"] for panel in panels for target in panel["targets"]
    )
    for metric in (
        "open_deep_research_runs_total",
        "open_deep_research_run_duration_seconds_bucket",
        "open_deep_research_llm_tokens_total",
        "open_deep_research_llm_estimated_cost_usd_total",
        "open_deep_research_llm_cache_requests_total",
        "open_deep_research_research_tasks_pending",
        "open_deep_research_research_task_queue_wait_seconds_bucket",
        "open_deep_research_terminal_rate_limits_total",
        "open_deep_research_api_rate_limited_total",
        "open_deep_research_retries_total",
        "open_deep_research_runs_dir_bytes",
    ):
        assert metric in expressions


def test_monitoring_readme_documents_scraping_and_alert_delivery() -> None:
    content = (REPO_ROOT / "deploy" / "README.md").read_text(encoding="utf-8")

    assert "honor_labels: true" in content
    assert "promtool check rules" in content
    assert "alertmanager:9093" in content
    assert "RUNS_DIR_MAX_BYTES" in content
