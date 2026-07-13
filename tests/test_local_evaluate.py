"""Unit tests for the local, LangSmith-free evaluation runner."""

import json
from pathlib import Path

from open_deep_research.report.coverage import derive_coverage_checklist
from tests.run_local_evaluate import (
    aggregate_score,
    assess_quality,
    build_run_config,
    reconcile_judge_metrics,
    recover_persisted_evidence,
)


def test_aggregate_score_ignores_unscored_and_duplicate_factual_accuracy() -> None:
    metrics = [
        {"key": "relevance_score", "score": 0.8, "status": "scored"},
        {"key": "groundedness_score", "score": 0.6, "status": "scored"},
        {"key": "factual_accuracy_score", "score": 0.6, "status": "scored"},
        {"key": "correctness_score", "score": None, "status": "not_scored"},
    ]

    assert aggregate_score(metrics) == 0.7


def test_quality_assessment_requires_strong_critical_metrics() -> None:
    metrics = [
        {"key": "source_quality_score", "score": 0.9, "status": "scored"},
        {"key": "groundedness_score", "score": 0.9, "status": "scored"},
        {"key": "citation_accuracy_score", "score": 0.9, "status": "scored"},
        {"key": "completeness_score", "score": 0.9, "status": "scored"},
        {"key": "relevance_score", "score": 0.9, "status": "scored"},
    ]

    assessment = assess_quality(metrics, aggregate=0.9)

    assert assessment["grade"] == "excellent"
    assert assessment["passed"] is True


def test_quality_assessment_cannot_average_away_missing_grounding() -> None:
    metrics = [
        {"key": "source_quality_score", "score": 0.9, "status": "scored"},
        {"key": "groundedness_score", "score": 0.0, "status": "scored"},
        {"key": "citation_accuracy_score", "score": 0.0, "status": "scored"},
        {"key": "completeness_score", "score": 0.9, "status": "scored"},
        {"key": "relevance_score", "score": 1.0, "status": "scored"},
    ]

    assessment = assess_quality(metrics, aggregate=0.65)

    assert assessment["grade"] == "failed"
    assert assessment["passed"] is False
    assert "groundedness_score below 0.50" in assessment["failures"]


def test_local_run_disables_clarification(monkeypatch) -> None:
    monkeypatch.setenv("ALLOW_CLARIFICATION", "true")

    config = build_run_config()

    assert config["configurable"]["allow_clarification"] is False
    assert config["configurable"]["web_pipeline_mode"] == "enforced"
    assert config["metadata"]["evaluation_mode"] == "local"


def test_local_run_uses_fresh_isolated_context() -> None:
    first = build_run_config()
    second = build_run_config()

    assert first["metadata"]["run_id"] == first["configurable"]["thread_id"]
    assert second["metadata"]["run_id"] == second["configurable"]["thread_id"]
    assert first["metadata"]["run_id"] != second["metadata"]["run_id"]
    assert first["configurable"]["enable_memory"] is False
    assert second["configurable"]["enable_memory"] is False


def test_recover_persisted_evidence_reads_latest_update(monkeypatch) -> None:
    records = [
        {"payload": {"update": {"notes": ["old"]}}},
        {"payload": {"update": {"notes": ["new"], "raw_notes": ["evidence"]}}},
        {"payload": {"update": {"notes": {"type": "override", "value": []}}}},
    ]
    monkeypatch.setattr(
        Path,
        "read_text",
        lambda _self, **_kwargs: "\n".join(json.dumps(record) for record in records),
    )

    assert recover_persisted_evidence("run-1") == {
        "notes": ["new"],
        "raw_notes": ["evidence"],
    }


def test_coverage_checklist_preserves_explicit_deliverables() -> None:
    checklist = derive_coverage_checklist(
        "请覆盖任务成功率、工具调用质量、事实性和引用质量；提出企业落地方案。"
    )

    assert any("任务成功率" in item for item in checklist)
    assert any("工具调用质量" in item for item in checklist)
    assert any("企业落地方案" in item for item in checklist)


def test_reconciler_caps_optimistic_source_quality() -> None:
    metrics = [
        {"key": "source_quality_score", "score": 1.0, "status": "scored"},
        {"key": "source_authority_score", "score": 0.5, "status": "scored"},
        {"key": "groundedness_score", "score": 0.8, "status": "scored"},
        {"key": "factual_accuracy_score", "score": 0.8, "status": "scored"},
    ]

    reconciled = reconcile_judge_metrics(metrics)

    assert next(
        item["score"] for item in reconciled if item["key"] == "source_quality_score"
    ) == 0.5
    assert next(
        item["score"] for item in reconciled if item["key"] == "judge_consistency_score"
    ) < 1.0
