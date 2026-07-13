"""Unit tests for the local, LangSmith-free evaluation runner."""

import json
from pathlib import Path

from open_deep_research.configuration import Configuration
from open_deep_research.report.coverage import (
    derive_coverage_checklist,
    derive_state_coverage_checklist,
)
from open_deep_research.sandbox.policy import allowed_domains
from open_deep_research.tools.utils import get_model_connection_kwargs
from tests.run_local_evaluate import (
    aggregate_score,
    assess_quality,
    build_argument_parser,
    build_run_config,
    reconcile_judge_metrics,
    recover_persisted_evidence,
)


def test_local_evaluation_defaults_to_one_question() -> None:
    args = build_argument_parser().parse_args([])

    assert args.question_limit == 1


def test_local_evaluation_uses_read_only_search_network_mode(monkeypatch) -> None:
    monkeypatch.delenv("SANDBOX_NETWORK_MODE", raising=False)

    config = build_run_config()["configurable"]

    assert config["web_pipeline_mode"] == "enforced"
    assert config["sandbox_network_mode"] == "allow-search-only"
    assert config["enable_memory"] is False
    env_example = (Path(__file__).resolve().parents[1] / ".env.example").read_text(
        encoding="utf-8"
    )
    assert "WEB_PIPELINE_MODE=enforced" in env_example


def test_deepseek_research_models_use_deepseek_credentials_and_endpoint(
    monkeypatch,
) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-test-key")
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://api.deepseek.test")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    config = build_run_config()

    for key in (
        "summarization_model",
        "research_model",
        "compression_model",
        "final_report_model",
        "web_rerank_model",
        "web_evidence_model",
    ):
        model_name = config["configurable"][key]
        connection = get_model_connection_kwargs(model_name, config)
        assert connection == {
            "api_key": "deepseek-test-key",
            "base_url": "https://api.deepseek.test",
        }

    assert "api.deepseek.com" in allowed_domains(
        Configuration.from_runnable_config(config)
    )
    assert get_model_connection_kwargs("openai:gpt-4.1", config) == {
        "api_key": None
    }


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
        {"key": "source_authority_score", "score": 0.9, "status": "scored"},
        {"key": "groundedness_score", "score": 0.9, "status": "scored"},
        {"key": "citation_accuracy_score", "score": 0.9, "status": "scored"},
        {"key": "completeness_score", "score": 0.9, "status": "scored"},
        {"key": "judge_consistency_score", "score": 0.9, "status": "scored"},
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


def test_report_coverage_includes_original_user_requirements_missing_from_brief() -> None:
    checklist = derive_state_coverage_checklist(
        {
            "messages": [
                {"role": "user", "content": "比较成本、安全性，并给出上线计划。"}
            ],
            "research_brief": "只分析成本。",
        }
    )

    assert any("安全性" in item for item in checklist)
    assert any("上线计划" in item for item in checklist)


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
