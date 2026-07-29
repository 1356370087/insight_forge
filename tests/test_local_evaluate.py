"""Unit tests for the local, LangSmith-free evaluation runner."""

import json
from pathlib import Path

import pytest

from open_deep_research.configuration import Configuration
from open_deep_research.report.coverage import (
    derive_coverage_checklist,
    derive_state_coverage_checklist,
)
from open_deep_research.sandbox.policy import allowed_domains
from open_deep_research.tools.utils import get_model_connection_kwargs
from tests import run_local_evaluate
from tests.run_local_evaluate import (
    aggregate_score,
    apply_quality_assessment,
    assess_quality,
    build_argument_parser,
    build_run_config,
    reconcile_judge_metrics,
    recover_persisted_evidence,
    refresh_quality_existing,
    rescore_existing,
)


def test_local_evaluation_defaults_to_one_question() -> None:
    args = build_argument_parser().parse_args([])

    assert args.question_limit == 1
    assert args.question is None


def test_local_evaluation_accepts_one_custom_question() -> None:
    args = build_argument_parser().parse_args([
        "--question",
        "Compare three citation evaluation methods.",
        "--question-title",
        "Citation methods",
    ])

    assert args.question == "Compare three citation evaluation methods."
    assert args.question_title == "Citation methods"


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
    assert "WEB_RERANK_MODEL=openai:deepseek-v4-flash\n" in env_example
    assert "WEB_EVIDENCE_MODEL=openai:deepseek-v4-flash\n" in env_example


def test_local_evaluation_web_models_follow_valid_summarization_model(
    monkeypatch,
) -> None:
    monkeypatch.setenv("SUMMARIZATION_MODEL", "openai:deepseek-v4-flash")
    monkeypatch.delenv("WEB_RERANK_MODEL", raising=False)
    monkeypatch.delenv("WEB_EVIDENCE_MODEL", raising=False)

    config = build_run_config()["configurable"]

    assert config["web_rerank_model"] == "openai:deepseek-v4-flash"
    assert config["web_evidence_model"] == "openai:deepseek-v4-flash"


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
    assert (
        "groundedness_score below 0.70 balanced target"
        in assessment["failures"]
    )


@pytest.mark.parametrize(
    ("rigor", "aggregate_floor", "critical_floor"),
    [
        ("very_relaxed", 0.60, 0.50),
        ("relaxed", 0.70, 0.60),
        ("balanced", 0.75, 0.70),
        ("strict", 0.85, 0.80),
        ("very_strict", 0.90, 0.90),
    ],
)
def test_outer_judge_uses_selected_rigor_thresholds(
    rigor: str,
    aggregate_floor: float,
    critical_floor: float,
) -> None:
    metrics = [
        {"key": key, "score": critical_floor, "status": "scored"}
        for key in (
            "source_quality_score",
            "source_authority_score",
            "groundedness_score",
            "citation_accuracy_score",
            "completeness_score",
            "judge_consistency_score",
        )
    ]

    passing = assess_quality(
        metrics,
        aggregate=aggregate_floor,
        rigor=rigor,
    )
    failing = assess_quality(
        metrics,
        aggregate=aggregate_floor - 0.01,
        rigor=rigor,
    )

    assert passing["passed"] is True
    assert passing["quality_rigor"] == rigor
    assert failing["passed"] is False


def test_outer_judge_execution_compliance_is_hard_in_very_relaxed_mode() -> None:
    metrics = [
        {"key": key, "score": 1.0, "status": "scored"}
        for key in (
            "source_quality_score",
            "source_authority_score",
            "groundedness_score",
            "citation_accuracy_score",
            "completeness_score",
            "judge_consistency_score",
        )
    ]
    metrics.append(
        {
            "key": "execution_compliance_score",
            "score": 0.0,
            "status": "scored",
        }
    )

    assessment = assess_quality(
        metrics,
        aggregate=1.0,
        rigor="very_relaxed",
    )

    assert assessment["passed"] is False
    assert "execution_compliance_score failed" in assessment["failures"]


def test_local_run_disables_clarification(monkeypatch) -> None:
    monkeypatch.setenv("ALLOW_CLARIFICATION", "true")
    monkeypatch.setenv("ENABLE_MEMORY", "true")
    monkeypatch.setenv("WEB_PIPELINE_MODE", "legacy")
    monkeypatch.setenv("SANDBOX_NETWORK_MODE", "full")

    config = build_run_config()

    assert config["configurable"]["allow_clarification"] is False
    assert config["configurable"]["web_pipeline_mode"] == "enforced"
    assert config["metadata"]["evaluation_mode"] == "local"
    with run_local_evaluate.evaluation_runtime_environment():
        resolved = Configuration.from_runnable_config(config)
        assert resolved.allow_clarification is False
        assert resolved.enable_memory is False
        assert resolved.web_pipeline_mode == "enforced"
        assert resolved.sandbox_network_mode == "allow-search-only"


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
        {
            "payload": {
                "update": {
                    "evaluation_snapshot": {
                        "schema_version": "1.0",
                        "evidence_registry": [{"evidence_id": "ev-1"}],
                        "tool_trace": {"completed_task_metrics": [{"task_id": "task-1"}]},
                    }
                }
            }
        },
        {"payload": {"update": {"notes": {"type": "override", "value": []}}}},
    ]
    monkeypatch.setattr(
        Path,
        "read_text",
        lambda _self, **_kwargs: "\n".join(json.dumps(record) for record in records),
    )

    assert recover_persisted_evidence("run-1") == {
        "notes": [],
        "raw_notes": ["evidence"],
        "evaluation_snapshot": {
            "schema_version": "1.0",
            "evidence_registry": [{"evidence_id": "ev-1"}],
            "tool_trace": {"completed_task_metrics": [{"task_id": "task-1"}]},
        },
    }


@pytest.mark.asyncio
async def test_rescore_writes_derived_artifact_without_mutating_source(
    monkeypatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "result.json"
    source_payload = {
        "id": "question-1",
        "title": "Question 1",
        "question": "Research claim A",
        "status": "success",
        "final_report": "Supported report.",
        "research_elapsed_seconds": 1.0,
        "evaluation_elapsed_seconds": 2.0,
        "metrics": [],
        "configuration": {},
        "run_result": {"status": "success"},
    }
    source.write_text(
        json.dumps(source_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    original_bytes = source.read_bytes()

    async def fake_evaluate_state(_inputs, _state):
        return [
            {
                "evaluator": "relevance",
                "key": "relevance_score",
                "score": 0.8,
                "comment": "rescored",
                "status": "scored",
            }
        ]

    monkeypatch.setattr(
        run_local_evaluate,
        "evaluate_state",
        fake_evaluate_state,
    )

    derived_path = await rescore_existing(source)

    assert isinstance(derived_path, Path)
    assert derived_path != source
    assert source.read_bytes() == original_bytes
    assert derived_path.exists()
    derived = json.loads(derived_path.read_text(encoding="utf-8"))
    assert derived["metrics"][0]["comment"] == "rescored"
    assert derived["rescore_provenance"]["source_artifact"] == str(source.resolve())
    assert derived["rescore_provenance"]["source_sha256"]


def test_quality_refresh_writes_derived_artifact_without_mutating_source(
    tmp_path: Path,
) -> None:
    source = tmp_path / "result.json"
    source_payload = {
        "id": "question-1",
        "title": "Question 1",
        "question": "Research claim A",
        "status": "success",
        "final_report": "Supported report.",
        "research_elapsed_seconds": 1.0,
        "evaluation_elapsed_seconds": 2.0,
        "aggregate_score": 0.8,
        "metrics": [],
    }
    source.write_text(
        json.dumps(source_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    original_bytes = source.read_bytes()

    derived_path = refresh_quality_existing(
        source,
        quality_rigor="very_strict",
    )

    assert derived_path != source
    assert source.read_bytes() == original_bytes
    derived = json.loads(derived_path.read_text(encoding="utf-8"))
    assert derived["quality_grade"] == "failed"
    assert derived["quality_refresh_provenance"]["source_artifact"] == str(
        source.resolve()
    )
    assert derived["quality_refresh_provenance"]["source_sha256"]
    assert derived["quality_gate"]["quality_rigor"] == "very_strict"
    assert derived["configuration"]["quality_evaluation_rigor"] == "very_strict"
    assert derived["quality_refresh_provenance"]["quality_evaluation_epoch"]


def test_quality_refresh_recomputes_execution_compliance_from_snapshot(
    tmp_path: Path,
) -> None:
    source = tmp_path / "r18-result.json"
    critical_keys = (
        "source_quality_score",
        "source_authority_score",
        "groundedness_score",
        "citation_accuracy_score",
        "completeness_score",
        "judge_consistency_score",
    )
    metrics = [
        {"key": key, "score": 1.0, "status": "scored", "comment": "ok"}
        for key in critical_keys
    ]
    metrics.append(
        {
            "key": "execution_compliance_score",
            "score": 0.0,
            "status": "scored",
            "comment": "stale parser result",
        }
    )
    question = (
        "必须只从一个ConductResearch子任务中分别使用fetch_url读取"
        "https://peps.python.org/pep-0008/与https://peps.python.org/pep-0257/，"
        "不得拆分为单来源子任务，不得使用搜索或二手来源。"
    )
    source_payload = {
        "id": "r18",
        "title": "R18",
        "question": question,
        "status": "success",
        "final_report": "Non-empty report.",
        "research_elapsed_seconds": 1.0,
        "evaluation_elapsed_seconds": 1.0,
        "aggregate_score": 1.0,
        "metrics": metrics,
        "run_result": {
            "status": "success",
            "quality_gate": {"status": "passed"},
        },
        "evaluation_snapshot": {
            "schema_version": "1.0",
            "evidence_registry": [
                {
                    "source_url": "https://peps.python.org/pep-0008/",
                    "security_status": "accepted",
                },
                {
                    "source_url": "https://peps.python.org/pep-0257/",
                    "security_status": "accepted",
                },
            ],
            "tool_trace": {
                "supervisor_tool_calls": [
                    {"name": "ConductResearch", "id": "task-1", "args": {}},
                ],
                "researcher_tool_calls": [
                    {
                        "name": "fetch_url",
                        "id": "fetch-8",
                        "task_id": "task-1",
                        "args": {"url": "https://peps.python.org/pep-0008/"},
                    },
                    {
                        "name": "fetch_url",
                        "id": "fetch-257",
                        "task_id": "task-1",
                        "args": {"url": "https://peps.python.org/pep-0257/"},
                    },
                ],
                "availability": {"researcher_tool_names_retained": True},
            },
        },
    }
    source.write_text(
        json.dumps(source_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    derived_path = refresh_quality_existing(source)
    derived = json.loads(derived_path.read_text(encoding="utf-8"))
    execution = next(
        metric
        for metric in derived["metrics"]
        if metric["key"] == "execution_compliance_score"
    )

    assert execution["score"] == 1.0
    assert derived["status"] == "success"
    assert derived["final_report"] == "Non-empty report."
    assert derived["quality_gate"]["status"] == "passed"
    assert derived["quality_refresh_provenance"]["deterministic_metrics"] == [
        "execution_compliance_score"
    ]


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
            "research_brief": "只分析成本，并额外比较部署地域。",
        }
    )

    assert any("安全性" in item for item in checklist)
    assert any("上线计划" in item for item in checklist)
    assert not any("部署地域" in item for item in checklist)


def test_completeness_judge_receives_content_requirements_only(monkeypatch) -> None:
    from tests import evaluators

    captured: dict = {}

    def fake_invoke(schema, messages, **_kwargs):
        raw_payload = messages[-1]["content"]
        inner = raw_payload.split("<untrusted_evaluation_payload>\n", 1)[1]
        inner = inner.rsplit("\n</untrusted_evaluation_payload>", 1)[0]
        captured.update(json.loads(inner))
        checklist = [
            {
                "requirement_id": item["requirement_id"],
                "status": "covered",
                "explanation": "Covered in the report.",
            }
            for item in captured["coverage_checklist"]
        ]
        return schema.model_validate(
            {"reasoning": "All report content is present.", "score": 5, "checklist": checklist}
        )

    monkeypatch.setattr(evaluators, "_invoke_structured_output", fake_invoke)
    question = (
        "说明 PEP 8 的行长并给出三条清单。必须只创建一个 ConductResearch "
        "子任务并使用 fetch_url；不得使用搜索。"
    )
    result = evaluators.eval_completeness(
        {"messages": [{"role": "user", "content": question}]},
        {
            "final_report": "PEP 8 行长说明。三条清单。",
            "evaluation_snapshot": {
                "schema_version": "1.0",
                "research_brief": question,
                "coverage_checklist": [
                    "说明 PEP 8 的行长并给出三条清单",
                    "必须只创建一个 ConductResearch 子任务并使用 fetch_url",
                    "不得使用搜索",
                ],
            },
        },
    )

    requirements = [item["requirement"] for item in captured["coverage_checklist"]]
    assert result["score"] == 1.0
    assert not any("ConductResearch" in item for item in requirements)
    assert not any("fetch_url" in item for item in requirements)
    assert not any("搜索" in item for item in requirements)


def test_outer_judge_failure_preserves_partial_report_and_status(monkeypatch) -> None:
    monkeypatch.setenv("EVALUATION_MODEL", "openai:qwen3.7-max")
    report = "# Non-empty recovered report\n\nEvidence remains visible."
    result = {
        "status": "partial",
        "final_report": report,
        "aggregate_score": 0.4,
        "metrics": [
            {
                "key": "completeness_score",
                "status": "scored",
                "score": 0.4,
            }
        ],
        "run_result": {
            "status": "partial",
            "quality_gate": {
                "status": "degraded",
                "reason_codes": ["quality_gate_recovery"],
            },
        },
    }

    apply_quality_assessment(result)

    assert result["status"] == "partial"
    assert result["final_report"] == report
    assert result["run_result"]["status"] == "partial"
    assert result["quality_gate"]["status"] == "failed"
    assert result["quality_gate"]["evaluator_model"] == "openai:qwen3.7-max"
    assert "quality_gate_recovery" in result["quality_gate"]["reason_codes"]


@pytest.mark.asyncio
async def test_run_question_does_not_promote_runtime_error_with_report(
    tmp_path,
    monkeypatch,
) -> None:
    class FailedEngine:
        run_id = "failed-after-report"
        config = {
            "configurable": {
                "search_api": "none",
                "max_concurrent_research_units": 1,
                "max_researcher_iterations": 1,
                "max_react_tool_calls": 1,
            }
        }

        def __init__(self, _config) -> None:
            pass

        async def submit_message(self, _messages, _config):
            return {
                "final_report": "A report generated before finalization failed.",
                "result": {
                    "status": "error",
                    "error": "memory extraction failed",
                },
            }

    async def no_metrics(_inputs, _state):
        return []

    monkeypatch.setattr(run_local_evaluate, "QueryEngine", FailedEngine)
    monkeypatch.setattr(run_local_evaluate, "evaluate_state", no_metrics)
    monkeypatch.setattr(
        run_local_evaluate,
        "apply_quality_assessment",
        lambda _result: None,
    )

    result = await run_local_evaluate.run_question(
        {"id": "runtime-error", "title": "Runtime error", "question": "Test"},
        tmp_path,
        1,
    )

    assert result["status"] == "failed"
    assert result["run_result"]["status"] == "error"
    assert result["final_report"]


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
