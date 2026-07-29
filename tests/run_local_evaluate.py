"""Run five local deep-research evaluations without LangSmith."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import statistics
import time
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterator

from dotenv import load_dotenv

from open_deep_research.agents.query_engine import QueryEngine
from open_deep_research.evaluation import (
    EvaluationMetric,
    JudgeConfig,
    MetricStatus,
    normalize_evaluator_metric,
)
from open_deep_research.quality_policy import (
    QualityEvaluationRigor,
    get_quality_rigor_policy,
    rigor_from_legacy_min_score,
)
from tests.evaluators import (
    eval_completeness,
    eval_evidence_integrity,
    eval_execution_compliance,
    eval_overall_quality,
    eval_relevance,
    eval_structure,
    eval_tool_efficiency,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_ROOT = ROOT / "tests" / "local_eval_results"

LOCAL_QUESTIONS = [
    {
        "id": "agent-evaluation",
        "title": "Agent 评估体系",
        "question": (
            "截至 2026 年 7 月，系统比较 LLM Agent 的主流评估方法。请覆盖任务成功率、轨迹与工具调用质量、"
            "事实性和引用质量、成本与延迟、安全性，以及 LLM-as-Judge 的偏差、可复现性和校准方法；"
            "结合公开基准或框架提出一套适合企业深度研究 Agent 的落地评估方案，并引用一手来源。"
        ),
    },
    {
        "id": "solid-state-batteries",
        "title": "全固态电池商业化判断",
        "question": (
            "截至 2026 年 7 月，评估全固态电池在乘用车领域于 2030 年前规模化商业落地的可行性。"
            "比较硫化物、氧化物和聚合物路线的能量密度、安全性、循环寿命、制造良率、成本与供应链，"
            "区分企业公告、实验室数据和已量产事实，并给出关键里程碑、主要风险和概率化结论。"
        ),
    },
    {
        "id": "eu-ai-act-gpai",
        "title": "欧盟 AI Act 合规路线",
        "question": (
            "一家向欧盟客户提供通用人工智能模型和下游 Agent 服务的中型企业，应如何制定 2026—2027 年"
            "EU AI Act 合规路线图？请区分 GPAI、系统性风险 GPAI、高风险下游系统和部署者义务，说明适用"
            "时间线、技术文档、版权政策、训练内容摘要、风险评估、事件报告与供应链合同要求，并引用欧盟官方来源。"
        ),
    },
    {
        "id": "long-context-vs-rag",
        "title": "长上下文与 RAG 架构决策",
        "question": (
            "面向包含 1000 万份多语言企业文档的知识助手，比较长上下文模型、传统 RAG、GraphRAG 和混合方案。"
            "从检索召回、跨文档推理、更新时效、权限隔离、提示注入风险、引用可追溯性、延迟与成本进行分析，"
            "基于公开研究和官方技术资料提出可验证的架构选择、评测集设计与分阶段上线方案。"
        ),
    },
    {
        "id": "carbon-removal-portfolio",
        "title": "碳移除采购组合",
        "question": (
            "一家计划在 2035 年抵消每年 10 万吨残余排放的科技公司，应如何构建高完整性的碳移除采购组合？"
            "比较造林与再造林、生物炭、BECCS、直接空气捕集和增强风化的额外性、永久性、MRV、泄漏风险、"
            "当前与预期成本及可扩展性，并结合权威标准提出采购比例、尽调清单和分阶段签约策略。"
        ),
    },
]

Evaluator = Callable[..., dict[str, Any] | list[dict[str, Any]]]
LOCAL_EVALUATORS: list[tuple[str, Evaluator]] = [
    ("overall_quality", eval_overall_quality),
    ("relevance", eval_relevance),
    ("structure", eval_structure),
    ("evidence_integrity", eval_evidence_integrity),
    ("completeness", eval_completeness),
    ("execution_compliance", eval_execution_compliance),
    ("tool_efficiency", eval_tool_efficiency),
]


def _env_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def build_run_config() -> dict[str, Any]:
    """Build the same QueryEngine configuration for every local question."""
    run_id = str(uuid.uuid4())
    default_model = "openai:deepseek-v4-flash"
    summarization_model = os.getenv("SUMMARIZATION_MODEL", default_model)
    research_model = os.getenv("RESEARCH_MODEL", default_model)
    configurable = {
        "thread_id": run_id,
        # Evaluations must not recall or write cross-run long-term memory.
        "enable_memory": False,
        "max_structured_output_retries": _env_int("MAX_STRUCTURED_OUTPUT_RETRIES", 3),
        "allow_clarification": False,
        "max_concurrent_research_units": _env_int("MAX_CONCURRENT_RESEARCH_UNITS", 3),
        "search_api": os.getenv("SEARCH_API", "tavily"),
        "max_researcher_iterations": _env_int("MAX_RESEARCHER_ITERATIONS", 4),
        "max_react_tool_calls": _env_int("MAX_REACT_TOOL_CALLS", 8),
        # Quality experiments must exercise the evidence admission path.
        "web_pipeline_mode": "enforced",
        # Read-only governed SEARCH tools may fetch their selected targets;
        # arbitrary shell/MCP egress remains outside this policy.
        "sandbox_network_mode": "allow-search-only",
        "web_min_source_authority": float(os.getenv("WEB_MIN_SOURCE_AUTHORITY", "0.65")),
        "web_rerank_model": os.getenv("WEB_RERANK_MODEL", summarization_model),
        "web_evidence_model": os.getenv("WEB_EVIDENCE_MODEL", summarization_model),
        "summarization_model": summarization_model,
        "summarization_model_max_tokens": _env_int("SUMMARIZATION_MODEL_MAX_TOKENS", 8192),
        "research_model": research_model,
        "research_model_max_tokens": _env_int("RESEARCH_MODEL_MAX_TOKENS", 10000),
        "compression_model": os.getenv("COMPRESSION_MODEL", research_model),
        "compression_model_max_tokens": _env_int("COMPRESSION_MODEL_MAX_TOKENS", 10000),
        "final_report_model": os.getenv("FINAL_REPORT_MODEL", research_model),
        "final_report_model_max_tokens": _env_int("FINAL_REPORT_MODEL_MAX_TOKENS", 10000),
    }
    return {
        "configurable": configurable,
        "metadata": {
            "evaluation_mode": "local",
            "run_id": run_id,
        },
    }


@contextmanager
def evaluation_runtime_environment() -> Iterator[None]:
    """Lock isolation settings that must beat project-level environment defaults."""
    overrides = {
        "ALLOW_CLARIFICATION": "false",
        "ENABLE_MEMORY": "false",
        "SANDBOX_NETWORK_MODE": "allow-search-only",
        "WEB_PIPELINE_MODE": "enforced",
    }
    previous = {name: os.environ.get(name) for name in overrides}
    os.environ.update(overrides)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _flatten_evaluation_result(
    evaluator_name: str,
    result: dict[str, Any] | list[dict[str, Any]],
) -> list[dict[str, Any]]:
    items = result if isinstance(result, list) else [result]
    return [normalize_evaluator_metric(evaluator_name, item) for item in items]


async def evaluate_state(inputs: dict[str, Any], state: dict[str, Any]) -> list[dict[str, Any]]:
    """Run local judges sequentially to avoid provider rate-limit bursts."""
    metrics: list[dict[str, Any]] = []
    for evaluator_name, evaluator in LOCAL_EVALUATORS:
        try:
            result = await asyncio.to_thread(evaluator, inputs, state)
            metrics.extend(_flatten_evaluation_result(evaluator_name, result))
        except Exception as exc:  # noqa: BLE001 - preserve the remaining local evaluation
            metrics.append(
                EvaluationMetric(
                    evaluator=evaluator_name,
                    key=f"{evaluator_name}_score",
                    score=None,
                    comment=f"Evaluator failed: {exc}",
                    status=MetricStatus.EVALUATOR_ERROR,
                ).model_dump(mode="json", exclude_none=False)
            )
    metrics.append(
        EvaluationMetric(
            evaluator="correctness",
            key="correctness_score",
            score=None,
            comment="Not scored because this local dataset has no independent golden answer.",
            status=MetricStatus.NOT_SCORED,
        ).model_dump(mode="json", exclude_none=False)
    )
    return reconcile_judge_metrics(metrics)


def _metric(metrics: list[dict[str, Any]], key: str) -> dict[str, Any] | None:
    return next((item for item in metrics if item.get("key") == key), None)


def reconcile_judge_metrics(metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Apply deterministic, conservative consistency rules across Judge outputs."""
    issues: list[str] = []
    evidence_keys = {
        "groundedness_score",
        "citation_accuracy_score",
        "source_authority_score",
    }
    available_evidence_keys = {
        item.get("key")
        for item in metrics
        if item.get("status") == "scored" and item.get("score") is not None
    }
    evidence_inventory_available = evidence_keys <= available_evidence_keys
    if not evidence_inventory_available:
        issues.append("canonical evidence inventory was unavailable or incomplete")
    source_quality = _metric(metrics, "source_quality_score")
    source_authority = _metric(metrics, "source_authority_score")
    if (
        source_quality
        and source_authority
        and source_quality.get("score") is not None
        and source_authority.get("score") is not None
        and float(source_quality["score"]) > float(source_authority["score"]) + 0.10
    ):
        original = float(source_quality["score"])
        source_quality["score"] = float(source_authority["score"])
        source_quality["comment"] = (
            f"{source_quality.get('comment', '')}\nConsistency calibration: source_quality "
            f"was capped from {original:.3f} to {source_authority['score']:.3f} because "
            "the canonical evidence inventory found lower source authority."
        ).strip()
        issues.append("source_quality exceeded audited source authority and was capped")

    consistency_score = (
        max(0.0, 1.0 - 0.25 * len(issues))
        if evidence_inventory_available
        else 0.0
    )
    metrics.append(
        {
            "evaluator": "consistency_reconciler",
            "key": "judge_consistency_score",
            "score": consistency_score,
            "comment": "; ".join(issues) if issues else "No cross-evaluator conflicts detected.",
            "status": "scored",
        }
    )
    return metrics


def aggregate_score(metrics: list[dict[str, Any]]) -> float | None:
    """Average scored dimensions without double-counting factual_accuracy."""
    scores = [
        float(metric["score"])
        for metric in metrics
        if metric.get("status") == "scored"
        and metric.get("score") is not None
        and metric.get("key") not in {
            "execution_compliance_score",
            "factual_accuracy_score",
            "source_authority_score",
        }
    ]
    return statistics.fmean(scores) if scores else None


def assess_quality(
    metrics: list[dict[str, Any]],
    *,
    aggregate: float | None,
    rigor: QualityEvaluationRigor | str = QualityEvaluationRigor.BALANCED,
) -> dict[str, Any]:
    """Apply non-compensatory evidence gates before assigning a quality grade."""
    policy = get_quality_rigor_policy(rigor)
    by_key = {metric["key"]: metric for metric in metrics}
    critical_keys = (
        "source_quality_score",
        "source_authority_score",
        "groundedness_score",
        "citation_accuracy_score",
        "completeness_score",
        "judge_consistency_score",
    )
    hard_failures = [
        f"{metric['key']} {metric.get('status')}"
        for metric in metrics
        if metric.get("status")
        in {
            "error",
            MetricStatus.EVALUATOR_ERROR.value,
            MetricStatus.RUN_FAILED.value,
        }
    ]
    critical_scores: dict[str, float] = {}
    for key in critical_keys:
        metric = by_key.get(key)
        if metric is None or metric.get("status") != "scored" or metric.get("score") is None:
            hard_failures.append(f"{key} not scored")
            continue
        score = float(metric["score"])
        critical_scores[key] = score

    execution_metric = by_key.get("execution_compliance_score")
    if execution_metric is not None:
        if (
            execution_metric.get("status") != "scored"
            or execution_metric.get("score") is None
        ):
            hard_failures.append("execution_compliance_score evaluator error")
        elif float(execution_metric["score"]) < 1.0:
            hard_failures.append("execution_compliance_score failed")

    if aggregate is None or hard_failures:
        grade = "failed"
    elif aggregate >= 0.85 and all(score >= 0.8 for score in critical_scores.values()):
        grade = "excellent"
    elif aggregate >= 0.75 and all(score >= 0.7 for score in critical_scores.values()):
        grade = "good"
    else:
        grade = "needs_improvement"
    policy_failures: list[str] = []
    for key, score in critical_scores.items():
        if score < policy.outer_critical_floor:
            policy_failures.append(
                f"{key} below {policy.outer_critical_floor:.2f} "
                f"{policy.rigor.value} target"
            )
    if aggregate is None:
        policy_failures.append("aggregate_score unavailable")
    elif aggregate < policy.outer_aggregate_floor:
        policy_failures.append(
            f"aggregate_score below {policy.outer_aggregate_floor:.2f} "
            f"{policy.rigor.value} target"
        )
    failures = [*hard_failures, *policy_failures]
    return {
        "grade": grade,
        "passed": not failures,
        "failures": failures,
        "quality_rigor": policy.rigor.value,
        "quality_thresholds": policy.as_dict(),
    }


def _quality_rigor_from_result(
    result: dict[str, Any],
) -> QualityEvaluationRigor:
    """Resolve the immutable run rigor, including legacy result artifacts."""
    run_result = result.get("run_result", {})
    runtime_gate = (
        run_result.get("quality_gate", {})
        if isinstance(run_result, dict)
        else {}
    )
    if isinstance(runtime_gate, dict) and runtime_gate.get("quality_rigor"):
        return QualityEvaluationRigor(str(runtime_gate["quality_rigor"]))
    configuration = result.get("configuration", {})
    if isinstance(configuration, dict):
        if configuration.get("quality_evaluation_rigor"):
            return QualityEvaluationRigor(
                str(configuration["quality_evaluation_rigor"])
            )
        if configuration.get("quality_evaluation_min_score") is not None:
            return rigor_from_legacy_min_score(
                configuration["quality_evaluation_min_score"]
            )
    return QualityEvaluationRigor.BALANCED


def _coerce_quality_rigor(
    value: QualityEvaluationRigor | str,
) -> QualityEvaluationRigor:
    """Normalize enum and string callers without leaking Enum repr strings."""
    return (
        value
        if isinstance(value, QualityEvaluationRigor)
        else QualityEvaluationRigor(str(value))
    )


def apply_quality_assessment(
    result: dict[str, Any],
    *,
    quality_rigor: QualityEvaluationRigor | str | None = None,
) -> dict[str, Any]:
    """Attach the quality gate outcome to a local question result."""
    original_status = result.get("status")
    original_report = result.get("final_report")
    resolved_rigor = (
        _coerce_quality_rigor(quality_rigor)
        if quality_rigor is not None
        else _quality_rigor_from_result(result)
    )
    assessment = assess_quality(
        result.get("metrics", []),
        aggregate=result.get("aggregate_score"),
        rigor=resolved_rigor,
    )
    result["quality_grade"] = assessment["grade"]
    result["quality_gate_passed"] = assessment["passed"]
    result["quality_gate_failures"] = assessment["failures"]
    run_result = result.get("run_result", {})
    runtime_gate = (
        run_result.get("quality_gate", {})
        if isinstance(run_result, dict)
        else {}
    )
    if not runtime_gate and isinstance(result.get("quality_gate"), dict):
        runtime_gate = result["quality_gate"].get("runtime", {})
    runtime_status = (
        str(runtime_gate.get("status", ""))
        if isinstance(runtime_gate, dict)
        else ""
    )
    if runtime_status == "failed":
        gate_status = "failed"
    elif assessment["passed"]:
        gate_status = "degraded" if runtime_status == "degraded" else "passed"
    else:
        gate_status = "failed"
    reason_codes = list(assessment["failures"])
    if isinstance(runtime_gate, dict):
        reason_codes.extend(
            str(item) for item in runtime_gate.get("reason_codes", [])
        )
    result["quality_gate"] = {
        "status": gate_status,
        "evaluator_model": JudgeConfig.from_env().model_spec,
        "policy_version": "offline-quality-gate-v3",
        "quality_rigor": assessment["quality_rigor"],
        "quality_thresholds": assessment["quality_thresholds"],
        "reason_codes": list(dict.fromkeys(reason_codes)),
        "assessment_refs": [
            {
                "key": str(metric.get("key", "")),
                "status": str(metric.get("status", "")),
            }
            for metric in result.get("metrics", [])
            if isinstance(metric, dict) and metric.get("key")
        ],
        **({"runtime": runtime_gate} if runtime_gate else {}),
    }
    # The Judge annotates quality only. It is never allowed to retract a
    # completed research product or rewrite its terminal result status.
    result["status"] = original_status
    result["final_report"] = original_report
    return result


def _json_safe(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def _write_new_json(path: Path, payload: Any) -> None:
    """Create a derived artifact without replacing any existing file."""
    serialized = json.dumps(_json_safe(payload), ensure_ascii=False, indent=2)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(serialized)


def recover_persisted_evidence(run_id: str) -> dict[str, Any]:
    """Replay evaluation fields using the same append/override state semantics."""
    journal = ROOT / ".runs" / run_id / "context" / "session_memory.jsonl"
    recovered: dict[str, Any] = {}
    reduced_list_fields = {
        "notes",
        "raw_notes",
        "completed_task_outputs",
        "supervisor_messages",
        "coverage_checklist",
        "evidence_registry",
    }
    for line in journal.read_text(encoding="utf-8").splitlines():
        update = json.loads(line).get("payload", {}).get("update", {})
        for key in (
            "notes",
            "raw_notes",
            "completed_task_outputs",
            "research_brief",
            "supervisor_messages",
            "coverage_checklist",
            "evidence_registry",
            "evaluation_snapshot",
        ):
            if key not in update:
                continue
            value = update[key]
            if isinstance(value, dict) and value.get("type") == "override":
                recovered[key] = value.get("value")
            elif key in reduced_list_fields and isinstance(value, list):
                current = recovered.get(key, [])
                recovered[key] = [
                    *(current if isinstance(current, list) else []),
                    *value,
                ]
            else:
                recovered[key] = value
    return recovered


def _default_derived_path(source: Path, label: str) -> Path:
    timestamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S-%f")
    return source.with_name(f"{source.stem}.{label}-{timestamp}.json")


def _supported_snapshot(result: dict[str, Any]) -> dict[str, Any] | None:
    snapshot = result.get("evaluation_snapshot")
    if isinstance(snapshot, dict) and snapshot.get("schema_version") == "1.0":
        return snapshot
    return None


def _minimize_persisted_runtime_fields(result: dict[str, Any]) -> None:
    """Drop transient runtime payloads once a versioned snapshot is available."""
    if _supported_snapshot(result) is None:
        return
    for key in (
        "raw_notes",
        "notes",
        "completed_task_outputs",
        "supervisor_messages",
        "evidence_registry",
    ):
        result.pop(key, None)


def _write_derived_result(
    source: Path,
    destination: Path,
    result: dict[str, Any],
) -> Path:
    """Persist a JSON/Markdown derivative while refusing every overwrite."""
    if destination.resolve() == source.resolve():
        raise ValueError("Derived output must differ from the source artifact")
    if destination.suffix.lower() != ".json":
        raise ValueError("Derived output must use a .json extension")
    markdown_path = destination.with_suffix(".md")
    if destination.exists() or markdown_path.exists():
        raise FileExistsError(
            f"Derived output already exists: {destination} or {markdown_path}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    _write_new_json(destination, result)
    with markdown_path.open("x", encoding="utf-8") as stream:
        stream.write(_render_question_markdown(result))
    return destination


async def rescore_existing(
    path: Path,
    run_id: str | None = None,
    *,
    output_path: Path | None = None,
    quality_rigor: QualityEvaluationRigor | str | None = None,
) -> Path:
    """Write a derived Judge artifact without mutating the completed research run."""
    source_bytes = path.read_bytes()
    result = json.loads(source_bytes)
    if run_id:
        recovered = recover_persisted_evidence(run_id)
        for key, value in recovered.items():
            result[key] = value
        result["run_id"] = run_id
    snapshot = _supported_snapshot(result)
    state = {
        "final_report": result.get("final_report", ""),
        "research_brief": (
            snapshot.get("research_brief")
            if snapshot is not None
            else result.get("research_brief", "")
        ),
        "raw_notes": result.get("raw_notes", []),
        "notes": result.get("notes", []),
        "completed_task_outputs": result.get("completed_task_outputs", []),
        "supervisor_messages": result.get("supervisor_messages", []),
        "result": result.get("run_result", {}),
        "evaluation_metadata": result.get("configuration", {}),
        "coverage_checklist": (
            snapshot.get("coverage_checklist", [])
            if snapshot is not None
            else result.get("coverage_checklist", [])
        ),
        "evidence_registry": result.get("evidence_registry", []),
        "evaluation_snapshot": snapshot,
    }
    inputs = {"messages": [{"role": "user", "content": result["question"]}]}
    started = time.perf_counter()
    result["metrics"] = await evaluate_state(inputs, state)
    result["evaluation_elapsed_seconds"] = time.perf_counter() - started
    result["aggregate_score"] = aggregate_score(result["metrics"])
    if quality_rigor is not None:
        resolved_requested_rigor = _coerce_quality_rigor(quality_rigor)
        configuration = dict(result.get("configuration", {}))
        configuration.pop("quality_evaluation_min_score", None)
        configuration["quality_evaluation_rigor"] = (
            resolved_requested_rigor.value
        )
        result["configuration"] = configuration
    apply_quality_assessment(result, quality_rigor=quality_rigor)
    resolved_rigor = result["quality_gate"]["quality_rigor"]
    result["rescore_provenance"] = {
        "source_artifact": str(path.resolve()),
        "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "rescored_at": datetime.now().astimezone().isoformat(),
        "run_id": run_id or result.get("run_id"),
        "evaluation_model": JudgeConfig.from_env().model_spec,
        "quality_policy_version": "offline-quality-gate-v3",
        "quality_rigor": resolved_rigor,
        "quality_evaluation_epoch": str(uuid.uuid4()),
    }
    _minimize_persisted_runtime_fields(result)
    destination = output_path or _default_derived_path(path, "rescore")
    return _write_derived_result(path, destination, result)


def refresh_quality_existing(
    path: Path,
    *,
    output_path: Path | None = None,
    quality_rigor: QualityEvaluationRigor | str | None = None,
) -> Path:
    """Recompute deterministic quality gates into a new immutable artifact."""
    source_bytes = path.read_bytes()
    result = json.loads(source_bytes)
    refreshed_metrics: list[str] = []
    snapshot = _supported_snapshot(result)
    if snapshot is not None:
        inputs = {
            "messages": [
                {"role": "user", "content": str(result.get("question", ""))}
            ]
        }
        outputs = {
            "final_report": result.get("final_report", ""),
            "result": result.get("run_result", {}),
            "evaluation_metadata": result.get("configuration", {}),
            "evaluation_snapshot": snapshot,
            "evidence_registry": result.get("evidence_registry", []),
        }
        execution_result = eval_execution_compliance(inputs, outputs)
        execution_metrics = _flatten_evaluation_result(
            "execution_compliance",
            execution_result,
        )
        result["metrics"] = [
            metric
            for metric in result.get("metrics", [])
            if metric.get("key") != "execution_compliance_score"
        ] + execution_metrics
        result["aggregate_score"] = aggregate_score(result["metrics"])
        refreshed_metrics.append("execution_compliance_score")
    if quality_rigor is not None:
        resolved_requested_rigor = _coerce_quality_rigor(quality_rigor)
        configuration = dict(result.get("configuration", {}))
        configuration.pop("quality_evaluation_min_score", None)
        configuration["quality_evaluation_rigor"] = (
            resolved_requested_rigor.value
        )
        result["configuration"] = configuration
    apply_quality_assessment(result, quality_rigor=quality_rigor)
    result["quality_refresh_provenance"] = {
        "source_artifact": str(path.resolve()),
        "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "refreshed_at": datetime.now().astimezone().isoformat(),
        "deterministic_metrics": refreshed_metrics,
        "quality_policy_version": "offline-quality-gate-v3",
        "quality_rigor": result["quality_gate"]["quality_rigor"],
        "quality_evaluation_epoch": str(uuid.uuid4()),
    }
    _minimize_persisted_runtime_fields(result)
    destination = output_path or _default_derived_path(path, "quality-refresh")
    return _write_derived_result(path, destination, result)


def _render_question_markdown(result: dict[str, Any]) -> str:
    score = result.get("aggregate_score")
    score_text = f"{score:.3f}" if isinstance(score, int | float) else "N/A"
    lines = [
        f"# {result['title']}",
        "",
        "## 研究问题",
        "",
        result["question"],
        "",
        "## 执行结果",
        "",
        f"- 状态：{result['status']}",
        f"- 本地综合得分：{score_text}",
        f"- 质量等级：{result.get('quality_grade', 'unknown')}",
        f"- 质量门禁：{'通过' if result.get('quality_gate_passed') else '未通过'}",
        f"- 研究耗时：{result['research_elapsed_seconds']:.1f} 秒",
        f"- 评估耗时：{result['evaluation_elapsed_seconds']:.1f} 秒",
        "",
        "## LLM-as-Judge 评分",
        "",
        "| 指标 | 得分 | 状态 | 反馈 |",
        "|:--|--:|:--|:--|",
    ]
    for metric in result["metrics"]:
        metric_score = metric.get("score")
        metric_score_text = f"{metric_score:.3f}" if isinstance(metric_score, int | float) else "N/A"
        comment = str(metric.get("comment", "")).replace("|", "\\|").replace("\n", "<br>")
        lines.append(f"| {metric['key']} | {metric_score_text} | {metric['status']} | {comment} |")
    lines.extend(["", "## 深度研究报告", "", result.get("final_report") or "未生成报告。", ""])
    return "\n".join(lines)


def _render_summary(results: list[dict[str, Any]], metadata: dict[str, Any]) -> str:
    scored = [item["aggregate_score"] for item in results if item.get("aggregate_score") is not None]
    overall = statistics.fmean(scored) if scored else None
    overall_text = f"{overall:.3f}" if overall is not None else "N/A"
    lines = [
        "# Open Deep Research 本地深度研究评估报告",
        "",
        f"生成时间：{metadata['completed_at']}",
        "",
        f"Judge 模型：{metadata['evaluation_model']}",
        "",
        f"成功完成：{sum(item['status'] == 'success' for item in results)}/{len(results)}",
        "",
        f"五题平均综合得分：{overall_text}",
        "",
        (
            "`correctness_score` 与 `factual_accuracy_score` 因没有独立黄金答案或外部事实核验源而不评分；"
            "综合得分不会计入仅用于来源门禁校准的 source_authority_score。"
        ),
        "",
        "## 汇总",
        "",
        "| # | 主题 | 状态 | 质量等级 | 综合得分 | 研究耗时（秒） | 评估耗时（秒） |",
        "|--:|:--|:--|:--|--:|--:|--:|",
    ]
    for index, item in enumerate(results, 1):
        score = item.get("aggregate_score")
        score_text = f"{score:.3f}" if isinstance(score, int | float) else "N/A"
        lines.append(
            f"| {index} | {item['title']} | {item['status']} | "
            f"{item.get('quality_grade', 'unknown')} | {score_text} | "
            f"{item['research_elapsed_seconds']:.1f} | {item['evaluation_elapsed_seconds']:.1f} |"
        )
    lines.extend(["", "## 文件索引", ""])
    for index, item in enumerate(results, 1):
        lines.append(f"- `{index:02d}_{item['id']}.md`：{item['title']}")
    lines.append("")
    return "\n".join(lines)


async def run_question(
    question: dict[str, str],
    output_dir: Path,
    index: int,
    *,
    total: int = 1,
) -> dict[str, Any]:
    """Run one complete research-and-evaluate cycle and checkpoint it."""
    inputs = {"messages": [{"role": "user", "content": question["question"]}]}
    print(f"[{index}/{total}] Research started: {question['title']}", flush=True)  # noqa: T201
    research_started = time.perf_counter()
    with evaluation_runtime_environment():
        config = build_run_config()
        engine = QueryEngine(config)
        try:
            state = await engine.submit_message(inputs["messages"], config)
        except Exception as exc:  # noqa: BLE001 - save a durable failed result
            state = {
                "result": {"status": "error", "error": str(exc)},
                "final_report": "",
            }
    research_elapsed = time.perf_counter() - research_started
    effective_config = engine.config.get("configurable", config["configurable"])
    state["evaluation_metadata"] = {
        key: effective_config[key]
        for key in (
            "search_api",
            "max_concurrent_research_units",
            "max_researcher_iterations",
            "max_react_tool_calls",
        )
    }
    print(f"[{index}/{total}] Judge started: {question['title']}", flush=True)  # noqa: T201
    evaluation_started = time.perf_counter()
    metrics = await evaluate_state(inputs, state)
    evaluation_elapsed = time.perf_counter() - evaluation_started
    run_result = state.get("result", {})
    if state.get("final_report"):
        runtime_status = str(run_result.get("status", "success"))
        status = (
            runtime_status
            if runtime_status in {"success", "partial"}
            else "failed"
        )
    else:
        status = "failed"
    result = {
        **question,
        "status": status,
        "aggregate_score": aggregate_score(metrics),
        "research_elapsed_seconds": research_elapsed,
        "evaluation_elapsed_seconds": evaluation_elapsed,
        "metrics": metrics,
        "research_brief": state.get("research_brief", ""),
        "final_report": state.get("final_report", ""),
        "raw_notes": state.get("raw_notes", []),
        "notes": state.get("notes", []),
        "run_result": run_result,
        "run_id": engine.run_id,
        "completed_task_outputs": state.get("completed_task_outputs", []),
        "supervisor_messages": state.get("supervisor_messages", []),
        "evidence_registry": state.get("evidence_registry", []),
        "evaluation_snapshot": state.get("evaluation_snapshot"),
        "configuration": effective_config,
        "coverage_checklist": state.get("coverage_checklist", []),
    }
    _minimize_persisted_runtime_fields(result)
    apply_quality_assessment(result)
    stem = f"{index:02d}_{question['id']}"
    _write_json(output_dir / f"{stem}.json", result)
    (output_dir / f"{stem}.md").write_text(_render_question_markdown(result), encoding="utf-8")
    print(f"[{index}/{total}] Completed: {question['title']}", flush=True)  # noqa: T201
    return result


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the CLI parser; a bare invocation intentionally runs one question."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, help="Directory for local JSON and Markdown results")
    parser.add_argument("--question-limit", type=int, default=1, choices=range(1, 6))
    parser.add_argument(
        "--question",
        help="Run one caller-supplied research question instead of the built-in evaluation set",
    )
    parser.add_argument(
        "--question-title",
        default="自定义研究主题",
        help="Display title used with --question",
    )
    parser.add_argument("--resume", action="store_true", help="Reuse per-question result files that already exist")
    parser.add_argument("--rescore-json", type=Path, help="Re-run only the judges for one existing result")
    parser.add_argument(
        "--rescore-output",
        type=Path,
        help="Optional new JSON path for a rescore artifact; existing files are refused",
    )
    parser.add_argument("--refresh-quality-json", type=Path, help="Recompute only aggregate quality gates")
    parser.add_argument(
        "--refresh-quality-output",
        type=Path,
        help="Optional new JSON path for refreshed quality gates; existing files are refused",
    )
    parser.add_argument("--run-id", help="Persisted run ID used to recover evidence while rescoring")
    parser.add_argument(
        "--quality-rigor",
        choices=[item.value for item in QualityEvaluationRigor],
        help=(
            "Create the derived rescore/refresh artifact under this quality "
            "rigor; the source run remains immutable."
        ),
    )
    return parser


async def main() -> Path:
    load_dotenv(ROOT / ".env")
    args = build_argument_parser().parse_args()

    if args.rescore_json:
        output_path = await rescore_existing(
            args.rescore_json,
            args.run_id,
            output_path=args.rescore_output,
            quality_rigor=args.quality_rigor,
        )
        print(f"Rescored local result: {output_path}", flush=True)  # noqa: T201
        return output_path.parent
    if args.refresh_quality_json:
        output_path = refresh_quality_existing(
            args.refresh_quality_json,
            output_path=args.refresh_quality_output,
            quality_rigor=args.quality_rigor,
        )
        print(f"Refreshed quality gates: {output_path}", flush=True)  # noqa: T201
        return output_path.parent

    timestamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    output_dir = args.output_dir or DEFAULT_RESULTS_ROOT / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now().astimezone().isoformat()
    results: list[dict[str, Any]] = []
    questions = (
        [{
            "id": "custom-research",
            "title": args.question_title,
            "question": args.question,
        }]
        if args.question
        else LOCAL_QUESTIONS[: args.question_limit]
    )
    for index, question in enumerate(questions, 1):
        existing_path = output_dir / f"{index:02d}_{question['id']}.json"
        if args.resume and existing_path.exists():
            results.append(json.loads(existing_path.read_text(encoding="utf-8")))
            print(f"[{index}/{args.question_limit}] Reused: {question['title']}", flush=True)  # noqa: T201
            continue
        results.append(await run_question(question, output_dir, index, total=len(questions)))

    metadata = {
        "started_at": started_at,
        "completed_at": datetime.now().astimezone().isoformat(),
        "evaluation_model": JudgeConfig.from_env().model_spec,
        "langsmith_used": False,
        "question_count": len(results),
    }
    payload = {"metadata": metadata, "results": results}
    _write_json(output_dir / "evaluation_results.json", payload)
    (output_dir / "evaluation_report.md").write_text(_render_summary(results, metadata), encoding="utf-8")
    print(f"Local evaluation written to: {output_dir}", flush=True)  # noqa: T201
    return output_dir


if __name__ == "__main__":
    asyncio.run(main())
