"""Quality-rigor policy and configuration compatibility tests."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from open_deep_research.configuration import (
    Configuration,
    freeze_run_config,
    run_config_fingerprint,
)
from open_deep_research.quality.gate import (
    evaluate_subagent_handoff,
    evaluate_tool_results,
)
from open_deep_research.quality.policy import (
    QualityEvaluationRigor,
    get_quality_rigor_policy,
)


@pytest.mark.parametrize(
    ("rigor", "dimension_floor", "average_floor", "outer_aggregate", "outer_critical"),
    [
        ("very_relaxed", 2, 2.5, 0.60, 0.50),
        ("relaxed", 2, 3.0, 0.70, 0.60),
        ("balanced", 3, 3.0, 0.75, 0.70),
        ("strict", 3, 4.0, 0.85, 0.80),
        ("very_strict", 4, 4.5, 0.90, 0.90),
    ],
)
def test_quality_rigor_profiles_are_canonical(
    rigor: str,
    dimension_floor: int,
    average_floor: float,
    outer_aggregate: float,
    outer_critical: float,
) -> None:
    policy = get_quality_rigor_policy(rigor)

    assert policy.rigor.value == rigor
    assert policy.runtime_dimension_floor == dimension_floor
    assert policy.runtime_average_floor == average_floor
    assert policy.outer_aggregate_floor == outer_aggregate
    assert policy.outer_critical_floor == outer_critical


def _strict_fail_open_config() -> dict:
    return {
        "configurable": {
            "quality_evaluation_rigor": "strict",
            "quality_evaluation_fail_open": True,
            "quality_evaluation_min_sources": 2,
        },
        "metadata": {
            "runtime_config_frozen": True,
            "quality_policy_version": "quality-gate-v3",
            "quality_evaluation_epoch": "epoch-strict-fail-open",
        },
    }


@pytest.mark.asyncio
async def test_strict_tool_gate_preserves_fail_open_on_judge_timeout(
    monkeypatch,
) -> None:
    async def fake_invoke(*_args, **_kwargs):
        raise TimeoutError("judge timed out")

    monkeypatch.setattr(
        "open_deep_research.quality.gate._build_quality_model",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        "open_deep_research.quality.gate.invoke_model_with_retry_observability",
        fake_invoke,
    )

    result = await evaluate_tool_results(
        "Synthetic topic",
        [{
            "name": "tavily_search",
            "content": "https://a.example/source https://b.example/source",
            "error": False,
        }],
        _strict_fail_open_config(),
    )

    assert result.decision == "continue"
    assert result.evaluator_error == "judge timed out"
    assert result.quality_rigor == "strict"


@pytest.mark.asyncio
async def test_strict_handoff_gate_preserves_fail_open_on_judge_timeout(
    monkeypatch,
) -> None:
    async def fake_invoke(*_args, **_kwargs):
        raise TimeoutError("judge timed out")

    monkeypatch.setattr(
        "open_deep_research.quality.gate._build_quality_model",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        "open_deep_research.quality.gate.invoke_model_with_retry_observability",
        fake_invoke,
    )
    handoff = {
        "compressed_research": (
            "Detailed evidence from https://a.example/source and "
            "https://b.example/source. "
        )
        * 6,
        "raw_notes": [],
    }

    result = await evaluate_subagent_handoff(
        "Synthetic topic",
        handoff,
        _strict_fail_open_config(),
    )

    assert result.accepted is True
    assert result.evaluator_error == "judge timed out"
    assert result.quality_rigor == "strict"


def test_quality_rigor_defaults_to_balanced(monkeypatch) -> None:
    monkeypatch.delenv("QUALITY_EVALUATION_RIGOR", raising=False)
    monkeypatch.delenv("QUALITY_EVALUATION_MIN_SCORE", raising=False)

    configurable = Configuration.from_runnable_config({})

    assert configurable.quality_evaluation_rigor is QualityEvaluationRigor.BALANCED


def test_legacy_min_score_maps_to_rigor_when_new_option_is_absent(
    monkeypatch,
) -> None:
    monkeypatch.delenv("QUALITY_EVALUATION_RIGOR", raising=False)
    monkeypatch.setenv("QUALITY_EVALUATION_MIN_SCORE", "4")

    frozen = freeze_run_config({})

    assert frozen["configurable"]["quality_evaluation_rigor"] == "strict"
    assert (
        frozen["metadata"]["quality_configuration_warnings"][0]["code"]
        == "legacy_quality_min_score_mapped"
    )
    assert "quality_evaluation_min_score" not in frozen["configurable"]


def test_new_rigor_wins_over_legacy_min_score(monkeypatch) -> None:
    monkeypatch.setenv("QUALITY_EVALUATION_RIGOR", "relaxed")
    monkeypatch.setenv("QUALITY_EVALUATION_MIN_SCORE", "5")

    frozen = freeze_run_config({})

    assert frozen["configurable"]["quality_evaluation_rigor"] == "relaxed"
    assert (
        frozen["metadata"]["quality_configuration_warnings"][0]["code"]
        == "legacy_quality_min_score_ignored"
    )


def test_frozen_v1_quality_policy_is_not_silently_upgraded() -> None:
    configurable = Configuration().model_dump(mode="json")
    configurable.pop("quality_evaluation_rigor")
    configurable["quality_evaluation_min_score"] = 4
    legacy = {
        "configurable": configurable,
        "metadata": {
            "runtime_config_frozen": True,
            "run_config_schema_version": 1,
            "quality_policy_version": "quality-gate-v2",
            "quality_evaluation_epoch": "legacy-epoch",
        },
    }
    legacy["metadata"]["run_config_fingerprint"] = run_config_fingerprint(
        legacy
    )

    restored = freeze_run_config(legacy)

    assert restored["metadata"]["quality_policy_version"] == "quality-gate-v2"
    assert restored["configurable"]["quality_evaluation_min_score"] == 4
    assert "quality_evaluation_rigor" not in restored["configurable"]


@pytest.mark.asyncio
async def test_runtime_judge_applies_frozen_rigor_and_records_thresholds(
    monkeypatch,
) -> None:
    complete = {
        "decision": "complete",
        "relevance": 3,
        "source_quality": 3,
        "evidence_coverage": 3,
        "corroboration": 3,
        "unresolved_conflicts": [],
        "missing_information": [],
        "suggested_queries": [],
        "reason": "The evidence meets the supplied policy.",
    }
    repaired_retry = {
        **complete,
        "decision": "retry",
        "suggested_queries": ["Obtain stronger corroboration."],
        "reason": "The evidence does not meet the strict average threshold.",
    }
    responses = [complete, complete, repaired_retry]

    async def fake_invoke(*_args, **_kwargs):
        return SimpleNamespace(content=json.dumps(responses.pop(0)))

    monkeypatch.setattr(
        "open_deep_research.quality.gate._build_quality_model",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        "open_deep_research.quality.gate.invoke_model_with_retry_observability",
        fake_invoke,
    )
    tool_results = [
        {
            "name": "tavily_search",
            "content": "https://a.example/source https://b.example/source",
            "error": False,
        }
    ]

    balanced = await evaluate_tool_results(
        "Synthetic topic",
        tool_results,
        {
            "configurable": {
                "quality_evaluation_rigor": "balanced",
                "quality_evaluation_min_sources": 2,
            },
            "metadata": {
                "runtime_config_frozen": True,
                "quality_policy_version": "quality-gate-v3",
                "quality_evaluation_epoch": "epoch-balanced",
            },
        },
    )
    strict = await evaluate_tool_results(
        "Synthetic topic",
        tool_results,
        {
            "configurable": {
                "quality_evaluation_rigor": "strict",
                "quality_evaluation_min_sources": 2,
            },
            "metadata": {
                "runtime_config_frozen": True,
                "quality_policy_version": "quality-gate-v3",
                "quality_evaluation_epoch": "epoch-strict",
            },
        },
    )

    assert balanced.decision == "complete"
    assert balanced.quality_rigor == "balanced"
    assert balanced.quality_thresholds["runtime_average_floor"] == 3.0
    assert strict.decision == "retry"
    assert strict.protocol_repair_count == 1
    assert strict.quality_rigor == "strict"
    assert strict.quality_thresholds["runtime_average_floor"] == 4.0
