"""Canonical quality-gate rigor policies shared by runtime and offline Judges."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any


class QualityEvaluationRigor(str, Enum):
    """User-selectable semantic approval rigor."""

    VERY_RELAXED = "very_relaxed"
    RELAXED = "relaxed"
    BALANCED = "balanced"
    STRICT = "strict"
    VERY_STRICT = "very_strict"


@dataclass(frozen=True, slots=True)
class QualityRigorPolicy:
    """Immutable thresholds for one quality-evaluation rigor."""

    rigor: QualityEvaluationRigor
    runtime_dimension_floor: int
    runtime_average_floor: float
    outer_aggregate_floor: float
    outer_critical_floor: float

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-native representation for manifests and assessments."""
        payload = asdict(self)
        payload["rigor"] = self.rigor.value
        return payload


_QUALITY_RIGOR_POLICIES = MappingProxyType(
    {
        QualityEvaluationRigor.VERY_RELAXED: QualityRigorPolicy(
            rigor=QualityEvaluationRigor.VERY_RELAXED,
            runtime_dimension_floor=2,
            runtime_average_floor=2.5,
            outer_aggregate_floor=0.60,
            outer_critical_floor=0.50,
        ),
        QualityEvaluationRigor.RELAXED: QualityRigorPolicy(
            rigor=QualityEvaluationRigor.RELAXED,
            runtime_dimension_floor=2,
            runtime_average_floor=3.0,
            outer_aggregate_floor=0.70,
            outer_critical_floor=0.60,
        ),
        QualityEvaluationRigor.BALANCED: QualityRigorPolicy(
            rigor=QualityEvaluationRigor.BALANCED,
            runtime_dimension_floor=3,
            runtime_average_floor=3.0,
            outer_aggregate_floor=0.75,
            outer_critical_floor=0.70,
        ),
        QualityEvaluationRigor.STRICT: QualityRigorPolicy(
            rigor=QualityEvaluationRigor.STRICT,
            runtime_dimension_floor=3,
            runtime_average_floor=4.0,
            outer_aggregate_floor=0.85,
            outer_critical_floor=0.80,
        ),
        QualityEvaluationRigor.VERY_STRICT: QualityRigorPolicy(
            rigor=QualityEvaluationRigor.VERY_STRICT,
            runtime_dimension_floor=4,
            runtime_average_floor=4.5,
            outer_aggregate_floor=0.90,
            outer_critical_floor=0.90,
        ),
    }
)

_LEGACY_MIN_SCORE_RIGOR = MappingProxyType(
    {
        1: QualityEvaluationRigor.VERY_RELAXED,
        2: QualityEvaluationRigor.RELAXED,
        3: QualityEvaluationRigor.BALANCED,
        4: QualityEvaluationRigor.STRICT,
        5: QualityEvaluationRigor.VERY_STRICT,
    }
)


def get_quality_rigor_policy(
    rigor: QualityEvaluationRigor | str,
) -> QualityRigorPolicy:
    """Resolve one rigor value to its canonical immutable policy."""
    resolved = (
        rigor
        if isinstance(rigor, QualityEvaluationRigor)
        else QualityEvaluationRigor(str(rigor).strip().lower())
    )
    return _QUALITY_RIGOR_POLICIES[resolved]


def get_run_quality_rigor_policy(
    rigor: QualityEvaluationRigor | str,
    *,
    policy_version: str,
    legacy_min_score: Any = None,
) -> QualityRigorPolicy:
    """Resolve a policy without silently changing frozen quality-gate-v2 runs."""
    if policy_version == "quality-gate-v2" and legacy_min_score is not None:
        mapped_rigor = rigor_from_legacy_min_score(legacy_min_score)
        score = int(legacy_min_score)
        return QualityRigorPolicy(
            rigor=mapped_rigor,
            runtime_dimension_floor=score,
            runtime_average_floor=float(score),
            outer_aggregate_floor=0.75,
            outer_critical_floor=0.70,
        )
    return get_quality_rigor_policy(rigor)


def rigor_from_legacy_min_score(value: Any) -> QualityEvaluationRigor:
    """Map the removed integer score option to its migration-equivalent rigor."""
    try:
        score = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "QUALITY_EVALUATION_MIN_SCORE must be an integer from 1 to 5"
        ) from exc
    try:
        return _LEGACY_MIN_SCORE_RIGOR[score]
    except KeyError as exc:
        raise ValueError(
            "QUALITY_EVALUATION_MIN_SCORE must be an integer from 1 to 5"
        ) from exc


def scores_meet_runtime_policy(
    scores: tuple[int, int, int, int],
    policy: QualityRigorPolicy,
) -> bool:
    """Return whether four semantic scores meet the selected rigor."""
    return (
        min(scores) >= policy.runtime_dimension_floor
        and sum(scores) / len(scores) >= policy.runtime_average_floor
    )
