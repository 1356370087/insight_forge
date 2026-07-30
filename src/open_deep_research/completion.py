"""Deterministic evidence-aware completion decisions for research loops."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from open_deep_research.evidence import eligible_evidence_records


def accepted_evidence(state: dict[str, Any]) -> list[dict[str, Any]]:
    """Return only structured evidence admitted to the current research state."""
    accepted = eligible_evidence_records(state.get("evidence_registry", []))
    for output in state.get("completed_task_outputs", []):
        if not isinstance(output, dict):
            continue
        if output.get("admission_status") == "rejected":
            continue
        accepted.extend(
            eligible_evidence_records(output.get("evidence_registry", []))
        )
    return accepted


def completion_policy_context(
    state: dict[str, Any],
    *,
    explicit_completion_succeeded: bool = False,
    explicit_completion_failed: bool = False,
    active_task_count: int = 0,
    has_remaining_budget: bool = True,
    exhausted_reason: str | None = None,
    cancelled: bool = False,
) -> CompletionPolicyContext:
    """Project runtime state into the deterministic completion-policy contract."""
    evidence = accepted_evidence(state)
    coverage_payload = state.get("coverage_contract", {})
    raw_requirements = (
        coverage_payload.get("requirements", [])
        if isinstance(coverage_payload, dict)
        else []
    )
    owned_requirement_ids = [
        str(item)
        for item in state.get("requirement_ids", [])
        if str(item)
    ]
    contract_requirement_ids = [
        str(item.get("requirement_id"))
        for item in raw_requirements
        if isinstance(item, dict) and item.get("requirement_id")
    ]
    required_ids = (
        owned_requirement_ids
        if owned_requirement_ids
        else contract_requirement_ids
    )
    coverage_ledger = state.get("coverage_ledger", {})
    uncovered_from_contract = (
        [
            requirement_id
            for requirement_id in required_ids
            if not isinstance(coverage_ledger, dict)
            or not isinstance(coverage_ledger.get(requirement_id), dict)
            or coverage_ledger[requirement_id].get("status")
            != "supported"
        ]
        if "coverage_ledger" in state
        else []
    )
    source_urls = {
        str(item.get("source_url", ""))
        for item in evidence
        if item.get("source_url")
    }
    return CompletionPolicyContext(
        explicit_completion_succeeded=explicit_completion_succeeded,
        explicit_completion_failed=explicit_completion_failed,
        evidence_count=len(evidence),
        independent_source_count=len(source_urls),
        active_task_count=active_task_count,
        unresolved_conflicts=tuple(
            state.get("result_assessment", {}).get("unresolved_conflicts", [])
        ),
        uncovered_requirements=tuple(
            dict.fromkeys(
                [
                    *uncovered_from_contract,
                    *state.get("result_assessment", {}).get(
                        "uncovered_requirements",
                        [],
                    ),
                ]
            )
        ),
        has_remaining_budget=has_remaining_budget,
        exhausted_reason=exhausted_reason,
        cancelled=cancelled,
    )


class CompletionDecision(str, Enum):
    """Domain actions returned by the completion policy."""

    CONTINUE_WITH_GAPS = "continue_with_gaps"
    COMPLETE = "complete"
    COMPLETE_PARTIAL = "complete_partial"
    TERMINATE = "terminate"


@dataclass(frozen=True, slots=True)
class CompletionPolicyContext:
    """Typed facts consumed by the deterministic completion policy."""

    explicit_completion_succeeded: bool = False
    explicit_completion_failed: bool = False
    evidence_count: int = 0
    independent_source_count: int = 0
    active_task_count: int = 0
    unresolved_conflicts: tuple[str, ...] = ()
    uncovered_requirements: tuple[str, ...] = ()
    has_remaining_budget: bool = True
    exhausted_reason: str | None = None
    cancelled: bool = False


@dataclass(frozen=True, slots=True)
class CompletionPolicyResult:
    """Immutable action, reason, and deterministic research gaps."""

    action: CompletionDecision
    reason: str
    gaps: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ResearchCompletionPolicy:
    """Require accepted evidence before a research loop may succeed."""

    min_evidence: int = 1
    min_sources: int = 1

    def evaluate(self, context: CompletionPolicyContext) -> CompletionPolicyResult:
        """Map observed research facts to one deterministic action."""
        if context.cancelled:
            return CompletionPolicyResult(
                CompletionDecision.TERMINATE,
                "cancelled",
            )

        gaps: list[str] = []
        if context.evidence_count < self.min_evidence:
            gaps.append("accepted_evidence")
        if context.independent_source_count < self.min_sources:
            gaps.append("independent_sources")
        if context.active_task_count:
            gaps.append("active_tasks")
        if context.unresolved_conflicts:
            gaps.append("unresolved_conflicts")
        if context.uncovered_requirements:
            gaps.append("coverage_gaps")

        has_usable_evidence = context.evidence_count > 0
        if context.explicit_completion_succeeded and not gaps:
            # A terminal signal may legitimately arrive on the final available
            # turn. Treat that completed work as success before projecting the
            # simultaneously exhausted turn budget into a partial result.
            return CompletionPolicyResult(
                CompletionDecision.COMPLETE,
                "explicit_completion",
            )
        if not context.has_remaining_budget:
            return CompletionPolicyResult(
                CompletionDecision.COMPLETE_PARTIAL
                if has_usable_evidence
                else CompletionDecision.TERMINATE,
                context.exhausted_reason or "budget_exhausted",
                tuple(gaps),
            )

        if context.explicit_completion_failed:
            return CompletionPolicyResult(
                CompletionDecision.CONTINUE_WITH_GAPS,
                "completion_signal_failed",
                tuple(gaps),
            )
        if context.explicit_completion_succeeded:
            return CompletionPolicyResult(
                CompletionDecision.CONTINUE_WITH_GAPS,
                "completion_requirements_unmet",
                tuple(gaps),
            )
        if not gaps:
            return CompletionPolicyResult(
                CompletionDecision.CONTINUE_WITH_GAPS,
                "explicit_completion_required",
                ("explicit_completion",),
            )
        return CompletionPolicyResult(
            CompletionDecision.CONTINUE_WITH_GAPS,
            "research_gaps_remain",
            tuple(gaps),
        )
