from __future__ import annotations

from open_deep_research.completion import (
    CompletionDecision,
    CompletionPolicyContext,
    ResearchCompletionPolicy,
    completion_policy_context,
)


def test_completion_policy_requires_evidence_for_explicit_completion():
    decision = ResearchCompletionPolicy().evaluate(CompletionPolicyContext(
        explicit_completion_succeeded=True,
        evidence_count=0,
        independent_source_count=0,
        has_remaining_budget=True,
    ))

    assert decision.action == CompletionDecision.CONTINUE_WITH_GAPS
    assert "accepted_evidence" in decision.gaps


def test_completion_policy_completes_with_successful_signal_and_evidence():
    decision = ResearchCompletionPolicy(min_evidence=1, min_sources=1).evaluate(
        CompletionPolicyContext(
            explicit_completion_succeeded=True,
            evidence_count=2,
            independent_source_count=2,
            has_remaining_budget=True,
        )
    )

    assert decision.action == CompletionDecision.COMPLETE
    assert decision.reason == "explicit_completion"


def test_completion_policy_preserves_success_on_last_available_turn():
    decision = ResearchCompletionPolicy(min_evidence=1, min_sources=1).evaluate(
        CompletionPolicyContext(
            explicit_completion_succeeded=True,
            evidence_count=2,
            independent_source_count=2,
            has_remaining_budget=False,
            exhausted_reason="max_turns",
        )
    )

    assert decision.action == CompletionDecision.COMPLETE, (
        "A successful completion signal with no research gaps must win over "
        "simultaneous turn-budget exhaustion."
    )
    assert decision.reason == "explicit_completion"


def test_completion_policy_never_accepts_failed_completion_signal():
    decision = ResearchCompletionPolicy().evaluate(CompletionPolicyContext(
        explicit_completion_succeeded=False,
        explicit_completion_failed=True,
        evidence_count=3,
        independent_source_count=3,
        has_remaining_budget=True,
    ))

    assert decision.action == CompletionDecision.CONTINUE_WITH_GAPS
    assert decision.reason == "completion_signal_failed"


def test_completion_policy_returns_partial_only_with_usable_evidence():
    policy = ResearchCompletionPolicy()

    partial = policy.evaluate(CompletionPolicyContext(
        evidence_count=1,
        independent_source_count=1,
        has_remaining_budget=False,
        exhausted_reason="max_turns",
    ))
    failed = policy.evaluate(CompletionPolicyContext(
        evidence_count=0,
        independent_source_count=0,
        has_remaining_budget=False,
        exhausted_reason="budget_exhausted",
    ))

    assert partial.action == CompletionDecision.COMPLETE_PARTIAL
    assert failed.action == CompletionDecision.TERMINATE


def test_completion_policy_rejects_completion_with_active_tasks_or_conflicts():
    decision = ResearchCompletionPolicy().evaluate(CompletionPolicyContext(
        explicit_completion_succeeded=True,
        evidence_count=2,
        independent_source_count=2,
        active_task_count=1,
        unresolved_conflicts=("claims disagree",),
        has_remaining_budget=True,
    ))

    assert decision.action == CompletionDecision.CONTINUE_WITH_GAPS
    assert "active_tasks" in decision.gaps
    assert "unresolved_conflicts" in decision.gaps


def test_completion_policy_does_not_count_quarantined_evidence():
    context = completion_policy_context(
        {
            "evidence_registry": [
                {
                    "evidence_id": "unsafe-1",
                    "source_url": "https://unsafe.example/source",
                    "security_status": "quarantined",
                }
            ]
        },
        explicit_completion_succeeded=True,
    )

    decision = ResearchCompletionPolicy().evaluate(context)

    assert context.evidence_count == 0
    assert context.independent_source_count == 0
    assert decision.action == CompletionDecision.CONTINUE_WITH_GAPS
    assert "accepted_evidence" in decision.gaps


def test_completion_context_ignores_non_factual_requirements_for_gaps():
    # Deliverable-format and process requirements are owned by the final
    # report / orchestration and never enter the research coverage ledger;
    # counting them would permanently force complete_partial outcomes.
    state = {
        "coverage_contract": {
            "requirements": [
                {
                    "requirement_id": "COV-01-aaa",
                    "text": "调查 Python 3.13 自由线程的生产可用性",
                    "kind": "factual",
                },
                {
                    "requirement_id": "COV-02-bbb",
                    "text": "调查 NumPy 2.1 的支持状态",
                    "kind": "factual",
                },
                {
                    "requirement_id": "COV-03-ccc",
                    "text": "风险矩阵",
                    "kind": "deliverable",
                },
                {
                    "requirement_id": "COV-04-ddd",
                    "text": "不需要澄清",
                    "kind": "process",
                },
            ]
        },
        "coverage_ledger": {
            "COV-01-aaa": {"status": "supported"},
            "COV-02-bbb": {"status": "supported"},
        },
        "evidence_registry": [
            {"evidence_id": "ev-1", "source_url": "https://example.org/1"}
        ],
    }

    context = completion_policy_context(state)
    assert context.uncovered_requirements == ()

    decision = ResearchCompletionPolicy().evaluate(
        CompletionPolicyContext(
            explicit_completion_succeeded=True,
            evidence_count=context.evidence_count,
            independent_source_count=context.independent_source_count,
            has_remaining_budget=False,
            exhausted_reason="max_turns",
        )
    )
    assert "coverage_gaps" not in decision.gaps


def test_completion_context_still_reports_uncovered_factual_requirements():
    state = {
        "coverage_contract": {
            "requirements": [
                {
                    "requirement_id": "COV-01-aaa",
                    "text": "调查 Python 3.13 自由线程的生产可用性",
                    "kind": "factual",
                },
                {
                    "requirement_id": "COV-02-bbb",
                    "text": "风险矩阵",
                    "kind": "deliverable",
                },
            ]
        },
        "coverage_ledger": {
            "COV-01-aaa": {"status": "supported"},
        },
    }

    context = completion_policy_context(state)
    assert context.uncovered_requirements == ()
    # A factual requirement missing from the ledger is still a real gap.
    state["coverage_ledger"] = {}
    context = completion_policy_context(state)
    assert list(context.uncovered_requirements) == ["COV-01-aaa"]
