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
