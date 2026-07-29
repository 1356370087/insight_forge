"""Regressions for automatic artifact reassessment and evidence recovery."""

from __future__ import annotations

import pytest

from open_deep_research.agents.query_engine import QueryEngine
from open_deep_research.completion import CompletionDecision
from open_deep_research.quality import HandoffAssessment


def _engine(tmp_path, run_id: str) -> QueryEngine:
    return QueryEngine({
        "configurable": {
            "runs_dir": str(tmp_path),
            "event_log_enabled": False,
            "observability_enabled": False,
            "quality_evaluation_enabled": True,
            "quality_evaluation_model": "openai:qwen3.7-max",
            "quality_evaluation_fail_open": False,
            "quality_evaluation_min_score": 3,
            "quality_evaluation_min_sources": 2,
        },
        "metadata": {
            "run_id": run_id,
            "runtime_config_frozen": True,
            "quality_policy_version": "quality-gate-v2",
            "quality_evaluation_epoch": "epoch-recovery",
        },
    })


def _artifact(*, security_status: str = "accepted", excerpt: str = "Exact excerpt") -> dict:
    return {
        "research_topic": "Verify two official sources.",
        "compressed_research": "REJECTED_COMPRESSED_SECRET " * 20,
        "raw_notes": ["REJECTED_RAW_NOTE_SECRET"],
        "metrics": {"sources_read": 2},
        "evidence_registry": [
            {
                "evidence_id": "ev-a",
                "claim": "Claim A",
                "supporting_excerpt": excerpt,
                "source_url": "https://a.example/source",
                "source_title": "Official A",
                "locator": "section A",
                "security_status": security_status,
            },
            {
                "evidence_id": "ev-b",
                "claim": "Claim B",
                "supporting_excerpt": "Exact excerpt B",
                "source_url": "https://b.example/source",
                "source_title": "Official B",
                "locator": "section B",
                "security_status": security_status,
            },
        ],
    }


def _termination_state(ref: dict) -> dict:
    return {
        "messages": [{"role": "user", "content": "Verify claims A and B."}],
        "research_brief": "Verify claims A and B.",
        "research_artifact_refs": {"task-1": ref},
        "handoff_assessments": [
            {
                "tool_call_id": "task-1",
                "accepted": False,
                "reason": "Initial handoff rejected.",
            }
        ],
        "completion_decision": {
            "action": CompletionDecision.TERMINATE.value,
            "reason": "max_turns",
            "gaps": ["accepted_evidence"],
        },
        "evidence_registry": [],
        "notes": [],
        "raw_notes": [],
        "completed_task_outputs": [],
    }


@pytest.mark.asyncio
async def test_automatic_reassessment_admits_sha_verified_artifact(
    tmp_path,
    monkeypatch,
) -> None:
    engine = _engine(tmp_path, "reassess-accepted")
    artifact = _artifact()
    digest = engine.context_store.persist_task_result("task-1", artifact)
    state = _termination_state({
        "path": "context/artifacts/research_tasks/task-1.json",
        "sha256": digest,
    })

    async def accept(*_args, **_kwargs):
        return HandoffAssessment(
            accepted=True,
            relevance=5,
            source_quality=5,
            evidence_coverage=5,
            groundedness=5,
            reason="SHA-verified evidence is sufficient.",
        )

    monkeypatch.setattr(
        "open_deep_research.agents.query_engine.evaluate_subagent_handoff",
        accept,
    )

    outcome = await engine._recover_quality_gate_termination(state)

    assert outcome["mode"] == "accepted"
    assert state["completion_decision"]["action"] == "complete"
    assert state["quality_gate"]["status"] == "passed"
    assert len(state["evidence_registry"]) == 2
    assert "REJECTED_COMPRESSED_SECRET" in state["notes"][0]
    assert state["handoff_assessments"][-1]["trigger"] == (
        "automatic_termination_reassessment"
    )


@pytest.mark.asyncio
async def test_rejected_handoff_with_safe_evidence_produces_full_partial_report(
    tmp_path,
    monkeypatch,
) -> None:
    engine = _engine(tmp_path, "reassess-partial")
    full_excerpt = "BEGIN-EXCERPT-" + ("x" * 5000) + "-END-EXCERPT"
    digest = engine.context_store.persist_task_result(
        "task-1",
        _artifact(excerpt=full_excerpt),
    )
    state = _termination_state({
        "path": "context/artifacts/research_tasks/task-1.json",
        "sha256": digest,
    })

    async def reject(*_args, **_kwargs):
        return HandoffAssessment(
            accepted=False,
            relevance=4,
            source_quality=4,
            evidence_coverage=2,
            groundedness=4,
            missing_information=["One presentation requirement remains."],
            follow_up_tasks=["Re-evaluate the artifact."],
            reason="Handoff protocol was not accepted.",
        )

    monkeypatch.setattr(
        "open_deep_research.agents.query_engine.evaluate_subagent_handoff",
        reject,
    )

    outcome = await engine._recover_quality_gate_termination(state)

    assert outcome["mode"] == "partial"
    assert state["completion_decision"] == {
        "action": "complete_partial",
        "reason": "quality_gate_recovery",
        "gaps": ["accepted_evidence"],
    }
    assert state["quality_gate"]["status"] == "degraded"
    assert state["final_report"]
    assert full_excerpt in state["final_report"]
    assert "REJECTED_COMPRESSED_SECRET" not in state["final_report"]
    assert "REJECTED_RAW_NOTE_SECRET" not in state["final_report"]
    assert state["evaluation_snapshot"]["evidence_registry"]


@pytest.mark.asyncio
async def test_only_quarantined_evidence_remains_failed_with_artifact_ref(
    tmp_path,
    monkeypatch,
) -> None:
    engine = _engine(tmp_path, "reassess-quarantined")
    digest = engine.context_store.persist_task_result(
        "task-1",
        _artifact(security_status="quarantined"),
    )
    state = _termination_state({
        "path": "context/artifacts/research_tasks/task-1.json",
        "sha256": digest,
    })

    async def reject(*_args, **_kwargs):
        return HandoffAssessment(
            accepted=False,
            relevance=1,
            source_quality=1,
            evidence_coverage=1,
            groundedness=1,
            missing_information=["No eligible evidence."],
            reason="Evidence is quarantined.",
        )

    monkeypatch.setattr(
        "open_deep_research.agents.query_engine.evaluate_subagent_handoff",
        reject,
    )

    outcome = await engine._recover_quality_gate_termination(state)

    assert outcome["mode"] == "failed"
    assert outcome["artifact_refs"][0]["sha256"] == digest
    assert state["quality_gate"]["status"] == "failed"
    assert not state.get("final_report")


@pytest.mark.asyncio
async def test_corrupted_artifact_is_not_reported_as_recoverable(
    tmp_path,
    monkeypatch,
) -> None:
    engine = _engine(tmp_path, "reassess-corrupted")
    digest = engine.context_store.persist_task_result("task-valid", _artifact())
    state = _termination_state({
        "path": "context/artifacts/research_tasks/task-valid.json",
        "sha256": digest,
    })
    state["research_artifact_refs"] = {
        "task-valid": state["research_artifact_refs"].pop("task-1")
    }
    state["research_artifact_refs"]["task-corrupted"] = {
        "path": "context/artifacts/research_tasks/task-corrupted.json",
        "sha256": "0" * 64,
    }

    async def reject(*_args, **_kwargs):
        return HandoffAssessment(
            accepted=False,
            relevance=2,
            source_quality=2,
            evidence_coverage=2,
            groundedness=2,
            reason="Keep only SHA-verified safe evidence.",
        )

    monkeypatch.setattr(
        "open_deep_research.agents.query_engine.evaluate_subagent_handoff",
        reject,
    )

    outcome = await engine._recover_quality_gate_termination(state)

    assert outcome["mode"] == "partial"
    assert [ref["task_id"] for ref in outcome["artifact_refs"]] == [
        "task-valid"
    ]
    assert [ref["task_id"] for ref in state["quality_gate"]["assessment_refs"]] == [
        "task-valid"
    ]
    assert "artifact_integrity_failed" in state["quality_gate"]["reason_codes"]


@pytest.mark.asyncio
async def test_legacy_migration_reassesses_sha_artifacts_in_new_epoch(
    tmp_path,
    monkeypatch,
) -> None:
    engine = _engine(tmp_path, "legacy-reassessment")
    digest = engine.context_store.persist_task_result("task-1", _artifact())
    state = {
        **_termination_state(
            {
                "path": "context/artifacts/research_tasks/task-1.json",
                "sha256": digest,
            }
        ),
        "evidence_registry": [{"evidence_id": "legacy-untrusted"}],
        "notes": ["legacy note"],
        "evaluation_snapshot": {"schema_version": "legacy"},
    }
    calls: list[str] = []

    async def accept(topic, _artifact_payload, _config):
        calls.append(topic)
        return HandoffAssessment(
            accepted=True,
            relevance=5,
            source_quality=5,
            evidence_coverage=5,
            groundedness=5,
            reason="Accepted in the migrated epoch.",
        )

    monkeypatch.setattr(
        "open_deep_research.agents.query_engine.evaluate_subagent_handoff",
        accept,
    )

    await engine._migrate_legacy_quality_artifacts(state)

    assert calls == ["Verify two official sources."]
    assert state["evaluation_snapshot"] == {}
    assert state["completion_decision"] == {}
    assert state["handoff_assessments"][0]["trigger"] == (
        "legacy_migration_reassessment"
    )
    assert {item["evidence_id"] for item in state["evidence_registry"]} == {
        "ev-a",
        "ev-b",
    }
    assert "legacy note" not in state["notes"]
