"""Regressions for automatic artifact reassessment and evidence recovery."""

from __future__ import annotations

import re

import pytest

from open_deep_research.agents.query_engine import QueryEngine
from open_deep_research.completion import CompletionDecision
from open_deep_research.quality import HandoffAssessment
from open_deep_research.report import recovery
from open_deep_research.report.recovery import build_evidence_recovery_report


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
    artifact = {**_artifact(), "requirement_ids": ["req-001"]}
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
    assert state["coverage_ledger"] == {
        "req-001": {
            "status": "partial",
            "evidence_ids": [],
            "task_ids": ["task-1"],
            "caveats": ["coverage_mapping_missing"],
        }
    }


@pytest.mark.asyncio
async def test_persisted_accepted_assessment_backfills_owned_coverage_ledger(
    tmp_path,
    monkeypatch,
) -> None:
    engine = _engine(tmp_path, "reassess-persisted-accepted")
    artifact = {**_artifact(), "requirement_ids": ["req-001"]}
    digest = engine.context_store.persist_task_result("task-1", artifact)
    accepted = HandoffAssessment(
        accepted=True,
        relevance=5,
        source_quality=5,
        evidence_coverage=5,
        groundedness=5,
        reason="Persisted assessment accepted the artifact.",
    )
    state = _termination_state({
        "path": "context/artifacts/research_tasks/task-1.json",
        "sha256": digest,
    })
    state["handoff_assessments"] = [
        {
            "tool_call_id": "task-1",
            "trigger": "artifact_read_reassessment",
            "artifact_sha256": digest,
            **accepted.model_dump(mode="json"),
        }
    ]

    async def unexpected_reassessment(*_args, **_kwargs):
        raise AssertionError("persisted accepted assessments must use the fast path")

    monkeypatch.setattr(
        "open_deep_research.agents.query_engine.evaluate_subagent_handoff",
        unexpected_reassessment,
    )

    outcome = await engine._recover_quality_gate_termination(state)

    assert outcome["mode"] == "accepted"
    assert state["coverage_ledger"] == {
        "req-001": {
            "status": "partial",
            "evidence_ids": [],
            "task_ids": ["task-1"],
            "caveats": ["coverage_mapping_missing"],
        }
    }


@pytest.mark.asyncio
async def test_admitted_artifact_with_uncovered_user_requirement_stays_partial(
    tmp_path,
    monkeypatch,
) -> None:
    engine = _engine(tmp_path, "reassess-contract-partial")
    contract = {
        "schema_version": 1,
        "original_query_sha256": "a" * 64,
        "requirements": [
            {
                "requirement_id": "req-001",
                "text": "Verify claim A.",
                "source_message_index": 0,
                "source_start": 0,
                "source_end": 15,
            },
            {
                "requirement_id": "req-002",
                "text": "Verify claim B.",
                "source_message_index": 0,
                "source_start": 16,
                "source_end": 31,
            },
        ],
        "advisory_dimensions": [],
    }
    artifact = {
        **_artifact(),
        "schema_version": 2,
        "coverage_contract": contract,
        "requirement_ids": ["req-001"],
    }
    digest = engine.context_store.persist_task_result("task-1", artifact)
    state = {
        **_termination_state({
            "path": "context/artifacts/research_tasks/task-1.json",
            "sha256": digest,
        }),
        "coverage_contract": contract,
        "coverage_ledger": {},
    }

    async def accept(*_args, **_kwargs):
        return HandoffAssessment(
            accepted=True,
            relevance=5,
            source_quality=5,
            evidence_coverage=5,
            groundedness=5,
            requirement_coverage=[
                {
                    "requirement_id": "req-001",
                    "status": "supported",
                    "evidence_ids": ["ev-a"],
                    "explanation": "Claim A is supported.",
                }
            ],
            reason="Owned requirement is supported.",
        )

    monkeypatch.setattr(
        "open_deep_research.agents.query_engine.evaluate_subagent_handoff",
        accept,
    )

    outcome = await engine._recover_quality_gate_termination(state)

    assert outcome["mode"] == "accepted"
    assert outcome["uncovered_requirement_ids"] == ["req-002"]
    assert state["completion_decision"] == {
        "action": "complete_partial",
        "reason": "coverage_requirements_incomplete",
        "gaps": ["req-002"],
    }


@pytest.mark.asyncio
async def test_rejected_v4_handoff_passes_safe_owned_mapping_to_partial_writer(
    tmp_path,
    monkeypatch,
) -> None:
    engine = _engine(tmp_path, "rejected-safe-coverage-map")
    contract = {
        "schema_version": 1,
        "original_query_sha256": "b" * 64,
        "requirements": [
            {
                "requirement_id": "req-001",
                "text": "Verify claim A.",
                "source_message_index": 0,
                "source_start": 0,
                "source_end": 15,
            },
            {
                "requirement_id": "req-002",
                "text": "Verify claim B.",
                "source_message_index": 0,
                "source_start": 16,
                "source_end": 31,
            },
        ],
        "advisory_dimensions": [],
    }
    artifact = {
        **_artifact(),
        "schema_version": 2,
        "coverage_contract": contract,
        "requirement_ids": ["req-001"],
    }
    digest = engine.context_store.persist_task_result("task-1", artifact)
    state = {
        **_termination_state({
            "path": "context/artifacts/research_tasks/task-1.json",
            "sha256": digest,
        }),
        "coverage_contract": contract,
        "coverage_ledger": {},
    }

    async def reject(*_args, **_kwargs):
        return HandoffAssessment(
            accepted=False,
            admission_status="rejected",
            relevance=4,
            source_quality=5,
            evidence_coverage=3,
            groundedness=4,
            requirement_coverage=[
                {
                    "requirement_id": "req-001",
                    "status": "supported",
                    "evidence_ids": ["ev-a", "unknown-evidence"],
                    "explanation": "The owned requirement has eligible support.",
                },
                {
                    "requirement_id": "req-002",
                    "status": "supported",
                    "evidence_ids": ["ev-b"],
                    "explanation": "This requirement is not owned by the task.",
                },
            ],
            unsupported_claims=["An unrelated claim is unsupported."],
            reason="Reject the full handoff without discarding safe evidence.",
        )

    captured: dict = {}

    async def capture_writer(_records, **kwargs):
        captured.update(kwargs)
        return "# Evidence-limited partial report"

    monkeypatch.setattr(
        "open_deep_research.agents.query_engine.evaluate_subagent_handoff",
        reject,
    )
    monkeypatch.setattr(
        "open_deep_research.agents.query_engine.build_evidence_limited_report",
        capture_writer,
    )

    outcome = await engine._recover_quality_gate_termination(state)

    assert outcome["mode"] == "partial"
    assert state["coverage_ledger"] == {}
    assert captured["coverage_ledger"] == {
        "req-001": {
            "status": "supported",
            "evidence_ids": ["ev-a"],
            "task_ids": ["task-1"],
            "caveats": [],
        }
    }
    assert captured["uncovered_requirement_ids"] == ["req-002"]


@pytest.mark.asyncio
async def test_partial_coverage_map_excludes_out_of_scope_evidence(
    tmp_path,
    monkeypatch,
) -> None:
    engine = _engine(tmp_path, "rejected-source-scoped-coverage-map")
    contract = {
        "schema_version": 1,
        "original_query_sha256": "c" * 64,
        "requirements": [
            {
                "requirement_id": "req-official",
                "text": (
                    "Based solely on the LangGraph official documentation, "
                    "verify checkpoint persistence."
                ),
                "source_message_index": 0,
                "source_start": 0,
                "source_end": 100,
            }
        ],
        "advisory_dimensions": [],
    }
    artifact = {
        "schema_version": 2,
        "research_topic": "Verify checkpoint persistence.",
        "compressed_research": "REJECTED_COMPRESSED_SECRET " * 20,
        "raw_notes": ["REJECTED_RAW_NOTE_SECRET"],
        "coverage_contract": contract,
        "requirement_ids": ["req-official"],
        "metrics": {"sources_read": 2},
        "evidence_registry": [
            {
                "evidence_id": "ev-docs",
                "claim": "Official checkpoint claim.",
                "supporting_excerpt": "Checkpoint state is persisted.",
                "source_url": (
                    "https://docs.langchain.com/oss/python/langgraph/"
                    "persistence"
                ),
                "source_title": "LangGraph docs",
                "security_status": "accepted",
            },
            {
                "evidence_id": "ev-forum",
                "claim": "Community checkpoint claim.",
                "supporting_excerpt": "A forum post describes persistence.",
                "source_url": "https://forum.langchain.com/t/checkpoints/123",
                "source_title": "Community forum",
                "security_status": "accepted",
            },
        ],
    }
    digest = engine.context_store.persist_task_result("task-1", artifact)
    state = {
        **_termination_state({
            "path": "context/artifacts/research_tasks/task-1.json",
            "sha256": digest,
        }),
        "coverage_contract": contract,
        "coverage_ledger": {},
    }

    async def reject(*_args, **_kwargs):
        return HandoffAssessment(
            accepted=False,
            admission_status="rejected",
            relevance=4,
            source_quality=4,
            evidence_coverage=3,
            groundedness=4,
            requirement_coverage=[
                {
                    "requirement_id": "req-official",
                    "status": "supported",
                    "evidence_ids": ["ev-forum"],
                    "explanation": (
                        "Only the out-of-scope forum evidence was mapped."
                    ),
                }
            ],
            reason="The full handoff is rejected.",
        )

    captured: dict = {}

    async def capture_writer(_records, **kwargs):
        captured.update(kwargs)
        return "# Evidence-limited partial report"

    monkeypatch.setattr(
        "open_deep_research.agents.query_engine.evaluate_subagent_handoff",
        reject,
    )
    monkeypatch.setattr(
        "open_deep_research.agents.query_engine.build_evidence_limited_report",
        capture_writer,
    )

    outcome = await engine._recover_quality_gate_termination(state)

    assert outcome["mode"] == "partial"
    assert state["coverage_ledger"] == {}
    assert captured["coverage_ledger"] == {}
    assert captured["uncovered_requirement_ids"] == ["req-official"]


@pytest.mark.asyncio
async def test_rejected_handoff_with_safe_evidence_produces_bounded_partial_report(
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
    assert "BEGIN-EXCERPT-" in state["final_report"]
    assert "-END-EXCERPT" not in state["final_report"]
    assert "REJECTED_COMPRESSED_SECRET" not in state["final_report"]
    assert "REJECTED_RAW_NOTE_SECRET" not in state["final_report"]
    assert state["evaluation_snapshot"]["evidence_registry"]


def test_evidence_recovery_report_has_a_hard_output_bound() -> None:
    report = build_evidence_recovery_report(
        [
            {
                "evidence_id": f"ev-{index}",
                "claim": f"Claim {index} " + ("c" * 2000),
                "supporting_excerpt": "x" * 5000,
                "source_url": f"https://example.test/{index}",
                "source_title": f"Source {index}",
                "locator": "section",
            }
            for index in range(100)
        ],
        gaps=["accepted_evidence"],
        rejection_reasons=["coverage"] * 100,
        artifact_refs=[],
    )

    assert len(report) < 100_000
    assert "其余 60 条证据保留在校验后的研究工件中" in report


def test_evidence_recovery_report_makes_embedded_urls_inert() -> None:
    allowed_url = "https://example.test/allowed"
    blocked_urls = {
        "https://evil.test/claim",
        "https://evil.test/excerpt",
        "https://evil.test/locator",
        "https://evil.test/rejection",
    }
    report = build_evidence_recovery_report(
        [
            {
                "evidence_id": "ev-safe",
                "claim": "Claim [details](https://evil.test/claim).",
                "supporting_excerpt": (
                    "Excerpt https://evil.test/excerpt and "
                    "[relative docs](/not-an-evidence-source)."
                ),
                "source_url": allowed_url,
                "source_title": "Allowed source",
                "locator": "See <https://evil.test/locator>.",
            }
        ],
        gaps=[],
        rejection_reasons=[
            "Rejected; inspect [reason](https://evil.test/rejection)."
        ],
        artifact_refs=[],
    )

    assert all(url not in report for url in blocked_urls)
    assert "](/not-an-evidence-source)" not in report
    assert set(re.findall(r"https?://[^\s)\]}>]+", report)) == {allowed_url}


def test_evidence_recovery_report_rechecks_final_url_allowlist(
    monkeypatch,
) -> None:
    original_sanitize = recovery.sanitize_report_markdown

    def inject_url(markdown: str) -> str:
        return original_sanitize(markdown) + "\nhttps://evil.test/injected"

    monkeypatch.setattr(recovery, "sanitize_report_markdown", inject_url)

    with pytest.raises(
        ValueError,
        match="evidence_recovery_rendered_url_not_allowlisted",
    ):
        build_evidence_recovery_report(
            [
                {
                    "evidence_id": "ev-safe",
                    "claim": "Safe claim.",
                    "source_url": "https://example.test/allowed",
                    "source_title": "Allowed source",
                }
            ],
            gaps=[],
            rejection_reasons=[],
            artifact_refs=[],
        )


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
    artifact = {**_artifact(), "requirement_ids": ["req-001"]}
    digest = engine.context_store.persist_task_result("task-1", artifact)
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

    async def accept(topic, _artifact_payload, _config, **_kwargs):
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
    assert state["coverage_ledger"] == {
        "req-001": {
            "status": "partial",
            "evidence_ids": [],
            "task_ids": ["task-1"],
            "caveats": ["coverage_mapping_missing"],
        }
    }
    assert "legacy note" not in state["notes"]
