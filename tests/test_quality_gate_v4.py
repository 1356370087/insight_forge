"""Regression tests for coverage-bound quality-gate v4 semantics."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from langchain_core.messages import HumanMessage

from open_deep_research.agents import deep_researcher
from open_deep_research.quality.contract import (
    AdmissionStatus,
    CoverageStatus,
    HandoffPolicyInput,
    RequirementCoverage,
    build_research_coverage_contract,
    classify_research_risk,
    merge_coverage_ledger,
    resolve_handoff_admission,
)
from open_deep_research.quality.gate import (
    HANDOFF_EVALUATION_PROMPT_V4,
    TOOL_RESULT_EVALUATION_PROMPT,
    HandoffAssessment,
    ToolResultAssessment,
    _bounded_quality_payload,
    deterministic_handoff_checks,
    deterministic_tool_checks,
    evaluate_subagent_handoff,
    evaluate_tool_results,
)
from open_deep_research.state import ResearchQuestion
from open_deep_research.tools.base import ToolContext


def test_contract_splits_final_chinese_conjunction_in_explicit_list() -> None:
    contract = build_research_coverage_contract([
        HumanMessage(content=(
            "验证并比较三项能力：异步 I/O、"
            "B-tree skip scan（跳跃扫描）和虚拟生成列。"
        ))
    ])
    texts = [requirement.text for requirement in contract.requirements]

    assert any("B-tree skip scan" in text for text in texts)
    assert "虚拟生成列" in texts
    assert not any(
        "B-tree skip scan" in text and "虚拟生成列" in text
        for text in texts
    )


def test_contract_fallback_never_copies_an_unbounded_full_message() -> None:
    original = "请比较成<b>本</b>、安全性。" + ("补充背景信息" * 300)

    contract = build_research_coverage_contract([
        HumanMessage(content=original)
    ])

    assert contract.requirements
    assert all(len(item.text) <= 500 for item in contract.requirements)
    assert all(item.text != original for item in contract.requirements)
    unlocated = next(item for item in contract.requirements if "成 本" in item.text)
    assert unlocated.source_start == unlocated.source_end == 0
    assert unlocated.source_located is False


def test_quality_prompts_explain_all_payload_truncation_markers() -> None:
    assert "input_truncated" in TOOL_RESULT_EVALUATION_PROMPT
    assert "input_truncated" in HANDOFF_EVALUATION_PROMPT_V4
    assert "compressed_research_truncated" in HANDOFF_EVALUATION_PROMPT_V4
    assert "raw_notes_truncated" in HANDOFF_EVALUATION_PROMPT_V4


def test_payload_budget_preserves_source_urls() -> None:
    source_url = "https://example.test/" + ("source/" * 110)
    payload = {
        "research_topic": "topic " + ("t" * 700),
        "evidence_registry": [{"source_url": source_url}],
    }

    bounded = _bounded_quality_payload(payload, max_chars=1_150)

    assert bounded["evidence_registry"][0]["source_url"] == source_url
    assert bounded["input_truncated"] is True


def test_payload_budget_fails_closed_instead_of_truncating_reason_codes() -> None:
    payload = {
        "deterministic_checks": {
            "failures": ["insufficient_traceable_sources"]
        }
    }
    encoded_chars = len(json.dumps({**payload, "input_truncated": False}))

    with pytest.raises(ValueError, match="quality_payload_budget_too_small"):
        _bounded_quality_payload(payload, max_chars=encoded_chars - 1)


def test_supervisor_contract_explains_aggregate_requirement_ownership() -> None:
    contract = build_research_coverage_contract([
        HumanMessage(content=(
            "分别比较能力 A、能力 B；至少提供 6 个官方链接。"
        ))
    ])

    rendered = deep_researcher._render_supervisor_coverage_contract(contract)

    assert "exactly one primary owner" in rendered
    assert "Aggregate final-output requirements" in rendered
    assert "must not be assigned to every parallel task" in rendered


def test_unique_coverage_ordinal_repairs_only_hash_suffix() -> None:
    contract = build_research_coverage_contract([
        HumanMessage(content="比较能力 A、能力 B，并仅使用官方来源。")
    ])
    allowed = list(contract.requirement_ids())
    target = allowed[0]
    ordinal, _separator, suffix = target.rpartition("-")
    typo = f"{ordinal}-{'0' if suffix[-1] != '0' else '1'}{suffix[:-1]}"

    normalized = deep_researcher._canonicalize_coverage_requirement_ids(
        [typo, "COV-99-deadbeef"],
        contract,
    )

    assert normalized[0] == target
    assert normalized[1] == "COV-99-deadbeef"


def test_supervisor_fills_missing_requirement_ownership_without_duplication() -> None:
    contract = build_research_coverage_contract([
        HumanMessage(content="分别研究能力 A、能力 B、能力 C。")
    ])
    first, *remaining = contract.requirement_ids()
    tool_calls = [
        {
            "id": "task-explicit",
            "name": "ConductResearch",
            "args": {
                "research_topic": "能力 A",
                "requirement_ids": [first],
            },
        },
        {
            "id": "task-fallback",
            "name": "ConductResearch",
            "args": {"research_topic": "其余能力"},
        },
    ]

    normalized = deep_researcher._canonicalize_supervisor_tool_call_requirements(
        tool_calls,
        contract,
    )

    assert normalized[0]["args"]["requirement_ids"] == [first]
    assert normalized[1]["args"]["requirement_ids"] == remaining


def test_accepted_empty_coverage_records_partial_ledger_entry() -> None:
    assessment = HandoffAssessment(
        accepted=True,
        admission_status=AdmissionStatus.ACCEPTED,
        relevance=5,
        source_quality=5,
        evidence_coverage=5,
        groundedness=5,
        reason="Legacy-compatible accepted handoff.",
    )

    ledger = merge_coverage_ledger(
        {},
        task_id="task-1",
        assessment=assessment,
        owned_requirement_ids=("COV-01",),
    )

    assert ledger == {
        "COV-01": {
            "status": "partial",
            "evidence_ids": [],
            "task_ids": ["task-1"],
            "caveats": ["coverage_mapping_missing"],
        }
    }


@pytest.mark.asyncio
async def test_official_only_tool_gate_projects_out_third_party_candidates(
    monkeypatch,
) -> None:
    contract = build_research_coverage_contract([
        HumanMessage(content=(
            "请仅依据 PostgreSQL 官方文档说明 skip scan，"
            "不得引用第三方来源。"
        ))
    ])
    captured: dict = {}

    async def capture_evaluation(
        _schema,
        _system_prompt,
        payload,
        _config,
        **_kwargs,
    ):
        captured.update(payload)
        return ToolResultAssessment(
            decision="complete",
            relevance=5,
            source_quality=5,
            evidence_coverage=5,
            corroboration=5,
            reason="Official evidence is complete.",
        )

    async def ignore_activity(*_args, **_kwargs):
        return None

    monkeypatch.setattr(
        "open_deep_research.quality.gate._evaluate_json",
        capture_evaluation,
    )
    monkeypatch.setattr(
        "open_deep_research.quality.gate.publish_task_activity",
        ignore_activity,
    )
    records = [
        {
            "evidence_id": "EV-OFFICIAL",
            "claim": "PostgreSQL 18 supports skip scan.",
            "supporting_excerpt": "Support for skip scan lookups.",
            "source_url": "https://www.postgresql.org/docs/release/18.0/",
            "security_status": "accepted",
        },
        {
            "evidence_id": "EV-BLOG",
            "claim": "A third-party explanation.",
            "supporting_excerpt": "Blog text.",
            "source_url": "https://example.com/postgresql-skip-scan",
            "security_status": "accepted",
        },
    ]
    tool_results = [{
        "name": "web_research",
        "content": json.dumps({"evidence": records}),
        "error": False,
    }]

    result = await evaluate_tool_results(
        "Advisory PostgreSQL task.",
        tool_results,
        {
            "configurable": {
                "quality_evaluation_min_sources": 1,
                "quality_evaluation_fail_open": False,
            },
            "metadata": {
                "quality_policy_version": "quality-gate-v4",
                "runtime_config_frozen": True,
                "run_id": "postgresql-source-scope",
            },
        },
        evidence_registry=records,
        coverage_contract=contract,
        requirement_ids=list(contract.requirement_ids()),
    )

    assert result.decision == "complete"
    assert captured["source_scope_enforced"] is True
    assert captured["deterministic_checks"]["source_count"] == 1
    assert [
        item["evidence_id"] for item in captured["cumulative_evidence"]
    ] == ["EV-OFFICIAL"]
    assert "example.com" not in json.dumps(captured["tool_results"])


def test_run3_artifact_replay_binds_acceptance_to_original_query() -> None:
    fixture_path = (
        Path(__file__).parent
        / "fixtures"
        / "quality_gate_v4_run3_replay.json"
    )
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    contract = build_research_coverage_contract(
        [HumanMessage(content=fixture["original_query"])],
        advisory_dimensions=[fixture["supervisor_research_topic"]],
    )
    requirement_ids = contract.requirement_ids()
    assessment = fixture["v3_result_assessment"]

    assert all(
        "specific latest LangGraph version" not in requirement.text
        for requirement in contract.requirements
    )
    assert all(
        "langchain-ai.github.io/langgraph" not in requirement.text
        for requirement in contract.requirements
    )
    english_time_requirement = next(
        requirement
        for requirement in contract.requirements
        if requirement.text.lower() == "as of july 2026"
    )
    assert (
        fixture["original_query"][
            english_time_requirement.source_start
            : english_time_requirement.source_end
        ]
        == english_time_requirement.text
    )

    result = resolve_handoff_admission(
        HandoffPolicyInput(
            requested_status=AdmissionStatus.REJECTED,
            requirement_coverage=tuple(
                RequirementCoverage(
                    requirement_id=requirement_id,
                    status=CoverageStatus.SUPPORTED,
                    evidence_ids=(f"ev-{index}",),
                    explanation="The original user requirement is supported.",
                )
                for index, requirement_id in enumerate(
                    requirement_ids,
                    start=1,
                )
            ),
            caveats=tuple(assessment["missing_information"]),
            missing_information=tuple(assessment["missing_information"]),
            unsupported_claims=(),
            deterministic_checks_passed=True,
            scores=(
                assessment["relevance"],
                assessment["source_quality"],
                assessment["evidence_coverage"],
                assessment["corroboration"],
            ),
            dimension_floor=3,
            average_floor=3.0,
            caveat_admission_enabled=True,
            high_risk=False,
        ),
        owned_requirement_ids=requirement_ids,
    )

    assert result.admission_status.value == fixture["expected_v4"][
        "admission_when_owned_user_requirements_are_supported"
    ]
    assert result.hard_rejection_reasons == ()


def test_run3_contract_keeps_numbered_deliverables_atomic() -> None:
    fixture_path = (
        Path(__file__).parent
        / "fixtures"
        / "quality_gate_v4_run3_replay.json"
    )
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    original_query = fixture["original_query"]

    contract = build_research_coverage_contract(
        [HumanMessage(content=original_query)]
    )
    texts = [requirement.text for requirement in contract.requirements]

    assert len(texts) <= 9
    assert not {"node results", "pending tasks", "database writes"} & set(texts)
    assert any(
        "Checkpointer state at super-step boundaries" in text
        and "pending tasks" in text
        for text in texts
    )
    assert any(
        "Interrupt and Command(resume=...) mechanics" in text
        and "restored vs. recalculated" in text
        for text in texts
    )
    assert any(
        "Idempotency design to avoid external side-effect duplication" in text
        and "database writes" in text
        and "file I/O" in text
        for text in texts
    )
    assert any(
        "officially guaranteed behavior versus engineering inference" in text
        for text in texts
    )
    assert any(
        "five-item Minimum Viable Reliability Checklist" in text
        for text in texts
    )
    for requirement in contract.requirements:
        assert (
            original_query[
                requirement.source_start : requirement.source_end
            ]
            == requirement.text
        )


def test_numbered_multiline_requirement_retains_exact_source_span() -> None:
    original_query = (
        "Please explain:\n"
        "1. Checkpoint state,\n"
        "   including pending tasks and metadata.\n"
        "2. Resume behavior and idempotency."
    )

    contract = build_research_coverage_contract(
        [HumanMessage(content=original_query)]
    )
    requirement = next(
        item
        for item in contract.requirements
        if "Checkpoint state" in item.text
    )

    assert "\n" in requirement.text
    assert requirement.text != original_query
    assert (
        original_query[requirement.source_start : requirement.source_end]
        == requirement.text
    )


def test_coverage_contract_preserves_explicit_chinese_time_constraint() -> None:
    original_query = (
        "截至2026年7月，说明 LangGraph checkpointer 在 super-step "
        "边界保存哪些状态。"
    )

    contract = build_research_coverage_contract(
        [HumanMessage(content=original_query)]
    )

    time_requirement = next(
        requirement
        for requirement in contract.requirements
        if "截至2026年7月" in requirement.text
    )
    assert (
        original_query[
            time_requirement.source_start : time_requirement.source_end
        ]
        == time_requirement.text
    )


@pytest.mark.asyncio
async def test_research_brief_exposes_contract_ids_to_supervisor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_query = (
        "截至2026年7月，说明 LangGraph checkpoint 恢复机制，"
        "并给出工程可靠性清单。"
    )
    advisory_brief = (
        "调查 checkpoint 恢复机制；额外查找一个精确版本号作为建议维度。"
    )

    class _FakeResearchModel:
        def with_structured_output(self, *_args, **_kwargs):
            return self

    async def _fake_invoke(*_args, **_kwargs):
        return ResearchQuestion(research_brief=advisory_brief)

    monkeypatch.setattr(
        deep_researcher,
        "get_model_connection_kwargs",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        deep_researcher,
        "apply_helicone_config",
        lambda model_config, *_args, **_kwargs: model_config,
    )
    monkeypatch.setattr(
        deep_researcher,
        "init_chat_model",
        lambda **_kwargs: _FakeResearchModel(),
    )
    monkeypatch.setattr(
        deep_researcher,
        "invoke_model_with_retry_observability",
        _fake_invoke,
    )

    command = await deep_researcher.write_research_brief(
        {"messages": [HumanMessage(content=original_query)]},
        {
            "configurable": {
                "research_model": "openai:gpt-4.1",
                "enable_async_research": False,
            },
            "metadata": {"run_id": "coverage-contract-visible"},
        },
    )

    contract = build_research_coverage_contract(
        [HumanMessage(content=original_query)],
        advisory_dimensions=[advisory_brief],
    )
    supervisor_messages = command.update["supervisor_messages"]["value"]
    visible_context = "\n".join(
        str(message.content) for message in supervisor_messages
    )

    for requirement in contract.requirements:
        assert (
            f"{requirement.requirement_id}: {requirement.text}"
            in visible_context
        )
    assert "精确版本号" not in "\n".join(
        requirement.text for requirement in contract.requirements
    )


def test_supervisor_advisory_requirement_cannot_hard_reject_handoff() -> None:
    contract = build_research_coverage_contract(
        [
            HumanMessage(
                content=(
                    "截至2026年7月，说明 LangGraph checkpointer 在 super-step "
                    "边界保存哪些状态，并区分官方保证与工程推断。"
                )
            )
        ],
        advisory_dimensions=[
            "必须给出精确 LangGraph 版本号",
            "必须只使用已迁移的旧文档域名",
        ],
    )
    requirement_ids = tuple(
        requirement.requirement_id for requirement in contract.requirements
    )

    result = resolve_handoff_admission(
        HandoffPolicyInput(
            requested_status=AdmissionStatus.REJECTED,
            requirement_coverage=tuple(
                RequirementCoverage(
                    requirement_id=requirement_id,
                    status=CoverageStatus.SUPPORTED,
                    evidence_ids=("ev-1",),
                    explanation="Official documentation supports the requirement.",
                )
                for requirement_id in requirement_ids
            ),
            caveats=(
                "A precise package version was not stated.",
                "The advisory legacy documentation hostname was not used.",
            ),
            missing_information=(
                "A precise package version was not stated.",
            ),
            unsupported_claims=(),
            deterministic_checks_passed=True,
            scores=(4, 5, 3, 4),
            dimension_floor=3,
            average_floor=3.0,
            caveat_admission_enabled=True,
            high_risk=False,
        ),
        owned_requirement_ids=requirement_ids,
    )

    assert result.admission_status is AdmissionStatus.ACCEPTED_WITH_CAVEATS
    assert result.accepted is True
    assert result.hard_rejection_reasons == ()


def test_explicit_user_requirement_remains_a_hard_gate() -> None:
    contract = build_research_coverage_contract(
        [
            HumanMessage(
                content=(
                    "请给出截至2026年7月的精确 LangGraph 版本号，并说明 "
                    "checkpointer 保存的字段。"
                )
            )
        ]
    )
    version_requirement = next(
        requirement
        for requirement in contract.requirements
        if "版本号" in requirement.text
    )

    result = resolve_handoff_admission(
        HandoffPolicyInput(
            requested_status=AdmissionStatus.ACCEPTED,
            requirement_coverage=(
                RequirementCoverage(
                    requirement_id=version_requirement.requirement_id,
                    status=CoverageStatus.UNSUPPORTED,
                    evidence_ids=(),
                    explanation="No official version evidence was supplied.",
                ),
            ),
            caveats=(),
            missing_information=("The exact version is missing.",),
            unsupported_claims=(),
            deterministic_checks_passed=True,
            scores=(4, 5, 4, 4),
            dimension_floor=3,
            average_floor=3.0,
            caveat_admission_enabled=True,
            high_risk=False,
        ),
        owned_requirement_ids=(version_requirement.requirement_id,),
    )

    assert result.admission_status is AdmissionStatus.REJECTED
    assert result.accepted is False
    assert result.hard_rejection_reasons == (
        f"required_coverage_missing:{version_requirement.requirement_id}",
    )


def test_unsupported_claim_never_enters_caveat_admission() -> None:
    contract = build_research_coverage_contract(
        [HumanMessage(content="说明 LangGraph 的 checkpoint 恢复机制。")]
    )
    requirement_id = contract.requirements[0].requirement_id

    result = resolve_handoff_admission(
        HandoffPolicyInput(
            requested_status=AdmissionStatus.ACCEPTED_WITH_CAVEATS,
            requirement_coverage=(
                RequirementCoverage(
                    requirement_id=requirement_id,
                    status=CoverageStatus.SUPPORTED,
                    evidence_ids=("ev-1",),
                    explanation="Covered.",
                ),
            ),
            caveats=("One optional detail is unavailable.",),
            missing_information=(),
            unsupported_claims=("Checkpoint writes are globally atomic.",),
            deterministic_checks_passed=True,
            scores=(5, 5, 4, 4),
            dimension_floor=3,
            average_floor=3.0,
            caveat_admission_enabled=True,
            high_risk=False,
        ),
        owned_requirement_ids=(requirement_id,),
    )

    assert result.admission_status is AdmissionStatus.REJECTED
    assert "unsupported_claims" in result.hard_rejection_reasons


def test_researcher_is_judged_only_on_owned_requirement_ids() -> None:
    contract = build_research_coverage_contract(
        [HumanMessage(content="- 说明检查点恢复机制。\n- 给出性能基准。")]
    )
    owned, other = contract.requirements[:2]

    result = resolve_handoff_admission(
        HandoffPolicyInput(
            requested_status=AdmissionStatus.ACCEPTED,
            requirement_coverage=(
                RequirementCoverage(
                    requirement_id=owned.requirement_id,
                    status=CoverageStatus.SUPPORTED,
                    evidence_ids=("ev-1",),
                    explanation="Covered by the assigned task.",
                ),
                RequirementCoverage(
                    requirement_id=other.requirement_id,
                    status=CoverageStatus.UNSUPPORTED,
                    evidence_ids=(),
                    explanation="Owned by a different task.",
                ),
            ),
            caveats=(),
            missing_information=(),
            unsupported_claims=(),
            deterministic_checks_passed=True,
            scores=(5, 5, 5, 5),
            dimension_floor=3,
            average_floor=3.0,
            caveat_admission_enabled=True,
            high_risk=False,
        ),
        owned_requirement_ids=(owned.requirement_id,),
    )

    assert result.admission_status is AdmissionStatus.ACCEPTED


def test_high_risk_keyword_disables_caveat_admission() -> None:
    risk = classify_research_risk(
        "请根据症状给出诊断和处方剂量建议。",
        mode="auto",
    )
    coverage = RequirementCoverage(
        requirement_id="COV-01",
        status=CoverageStatus.SUPPORTED,
        evidence_ids=("ev-1",),
        explanation="Covered.",
    )

    result = resolve_handoff_admission(
        HandoffPolicyInput(
            requested_status=AdmissionStatus.ACCEPTED_WITH_CAVEATS,
            requirement_coverage=(coverage,),
            caveats=("One secondary source is unavailable.",),
            missing_information=(),
            unsupported_claims=(),
            deterministic_checks_passed=True,
            scores=(5, 5, 5, 5),
            dimension_floor=3,
            average_floor=3.0,
            caveat_admission_enabled=True,
            high_risk=risk.level == "high",
        ),
        owned_requirement_ids=("COV-01",),
    )

    assert risk.level == "high"
    assert any(
        rule_id.startswith("medical.")
        for rule_id in risk.matched_rule_ids
    )
    assert result.admission_status is AdmissionStatus.REJECTED
    assert "high_risk_caveats_disallowed" in result.hard_rejection_reasons


def test_trade_economics_is_not_misclassified_as_personal_finance_risk() -> None:
    economics = classify_research_risk(
        "Compare international trade flows and trading volumes in economic history.",
        mode="auto",
    )
    advice = classify_research_risk(
        "Recommend a trading strategy and whether I should buy or sell today.",
        mode="auto",
    )

    assert economics.level == "standard"
    assert advice.level == "high"
    assert "finance.trading" in advice.matched_rule_ids


def test_finance_risk_handles_securities_without_flagging_laptop_purchase() -> None:
    securities = classify_research_risk(
        "Should I buy or sell these securities?",
        mode="auto",
    )
    laptop = classify_research_risk(
        "Compare reviews before I buy a laptop now.",
        mode="auto",
    )

    assert securities.level == "high"
    assert laptop.level == "standard"


def test_plain_text_none_error_type_is_not_a_tool_failure() -> None:
    checks = deterministic_tool_checks(
        [{
            "name": "legacy_tool",
            "content": 'prefix {"error_type": "none"} suffix',
            "error": False,
        }],
        min_sources=0,
    )

    assert checks["passed"] is True
    assert checks["error_count"] == 0


@pytest.mark.asyncio
async def test_v4_fail_open_evaluator_error_does_not_admit_empty_coverage(
    monkeypatch,
) -> None:
    contract = build_research_coverage_contract(
        [HumanMessage(content="说明 LangGraph 的 checkpoint 恢复机制。")]
    )
    requirement_id = contract.requirements[0].requirement_id

    async def fail_judge(*_args, **_kwargs):
        raise TimeoutError("quality judge unavailable")

    monkeypatch.setattr(
        "open_deep_research.quality.gate._build_quality_model",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        "open_deep_research.quality.gate.invoke_model_with_retry_observability",
        fail_judge,
    )
    handoff = {
        "compressed_research": (
            "Detailed checkpoint evidence from two official sources. "
        )
        * 10,
        "evidence_registry": [
            {
                "evidence_id": "ev-a",
                "claim": "Checkpoint state is persisted.",
                "supporting_excerpt": "Checkpoint state is persisted.",
                "source_url": "https://docs.example/checkpoints",
                "source_title": "Official checkpoints",
                "security_status": "accepted",
            },
            {
                "evidence_id": "ev-b",
                "claim": "Resume restores persisted state.",
                "supporting_excerpt": "Resume restores persisted state.",
                "source_url": "https://api.example/checkpoints",
                "source_title": "Official API",
                "security_status": "accepted",
            },
        ],
        "metrics": {"sources_read": 2},
    }
    config = {
        "configurable": {
            "quality_evaluation_fail_open": True,
            "quality_evaluation_min_sources": 2,
        },
        "metadata": {
            "quality_policy_version": "quality-gate-v4",
            "runtime_config_frozen": True,
            "run_id": "v4-fail-open-empty-coverage",
        },
    }

    result = await evaluate_subagent_handoff(
        "Advisory checkpoint task.",
        handoff,
        config,
        coverage_contract=contract,
        requirement_ids=[requirement_id],
    )

    assert result.accepted is False
    assert result.admission_status is AdmissionStatus.REJECTED
    assert result.requirement_coverage == []
    assert result.evaluator_error == "quality judge unavailable"
    assert "quality_evaluator_unavailable" in result.hard_rejection_reasons
    assert "free-text handoff is not admitted" in result.reason


@pytest.mark.asyncio
async def test_v4_fail_open_does_not_bypass_required_coverage(
    monkeypatch,
) -> None:
    """An outer judge outage must not admit free text without coverage mapping."""
    contract = build_research_coverage_contract(
        [HumanMessage(content="说明 LangGraph 的 checkpoint 恢复机制。")]
    )
    requirement_id = contract.requirements[0].requirement_id

    async def fail_judge(*_args, **_kwargs):
        raise TimeoutError("quality judge unavailable")

    monkeypatch.setattr(
        "open_deep_research.quality.gate._build_quality_model",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        "open_deep_research.quality.gate.invoke_model_with_retry_observability",
        fail_judge,
    )
    handoff = {
        "compressed_research": "Detailed checkpoint evidence. " * 20,
        "evidence_registry": [
            {
                "evidence_id": "ev-a",
                "claim": "Checkpoint state is persisted.",
                "supporting_excerpt": "Checkpoint state is persisted.",
                "source_url": "https://docs.example/checkpoints",
                "source_title": "Official checkpoints",
                "security_status": "accepted",
            },
            {
                "evidence_id": "ev-b",
                "claim": "Resume restores persisted state.",
                "supporting_excerpt": "Resume restores persisted state.",
                "source_url": "https://api.example/checkpoints",
                "source_title": "Official API",
                "security_status": "accepted",
            },
        ],
        "metrics": {"sources_read": 2},
        "result_assessment": {
            "decision": "complete",
            "relevance": 5,
            "source_quality": 5,
            "evidence_coverage": 4,
            "corroboration": 4,
            "deterministic_checks": {"passed": True},
            "evaluator_error": None,
        },
    }
    config = {
        "configurable": {
            "quality_evaluation_fail_open": True,
            "quality_evaluation_min_sources": 2,
            "quality_caveat_admission_enabled": True,
        },
        "metadata": {
            "quality_policy_version": "quality-gate-v4",
            "runtime_config_frozen": True,
            "run_id": "v4-fail-open-inner-assessment",
        },
    }

    result = await evaluate_subagent_handoff(
        "Advisory checkpoint task.",
        handoff,
        config,
        coverage_contract=contract,
        requirement_ids=[requirement_id],
    )

    assert result.accepted is False
    assert result.admission_status is AdmissionStatus.REJECTED
    assert result.evaluator_error == "quality judge unavailable"
    assert any(
        reason.startswith("required_coverage_missing:")
        for reason in result.hard_rejection_reasons
    )
    assert "quality_evaluator_unavailable" in result.hard_rejection_reasons
    assert "admitting with caveats" not in result.reason


@pytest.mark.asyncio
async def test_v4_policy_without_contract_fails_closed_on_judge_outage(
    monkeypatch,
) -> None:
    async def fail_judge(*_args, **_kwargs):
        raise TimeoutError("quality judge unavailable")

    monkeypatch.setattr(
        "open_deep_research.quality.gate._evaluate_json",
        fail_judge,
    )
    result = await evaluate_subagent_handoff(
        "Advisory checkpoint task.",
        {
            "compressed_research": "Detailed checkpoint evidence. " * 20,
            "evidence_registry": [],
            "metrics": {"sources_read": 2},
        },
        {
            "configurable": {
                "quality_evaluation_fail_open": True,
                "quality_evaluation_min_sources": 0,
            },
            "metadata": {"quality_policy_version": "quality-gate-v4"},
        },
    )

    assert result.accepted is False
    assert result.admission_status is AdmissionStatus.REJECTED
    assert "quality_evaluator_unavailable" in result.hard_rejection_reasons


@pytest.mark.asyncio
async def test_v4_malformed_handoff_contract_is_rejected_not_raised(
    monkeypatch,
) -> None:
    async def fail_if_called(*_args, **_kwargs):
        raise AssertionError("invalid contracts must not reach the Judge")

    monkeypatch.setattr(
        "open_deep_research.quality.gate._evaluate_json",
        fail_if_called,
    )
    result = await evaluate_subagent_handoff(
        "Advisory task.",
        {
            "compressed_research": "Grounded handoff. " * 30,
            "evidence_registry": [],
            "metrics": {"sources_read": 2},
        },
        {
            "configurable": {
                "quality_evaluation_fail_open": True,
                "quality_evaluation_min_sources": 0,
            },
            "metadata": {"quality_policy_version": "quality-gate-v4"},
        },
        coverage_contract={"requirements": "not-a-list"},
        requirement_ids=["COV-01"],
    )

    assert result.accepted is False
    assert result.admission_status is AdmissionStatus.REJECTED
    assert "coverage_contract_invalid" in result.hard_rejection_reasons


@pytest.mark.asyncio
async def test_v4_successful_handoff_judge_does_not_add_unavailable_caveat(
    monkeypatch,
) -> None:
    contract = build_research_coverage_contract(
        [HumanMessage(content="说明 LangGraph 的 checkpoint 恢复机制。")]
    )
    requirement_id = contract.requirements[0].requirement_id

    async def pass_judge(*_args, **_kwargs):
        return HandoffAssessment(
            accepted=True,
            admission_status="accepted",
            relevance=5,
            source_quality=5,
            evidence_coverage=5,
            groundedness=5,
            requirement_coverage=[
                {
                    "requirement_id": requirement_id,
                    "status": "supported",
                    "evidence_ids": ["ev-a"],
                    "explanation": "The official evidence supports it.",
                }
            ],
            reason="All owned requirements are supported.",
        )

    monkeypatch.setattr(
        "open_deep_research.quality.gate._evaluate_json",
        pass_judge,
    )
    handoff = {
        "compressed_research": "Detailed checkpoint evidence. " * 20,
        "evidence_registry": [
            {
                "evidence_id": "ev-a",
                "claim": "Checkpoint state is persisted.",
                "supporting_excerpt": "Checkpoint state is persisted.",
                "source_url": "https://docs.example/checkpoints",
                "source_title": "Official checkpoints",
                "security_status": "accepted",
            },
            {
                "evidence_id": "ev-b",
                "claim": "Resume restores persisted state.",
                "supporting_excerpt": "Resume restores persisted state.",
                "source_url": "https://api.example/checkpoints",
                "source_title": "Official API",
                "security_status": "accepted",
            },
        ],
        "metrics": {"sources_read": 2},
        "result_assessment": {
            "decision": "complete",
            "relevance": 5,
            "source_quality": 5,
            "evidence_coverage": 5,
            "corroboration": 5,
            "deterministic_checks": {"passed": True},
            "evaluator_error": None,
        },
    }
    config = {
        "configurable": {
            "quality_evaluation_fail_open": True,
            "quality_evaluation_min_sources": 2,
            "quality_caveat_admission_enabled": True,
        },
        "metadata": {
            "quality_policy_version": "quality-gate-v4",
            "runtime_config_frozen": True,
            "run_id": "v4-success-with-fail-open-enabled",
        },
    }

    result = await evaluate_subagent_handoff(
        "Advisory checkpoint task.",
        handoff,
        config,
        coverage_contract=contract,
        requirement_ids=[requirement_id],
    )

    assert result.accepted is True
    assert result.admission_status is AdmissionStatus.ACCEPTED
    assert result.evaluator_error is None
    assert "quality_evaluator_unavailable" not in result.caveats
    assert result.hard_rejection_reasons == []


@pytest.mark.asyncio
async def test_v4_structural_requirements_do_not_need_external_evidence_ids(
    monkeypatch,
) -> None:
    contract = build_research_coverage_contract([
        HumanMessage(content=(
            "把下列内容作为一个不可拆分的单一研究任务；"
            "核验 PostgreSQL 18 skip scan；不得引用第三方来源。"
        ))
    ])
    factual_requirement = next(
        item for item in contract.requirements if "skip scan" in item.text
    )
    structural_ids = {
        item.requirement_id
        for item in contract.requirements
        if item.requirement_id != factual_requirement.requirement_id
    }
    captured: dict = {}

    async def pass_judge(_schema, _prompt, payload, _config, **_kwargs):
        captured.update(payload)
        return HandoffAssessment(
            accepted=True,
            admission_status="accepted",
            relevance=5,
            source_quality=5,
            evidence_coverage=5,
            groundedness=5,
            requirement_coverage=[
                {
                    "requirement_id": item.requirement_id,
                    "status": "supported",
                    "evidence_ids": (
                        ["ev-a"]
                        if item.requirement_id
                        == factual_requirement.requirement_id
                        else []
                    ),
                    "explanation": "Satisfied by evidence or output structure.",
                }
                for item in contract.requirements
            ],
            reason="All requirements are supported.",
        )

    monkeypatch.setattr(
        "open_deep_research.quality.gate._evaluate_json",
        pass_judge,
    )
    handoff = {
        "compressed_research": "Grounded PostgreSQL skip scan evidence. " * 20,
        "evidence_registry": [
            {
                "evidence_id": "ev-a",
                "claim": "PostgreSQL 18 supports skip scan.",
                "supporting_excerpt": "Support for skip scan lookups.",
                "source_url": "https://www.postgresql.org/docs/release/18.0/",
                "security_status": "accepted",
            }
        ],
        "metrics": {"sources_read": 1},
    }
    result = await evaluate_subagent_handoff(
        "Advisory task.",
        handoff,
        {
            "configurable": {
                "quality_evaluation_fail_open": False,
                "quality_evaluation_min_sources": 1,
            },
            "metadata": {
                "quality_policy_version": "quality-gate-v4",
                "runtime_config_frozen": True,
            },
        },
        coverage_contract=contract,
        requirement_ids=list(contract.requirement_ids()),
    )

    assert result.accepted is True
    assert result.admission_status is AdmissionStatus.ACCEPTED
    assert set(captured["evidence_optional_requirement_ids"]) == structural_ids
    assert captured["owned_requirement_ids"] == [
        factual_requirement.requirement_id
    ]
    assert result.hard_rejection_reasons == []


@pytest.mark.asyncio
async def test_v4_parallel_delegation_and_final_table_are_run_level(
    monkeypatch,
) -> None:
    contract = build_research_coverage_contract([
        HumanMessage(content=(
            "最终修复后 E2E：请并行委派两个 Subagent；"
            "A 仅根据 https://peps.python.org/pep-0703/ 总结状态与风险；"
            "不得引用其他 URL；最终用中文给出对照表；"
            "用中文输出；每个事实结论都附来源。"
        ))
    ])
    factual_requirement = next(
        item for item in contract.requirements if "peps.python.org" in item.text
    )
    citation_requirement = next(
        item for item in contract.requirements if "每个事实" in item.text
    )
    run_level_ids = {
        item.requirement_id
        for item in contract.requirements
        if (
            "并行委派" in item.text
            or "不得引用其他 URL" in item.text
            or "对照表" in item.text
            or "用中文输出" in item.text
        )
    }
    captured: dict = {}

    async def pass_judge(_schema, _prompt, payload, _config, **_kwargs):
        captured.update(payload)
        return HandoffAssessment(
            accepted=True,
            admission_status="accepted",
            relevance=5,
            source_quality=5,
            evidence_coverage=5,
            groundedness=5,
            requirement_coverage=[
                {
                    "requirement_id": factual_requirement.requirement_id,
                    "status": "supported",
                    "evidence_ids": ["ev-a"],
                    "explanation": "Grounded leaf-task finding.",
                },
                {
                    "requirement_id": citation_requirement.requirement_id,
                    "status": "supported",
                    "evidence_ids": ["ev-a"],
                    "explanation": "Every factual leaf finding is cited.",
                },
            ],
            reason="The factual leaf requirement is supported.",
        )

    monkeypatch.setattr(
        "open_deep_research.quality.gate._evaluate_json",
        pass_judge,
    )
    result = await evaluate_subagent_handoff(
        "Advisory task A.",
        {
            "compressed_research": "Grounded PEP 703 finding [ev-a]. " * 20,
            "evidence_registry": [
                {
                    "evidence_id": "ev-a",
                    "claim": "PEP 703 defines the free-threading design.",
                    "source_url": "https://peps.python.org/pep-0703/",
                    "security_status": "accepted",
                }
            ],
            "metrics": {"sources_read": 1},
        },
        {
            "configurable": {
                "quality_evaluation_fail_open": False,
                "quality_evaluation_min_sources": 1,
            },
            "metadata": {
                "quality_policy_version": "quality-gate-v4",
                "runtime_config_frozen": True,
            },
        },
        coverage_contract=contract,
        requirement_ids=list(contract.requirement_ids()),
    )

    assert result.accepted is True
    assert run_level_ids <= set(captured["evidence_optional_requirement_ids"])
    assert captured["owned_requirement_ids"] == [
        factual_requirement.requirement_id,
        citation_requirement.requirement_id,
    ]


def test_explicit_url_leaf_handoff_does_not_require_global_source_floor() -> None:
    contract = build_research_coverage_contract([
        HumanMessage(content=(
            "A 仅根据 https://peps.python.org/pep-0703/ 总结状态；"
            "B 仅根据 https://numpy.org/doc/2.1/release/2.1.0-notes.html "
            "总结支持情况；不得引用其他 URL。"
        ))
    ])
    checks = deterministic_handoff_checks(
        {
            "compressed_research": "Grounded finding [ev-a]. " * 20,
            "evidence_registry": [
                {
                    "evidence_id": "ev-a",
                    "claim": "PEP 703 defines the design.",
                    "source_url": "https://peps.python.org/pep-0703/",
                    "security_status": "accepted",
                }
            ],
        },
        min_sources=3,
        coverage_contract=contract,
    )

    assert checks["passed"] is True
    assert checks["source_count"] == 1
    assert checks["required_source_count"] == 1


@pytest.mark.asyncio
async def test_v4_official_only_handoff_cannot_map_to_out_of_scope_evidence(
    monkeypatch,
) -> None:
    contract = build_research_coverage_contract(
        [
            HumanMessage(
                content=(
                    "Based solely on the LangGraph official documentation, "
                    "official API reference, and official GitHub repository, "
                    "explain checkpoint persistence."
                )
            )
        ]
    )
    requirement_ids = list(contract.requirement_ids())
    captured: dict = {}

    async def fake_evaluate(
        _schema,
        _prompt,
        payload,
        _config,
        **_kwargs,
    ):
        captured.update(payload)
        return HandoffAssessment(
            accepted=True,
            admission_status="accepted",
            relevance=5,
            source_quality=5,
            evidence_coverage=5,
            groundedness=5,
            requirement_coverage=[
                {
                    "requirement_id": requirement_id,
                    "status": "supported",
                    "evidence_ids": ["EV-MINTLIFY"],
                    "explanation": "The temporary mirror supports this.",
                }
                for requirement_id in requirement_ids
            ],
            reason="All requirements are supported.",
        )

    monkeypatch.setattr(
        "open_deep_research.quality.gate._evaluate_json",
        fake_evaluate,
    )
    handoff = {
        "compressed_research": (
            "MINTLIFY_FREE_TEXT_SECRET "
            "https://langchain-5e9cc07a.mintlify.app/oss/python/langgraph "
        )
        * 10,
        "raw_notes": ["MINTLIFY_RAW_NOTE_SECRET"],
        "evidence_registry": [
            {
                "evidence_id": "EV-DOCS",
                "claim": "Official checkpoint claim. " + ("c" * 200),
                "supporting_excerpt": "Official excerpt. " + ("e" * 200),
                "source_url": (
                    "https://docs.langchain.com/oss/python/langgraph/"
                    "persistence"
                ),
                "source_title": "LangGraph docs",
                "security_status": "accepted",
            },
            {
                "evidence_id": "EV-MINTLIFY",
                "claim": "Temporary mirror claim.",
                "supporting_excerpt": "Temporary mirror excerpt.",
                "source_url": (
                    "https://langchain-5e9cc07a.mintlify.app/oss/python/"
                    "langgraph/persistence"
                ),
                "source_title": "Temporary mirror",
                "security_status": "accepted",
            },
        ],
        "metrics": {"sources_read": 99},
    }
    config = {
        "configurable": {
            "quality_evaluation_fail_open": False,
            "quality_evaluation_min_sources": 1,
        },
        "metadata": {
            "quality_policy_version": "quality-gate-v4",
            "runtime_config_frozen": True,
            "run_id": "official-only-runtime-gate",
        },
    }

    result = await evaluate_subagent_handoff(
        "Advisory task.",
        handoff,
        config,
        coverage_contract=contract,
        requirement_ids=requirement_ids,
    )

    assert result.accepted is False
    assert result.admission_status is AdmissionStatus.REJECTED
    assert "deterministic_checks_failed" in result.hard_rejection_reasons
    assert all(
        reason.startswith("supported_requirement_has_invalid_evidence:")
        for reason in result.hard_rejection_reasons
        if reason != "deterministic_checks_failed"
    )
    assert [item["evidence_id"] for item in captured["evidence_registry"]] == [
        "EV-DOCS"
    ]
    assert "MINTLIFY_FREE_TEXT_SECRET" in captured["compressed_research"]
    assert captured["raw_notes"] == ""
    assert captured["deterministic_checks"]["source_count"] == 1
    assert captured["deterministic_checks"]["source_scope_enforced"] is True
    assert (
        "handoff_contains_out_of_scope_source_url"
        in captured["deterministic_checks"]["failures"]
    )
    assert captured["deterministic_checks"]["out_of_scope_source_count"] > 0


@pytest.mark.asyncio
async def test_v4_official_only_judge_can_evaluate_candidate_structure(
    monkeypatch,
) -> None:
    contract = build_research_coverage_contract(
        [
            HumanMessage(
                content=(
                    "Based solely on the LangGraph official documentation, "
                    "explain checkpoint persistence and provide a checklist "
                    "that labels guarantees versus engineering inference."
                )
            )
        ]
    )
    requirement_ids = list(contract.requirement_ids())
    captured: dict = {}

    async def fake_evaluate(
        _schema,
        _prompt,
        payload,
        _config,
        **_kwargs,
    ):
        captured.update(payload)
        return HandoffAssessment(
            accepted=True,
            admission_status="accepted",
            relevance=5,
            source_quality=5,
            evidence_coverage=5,
            groundedness=5,
            requirement_coverage=[
                {
                    "requirement_id": requirement_id,
                    "status": "supported",
                    "evidence_ids": ["EV-DOCS"],
                    "explanation": "The official evidence supports it.",
                }
                for requirement_id in requirement_ids
            ],
            reason="All owned requirements are supported.",
        )

    monkeypatch.setattr(
        "open_deep_research.quality.gate._evaluate_json",
        fake_evaluate,
    )
    official_url = (
        "https://docs.langchain.com/oss/python/langgraph/persistence"
    )
    candidate = (
        "Officially Guaranteed: checkpoint state can be persisted. "
        "Engineering Inference: use durable storage in production. "
        "Five-item Checklist: configure, persist, resume, inspect, verify. "
        f"Source: {official_url}. "
    ) * 3
    handoff = {
        "compressed_research": candidate,
        "raw_notes": ["UNTRUSTED_RAW_NOTE"],
        "evidence_registry": [
            {
                "evidence_id": "EV-DOCS",
                "claim": "Checkpoint state can be persisted.",
                "supporting_excerpt": (
                    "Checkpointers save graph state at every super-step."
                ),
                "source_url": official_url,
                "source_title": "LangGraph persistence",
                "security_status": "accepted",
            }
        ],
        "metrics": {"sources_read": 1},
    }
    config = {
        "configurable": {
            "quality_evaluation_fail_open": False,
            "quality_evaluation_min_sources": 1,
        },
        "metadata": {
            "quality_policy_version": "quality-gate-v4",
            "runtime_config_frozen": True,
            "run_id": "official-candidate-structure",
        },
    }

    result = await evaluate_subagent_handoff(
        "Advisory task.",
        handoff,
        config,
        coverage_contract=contract,
        requirement_ids=requirement_ids,
    )

    assert result.accepted is True
    assert result.admission_status is AdmissionStatus.ACCEPTED
    assert captured["compressed_research"] == candidate
    assert captured["raw_notes"] == ""
    assert [
        item["evidence_id"] for item in captured["evidence_registry"]
    ] == ["EV-DOCS"]
    assert captured["deterministic_checks"]["passed"] is True
    assert captured["deterministic_checks"]["out_of_scope_source_count"] == 0


@pytest.mark.asyncio
async def test_conduct_research_rejects_unknown_requirement_id() -> None:
    contract = build_research_coverage_contract(
        [HumanMessage(content="说明检查点恢复机制。")]
    )
    tool = next(
        tool
        for tool in deep_researcher.build_supervisor_tools({
            "coverage_contract": contract.model_dump(mode="json"),
            "research_risk_profile": {"level": "standard"},
        })
        if tool.name == "ConductResearch"
    )
    tool_input = tool.input_schema(
        research_topic="advisory task",
        requirement_ids=["COV-UNKNOWN"],
    )

    with pytest.raises(ValueError, match="unknown_coverage_requirement_ids"):
        await tool.call(
            tool_input,
            ToolContext(
                config={
                    "configurable": {},
                    "metadata": {
                        "run_id": "coverage-v4",
                        "quality_policy_version": "quality-gate-v4",
                    },
                },
                role="supervisor",
                tool_call_id="task-1",
            ),
        )


def test_research_tool_schemas_enumerate_contract_requirement_ids() -> None:
    contract = build_research_coverage_contract(
        [HumanMessage(content="说明检查点恢复机制，并比较失败恢复策略。")]
    )
    expected_ids = list(contract.requirement_ids())

    sync_tool = next(
        tool
        for tool in deep_researcher.build_supervisor_tools({
            "coverage_contract": contract.model_dump(mode="json"),
            "research_risk_profile": {"level": "standard"},
        })
        if tool.name == "ConductResearch"
    )
    async_tool = next(
        tool
        for tool in deep_researcher.build_supervisor_tools({
            "coverage_contract": contract.model_dump(mode="json"),
            "research_risk_profile": {"level": "standard"},
            "enable_async_research": True,
        })
        if tool.name == "StartResearchTask"
    )

    for tool in (sync_tool, async_tool):
        requirement_schema = tool.input_schema.model_json_schema()["properties"][
            "requirement_ids"
        ]
        assert requirement_schema["items"]["enum"] == expected_ids
