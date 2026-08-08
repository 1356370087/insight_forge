"""Tests for sectioned report assembly (Phase 4)."""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from open_deep_research.report import assembly as assembly_module
from open_deep_research.report import build_report
from open_deep_research.report.assembly import (
    AssemblyResult,
    ReportContext,
    SectionedStrategy,
)
from open_deep_research.report.models import ReportOutline, SectionSpec, WrittenSection
from open_deep_research.report.profiles import AssemblyMode, get_profile


def _config(**configurable: Any) -> RunnableConfig:
    return {
        "configurable": {"quality_evaluation_enabled": False, **configurable},
        "metadata": {"run_id": "assembly-test"},
    }


class _FakeStructuredModel:
    """Stands in for a concrete chat model so build_structured_model doesn't need an API key.

    The actual LLM call is mocked (invoke_model_with_retry_observability), so the
    model object only needs to support .with_structured_output().
    """

    def with_structured_output(self, _schema, **_kwargs):
        return self


def _patch_init_chat_model(monkeypatch):
    monkeypatch.setattr(assembly_module, "init_chat_model", lambda **_kw: _FakeStructuredModel())


def test_sectioned_assemble_ordering():
    strategy = SectionedStrategy()
    outline = ReportOutline(
        title="T", sections=[SectionSpec(name="Alpha"), SectionSpec(name="Beta")]
    )
    sections = [
        WrittenSection(name="Alpha", content="alpha body"),
        WrittenSection(name="Beta", content="beta body"),
    ]
    body = strategy._assemble(None, outline, sections, "intro text", "conclusion text")

    assert body.startswith("# T\n")
    # Ordering: Introduction -> Table of Contents -> content sections -> Conclusion
    assert body.index("## Introduction") < body.index("## Table of Contents")
    assert body.index("## Table of Contents") < body.index("## Alpha")
    assert body.index("## Alpha") < body.index("## Beta")
    assert body.index("## Beta") < body.index("## Conclusion")
    # TOC lists both sections
    assert "1. Alpha" in body and "2. Beta" in body
    # Bodies + intro/conclusion present
    assert "alpha body" in body and "beta body" in body
    assert "intro text" in body and "conclusion text" in body


def test_sectioned_profiles_are_sectioned():
    assert get_profile("comparison_matrix").assembly == AssemblyMode.SECTIONED
    assert get_profile("pros_cons").assembly == AssemblyMode.SECTIONED
    assert get_profile("literature_review").assembly == AssemblyMode.SECTIONED


def test_literature_review_uses_bibtex_style():
    assert get_profile("literature_review").reference_style.value == "bibtex_like"


def test_report_context_excludes_quarantined_evidence_sources():
    context = ReportContext.from_state(
        {
            "evidence_registry": [
                {
                    "source_title": "Accepted",
                    "source_url": "https://accepted.example/source",
                    "security_status": "accepted",
                },
                {
                    "source_title": "Quarantined",
                    "source_url": "https://quarantined.example/source",
                    "security_status": "quarantined",
                },
            ]
        },
        _config(),
        get_profile("default"),
    )

    assert [source.url for source in context.sources] == [
        "https://accepted.example/source"
    ]


def test_official_only_report_context_excludes_out_of_scope_handoff_text():
    contract = {
        "schema_version": 1,
        "original_query_sha256": "a" * 64,
        "requirements": [
            {
                "requirement_id": "COV-01",
                "text": (
                    "Based solely on the LangGraph official documentation, "
                    "official API reference, and official GitHub repository."
                ),
                "source_message_index": 0,
                "source_start": 0,
                "source_end": 100,
            }
        ],
        "advisory_dimensions": [],
    }
    context = ReportContext.from_state(
        {
            "coverage_contract": contract,
            "notes": [
                "MINTLIFY_FREE_TEXT_SECRET "
                "https://langchain-5e9cc07a.mintlify.app/oss/python/langgraph"
            ],
            "evidence_registry": [
                {
                    "evidence_id": "EV-DOCS",
                    "claim": "Official checkpoint claim.",
                    "supporting_excerpt": "Official checkpoint excerpt.",
                    "source_title": "LangGraph docs",
                    "source_url": (
                        "https://docs.langchain.com/oss/python/langgraph/"
                        "persistence"
                    ),
                    "security_status": "accepted",
                },
                {
                    "evidence_id": "EV-MINTLIFY",
                    "claim": "MINTLIFY_EVIDENCE_SECRET",
                    "source_title": "Temporary mirror",
                    "source_url": (
                        "https://langchain-5e9cc07a.mintlify.app/oss/python/"
                        "langgraph/persistence"
                    ),
                    "security_status": "accepted",
                },
                {
                    "evidence_id": "EV-ISSUE",
                    "claim": "ISSUE_EVIDENCE_SECRET",
                    "source_title": "Community issue",
                    "source_url": (
                        "https://github.com/langchain-ai/langgraph/issues/8405"
                    ),
                    "security_status": "accepted",
                },
            ],
        },
        _config(quality_evaluation_enabled=True),
        get_profile("default"),
    )

    assert [source.url for source in context.sources] == [
        "https://docs.langchain.com/oss/python/langgraph/persistence"
    ]
    assert "Official checkpoint claim." in context.findings
    assert "MINTLIFY_FREE_TEXT_SECRET" not in context.findings
    assert "MINTLIFY_EVIDENCE_SECRET" not in context.findings
    assert "ISSUE_EVIDENCE_SECRET" not in context.findings


@pytest.mark.asyncio
async def test_comparison_matrix_builds_multi_section_report(monkeypatch):
    monkeypatch.setenv("QUALITY_EVALUATION_ENABLED", "false")
    outline = ReportOutline(
        title="A vs B",
        sections=[
            SectionSpec(name="Overview"),
            SectionSpec(name="Comparison"),
            SectionSpec(name="Recommendation"),
        ],
    )

    async def fake_invoke(model, messages, config, *, span_name, agent_role=None, model_name=None, **_kw):
        if span_name == "lead.report_outline":
            return outline
        return AIMessage(content=f"[{span_name}] body")

    monkeypatch.setattr(
        assembly_module, "invoke_model_with_retry_observability", fake_invoke
    )
    _patch_init_chat_model(monkeypatch)

    state = {
        "messages": [],
        "research_brief": "compare A and B",
        "notes": ["finding about A", "finding about B"],
        "completed_task_outputs": [],
    }
    update = await build_report(state, _config(report_type="comparison_matrix"))

    md = update["final_report"]
    assert md.startswith("# A vs B\n")
    # Multiple ## sections (Introduction, TOC, 3 content sections, Conclusion)
    assert md.count("\n## ") >= 4
    assert "## Table of Contents" in md
    assert "## Overview" in md and "## Comparison" in md and "## Recommendation" in md
    assert "## Introduction" in md and "## Conclusion" in md
    # Content sections appear in outline order
    assert md.index("## Overview") < md.index("## Comparison") < md.index("## Recommendation")
    # messages carries the assembled markdown (no single writer AIMessage for sectioned)
    assert update["messages"][0].content == md


@pytest.mark.asyncio
async def test_sectioned_report_appends_sources_from_findings(monkeypatch):
    monkeypatch.setenv("QUALITY_EVALUATION_ENABLED", "false")
    outline = ReportOutline(title="T", sections=[SectionSpec(name="Findings")])

    async def fake_invoke(model, messages, config, *, span_name, agent_role=None, model_name=None, **_kw):
        if span_name == "lead.report_outline":
            return outline
        return AIMessage(content="section text [1]")

    monkeypatch.setattr(
        assembly_module, "invoke_model_with_retry_observability", fake_invoke
    )
    _patch_init_chat_model(monkeypatch)

    state = {
        "messages": [],
        "research_brief": "b",
        "notes": ["finding with [Source](https://x.io) inline"],
        "completed_task_outputs": [],
    }
    update = await build_report(state, _config(report_type="comparison_matrix"))

    # Sectioned assembly has no model-emitted Sources section, so the orchestrator
    # appends one (numbered by default) from the recovered sources.
    assert "### Sources" in update["final_report"]
    assert "https://x.io" in update["final_report"]
    assert update["sources"]["value"][0]["url"] == "https://x.io"


@pytest.mark.asyncio
async def test_quality_enabled_report_rejects_state_without_accepted_evidence(monkeypatch):
    monkeypatch.setenv("QUALITY_EVALUATION_ENABLED", "true")
    async def fake_assemble(_ctx):
        return AssemblyResult(body_markdown="unsupported report")

    monkeypatch.setattr("open_deep_research.report.orchestrator.assemble", fake_assemble)
    state = {
        "messages": [],
        "research_brief": "research A",
        "notes": ["rejected_by_supervisor_quality_gate"],
        "raw_notes": [],
        "completed_task_outputs": [],
        "evidence_registry": [],
    }

    with pytest.raises(RuntimeError, match="accepted research evidence"):
        await build_report(state, _config(quality_evaluation_enabled=True))


@pytest.mark.asyncio
async def test_quality_enabled_report_rejects_quarantined_only_evidence(monkeypatch):
    monkeypatch.setenv("QUALITY_EVALUATION_ENABLED", "true")

    async def fake_assemble(_ctx):
        return AssemblyResult(body_markdown="must not be generated")

    monkeypatch.setattr("open_deep_research.report.orchestrator.assemble", fake_assemble)
    state = {
        "messages": [],
        "research_brief": "research A",
        "notes": ["untrusted finding https://quarantined.example/source"],
        "raw_notes": ["untrusted finding https://quarantined.example/source"],
        "completed_task_outputs": [],
        "supervisor_messages": [
            {"type": "tool", "name": "ConductResearch", "content": "done"}
        ],
        "evidence_registry": [
            {
                "source_url": "https://quarantined.example/source",
                "security_status": "quarantined",
            }
        ],
    }

    with pytest.raises(RuntimeError, match="accepted research evidence"):
        await build_report(state, _config(quality_evaluation_enabled=True))


@pytest.mark.asyncio
async def test_official_only_report_does_not_treat_free_text_as_evidence(
    monkeypatch,
):
    async def fake_assemble(_ctx):
        return AssemblyResult(body_markdown="must not be generated")

    monkeypatch.setattr(
        "open_deep_research.report.orchestrator.assemble",
        fake_assemble,
    )
    state = {
        "messages": [],
        "research_brief": "research A",
        "notes": ["unverified official-looking free text"],
        "raw_notes": ["https://docs.langchain.com/unverified"],
        "completed_task_outputs": [],
        "supervisor_messages": [
            {"type": "tool", "name": "ConductResearch", "content": "done"}
        ],
        "evidence_registry": [],
        "coverage_contract": {
            "schema_version": 1,
            "original_query_sha256": "a" * 64,
            "requirements": [
                {
                    "requirement_id": "COV-01",
                    "text": (
                        "Based solely on the official documentation and "
                        "official GitHub repository."
                    ),
                    "source_message_index": 0,
                    "source_start": 0,
                    "source_end": 64,
                }
            ],
            "advisory_dimensions": [],
        },
    }

    with pytest.raises(RuntimeError, match="accepted research evidence"):
        await build_report(
            state,
            _config(quality_evaluation_enabled=True),
        )


@pytest.mark.asyncio
async def test_caveat_admission_forces_limitations_section(monkeypatch):
    async def fake_assemble(_ctx):
        return AssemblyResult(body_markdown="# Report\n\nSupported result [1].")

    monkeypatch.setattr(
        "open_deep_research.report.orchestrator.assemble",
        fake_assemble,
    )
    state = {
        "messages": [],
        "research_brief": "research A",
        "notes": ["supported finding"],
        "completed_task_outputs": [],
        "evidence_registry": [{
            "evidence_id": "EV-01",
            "claim": "Supported result.",
            "source_url": "https://example.com/evidence",
            "source_title": "Evidence",
            "security_status": "accepted",
        }],
        "handoff_assessments": [{
            "admission_status": "accepted_with_caveats",
            "accepted": True,
            "caveats": ["The precise package version was not confirmed."],
        }],
    }

    update = await build_report(
        state,
        _config(quality_evaluation_enabled=True),
    )

    assert "## 限制与不确定性" in update["final_report"]
    assert "precise package version" in update["final_report"]


@pytest.mark.asyncio
async def test_quality_enabled_report_rewrites_sources_to_evidence_allowlist(monkeypatch):
    monkeypatch.setenv("QUALITY_EVALUATION_ENABLED", "true")
    async def fake_assemble(_ctx):
        return AssemblyResult(
            body_markdown=(
                "Claim [1] [Allowed](https://allowed.example/paper) "
                "[Fabricated](https://fabricated.example/post)\n\n"
                "### Sources\n[1] Fabricated: https://fabricated.example/post"
            )
        )

    monkeypatch.setattr("open_deep_research.report.orchestrator.assemble", fake_assemble)
    state = {
        "messages": [],
        "research_brief": "research A",
        "notes": ["supported finding"],
        "raw_notes": ["supported finding https://allowed.example/paper"],
        "evidence_registry": [
            {
                "source_title": "Allowed primary source",
                "source_url": "https://allowed.example/paper",
            }
        ],
    }

    update = await build_report(state, _config(quality_evaluation_enabled=True))

    assert "https://allowed.example/paper" in update["final_report"]
    assert "https://fabricated.example/post" not in update["final_report"]
    assert "[1] Allowed primary source" in update["final_report"]


@pytest.mark.asyncio
async def test_quality_report_without_body_citations_uses_evidence_limited_fallback(
    monkeypatch,
):
    async def fake_assemble(_ctx):
        return AssemblyResult(
            body_markdown="# Unsupported normal draft\n\nA factual claim without a citation."
        )

    captured: dict[str, Any] = {}

    async def fake_evidence_limited(records, **kwargs):
        captured["records"] = records
        captured.update(kwargs)
        return (
            "# Evidence-limited partial report\n\n"
            "- Supported claim [EV-01]\n\n"
            "## 来源\n\n"
            "- [EV-01] [Official source](https://allowed.example/paper)"
        )

    monkeypatch.setattr(
        "open_deep_research.report.orchestrator.assemble",
        fake_assemble,
    )
    monkeypatch.setattr(
        "open_deep_research.report.orchestrator.build_evidence_limited_report",
        fake_evidence_limited,
        raising=False,
    )
    async def leave_uncited(markdown, _ctx):
        return markdown

    monkeypatch.setattr(
        "open_deep_research.report.orchestrator._repair_missing_report_citations",
        leave_uncited,
    )
    state = {
        "messages": [],
        "research_brief": "research A",
        "notes": ["UNTRUSTED_FREE_TEXT_HANDOFF"],
        "raw_notes": ["UNTRUSTED_RAW_NOTE"],
        "evidence_registry": [
            {
                "evidence_id": "EV-01",
                "claim": "Supported claim.",
                "supporting_excerpt": "Supported claim.",
                "source_title": "Official source",
                "source_url": "https://allowed.example/paper",
                "security_status": "accepted",
            }
        ],
        "coverage_contract": {
            "schema_version": 1,
            "original_query_sha256": "a" * 64,
            "requirements": [
                {
                    "requirement_id": "req-001",
                    "text": "Explain A.",
                    "source_message_index": 0,
                    "source_start": 0,
                    "source_end": 10,
                }
            ],
            "advisory_dimensions": [],
        },
        "coverage_ledger": {
            "req-001": {
                "status": "supported",
                "evidence_ids": ["EV-01"],
                "task_ids": ["task-1"],
                "caveats": [],
            }
        },
    }

    update = await build_report(
        state,
        _config(quality_evaluation_enabled=True),
    )

    assert update["final_report"].startswith("# Evidence-limited partial report")
    assert "UNTRUSTED_FREE_TEXT_HANDOFF" not in update["final_report"]
    assert [
        record["evidence_id"] for record in captured["records"]
    ] == ["EV-01"]
    assert captured["records"][0]["source_url"] == (
        "https://allowed.example/paper"
    )
    assert captured["records"][0]["source_scope_status"] == "not_constrained"
    assert update["completion_decision"]["value"]["action"] == "complete_partial"
    assert "report_missing_verifiable_citations" in (
        update["quality_gate"]["reason_codes"]
    )


@pytest.mark.asyncio
async def test_quality_report_repairs_missing_body_citations(monkeypatch):
    async def fake_assemble(_ctx):
        return AssemblyResult(
            body_markdown="# Draft\n\nA supported claim without a citation."
        )

    async def fake_repair(_markdown, _ctx):
        return "# Draft\n\nA supported claim [Official](https://allowed.example/paper)."

    monkeypatch.setattr(
        "open_deep_research.report.orchestrator.assemble",
        fake_assemble,
    )
    monkeypatch.setattr(
        "open_deep_research.report.orchestrator._repair_missing_report_citations",
        fake_repair,
    )
    state = {
        "messages": [],
        "research_brief": "research A",
        "notes": [],
        "evidence_registry": [{
            "evidence_id": "EV-01",
            "claim": "A supported claim.",
            "supporting_excerpt": "A supported claim.",
            "source_title": "Official",
            "source_url": "https://allowed.example/paper",
            "security_status": "accepted",
        }],
    }

    update = await build_report(
        state,
        _config(quality_evaluation_enabled=True),
    )

    assert "A supported claim [Official]" in update["final_report"]
    assert "quality_gate" not in update


@pytest.mark.asyncio
async def test_quality_report_accepts_allowlisted_trailing_slash_variant(monkeypatch):
    async def fake_assemble(_ctx):
        return AssemblyResult(
            body_markdown=(
                "Supported claim [Official](https://allowed.example/paper/)."
            )
        )

    monkeypatch.setattr(
        "open_deep_research.report.orchestrator.assemble",
        fake_assemble,
    )
    state = {
        "messages": [],
        "research_brief": "research A",
        "notes": [],
        "evidence_registry": [{
            "source_title": "Official",
            "source_url": "https://allowed.example/paper",
            "security_status": "accepted",
        }],
    }

    update = await build_report(
        state,
        _config(quality_evaluation_enabled=True),
    )

    assert "https://allowed.example/paper" in update["final_report"]


@pytest.mark.asyncio
async def test_disallowed_url_inside_fenced_code_fails_closed_to_safe_fallback(
    monkeypatch,
):
    async def fake_assemble(_ctx):
        return AssemblyResult(
            body_markdown=(
                "Supported claim [1].\n\n"
                "```python\n"
                'endpoint = "https://fabricated.example/api"\n'
                "```\n"
            )
        )

    fallback_calls = 0

    async def fake_evidence_limited(_records, **_kwargs):
        nonlocal fallback_calls
        fallback_calls += 1
        return (
            "# Safe partial report\n\n"
            "- Supported claim [EV-01]\n\n"
            "## 来源\n\n"
            "- [EV-01] [Official source](https://allowed.example/paper)"
        )

    monkeypatch.setattr(
        "open_deep_research.report.orchestrator.assemble",
        fake_assemble,
    )
    monkeypatch.setattr(
        "open_deep_research.report.orchestrator.build_evidence_limited_report",
        fake_evidence_limited,
        raising=False,
    )
    state = {
        "messages": [],
        "research_brief": "research A",
        "notes": ["supported finding"],
        "raw_notes": ["supported finding https://allowed.example/paper"],
        "evidence_registry": [
            {
                "evidence_id": "EV-01",
                "claim": "Supported claim.",
                "supporting_excerpt": "Supported claim.",
                "source_title": "Official source",
                "source_url": "https://allowed.example/paper",
                "security_status": "accepted",
            }
        ],
    }

    update = await build_report(
        state,
        _config(quality_evaluation_enabled=True),
    )

    assert fallback_calls == 1
    assert "https://fabricated.example/api" not in update["final_report"]
    assert 'endpoint = ""' not in update["final_report"]


@pytest.mark.asyncio
async def test_report_sanitizer_preserves_fenced_code_and_cleans_prose(
    monkeypatch,
):
    async def fake_assemble(_ctx):
        return AssemblyResult(
            body_markdown=(
                "Supported claim [1].\n\n"
                "<script>alert('outside')</script>\n\n"
                "```html\n"
                "<script>const scheme = 'javascript:example'</script>\n"
                "```\n"
            )
        )

    monkeypatch.setattr(
        "open_deep_research.report.orchestrator.assemble",
        fake_assemble,
    )
    state = {
        "messages": [],
        "research_brief": "research A",
        "notes": ["supported finding"],
        "evidence_registry": [
            {
                "evidence_id": "EV-01",
                "claim": "Supported claim.",
                "supporting_excerpt": "Supported claim.",
                "source_title": "Official source",
                "source_url": "https://allowed.example/paper",
                "security_status": "accepted",
            }
        ],
    }

    update = await build_report(
        state,
        _config(quality_evaluation_enabled=True),
    )

    assert "alert('outside')" not in update["final_report"]
    assert (
        "<script>const scheme = 'javascript:example'</script>"
        in update["final_report"]
    )


@pytest.mark.asyncio
async def test_enforced_report_uses_evidence_allowlist_when_quality_gate_is_disabled(
    monkeypatch,
):
    monkeypatch.setenv("QUALITY_EVALUATION_ENABLED", "false")
    monkeypatch.setenv("WEB_PIPELINE_MODE", "enforced")

    async def fake_assemble(_ctx):
        return AssemblyResult(
            body_markdown=(
                "[Allowed](https://allowed.example/paper) "
                "[Fabricated](https://fabricated.example/post) "
                "bare https://unknown.example/post [99]"
            )
        )

    monkeypatch.setattr("open_deep_research.report.orchestrator.assemble", fake_assemble)
    state = {
        "messages": [],
        "research_brief": "research A",
        "notes": ["supported finding"],
        "raw_notes": ["supported finding https://allowed.example/paper"],
        "evidence_registry": [
            {
                "source_title": "Allowed primary source",
                "source_url": "https://allowed.example/paper",
            }
        ],
    }

    update = await build_report(
        state,
        _config(
            quality_evaluation_enabled=False,
            web_pipeline_mode="enforced",
        ),
    )

    assert "https://allowed.example/paper" in update["final_report"]
    assert "https://fabricated.example/post" not in update["final_report"]
    assert "https://unknown.example/post" not in update["final_report"]
    assert "[99]" not in update["final_report"]


@pytest.mark.asyncio
async def test_enforced_report_removes_all_citations_without_accepted_evidence(
    monkeypatch,
):
    monkeypatch.setenv("QUALITY_EVALUATION_ENABLED", "false")
    monkeypatch.setenv("WEB_PIPELINE_MODE", "enforced")

    async def fake_assemble(_ctx):
        return AssemblyResult(
            body_markdown=(
                "Unsupported [source](https://unknown.example/post), "
                "bare https://other.example/page and numeric [1]."
            )
        )

    monkeypatch.setattr("open_deep_research.report.orchestrator.assemble", fake_assemble)
    state = {
        "messages": [],
        "research_brief": "research A",
        "notes": ["no accepted evidence"],
        "raw_notes": [],
        "evidence_registry": [],
    }

    update = await build_report(
        state,
        _config(quality_evaluation_enabled=False, web_pipeline_mode="enforced"),
    )

    assert "https://" not in update["final_report"]
    assert "[1]" not in update["final_report"]
