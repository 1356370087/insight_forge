"""Tests for sectioned report assembly (Phase 4)."""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from open_deep_research.report import assembly as assembly_module
from open_deep_research.report import build_report
from open_deep_research.report.assembly import AssemblyResult, SectionedStrategy
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
        return AIMessage(content="section text")

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
