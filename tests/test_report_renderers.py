"""Unit tests for output-format renderers (Phase 3: structured_json)."""

from __future__ import annotations

from dataclasses import replace

from pydantic import BaseModel

from open_deep_research.report.assembly import AssemblyResult, ReportContext
from open_deep_research.report.models import SourceRef, StructuredReport
from open_deep_research.report.profiles import OutputFormat, get_profile
from open_deep_research.report.renderers import (
    _markdown_to_structured,
    render_artifacts,
)


def _ctx(profile_key: str = "default", research_brief: str = "brief") -> ReportContext:
    config = {"configurable": {}, "metadata": {"run_id": "renderer-test"}}
    state = {"messages": [], "notes": [], "research_brief": research_brief}
    return ReportContext.from_state(state, config, get_profile(profile_key))


def test_markdown_to_structured_splits_h2_sections():
    body = (
        "# My Title\n\nIntro paragraph here.\n\n"
        "## Section A\n\nContent A.\n\n"
        "## Section B\n\nContent B.\n\n"
        "### Sources\n[1] X: https://x.io\n"
    )
    rep = _markdown_to_structured(body, [], research_brief="brief")
    assert isinstance(rep, StructuredReport)
    assert rep.title == "My Title"
    assert rep.summary == "Intro paragraph here."
    assert [s.name for s in rep.sections] == ["Section A", "Section B"]
    assert rep.sections[0].content == "Content A."
    # The Sources section must not appear as a content section.
    assert all("Sources" != s.name for s in rep.sections)


def test_markdown_to_structured_title_falls_back_to_brief():
    body = "## Only section\n\ntext"
    rep = _markdown_to_structured(body, [], research_brief="the brief")
    assert rep.title == "the brief"


def test_markdown_to_structured_includes_sources():
    rep = _markdown_to_structured(
        "# T\n\n## S\n\nc",
        [SourceRef(title="X", url="https://x.io")],
    )
    assert rep.sources[0].url == "https://x.io"


def test_structured_json_renderer_output_shape():
    body = "# T\n\nintro\n\n## S1\n\nc1"
    result = AssemblyResult(body_markdown=body)
    out = render_artifacts(result, OutputFormat.STRUCTURED_JSON, _ctx())
    assert out["markdown"] == body
    assert "structured_json" in out
    # Round-trips through the StructuredReport schema (machine-readable).
    validated = StructuredReport(**out["structured_json"])
    assert validated.title == "T"
    assert validated.sections[0].name == "S1"


def test_slides_renderer_produces_deck():
    body = (
        "# Deck Title\n\nsummary line one. summary line two.\n\n"
        "## Section A\n\npoint one. point two.\n\n"
        "## Section B\n\nalpha"
    )
    result = AssemblyResult(body_markdown=body)
    out = render_artifacts(result, OutputFormat.SLIDES, _ctx())
    assert out["markdown"] == body
    slides = out["slides"]
    # title slide + one slide per section
    assert slides[0]["title"] == "Deck Title"
    assert isinstance(slides[0]["bullets"], list)
    assert len(slides) == 3
    assert slides[1]["title"] == "Section A"
    assert slides[2]["title"] == "Section B"


def test_one_pager_renderer_condenses():
    body = "# T\n\nsum.\n\n## S1\n\nc\n\n## S2\n\nc2"
    result = AssemblyResult(body_markdown=body)
    out = render_artifacts(result, OutputFormat.ONE_PAGER, _ctx())
    assert "one_pager" in out
    assert out["one_pager"].startswith("# T")
    assert "S1" in out["one_pager"] and "S2" in out["one_pager"]


def test_structured_json_renderer_consumes_profile_schema():
    class CustomStructuredReport(BaseModel):
        title: str
        summary: str
        sections: list
        key_findings: list
        sources: list
        schema_marker: str = "custom-schema"

    ctx = _ctx("faq")
    ctx.profile = replace(ctx.profile, structured_schema=CustomStructuredReport)
    result = AssemblyResult(body_markdown="# T\n\nintro\n\n## S\n\nbody")

    out = render_artifacts(result, OutputFormat.STRUCTURED_JSON, ctx)

    assert out["structured_json"]["schema_marker"] == "custom-schema"
