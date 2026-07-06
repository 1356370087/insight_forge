"""Output-format renderers — registry-based (no if/elif).

Each renderer takes an :class:`AssemblyResult` (plus the :class:`ReportContext`)
and returns a dict of named artifacts (e.g. ``{"markdown": ...}``). Phase 1
shipped only the markdown renderer; Phase 3 adds the structured-json renderer.
Slides / one-pager renderers are added in later phases by registering them in
:data:`RENDERERS`.

The structured-json renderer is deterministic: it derives a
:class:`StructuredReport` from the already-assembled markdown body + recovered
sources, so it costs no extra LLM call and stays a pure function.
"""

from __future__ import annotations

import re
from typing import Callable, Dict

from .assembly import AssemblyResult, ReportContext
from .models import SourceRef, StructuredReport, WrittenSection
from .profiles import OutputFormat


def _render_markdown(result: AssemblyResult, ctx: ReportContext) -> dict:
    """Render the assembled Markdown body."""
    return {"markdown": result.body_markdown}


_H1_RE = re.compile(r"^#\s+(.+)$", re.MULTILINE)
# Splits on "## " headings (start of a line).
_H2_SPLIT_RE = re.compile(r"\n(?=##\s+)")


def _markdown_to_structured(
    body: str, sources, research_brief: str = ""
) -> StructuredReport:
    """Derive a StructuredReport from a markdown body + sources (heuristic).

    Splits the body into ``## `` sections, uses the first ``# `` heading (or the
    research brief) as the title, the preamble before the first section as the
    summary, and drops the trailing ``### Sources`` section from section parsing.
    """
    # Drop the trailing Sources section so it isn't parsed as a content section.
    body_for_sections = re.split(r"\n###\s*Sources\b", body, maxsplit=1)[0]

    title_match = _H1_RE.search(body_for_sections)
    title = (
        title_match.group(1).strip()
        if title_match
        else (research_brief.strip() or "Research Report")
    )
    # Remove the H1 title line so it doesn't pollute the summary preamble.
    if title_match:
        body_for_sections = (
            body_for_sections[: title_match.start()] + body_for_sections[title_match.end() :]
        )

    sections = []
    summary = ""
    for part in _H2_SPLIT_RE.split(body_for_sections):
        part = part.strip()
        if not part:
            continue
        if part.startswith("## "):
            first_line, _, rest = part.partition("\n")
            name = first_line[3:].strip()
            sections.append(WrittenSection(name=name, content=rest.strip()))
        elif not summary:
            # Preamble before the first ## section → use as the summary.
            summary = part

    if not summary and sections:
        summary = sections[0].content[:300]

    source_refs = [
        s if isinstance(s, SourceRef) else SourceRef(**s) if isinstance(s, dict) else SourceRef(url=str(s))
        for s in (sources or [])
    ]

    return StructuredReport(
        title=title,
        summary=summary.strip(),
        sections=sections,
        sources=source_refs,
        key_findings=[],
    )


def _render_structured_json(result: AssemblyResult, ctx: ReportContext) -> dict:
    """Render a machine-readable report alongside the Markdown body."""
    structured = _markdown_to_structured(
        result.body_markdown,
        result.sources,
        ctx.state.get("research_brief", ""),
    )
    schema = ctx.profile.structured_schema or StructuredReport
    structured = schema.model_validate(structured.model_dump())
    return {"markdown": result.body_markdown, "structured_json": structured.model_dump()}


def _to_bullets(text: str, limit: int = 5) -> list:
    """Heuristically turn a block of text into <= `limit` bullet strings."""
    if not text:
        return []
    lines = [ln.strip().lstrip("-*").strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) > 1:
        bullets = lines
    else:
        bullets = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    return [b for b in bullets if b][:limit]


def _render_slides(result: AssemblyResult, ctx: ReportContext) -> dict:
    """Render one slide per section with heuristic bullets."""
    structured = _markdown_to_structured(
        result.body_markdown, result.sources, ctx.state.get("research_brief", "")
    )
    slides = [{"title": structured.title, "bullets": _to_bullets(structured.summary)}]
    slides.extend(
        {"title": s.name, "bullets": _to_bullets(s.content)} for s in structured.sections
    )
    return {"markdown": result.body_markdown, "slides": slides}


def _render_one_pager(result: AssemblyResult, ctx: ReportContext) -> dict:
    """Render a condensed view with title, summary, sections, and source count."""
    structured = _markdown_to_structured(
        result.body_markdown, result.sources, ctx.state.get("research_brief", "")
    )
    parts = [f"# {structured.title}"]
    if structured.summary:
        parts.append(structured.summary)
    if structured.sections:
        parts.append("**Sections:** " + ", ".join(s.name for s in structured.sections[:5]))
    parts.append(f"*{len(structured.sources)} source(s)*")
    return {"markdown": result.body_markdown, "one_pager": "\n\n".join(parts)}


RENDERERS: Dict[OutputFormat, Callable[[AssemblyResult, ReportContext], dict]] = {
    OutputFormat.MARKDOWN: _render_markdown,
    OutputFormat.STRUCTURED_JSON: _render_structured_json,
    OutputFormat.SLIDES: _render_slides,
    OutputFormat.ONE_PAGER: _render_one_pager,
}


def render_artifacts(
    result: AssemblyResult, fmt: OutputFormat, ctx: ReportContext
) -> dict:
    """Render ``result`` for ``fmt``, falling back to markdown when unsupported."""
    renderer = RENDERERS.get(fmt, RENDERERS[OutputFormat.MARKDOWN])
    return renderer(result, ctx)
