"""Pydantic schemas for structured report outputs and section planning.

These models are used for LLM structured-output calls (``.with_structured_output``)
and as the typed shape of assembled reports. They live here rather than in
``state.py`` to avoid import cycles between the ``agents/`` package and the
``report/`` package, while mirroring the structured-output convention already
established in ``state.py`` (e.g. ``Summary``, ``ClarifyWithUser``).
"""

from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


class SourceRef(BaseModel):
    """A single source reference (title + url)."""

    title: str = ""
    url: str


class SectionSpec(BaseModel):
    """One entry in an LLM-generated report outline (sectioned assembly)."""

    name: str
    description: str = ""
    needs_research: bool = True


class ReportOutline(BaseModel):
    """Structured output from the outline-planning call (sectioned assembly)."""

    title: str
    sections: List[SectionSpec]


class WrittenSection(BaseModel):
    """A fully written report section."""

    name: str
    content: str


class StructuredReport(BaseModel):
    """Schema for the deterministic ``output_format=structured_json`` artifact."""

    title: str
    summary: str
    sections: List[WrittenSection]
    key_findings: List[str] = Field(default_factory=list)
    sources: List[SourceRef] = Field(default_factory=list)
