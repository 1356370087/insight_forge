"""Report product profiles — a registry-based dispatch (no if/elif chains).

A :class:`ReportProfile` bundles everything that distinguishes one report
*product form* from another: the prompt template, the assembly mode, the
default output format, the reference style, and an optional structured-output
schema. Adding a new report genre = adding one entry to :data:`REPORT_PROFILES`.

This is deliberately cleaner than gpt-researcher's scattered three-layer
``if/elif`` dispatch: there is exactly one lookup (``get_profile``) and every
extensible axis (genre / assembly mode / output format / reference style) is an
enum consumed by a small dispatch table.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple, Type

from .models import StructuredReport


class AssemblyMode(str, Enum):
    """How a report is assembled from findings."""

    ONE_SHOT = "one_shot"  # current behavior: a single LLM synthesis call
    SECTIONED = "sectioned"  # outline -> per-section writes -> assemble


class OutputFormat(str, Enum):
    """Render format of the report deliverable."""

    MARKDOWN = "markdown"
    STRUCTURED_JSON = "structured_json"
    SLIDES = "slides"
    ONE_PAGER = "one_pager"


class ReferenceStyle(str, Enum):
    """Citation/reference rendering style for the Sources section."""

    NUMBERED = "numbered"  # [1] Title: url (model emits this by default)
    BIBTEX_LIKE = "bibtex_like"  # @misc{key, title=..., howpublished=url}


@dataclass(frozen=True)
class ReportProfile:
    """A report product form.

    Attributes:
        key: Stable identifier matching the ``report_type`` config value.
        prompt_template: Name of the module-level prompt constant in
            ``open_deep_research.prompts`` to ``.format()`` for one-shot genres.
        assembly: How findings are turned into a report body.
        default_format: Default output format when the caller doesn't override.
        reference_style: How the Sources section is rendered.
        structured_schema: Optional Pydantic model used to validate and shape
            the deterministic ``structured_json`` artifact.
        section_skeleton: Optional ordered section-name hints injected into the
            outline-planner prompt for sectioned genres.
    """

    key: str
    prompt_template: str
    assembly: AssemblyMode
    default_format: OutputFormat
    reference_style: ReferenceStyle
    structured_schema: Optional[Type] = None
    section_skeleton: Tuple[str, ...] = field(default_factory=tuple)


# Phase 1 ships only the ``default`` profile so that the refactored
# ``final_report_generation`` is byte-for-byte identical to the original
# single-call synthesis. Additional genres are registered in later phases.
REPORT_PROFILES = {
    "default": ReportProfile(
        key="default",
        prompt_template="final_report_generation_prompt",
        assembly=AssemblyMode.ONE_SHOT,
        default_format=OutputFormat.MARKDOWN,
        reference_style=ReferenceStyle.NUMBERED,
    ),
    "executive_summary": ReportProfile(
        key="executive_summary",
        prompt_template="executive_summary_prompt",
        assembly=AssemblyMode.ONE_SHOT,
        default_format=OutputFormat.MARKDOWN,
        reference_style=ReferenceStyle.NUMBERED,
    ),
    "decision_brief": ReportProfile(
        key="decision_brief",
        prompt_template="decision_brief_prompt",
        assembly=AssemblyMode.ONE_SHOT,
        default_format=OutputFormat.MARKDOWN,
        reference_style=ReferenceStyle.NUMBERED,
        section_skeleton=(
            "Recommendation",
            "Rationale",
            "Alternatives Considered",
            "Risks",
            "Next Actions",
        ),
    ),
    "faq": ReportProfile(
        key="faq",
        prompt_template="faq_prompt",
        assembly=AssemblyMode.ONE_SHOT,
        default_format=OutputFormat.STRUCTURED_JSON,
        reference_style=ReferenceStyle.NUMBERED,
        structured_schema=StructuredReport,
    ),
    "comparison_matrix": ReportProfile(
        key="comparison_matrix",
        prompt_template="section_writer_prompt",
        assembly=AssemblyMode.SECTIONED,
        default_format=OutputFormat.MARKDOWN,
        reference_style=ReferenceStyle.NUMBERED,
        section_skeleton=(
            "Overview",
            "Comparison Criteria",
            "Side-by-Side Comparison",
            "Analysis",
            "Recommendation",
        ),
    ),
    "pros_cons": ReportProfile(
        key="pros_cons",
        prompt_template="section_writer_prompt",
        assembly=AssemblyMode.SECTIONED,
        default_format=OutputFormat.MARKDOWN,
        reference_style=ReferenceStyle.NUMBERED,
        section_skeleton=("Overview", "Pros", "Cons", "Verdict"),
    ),
    "literature_review": ReportProfile(
        key="literature_review",
        prompt_template="section_writer_prompt",
        assembly=AssemblyMode.SECTIONED,
        default_format=OutputFormat.MARKDOWN,
        reference_style=ReferenceStyle.BIBTEX_LIKE,
        section_skeleton=(
            "Background",
            "Key Themes",
            "Methodological Notes",
            "Gaps and Future Work",
        ),
    ),
}


def get_profile(key: Optional[str]) -> ReportProfile:
    """Return the profile for ``key``, falling back to ``default`` when unknown.

    Never raises: an unknown / empty ``report_type`` silently uses ``default``,
    preserving backward compatibility.
    """
    return REPORT_PROFILES.get(key or "default", REPORT_PROFILES["default"])
