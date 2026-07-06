"""Report product system for Open Deep Research.

A registry-based dispatch that turns collected research findings into a report
*product form* selected by configuration: report type (genre), output format,
assembly mode, and reference style. Domain skills plug in later to provide
domain-specific context orchestration.

The single entry point is :func:`build_report`, which replaces the body of the
original ``final_report_generation`` node. The ``default`` report type
reproduces the pre-refactor single-call synthesis byte-for-byte; all other
product forms are opt-in via ``Configuration``.
"""

from open_deep_research.report.orchestrator import build_report

__all__ = ["build_report"]
