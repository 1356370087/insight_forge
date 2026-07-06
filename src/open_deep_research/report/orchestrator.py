"""Single entry point for report generation.

:func:`build_report` replaces the body of the original ``final_report_generation``
node in ``agents/deep_researcher.py``. It dispatches by report profile (a
registry lookup, not an if/elif chain) and renders the requested output format.
The ``default`` report type reproduces the pre-refactor single-call synthesis
byte-for-byte.

Contract preserved for backward compatibility:

* ``final_report`` is always a markdown string.
* ``messages`` carries the writer AIMessage (or an error AIMessage on failure).
* ``notes`` and ``completed_task_outputs`` are cleared via the override reducer,
  exactly as before.
* For the default profile (markdown, one-shot) no extra state/SSE keys are
  produced — the output is identical to the original node.
"""

from __future__ import annotations

from typing import Optional

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from open_deep_research.configuration import Configuration

from .assembly import ReportContext, assemble
from .profiles import (
    AssemblyMode,
    OutputFormat,
    ReferenceStyle,
    ReportProfile,
    get_profile,
)
from .references import (
    parse_sources_from_text,
    render_references,
    replace_sources_section,
)
from .renderers import render_artifacts


def _cleared_state() -> dict:
    """Return the notes/task-outputs override used by the original node."""
    return {
        "notes": {"type": "override", "value": []},
        "completed_task_outputs": {"type": "override", "value": []},
    }


def _resolve_output_format(cfg: Configuration, profile: ReportProfile) -> OutputFormat:
    """Resolve the output format from config, falling back to the profile default."""
    raw: Optional[str] = getattr(cfg, "output_format", None)
    if raw:
        try:
            return OutputFormat(raw)
        except ValueError:
            return profile.default_format
    return profile.default_format


def _resolve_reference_style(cfg: Configuration, profile: ReportProfile) -> ReferenceStyle:
    """Resolve the reference style from config, falling back to the profile default."""
    raw: Optional[str] = getattr(cfg, "reference_style", None)
    if raw:
        try:
            return ReferenceStyle(raw)
        except ValueError:
            return profile.reference_style
    return profile.reference_style


async def build_report(state: dict, config: RunnableConfig) -> dict:
    """Build the final report product from collected research state.

    Args:
        state: Agent state containing ``notes`` / ``completed_task_outputs`` /
            ``research_brief`` / ``messages`` (a plain dict mutated in place by
            the orchestrator upstream).
        config: The runnable config carrying ``configurable`` (report_type,
            output_format, reference_style, model settings, ...).

    Returns:
        The state update dict (``final_report``, ``messages``, cleared notes),
        plus ``report_artifacts`` only when a non-markdown format is produced,
        and ``sources`` only when non-default reference handling runs.
    """
    cfg = Configuration.from_runnable_config(config)
    profile = get_profile(getattr(cfg, "report_type", None))
    fmt = _resolve_output_format(cfg, profile)
    ref_style = _resolve_reference_style(cfg, profile)

    ctx = ReportContext.from_state(state, config, profile)
    result = await assemble(ctx)

    # Recover structured sources whenever a non-default output format, a
    # non-numbered reference style, or sectioned assembly needs them. The pure
    # default (one-shot markdown + numbered) skips this entirely, so its output
    # stays byte-identical to the original node.
    is_sectioned = profile.assembly == AssemblyMode.SECTIONED
    needs_sources = (
        (fmt != OutputFormat.MARKDOWN)
        or (ref_style != ReferenceStyle.NUMBERED)
        or is_sectioned
    )
    if needs_sources and not result.sources:
        result.sources = parse_sources_from_text(result.body_markdown + "\n" + ctx.findings)

    markdown = result.body_markdown
    if result.sources and (is_sectioned or ref_style == ReferenceStyle.BIBTEX_LIKE):
        # Sectioned bodies have no Sources section -> append one in the resolved
        # style. One-shot bibtex -> replace the model-emitted Sources section.
        # One-shot numbered is left untouched (byte-identical to the original).
        markdown = replace_sources_section(
            markdown, render_references(result.sources, ref_style)
        )

    # Render every artifact from the same canonical markdown exposed through
    # ``final_report`` so reference rewriting cannot create divergent payloads.
    result.body_markdown = markdown
    artifacts = render_artifacts(result, fmt, ctx)

    update: dict = {
        "final_report": markdown,
        "messages": (
            [result.message] if result.message is not None else [AIMessage(content=markdown)]
        ),
        **_cleared_state(),
    }

    if needs_sources and result.sources:
        update["sources"] = {
            "type": "override",
            "value": [s.model_dump() for s in result.sources],
        }

    # Only surface non-default artifacts. The markdown body already lives in
    # ``final_report`` (and the SSE ``result``), so for the default markdown
    # profile we add nothing — keeping the output byte-identical to before.
    if fmt != OutputFormat.MARKDOWN:
        update["report_artifacts"] = {"format": fmt.value, **artifacts}

    return update
