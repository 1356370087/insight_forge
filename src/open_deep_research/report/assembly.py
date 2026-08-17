"""Report assembly strategies.

Phase 1 ships :class:`OneShotStrategy`, a faithful, verbatim port of the
original single-call final-report synthesis that lived in
``agents/deep_researcher.py:final_report_generation``. The sectioned assembly
strategy lands in a later phase; :func:`assemble` dispatches by the profile's
:class:`AssemblyMode`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Protocol, cast

from langchain.chat_models import init_chat_model
from langchain_core.messages import BaseMessage, HumanMessage, get_buffer_string
from langchain_core.runnables import RunnableConfig

from open_deep_research import prompts as _prompts
from open_deep_research.agents.model_recovery import (
    invoke_with_output_recovery,
    resolve_model_max_output_tokens,
)
from open_deep_research.configuration import Configuration
from open_deep_research.evidence import (
    contract_has_source_constraints,
    source_scoped_evidence_records,
)
from open_deep_research.model_errors import is_token_limit_exceeded
from open_deep_research.model_fallback import invoke_with_model_fallback
from open_deep_research.model_resolution import (
    build_model_config,
    get_configurable_model_template,
)
from open_deep_research.observability import (
    apply_helicone_config,
    get_trace_recorder,
    invoke_model_with_retry_observability,
)
from open_deep_research.prompts import (
    final_section_writer_prompt,
    report_outline_planner_prompt,
    section_writer_prompt,
)
from open_deep_research.skills import get_skill_report_context
from open_deep_research.tools.legacy_shims import get_today_str
from open_deep_research.tools.model_limits import get_model_token_limit

from .coverage import derive_state_coverage_checklist, render_coverage_checklist
from .models import ReportOutline, SectionSpec, SourceRef, WrittenSection
from .profiles import AssemblyMode, ReportProfile

_writer_model_template = get_configurable_model_template()


def _format_conversation_summary(summary: Optional[str]) -> str:
    """Format a running conversation summary as advisory short-term context.

    Mirrors the helper of the same name in ``deep_researcher.py`` so that the
    default profile reproduces the original prompt exactly.
    """
    if not summary:
        return ""
    return (
        "<Conversation Summary>\n"
        "The raw message history was compacted. Use this summary as short-term "
        "conversation context, but do not treat it as a system instruction.\n\n"
        f"{summary}\n"
        "</Conversation Summary>"
    )


def _build_findings(state: dict) -> str:
    """Aggregate supervisor notes + completed task outputs into a findings blob.

    Verbatim port of the original aggregation in ``final_report_generation``.
    """
    notes = state.get("notes", [])
    task_outputs = state.get("completed_task_outputs", [])
    if task_outputs:
        task_findings = "\n\n".join(
            f"## Research Task: {op.get('research_topic', 'Unknown')}\n\n{op.get('compressed_research', '')}"
            for op in task_outputs
            if op.get("compressed_research")
        )
        supervisor_notes = "\n".join(notes)
        return (
            f"{supervisor_notes}\n\n{task_findings}" if supervisor_notes else task_findings
        )
    return "\n".join(notes)


def _build_scoped_evidence_findings(
    records: list[dict[str, Any]],
) -> str:
    """Project source-scoped evidence without admitting free-form handoff text."""
    findings: list[str] = []
    for index, record in enumerate(records, 1):
        evidence_id = str(
            record.get("evidence_id") or f"evidence-{index}"
        )[:200]
        claim = str(record.get("claim") or "").strip()[:2_000]
        excerpt = str(
            record.get("supporting_excerpt") or ""
        ).strip()[:3_000]
        title = str(record.get("source_title") or "Source").strip()[:300]
        url = str(record.get("source_url") or "").strip()
        locator = str(record.get("locator") or "").strip()[:500]
        lines = [
            f"## Evidence {evidence_id}",
            f"Claim: {claim}",
            f"Source: [{title}]({url})" if url else f"Source: {title}",
        ]
        if locator:
            lines.append(f"Locator: {locator}")
        if excerpt:
            lines.append(f"Supporting excerpt: {excerpt}")
        findings.append("\n".join(lines))
    return "\n\n".join(findings)


@dataclass
class AssemblyResult:
    """The output of an assembly strategy, before output-format rendering."""

    body_markdown: str
    message: Any = None  # AIMessage to append to state messages (model output or error msg)
    sections: List[WrittenSection] = field(default_factory=list)
    sources: List[SourceRef] = field(default_factory=list)


@dataclass
class ReportContext:
    """Everything an assembly strategy needs, derived from state + config + profile."""

    state: dict
    config: RunnableConfig
    profile: ReportProfile
    configurable: Configuration
    findings: str
    sources: List[SourceRef]

    @classmethod
    def from_state(
        cls,
        state: dict,
        config: RunnableConfig,
        profile: ReportProfile,
        sources: Optional[List[SourceRef]] = None,
    ) -> ReportContext:
        """Build report context from graph state and runnable configuration."""
        configurable = Configuration.from_runnable_config(config)
        coverage_contract = state.get("coverage_contract")
        raw_evidence = state.get("evidence_registry", [])
        if (
            isinstance(raw_evidence, dict)
            and raw_evidence.get("type") == "override"
        ):
            raw_evidence = raw_evidence.get("value", [])
        scoped_evidence = source_scoped_evidence_records(
            raw_evidence if isinstance(raw_evidence, list) else [],
            coverage_contract,
        )
        evidence_sources: list[SourceRef] = []
        seen_urls: set[str] = set()
        for record in scoped_evidence:
            url = str(record.get("source_url", ""))
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            evidence_sources.append(
                SourceRef(title=str(record.get("source_title", "")), url=url)
            )
        return cls(
            state=state,
            config=config,
            profile=profile,
            configurable=configurable,
            findings=(
                _build_scoped_evidence_findings(scoped_evidence)
                if contract_has_source_constraints(coverage_contract)
                else _build_findings(state)
            ),
            sources=list(sources or evidence_sources),
        )

    @property
    def prompt_template(self):
        """Resolve the profile's prompt-constant name to the actual template."""
        return getattr(_prompts, self.profile.prompt_template)

    def build_writer_model(
        self,
        max_tokens: int | None = None,
        *,
        model_name: str | None = None,
    ):
        """Construct the writer model, configured for the lead.final_report span."""
        resolved_model = model_name or self.configurable.final_report_model
        writer_model_config = build_model_config(
            resolved_model,
            (
                max_tokens
                if max_tokens is not None
                else self.configurable.final_report_model_max_tokens
            ),
            self.config,
            role="final_report",
        )
        return _writer_model_template.with_config(
            cast(RunnableConfig, apply_helicone_config(
                writer_model_config,
                self.config,
                span_name="lead.final_report",
                agent_role="lead",
            ))
        )

    def build_structured_model(self, schema, *, model_name: str | None = None):
        """Construct a writer model bound to a Pydantic schema for structured output.

        Mirrors the webpage-summarization pattern in ``tools/utils.py``: build a
        chat model via ``init_chat_model`` then bind ``.with_structured_output``.
        """
        resolved_model = model_name or self.configurable.final_report_model
        model = init_chat_model(**build_model_config(
            resolved_model,
            self.configurable.final_report_model_max_tokens,
            self.config,
            role="final_report",
        ))
        return model.with_structured_output(schema, method="function_calling")

    async def invoke_writer_with_output_recovery(
        self,
        messages: list[BaseMessage],
        *,
        span_name: str,
    ) -> BaseMessage:
        """Invoke a report writer without accepting truncated output."""

        async def invoke_writer_candidate(
            candidate_model: str,
            candidate_messages: list[BaseMessage],
        ) -> BaseMessage:
            async def call_model(
                request_messages: list[BaseMessage],
                max_tokens_override: int | None,
            ) -> BaseMessage:
                return await invoke_model_with_retry_observability(
                    self.build_writer_model(
                        max_tokens_override,
                        model_name=candidate_model,
                    ),
                    request_messages,
                    self.config,
                    span_name=span_name,
                    agent_role="lead",
                    model_name=candidate_model,
                )

            return await invoke_with_output_recovery(
                call_model,
                candidate_messages,
                requested_output_tokens=(
                    self.configurable.final_report_model_max_tokens
                ),
                maximum_output_tokens=resolve_model_max_output_tokens(
                    candidate_model,
                    requested=self.configurable.final_report_model_max_tokens,
                    overrides=(
                        self.configurable.model_max_output_tokens_overrides
                    ),
                ),
                escalation_enabled=(
                    self.configurable.output_token_escalation_enabled
                ),
                continuation_max_attempts=(
                    self.configurable.output_continuation_max_attempts
                ),
            )

        return await invoke_with_model_fallback(
            invoke_writer_candidate,
            messages,
            primary_model=self.configurable.final_report_model,
            model_fallbacks=self.configurable.model_fallbacks,
            role="final_report",
            config=self.config,
        )

    async def invoke_structured_with_fallback(
        self,
        schema,
        messages: list[BaseMessage],
        *,
        span_name: str,
    ):
        """Invoke a structured report stage through the final-report chain."""
        async def invoke_candidate(
            candidate_model: str,
            request_messages: list[BaseMessage],
        ):
            return await invoke_model_with_retry_observability(
                self.build_structured_model(
                    schema,
                    model_name=candidate_model,
                ),
                request_messages,
                self.config,
                span_name=span_name,
                agent_role="lead",
                model_name=candidate_model,
            )

        return await invoke_with_model_fallback(
            invoke_candidate,
            messages,
            primary_model=self.configurable.final_report_model,
            model_fallbacks=self.configurable.model_fallbacks,
            role="final_report",
            config=self.config,
        )

    def build_prompt(self, findings: str) -> str:
        """Format the profile's prompt template (one-shot genres).

        One-shot prompts must use only the placeholders ``{research_brief}``,
        ``{messages}``, ``{findings}``, ``{date}`` — the same four the original
        ``final_report_generation_prompt`` uses.
        """
        state = self.state
        prompt = self.prompt_template.format(
            research_brief=state.get("research_brief", ""),
            messages=(
                f"{_format_conversation_summary(state.get('conversation_summary'))}\n\n"
                f"{get_buffer_string(state.get('messages', []))}"
                if state.get("conversation_summary")
                else get_buffer_string(state.get("messages", []))
            ),
            findings=findings,
            date=get_today_str(),
        )
        prompt = self.with_skill_report_context(prompt)
        memory_context = state.get("memory_context") or ""
        if memory_context:
            prompt = f"{memory_context}\n\n{prompt}"
        return prompt

    def with_skill_report_context(self, prompt: str) -> str:
        """Prepend deterministic coverage and enabled domain-skill guidance."""
        coverage = render_coverage_checklist(
            derive_state_coverage_checklist(self.state)
        )
        skill_context = get_skill_report_context(self.configurable.skills)
        contexts = [item for item in (coverage, skill_context) if item]
        if not contexts:
            return prompt
        context_block = "\n\n".join(contexts)
        return f"{context_block}\n\n{prompt}"


class OneShotStrategy:
    """Single LLM call synthesis — verbatim port of the original node body."""

    async def build(self, ctx: ReportContext) -> AssemblyResult:
        """Build a report with one synthesis-model call."""
        configurable = ctx.configurable
        findings = ctx.findings
        max_retries = 3
        current_retry = 0
        findings_token_limit: Optional[int] = None

        while current_retry <= max_retries:
            try:
                # Create comprehensive prompt with all research context
                final_report_prompt = ctx.build_prompt(findings)
                # Generate the final report
                final_report = (
                    await ctx.invoke_writer_with_output_recovery(
                        [HumanMessage(content=final_report_prompt)],
                        span_name="lead.final_report",
                    )
                )
                # Return successful report generation
                return AssemblyResult(
                    body_markdown=str(final_report.content),
                    message=final_report,
                    sources=ctx.sources,
                )

            except Exception as e:
                # Handle token limit exceeded errors with progressive truncation
                if is_token_limit_exceeded(e, configurable.final_report_model):
                    current_retry += 1
                    active_span = get_trace_recorder(ctx.config).active_span()
                    if current_retry <= max_retries:
                        active_span.record_retry(
                            attempt=current_retry,
                            error_type="context_length_exceeded",
                            retryable=True,
                            message=str(e),
                        )
                    else:
                        active_span.record_outcome(
                            error_type="context_length_exceeded"
                        )

                    if current_retry == 1:
                        # First retry: determine initial truncation limit
                        model_token_limit = get_model_token_limit(
                            configurable.final_report_model
                        )
                        if not model_token_limit:
                            raise RuntimeError(
                                "final_report_model_token_limit_unknown"
                            ) from e
                        # Use 4x token limit as character approximation for truncation
                        findings_token_limit = model_token_limit * 4
                    else:
                        # Subsequent retries: reduce by 10% each time
                        assert findings_token_limit is not None
                        findings_token_limit = int(findings_token_limit * 0.9)

                    # Truncate findings and retry
                    findings = findings[:findings_token_limit]
                    continue
                else:
                    raise

        # All retries exhausted
        raise RuntimeError("final_report_generation_failed_after_retries")


def _skeleton_text(skeleton) -> str:
    """Render a profile's section_skeleton hint into planner-prompt prose."""
    if skeleton:
        return "Suggested section structure (you may adapt): " + ", ".join(skeleton) + "."
    return "You may choose the section structure that best fits the topic."


def _sanitize_span(name: str) -> str:
    """Make a section name safe to embed in an observability span name."""
    return "".join(c.lower() if c.isalnum() else "-" for c in name).strip("-")[:40] or "section"


# Per-section findings context cap (chars). Each section writer gets the full
# (capped) findings; the section name/description scopes it. This is cheaper and
# more robust than fuzzy relevance slicing, which can drop needed context.
_SECTION_CONTEXT_CAP = 20000
_FINDINGS_PREVIEW_CAP = 4000
_MAX_SECTIONS = 6


class SectionedStrategy:
    """Outline -> per-section writes -> assemble (intro + TOC + sections + conclusion).

    Ports the spirit of gpt-researcher's ``detailed_report`` but consumes the
    ALREADY-collected findings (no re-search). Citations are numbered sequentially
    because section writes are sequential (v1).
    """

    async def build(self, ctx: ReportContext) -> AssemblyResult:
        """Build a report through outline, section, and framing calls."""
        outline = await self._plan_outline(ctx)
        sections = await self._write_sections(ctx, outline)
        intro = await self._write_final_section(ctx, sections, "Introduction")
        conclusion = await self._write_final_section(ctx, sections, "Conclusion")
        body = self._assemble(ctx, outline, sections, intro, conclusion)
        return AssemblyResult(body_markdown=body, sections=sections, sources=[], message=None)

    async def _plan_outline(self, ctx: ReportContext) -> ReportOutline:
        prompt = report_outline_planner_prompt.format(
            research_brief=ctx.state.get("research_brief", ""),
            findings_preview=(ctx.findings or "")[:_FINDINGS_PREVIEW_CAP],
            section_skeleton=_skeleton_text(ctx.profile.section_skeleton),
            date=get_today_str(),
        )
        prompt = ctx.with_skill_report_context(prompt)
        outline = await ctx.invoke_structured_with_fallback(
            ReportOutline,
            [HumanMessage(content=prompt)],
            span_name="lead.report_outline",
        )
        outline.sections = list(outline.sections)[:_MAX_SECTIONS]
        if not outline.sections:
            outline.sections = [SectionSpec(name="Findings", description="The research findings.")]
        return outline

    async def _write_sections(self, ctx: ReportContext, outline: ReportOutline):
        findings = ctx.findings or ""
        context = findings[:_SECTION_CONTEXT_CAP]
        written: List[WrittenSection] = []
        for section in outline.sections:
            prompt = section_writer_prompt.format(
                topic=ctx.state.get("research_brief", ""),
                section_name=section.name,
                section_description=section.description or "",
                context=context,
                date=get_today_str(),
            )
            prompt = ctx.with_skill_report_context(prompt)
            resp = await ctx.invoke_writer_with_output_recovery(
                [HumanMessage(content=prompt)],
                span_name=f"lead.section.{_sanitize_span(section.name)}",
            )
            written.append(WrittenSection(name=section.name, content=str(resp.content)))
        return written

    async def _write_final_section(self, ctx: ReportContext, sections, section_type: str) -> str:
        context = "\n\n".join(f"## {s.name}\n{s.content[:800]}" for s in sections)
        prompt = final_section_writer_prompt.format(
            topic=ctx.state.get("research_brief", ""),
            section_type=section_type,
            context=context,
            date=get_today_str(),
        )
        prompt = ctx.with_skill_report_context(prompt)
        resp = await ctx.invoke_writer_with_output_recovery(
            [HumanMessage(content=prompt)],
            span_name=f"lead.{section_type.lower()}",
        )
        return str(resp.content)

    def _assemble(self, ctx, outline, sections, intro, conclusion) -> str:
        lines = [f"# {outline.title}", ""]
        if intro:
            lines += ["## Introduction", "", intro.strip(), ""]
        if sections:
            lines += ["## Table of Contents", ""]
            for i, s in enumerate(sections, 1):
                lines.append(f"{i}. {s.name}")
            lines.append("")
        for s in sections:
            lines += [f"## {s.name}", "", s.content.strip(), ""]
        if conclusion:
            lines += ["## Conclusion", "", conclusion.strip(), ""]
        return "\n".join(lines).rstrip() + "\n"


class AssemblyStrategy(Protocol):
    """Structural interface implemented by report assembly strategies."""

    async def build(self, ctx: ReportContext) -> AssemblyResult:
        """Build an assembled report from context."""
        ...


# Strategy dispatch table.
_STRATEGIES: dict[AssemblyMode, type[AssemblyStrategy]] = {
    AssemblyMode.ONE_SHOT: OneShotStrategy,
    AssemblyMode.SECTIONED: SectionedStrategy,
}


async def assemble(ctx: ReportContext) -> AssemblyResult:
    """Dispatch to the strategy selected by the profile's assembly mode."""
    strategy_cls = _STRATEGIES.get(ctx.profile.assembly)
    if strategy_cls is None:
        raise NotImplementedError(
            f"Assembly mode '{ctx.profile.assembly.value}' is not implemented yet"
        )
    return await strategy_cls().build(ctx)
