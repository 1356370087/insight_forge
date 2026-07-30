"""Fail-closed synthesis from SHA-verified, eligible evidence records only."""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict, deque
from typing import Any, cast
from urllib.parse import urlsplit

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from open_deep_research.agents.model_recovery import (
    resolve_model_context_window,
)
from open_deep_research.agents.research_context import (
    response_was_truncated,
)
from open_deep_research.configuration import Configuration
from open_deep_research.evidence import source_scoped_evidence_records
from open_deep_research.observability import (
    invoke_model_with_retry_observability,
)
from open_deep_research.quality import _evaluate_json
from open_deep_research.quality_contract import ResearchCoverageContract
from open_deep_research.report.recovery import (
    build_evidence_recovery_report,
)
from open_deep_research.security.content import sanitize_report_markdown

_URL_RE = re.compile(r"https?://[^\s)\]}>]+", re.IGNORECASE)
_LOGGER = logging.getLogger(__name__)
_DRAFT_PROMPT = """You write a partial research report from a closed evidence set.
Return one JSON object matching the requested schema. Use only the supplied
evidence records. Every factual summary or section claim must cite one or more
evidence_ids. Never infer absence from a failed search. Put uncertainty in the
qualification field and preserve unresolved user requirement IDs. Do not add
URLs that are not present in the evidence records. A community_issue is
anecdotal evidence and must never support a claim framed as an official
guarantee."""
_GROUNDING_PROMPT = """You are an independent fail-closed grounding judge.
Check whether every factual statement in the draft is supported by its bound
evidence records and whether qualifications accurately express uncertainty.
Return JSON only. Set supported=false for any unsupported, overstated, or
mis-cited claim, including any official-guarantee claim supported only by a
community_issue."""


class EvidenceBoundClaim(BaseModel):
    """One report claim with explicit evidence provenance."""

    text: str
    evidence_ids: list[str] = Field(min_length=1)
    qualification: str | None = None


class EvidenceSynthesisSection(BaseModel):
    """A report section made exclusively of evidence-bound claims."""

    heading: str
    claims: list[EvidenceBoundClaim]


class EvidenceSynthesisDraft(BaseModel):
    """Structured partial report produced from the eligible evidence allowlist."""

    title: str
    summary: str
    summary_evidence_ids: list[str] = Field(min_length=1)
    sections: list[EvidenceSynthesisSection]
    unresolved_requirements: list[str] = Field(default_factory=list)


class EvidenceSynthesisGroundingAssessment(BaseModel):
    """Independent grounding verdict for an evidence-limited draft."""

    supported: bool
    unsupported_claim_paths: list[str] = Field(default_factory=list)
    reason: str


def _draft_system_prompt() -> str:
    """Return provider-neutral instructions with the exact output schema."""
    schema = json.dumps(
        EvidenceSynthesisDraft.model_json_schema(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return (
        f"{_DRAFT_PROMPT}\n\n"
        "The exact JSON Schema for the single response object is below. "
        "Include every required field with the specified type. Do not wrap "
        "the object in another field and do not return prose or Markdown "
        "outside the object.\n"
        f"{schema}"
    )


def _safe_markdown_link_label(value: Any) -> str:
    """Return untrusted source metadata as a single inert Markdown label."""
    label = " ".join(str(value or "").split())
    label = _URL_RE.sub("[URL omitted]", label)
    label = (
        label.replace("\\", "\\\\")
        .replace("[", "\\[")
        .replace("]", "\\]")
    )
    return label or "来源"


def _validate_rendered_urls(report: str, allowed_urls: set[str]) -> None:
    """Fail closed if rendering introduced a URL outside the evidence set."""
    if any(
        url.rstrip(".,;:") not in allowed_urls
        for url in _URL_RE.findall(report)
    ):
        raise ValueError("evidence_synthesis_rendered_url_not_allowlisted")


def _project_evidence(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Project evidence to the only fields the restricted writer may read."""
    text_limits = {
        "evidence_id": 200,
        "claim": 1_000,
        "supporting_excerpt": 2_000,
        "source_url": 2_000,
        "source_title": 500,
        "locator": 300,
        "source_kind": 100,
        "source_scope_status": 100,
    }
    projected: list[dict[str, Any]] = []
    for record in records:
        item: dict[str, Any] = {}
        for field_name, limit in text_limits.items():
            value = record.get(field_name)
            if value not in (None, ""):
                item[field_name] = str(value)[:limit]
        for field_name in ("source_authority", "confidence"):
            value = record.get(field_name)
            if value not in (None, ""):
                item[field_name] = value
        projected.append(item)
    return projected


def _source_bucket(record: dict[str, Any]) -> str:
    """Return a stable source bucket for deterministic diversity selection."""
    url = str(record.get("source_url") or "")
    return (urlsplit(url).hostname or url or "unknown").casefold()


def _rank_record(record: dict[str, Any]) -> tuple[float, float]:
    """Prefer stronger evidence without letting one source monopolize input."""
    try:
        authority = float(record.get("source_authority", 0.0))
    except (TypeError, ValueError):
        authority = 0.0
    try:
        confidence = float(record.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    return authority, confidence


def _select_evidence_for_budget(
    records: list[dict[str, Any]],
    *,
    token_budget: int,
    max_records: int = 56,
) -> list[dict[str, Any]]:
    """Select a bounded, source-diverse evidence projection."""
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    bucket_order: list[str] = []
    for record in records:
        bucket = _source_bucket(record)
        if bucket not in buckets:
            bucket_order.append(bucket)
        buckets[bucket].append(record)
    queues = {
        bucket: deque(
            sorted(
                buckets[bucket],
                key=_rank_record,
                reverse=True,
            )
        )
        for bucket in bucket_order
    }

    selected: list[dict[str, Any]] = []
    used_tokens = 0
    while queues and len(selected) < max_records:
        made_progress = False
        for bucket in tuple(bucket_order):
            queue = queues.get(bucket)
            if not queue:
                queues.pop(bucket, None)
                continue
            record = queue.popleft()
            record_tokens = max(
                1,
                count_tokens_approximately([
                    HumanMessage(
                        content=json.dumps(
                            record,
                            ensure_ascii=False,
                            sort_keys=True,
                        )
                    )
                ]),
            )
            if used_tokens + record_tokens <= token_budget or not selected:
                selected.append(record)
                used_tokens += record_tokens
                made_progress = True
            if not queue:
                queues.pop(bucket, None)
            if len(selected) >= max_records:
                break
        if not made_progress:
            break
    return selected


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(block.get("text", ""))
            if isinstance(block, dict)
            else str(getattr(block, "text", block))
            for block in content
        )
    return str(content)


def _parse_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    fenced = re.fullmatch(
        r"```(?:json)?\s*(.*?)\s*```",
        stripped,
        re.DOTALL,
    )
    if fenced:
        stripped = fenced.group(1).strip()
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end <= start:
            raise
        payload = json.loads(stripped[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("evidence_synthesis_writer_must_return_object")
    return payload


async def _invoke_draft(
    payload: dict[str, Any],
    config: RunnableConfig,
) -> EvidenceSynthesisDraft:
    """Invoke the configured report writer without exposing rejected prose."""
    from open_deep_research.agents.deep_researcher import (
        configurable_model,
        get_model_compatibility_kwargs,
        get_model_connection_kwargs,
    )

    configurable = Configuration.from_runnable_config(config)
    model_name = configurable.final_report_model
    model = configurable_model.with_config(
        cast(
            RunnableConfig,
            {
                "model": model_name,
                "max_tokens": configurable.final_report_model_max_tokens,
                **get_model_connection_kwargs(model_name, config),
                "tags": ["langsmith:nostream"],
                **get_model_compatibility_kwargs(model_name),
            },
        )
    )
    response = await invoke_model_with_retry_observability(
        model,
        [
            SystemMessage(content=_draft_system_prompt()),
            HumanMessage(
                content=(
                    "Write the evidence-limited draft from this JSON payload:\n"
                    + json.dumps(payload, ensure_ascii=False)
                )
            ),
        ],
        config,
        span_name="lead.evidence_limited_writer",
        agent_role="lead",
        model_name=model_name,
    )
    if response_was_truncated(response):
        raise ValueError("evidence_synthesis_writer_truncated")
    return EvidenceSynthesisDraft.model_validate(
        _parse_json_object(_content_text(response.content))
    )


async def _invoke_grounding_judge(
    payload: dict[str, Any],
    config: RunnableConfig,
) -> EvidenceSynthesisGroundingAssessment:
    result = await _evaluate_json(
        EvidenceSynthesisGroundingAssessment,
        _GROUNDING_PROMPT,
        payload,
        config,
        span_name="lead.evidence_limited_grounding",
    )
    return EvidenceSynthesisGroundingAssessment.model_validate(result)


def _validate_draft(
    draft: EvidenceSynthesisDraft,
    *,
    evidence_ids: set[str],
    allowed_urls: set[str],
    requirement_ids: set[str],
) -> None:
    """Enforce provenance, URL and requirement boundaries before rendering."""
    referenced_ids = [
        *draft.summary_evidence_ids,
        *[
            evidence_id
            for section in draft.sections
            for claim in section.claims
            for evidence_id in claim.evidence_ids
        ],
    ]
    if not referenced_ids or any(
        evidence_id not in evidence_ids
        for evidence_id in referenced_ids
    ):
        raise ValueError("evidence_synthesis_unknown_evidence_id")
    if any(
        requirement_id not in requirement_ids
        for requirement_id in draft.unresolved_requirements
    ):
        raise ValueError("evidence_synthesis_unknown_requirement_id")
    encoded = draft.model_dump_json()
    if any(
        url.rstrip(".,;:") not in allowed_urls
        for url in _URL_RE.findall(encoded)
    ):
        raise ValueError("evidence_synthesis_url_not_allowlisted")


def _render_draft(
    draft: EvidenceSynthesisDraft,
    evidence_by_id: dict[str, dict[str, Any]],
    requirement_text: dict[str, str],
) -> str:
    lines = [
        f"# {draft.title}",
        "",
        draft.summary,
        "",
    ]
    for section in draft.sections:
        lines.extend([f"## {section.heading}", ""])
        for claim in section.claims:
            citations = " ".join(
                f"[{evidence_id}]" for evidence_id in claim.evidence_ids
            )
            qualification = (
                f"（{claim.qualification}）"
                if claim.qualification
                else ""
            )
            lines.append(
                f"- {claim.text}{qualification} {citations}".rstrip()
            )
        lines.append("")
    lines.extend(["## 限制与不确定性", ""])
    if draft.unresolved_requirements:
        lines.extend(
            f"- [{requirement_id}] "
            f"{requirement_text.get(requirement_id, requirement_id)}："
            "现有合格证据不足，未作确定性结论。"
            for requirement_id in draft.unresolved_requirements
        )
    else:
        lines.append("- 本报告仅基于已通过完整性与安全校验的证据记录。")
    lines.extend(["", "## 来源", ""])
    used_ids = list(dict.fromkeys([
        *draft.summary_evidence_ids,
        *[
            evidence_id
            for section in draft.sections
            for claim in section.claims
            for evidence_id in claim.evidence_ids
        ],
    ]))
    for evidence_id in used_ids:
        record = evidence_by_id[evidence_id]
        title = _safe_markdown_link_label(
            record.get("source_title") or evidence_id
        )
        url = str(record.get("source_url") or "")
        lines.append(
            f"- [{evidence_id}] [{title}]({url})"
            if url
            else f"- [{evidence_id}] {title}"
        )
    return sanitize_report_markdown("\n".join(lines).strip())


async def build_evidence_limited_report(
    evidence_records: list[dict[str, Any]],
    *,
    coverage_contract: ResearchCoverageContract | dict[str, Any] | None,
    coverage_ledger: dict[str, Any] | None,
    caveats: list[str],
    uncovered_requirement_ids: list[str],
    rejection_reasons: list[str],
    artifact_refs: list[dict[str, Any]],
    config: RunnableConfig,
) -> str:
    """Return restricted synthesis, falling back deterministically on any fault."""
    try:
        contract = (
            coverage_contract
            if isinstance(coverage_contract, ResearchCoverageContract)
            else ResearchCoverageContract.model_validate(coverage_contract)
            if isinstance(coverage_contract, dict) and coverage_contract
            else None
        )
    except Exception:  # noqa: BLE001 - invalid contract must fail closed
        contract = None
    eligible = source_scoped_evidence_records(
        evidence_records,
        contract or {},
    )
    fallback = lambda: build_evidence_recovery_report(  # noqa: E731
        eligible,
        gaps=uncovered_requirement_ids,
        rejection_reasons=rejection_reasons,
        artifact_refs=artifact_refs,
    )
    if not eligible:
        return fallback()
    try:
        if contract is None:
            return fallback()
        configurable = Configuration.from_runnable_config(config)
        context_window = resolve_model_context_window(
            configurable.final_report_model,
            overrides=configurable.model_context_window_overrides,
            unknown_default=configurable.unknown_model_context_window_tokens,
        )
        safety_margin = max(2_048, int(context_window * 0.05))
        reserved_output = min(
            configurable.final_report_model_max_tokens,
            max(1_024, context_window // 3),
        )
        contract_payload = contract.model_dump(mode="json")
        base_payload = {
            "coverage_contract": contract_payload,
            "eligible_evidence": [],
            "requirement_to_evidence": dict(coverage_ledger or {}),
            "caveats": [str(item)[:500] for item in caveats[:20]],
            "uncovered_requirement_ids": list(
                dict.fromkeys(uncovered_requirement_ids)
            ),
            "allowed_source_urls": [],
        }
        fixed_tokens = count_tokens_approximately([
            SystemMessage(content=_draft_system_prompt()),
            HumanMessage(
                content=(
                    "Write the evidence-limited draft from this JSON payload:\n"
                    + json.dumps(base_payload, ensure_ascii=False)
                )
            ),
        ])
        available_evidence_tokens = (
            context_window
            - reserved_output
            - safety_margin
            - fixed_tokens
        )
        if available_evidence_tokens < 256:
            return fallback()
        projected_evidence = _select_evidence_for_budget(
            _project_evidence(eligible),
            token_budget=max(256, int(available_evidence_tokens * 0.7)),
        )
        evidence_by_id = {
            str(record.get("evidence_id", "")): record
            for record in projected_evidence
            if record.get("evidence_id")
        }
        if not evidence_by_id:
            return fallback()
        requirement_text = {
            requirement.requirement_id: requirement.text
            for requirement in contract.requirements
        }
        selected_evidence_ids = set(evidence_by_id)
        projected_ledger: dict[str, Any] = {}
        for requirement_id, entry in dict(coverage_ledger or {}).items():
            if not isinstance(entry, dict):
                continue
            projected_entry = dict(entry)
            projected_entry["evidence_ids"] = [
                str(evidence_id)
                for evidence_id in entry.get("evidence_ids", [])
                if str(evidence_id) in selected_evidence_ids
            ]
            projected_ledger[str(requirement_id)] = projected_entry
        payload: dict[str, Any] = {
            "coverage_contract": contract_payload,
            "eligible_evidence": projected_evidence,
            "requirement_to_evidence": projected_ledger,
            "caveats": [str(item)[:500] for item in caveats[:20]],
            "uncovered_requirement_ids": list(
                dict.fromkeys(uncovered_requirement_ids)
            ),
            "allowed_source_urls": sorted({
                str(record.get("source_url"))
                for record in projected_evidence
                if record.get("source_url")
            }),
        }
        draft = await _invoke_draft(payload, config)
        _validate_draft(
            draft,
            evidence_ids=set(evidence_by_id),
            allowed_urls=set(payload["allowed_source_urls"]),
            requirement_ids=set(requirement_text),
        )
        grounding = await _invoke_grounding_judge(
            {
                "draft": draft.model_dump(mode="json"),
                "eligible_evidence": projected_evidence,
            },
            config,
        )
        if not grounding.supported or grounding.unsupported_claim_paths:
            raise ValueError("evidence_synthesis_grounding_failed")
        rendered = _render_draft(draft, evidence_by_id, requirement_text)
        _validate_rendered_urls(
            rendered,
            set(payload["allowed_source_urls"]),
        )
        return rendered
    except Exception as exc:  # noqa: BLE001 - recovery boundary is fail-closed
        reason_code = (
            str(exc)
            if isinstance(exc, ValueError)
            and str(exc).startswith("evidence_synthesis_")
            else type(exc).__name__
        )
        _LOGGER.warning(
            "Evidence-limited synthesis used deterministic fallback: %s",
            reason_code,
        )
        return fallback()
