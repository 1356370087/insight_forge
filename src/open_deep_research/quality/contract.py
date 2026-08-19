"""Coverage-bound quality contracts and deterministic admission policy."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Literal, Mapping, Sequence, cast

from langchain_core.messages import BaseMessage
from pydantic import BaseModel, ConfigDict, Field

from open_deep_research.report.coverage import derive_coverage_checklist

COVERAGE_CONTRACT_SCHEMA_VERSION = 1
QUALITY_RISK_POLICY_VERSION = "quality-risk-v1"

RequirementKind = Literal["factual", "process", "deliverable"]

# Orchestration/interaction directives the engine itself satisfies; a research
# subtask can never prove them with web evidence ("至少并行两个研究员",
# "不需要澄清", citation-scope rules).
_PROCESS_REQUIREMENT_RE = re.compile(
    r"(?:不可拆分|单一研究任务)|"
    r"(?:至少.{0,48}(?:并行.{0,16})?(?:Subagent|子智能体|研究员))|"
    r"(?:并行.{0,24}(?:委派|开展|执行|运行)?.{0,16}(?:两个|多个|多名|两名)?.{0,16}(?:Subagent|子智能体|研究员))|"
    r"(?:每个.{0,24}(?:Subagent|子智能体|研究员).{0,80}(?:读取|引用|来源))|"
    r"(?:(?:只|仅|严格).{0,480}(?:URL|网址|链接).{0,80}(?:证据|来源))|"
    r"(?:不得.{0,32}(?:引用|使用).{0,32}(?:其他|额外|未指定).{0,16}(?:URL|网址|链接|来源))|"
    r"(?:每项.{0,48}(?:引用|来源))|"
    r"(?:(?:必须|需要).{0,48}标(?:为|注为).{0,24}未证实)|"
    r"(?:不得.{0,80}(?:引用第三方|搜索候选摘要|当作证据))|"
    r"(?:不?需要澄清|无需澄清|不?需要确认|无需确认)|"
    r"(?:直接(?:执行|开始|研究|运行)|立即(?:执行|开始)|不必等待)|"
    r"(?:single\s+(?:indivisible\s+)?research\s+task)|"
    r"(?:must\s+(?:cite|label).{0,80})|"
    r"(?:do\s+not\s+(?:cite|use).{0,80})|"
    r"(?:no\s+clarification.{0,40})|"
    r"(?:proceed\s+directly.{0,40})",
    flags=re.IGNORECASE | re.DOTALL,
)

# Output-format obligations owned by the final report stage ("风险矩阵",
# "检查清单", "用中文输出"); evidence cannot prove a deliverable's existence.
_DELIVERABLE_REQUIREMENT_RE = re.compile(
    r"(?:最终.{0,32}(?:中文.{0,16})?(?:对照表|比较表|表格|报告|输出|呈现))|"
    r"(?:(?:用|使用|以).{0,12}(?:中文|英文).{0,24}(?:输出|撰写|呈现|回答|报告))|"
    r"(?:执行摘要|executive\s+summary)|"
    r"(?:风险矩阵|risk\s+matrix)|"
    r"(?:(?:检查|核对|排查)清单|(?:pre-?)?(?:launch|production|go-?live).{0,24}checklist)|"
    r"(?:(?:对照|比较)表|对比表格)|"
    r"(?:可点击.{0,8}(?:引用|链接)|clickable\s+(?:citation|link))|"
    r"(?:输出.{0,16}(?:表格|清单|矩阵|摘要))|"
    r"(?:报告(?:末尾|结尾|中).{0,24}(?:附|包含|提供))",
    flags=re.IGNORECASE | re.DOTALL,
)


def classify_requirement_kind(text: str) -> RequirementKind:
    """Classify one requirement text as factual, process, or deliverable.

    Factual requirements need external evidence; process requirements are
    orchestration directives the engine satisfies itself; deliverable
    requirements are output-format obligations owned by the final report.
    """
    value = str(text or "")
    if _PROCESS_REQUIREMENT_RE.search(value):
        return "process"
    if _DELIVERABLE_REQUIREMENT_RE.search(value):
        return "deliverable"
    return "factual"


def is_delegable_requirement(
    requirement: CoverageRequirement | Mapping[str, Any],
) -> bool:
    """Return whether a research subtask can own this requirement.

    Only factual requirements are delegable. An explicit ``process`` /
    ``deliverable`` kind is authoritative; an explicit ``factual`` kind (or a
    payload compiled before kinds existed) still gets the pattern fallback so
    hand-built and legacy requirements classify the same way the compiler
    would have classified them.
    """
    if isinstance(requirement, CoverageRequirement):
        kind: RequirementKind | None = requirement.kind
        text = requirement.text
    else:
        raw_kind = requirement.get("kind")
        kind = None
        if isinstance(raw_kind, str) and raw_kind in (
            "factual",
            "process",
            "deliverable",
        ):
            kind = cast(RequirementKind, raw_kind)
        text = str(requirement.get("text", ""))
    if kind is not None and kind != "factual":
        return False
    return classify_requirement_kind(text) == "factual"


class AdmissionStatus(str, Enum):
    """Supervisor admission result for one Researcher handoff."""

    ACCEPTED = "accepted"
    ACCEPTED_WITH_CAVEATS = "accepted_with_caveats"
    REJECTED = "rejected"


class CoverageStatus(str, Enum):
    """Evidence coverage for one explicit user requirement."""

    SUPPORTED = "supported"
    PARTIAL = "partial"
    UNSUPPORTED = "unsupported"


class CoverageRequirement(BaseModel):
    """One requirement copied from an original human message."""

    model_config = ConfigDict(frozen=True)

    requirement_id: str
    text: str
    kind: RequirementKind = "factual"
    source_message_index: int = Field(ge=0)
    source_start: int = Field(ge=0)
    source_end: int = Field(ge=0)
    source_located: bool = True


class ResearchCoverageContract(BaseModel):
    """Immutable contract separating user requirements from model advice."""

    model_config = ConfigDict(frozen=True)

    schema_version: int = COVERAGE_CONTRACT_SCHEMA_VERSION
    original_query_sha256: str
    requirements: tuple[CoverageRequirement, ...]
    advisory_dimensions: tuple[str, ...] = ()

    def requirement_ids(self) -> tuple[str, ...]:
        """Return stable requirement identifiers in source order."""
        return tuple(item.requirement_id for item in self.requirements)

    def delegable_requirement_ids(self) -> tuple[str, ...]:
        """Return IDs of factual requirements a research subtask may own."""
        return tuple(
            item.requirement_id
            for item in self.requirements
            if item.kind == "factual"
        )


class RequirementCoverage(BaseModel):
    """Judge-proposed evidence mapping for one explicit requirement."""

    model_config = ConfigDict(frozen=True)

    requirement_id: str
    status: CoverageStatus
    evidence_ids: tuple[str, ...] = ()
    explanation: str = ""


@dataclass(frozen=True, slots=True)
class HandoffPolicyInput:
    """Inputs consumed by the deterministic v4 admission reducer."""

    requested_status: AdmissionStatus
    requirement_coverage: tuple[RequirementCoverage, ...]
    caveats: tuple[str, ...]
    missing_information: tuple[str, ...]
    unsupported_claims: tuple[str, ...]
    deterministic_checks_passed: bool
    scores: tuple[int, int, int, int]
    dimension_floor: int
    average_floor: float
    caveat_admission_enabled: bool
    high_risk: bool
    evaluator_failed_closed: bool = False
    additional_hard_rejection_reasons: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class HandoffPolicyResult:
    """Final admission result after applying non-model policy."""

    admission_status: AdmissionStatus
    accepted: bool
    caveats: tuple[str, ...]
    hard_rejection_reasons: tuple[str, ...]


class ResearchRiskProfile(BaseModel):
    """Persistable result of deterministic high-risk classification."""

    model_config = ConfigDict(frozen=True)

    policy_version: str = QUALITY_RISK_POLICY_VERSION
    level: str
    categories: tuple[str, ...] = ()
    matched_rule_ids: tuple[str, ...] = ()

    @property
    def high_risk(self) -> bool:
        """Return whether caveat admission must be disabled."""
        return self.level == "high"


_HIGH_RISK_RULES: dict[str, tuple[tuple[str, str], ...]] = {
    "medical": (
        ("medical.diagnosis", r"\bdiagnos(?:is|e)\b|诊断|确诊"),
        ("medical.treatment", r"\btreat(?:ment)?\b|治疗方案|用药"),
        ("medical.prescription", r"\bprescri(?:be|ption)\b|处方|剂量"),
        ("medical.emergency", r"\bemergency\b|急救|急诊"),
    ),
    "legal": (
        ("legal.advice", r"\blegal advice\b|法律意见|法律建议"),
        ("legal.litigation", r"\blitigation\b|诉讼|起诉"),
        ("legal.liability", r"\bliability\b|法律责任|侵权责任"),
        ("legal.contract", r"\bcontract interpretation\b|合同解释"),
    ),
    "finance": (
        ("finance.investment", r"\binvest(?:ment|ing)\b|投资建议|证券"),
        (
            "finance.trading",
            r"\b(?:trade|trading)\s+(?:advice|recommendation|strategy|signal)s?\b|"
            r"\b(?:buy|sell)\b.{0,48}\b(?:stocks?|securit(?:y|ies)|"
            r"crypto(?:currenc(?:y|ies))?|funds?|bonds?|shares?|options?|futures?)\b|"
            r"交易策略|交易建议|买入|卖出",
        ),
        ("finance.credit", r"\bcredit\b|\bloan\b|信贷|贷款"),
        ("finance.tax", r"\btax advice\b|税务建议"),
        ("finance.insurance", r"\binsurance advice\b|保险建议"),
    ),
}

_EXPLICIT_TIME_CONSTRAINT_PATTERNS = (
    re.compile(
        r"截至\s*(?:"
        r"\d{4}年(?:\d{1,2}月(?:\d{1,2}日)?)?"
        r"|目前|当前|今日|现在"
        r")"
    ),
    re.compile(
        r"\bas of\s+(?:"
        r"(?:january|february|march|april|may|june|july|august|"
        r"september|october|november|december)"
        r"(?:\s+\d{1,2},?)?\s+\d{4}"
        r"|\d{4}(?:-\d{1,2}(?:-\d{1,2})?)?"
        r"|today|now|the current date"
        r")\b",
        flags=re.IGNORECASE,
    ),
)


def _message_role(message: Any) -> str:
    if isinstance(message, dict):
        return str(message.get("role") or message.get("type") or "").lower()
    return str(getattr(message, "type", "")).lower()


def _message_content(message: Any) -> str:
    if isinstance(message, dict):
        return str(message.get("content") or "")
    return str(getattr(message, "content", "") or "")


def _stable_requirement_id(
    message_index: int,
    text: str,
    ordinal: int,
) -> str:
    digest = hashlib.sha256(
        f"{message_index}:{ordinal}:{text}".encode()
    ).hexdigest()[:12]
    return f"COV-{ordinal:02d}-{digest}"


def _extract_explicit_time_constraints(
    content: str,
) -> tuple[tuple[str, int, int], ...]:
    """Return exact, source-addressable time constraints in source order."""
    matches: list[tuple[str, int, int]] = []
    occupied: set[tuple[int, int]] = set()
    for pattern in _EXPLICIT_TIME_CONSTRAINT_PATTERNS:
        for match in pattern.finditer(content):
            span = match.span()
            if span in occupied:
                continue
            occupied.add(span)
            matches.append((match.group(0), span[0], span[1]))
    return tuple(sorted(matches, key=lambda item: (item[1], item[2])))


def _locate_requirement_source(
    content: str,
    requirement: str,
) -> tuple[str, int, int]:
    """Locate a normalized checklist item in its original source text."""
    direct_start = content.find(requirement)
    if direct_start >= 0:
        direct_end = direct_start + len(requirement)
        return content[direct_start:direct_end], direct_start, direct_end

    tokens = requirement.split()
    if tokens:
        whitespace_tolerant = re.compile(
            r"\s+".join(re.escape(token) for token in tokens),
        )
        match = whitespace_tolerant.search(content)
        if match is not None:
            start, end = match.span()
            return content[start:end], start, end

    # The checklist extractor already bounds semantic items to 500 characters.
    # If normalization removed markup or prefixes so thoroughly that no honest
    # source span can be recovered, retain that bounded item and mark the span
    # as unavailable instead of duplicating the entire user message.
    return requirement[:500], 0, 0


def build_research_coverage_contract(
    messages: Sequence[BaseMessage | dict[str, Any]],
    *,
    advisory_dimensions: Iterable[str] = (),
    max_requirements: int = 20,
) -> ResearchCoverageContract:
    """Build an immutable contract only from original human messages."""
    human_payloads = [
        (index, _message_content(message))
        for index, message in enumerate(messages)
        if _message_role(message) in {"human", "user"}
        and _message_content(message).strip()
    ]
    query_text = "\n\n".join(text for _, text in human_payloads)
    requirements: list[CoverageRequirement] = []
    seen: set[str] = set()
    ordinal = 1
    for message_index, content in human_payloads:
        for item, start, end in _extract_explicit_time_constraints(content):
            normalized = re.sub(r"\W+", "", item).lower()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            requirements.append(
                CoverageRequirement(
                    requirement_id=_stable_requirement_id(
                        message_index,
                        item,
                        ordinal,
                    ),
                    text=item,
                    kind=classify_requirement_kind(item),
                    source_message_index=message_index,
                    source_start=start,
                    source_end=end,
                )
            )
            ordinal += 1
            if len(requirements) >= max_requirements:
                break
        if len(requirements) >= max_requirements:
            break
        for item in derive_coverage_checklist(
            content,
            max_items=max_requirements,
        ):
            normalized = re.sub(r"\W+", "", item).lower()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            source_item, start, end = _locate_requirement_source(content, item)
            requirements.append(
                CoverageRequirement(
                    requirement_id=_stable_requirement_id(
                        message_index,
                        source_item,
                        ordinal,
                    ),
                    text=source_item,
                    kind=classify_requirement_kind(source_item),
                    source_message_index=message_index,
                    source_start=start,
                    source_end=end,
                    source_located=end > start,
                )
            )
            ordinal += 1
            if len(requirements) >= max_requirements:
                break
        if len(requirements) >= max_requirements:
            break
    if not requirements and query_text.strip():
        fallback = query_text.strip()[:240]
        requirements.append(
            CoverageRequirement(
                requirement_id=_stable_requirement_id(0, fallback, 1),
                text=fallback,
                kind=classify_requirement_kind(fallback),
                source_message_index=human_payloads[0][0],
                source_start=0,
                source_end=len(fallback),
                source_located=True,
            )
        )
    advisories = tuple(
        dict.fromkeys(
            str(item).strip()[:500]
            for item in advisory_dimensions
            if str(item).strip()
        )
    )
    return ResearchCoverageContract(
        original_query_sha256=hashlib.sha256(
            query_text.encode("utf-8")
        ).hexdigest(),
        requirements=tuple(requirements),
        advisory_dimensions=advisories,
    )


def classify_research_risk(
    text: str,
    *,
    mode: str = "auto",
    skills: Iterable[str] = (),
) -> ResearchRiskProfile:
    """Classify high-risk research using versioned deterministic rules."""
    normalized_mode = str(mode).strip().lower()
    if normalized_mode not in {"auto", "high", "standard"}:
        raise ValueError(f"unsupported_quality_risk_mode:{mode}")
    if normalized_mode == "high":
        return ResearchRiskProfile(
            level="high",
            matched_rule_ids=("config.force_high",),
        )
    if normalized_mode == "standard":
        return ResearchRiskProfile(
            level="standard",
            matched_rule_ids=("config.force_standard",),
        )

    categories: set[str] = {
        skill
        for skill in (str(item).strip().lower() for item in skills)
        if skill in _HIGH_RISK_RULES
    }
    matched_rules: set[str] = {
        f"skill.{category}" for category in categories
    }
    for category, rules in _HIGH_RISK_RULES.items():
        for rule_id, pattern in rules:
            if re.search(pattern, text, flags=re.IGNORECASE):
                categories.add(category)
                matched_rules.add(rule_id)
    return ResearchRiskProfile(
        level="high" if categories else "standard",
        categories=tuple(sorted(categories)),
        matched_rule_ids=tuple(sorted(matched_rules)),
    )


def resolve_handoff_admission(
    policy_input: HandoffPolicyInput,
    *,
    owned_requirement_ids: Iterable[str],
) -> HandoffPolicyResult:
    """Apply hard safety and coverage rules after the model assessment."""
    owned = tuple(dict.fromkeys(str(item) for item in owned_requirement_ids))
    coverage_by_id = {
        item.requirement_id: item
        for item in policy_input.requirement_coverage
    }
    hard_reasons: list[str] = []
    hard_reasons.extend(policy_input.additional_hard_rejection_reasons)
    if not owned:
        hard_reasons.append("owned_requirements_missing")
    if not policy_input.deterministic_checks_passed:
        hard_reasons.append("deterministic_checks_failed")
    if policy_input.evaluator_failed_closed:
        hard_reasons.append("quality_evaluator_failed_closed")
    if policy_input.unsupported_claims:
        hard_reasons.append("unsupported_claims")
    if min(policy_input.scores) < policy_input.dimension_floor:
        hard_reasons.append("score_below_dimension_floor")
    if (
        sum(policy_input.scores) / len(policy_input.scores)
        < policy_input.average_floor
    ):
        hard_reasons.append("score_below_average_floor")
    for requirement_id in owned:
        coverage = coverage_by_id.get(requirement_id)
        if coverage is None or coverage.status is not CoverageStatus.SUPPORTED:
            hard_reasons.append(
                f"required_coverage_missing:{requirement_id}"
            )

    caveats = tuple(
        dict.fromkeys(
            str(item).strip()
            for item in (
                *policy_input.caveats,
                *policy_input.missing_information,
            )
            if str(item).strip()
        )
    )
    if hard_reasons:
        return HandoffPolicyResult(
            admission_status=AdmissionStatus.REJECTED,
            accepted=False,
            caveats=caveats,
            hard_rejection_reasons=tuple(dict.fromkeys(hard_reasons)),
        )
    if caveats:
        if (
            not policy_input.caveat_admission_enabled
            or policy_input.high_risk
        ):
            reason = (
                "high_risk_caveats_disallowed"
                if policy_input.high_risk
                else "caveat_admission_disabled"
            )
            return HandoffPolicyResult(
                admission_status=AdmissionStatus.REJECTED,
                accepted=False,
                caveats=caveats,
                hard_rejection_reasons=(reason,),
            )
        return HandoffPolicyResult(
            admission_status=AdmissionStatus.ACCEPTED_WITH_CAVEATS,
            accepted=True,
            caveats=caveats,
            hard_rejection_reasons=(),
        )
    return HandoffPolicyResult(
        admission_status=AdmissionStatus.ACCEPTED,
        accepted=True,
        caveats=(),
        hard_rejection_reasons=(),
    )


def merge_coverage_ledger(
    ledger: dict[str, dict[str, Any]],
    *,
    task_id: str,
    assessment: Any,
    owned_requirement_ids: Iterable[str] = (),
) -> dict[str, dict[str, Any]]:
    """Merge an admitted assessment into a reducer-safe coverage ledger."""
    merged = {
        str(key): {
            "status": str(value.get("status", CoverageStatus.UNSUPPORTED.value)),
            "evidence_ids": list(value.get("evidence_ids", [])),
            "task_ids": list(value.get("task_ids", [])),
            "caveats": list(value.get("caveats", [])),
        }
        for key, value in ledger.items()
        if isinstance(value, dict)
    }
    admission_status = str(
        getattr(
            getattr(assessment, "admission_status", ""),
            "value",
            getattr(assessment, "admission_status", ""),
        )
    )
    if admission_status == AdmissionStatus.REJECTED.value:
        return merged
    assessment_caveats = [
        str(item) for item in getattr(assessment, "caveats", [])
    ]
    statuses = {
        CoverageStatus.UNSUPPORTED.value: 0,
        CoverageStatus.PARTIAL.value: 1,
        CoverageStatus.SUPPORTED.value: 2,
    }
    mapped_requirement_ids: set[str] = set()
    for coverage in getattr(assessment, "requirement_coverage", []):
        requirement_id = str(coverage.requirement_id)
        mapped_requirement_ids.add(requirement_id)
        current = merged.setdefault(
            requirement_id,
            {
                "status": CoverageStatus.UNSUPPORTED.value,
                "evidence_ids": [],
                "task_ids": [],
                "caveats": [],
            },
        )
        new_status = coverage.status.value
        if statuses[new_status] >= statuses.get(str(current["status"]), 0):
            current["status"] = new_status
        current["evidence_ids"] = list(
            dict.fromkeys(
                [
                    *current["evidence_ids"],
                    *(str(item) for item in coverage.evidence_ids),
                ]
            )
        )
        current["task_ids"] = list(
            dict.fromkeys([*current["task_ids"], task_id])
        )
        current["caveats"] = list(
            dict.fromkeys([*current["caveats"], *assessment_caveats])
        )
    for requirement_id in dict.fromkeys(
        str(item) for item in owned_requirement_ids if str(item)
    ):
        if requirement_id in mapped_requirement_ids:
            continue
        current = merged.setdefault(
            requirement_id,
            {
                "status": CoverageStatus.UNSUPPORTED.value,
                "evidence_ids": [],
                "task_ids": [],
                "caveats": [],
            },
        )
        if statuses.get(str(current["status"]), 0) < statuses[
            CoverageStatus.PARTIAL.value
        ]:
            current["status"] = CoverageStatus.PARTIAL.value
        current["task_ids"] = list(
            dict.fromkeys([*current["task_ids"], task_id])
        )
        current["caveats"] = list(
            dict.fromkeys([
                *current["caveats"],
                *assessment_caveats,
                "coverage_mapping_missing",
            ])
        )
    return merged
