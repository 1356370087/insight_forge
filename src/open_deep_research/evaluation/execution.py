"""Deterministic evaluation of user-specified execution constraints."""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Literal
from urllib.parse import urlsplit, urlunsplit

from pydantic import BaseModel, Field

from open_deep_research.tool_taxonomy import classify_tool_name

_URL_RE = re.compile(
    r"https?://[^\s，。；、;,<>\"'()（）【】与和及]+",
    re.IGNORECASE,
)
_DOMAIN_RE = re.compile(
    r"(?:域名|domain)\s*(?:为|是|[:：])?\s*([a-z0-9.-]+\.[a-z]{2,})",
    re.IGNORECASE,
)
_NAMED_TOOL_RE = re.compile(
    r"(?:使用|调用|use|call)\s*(?:the\s+)?`?([A-Za-z][A-Za-z0-9_.:-]*)`?",
    re.IGNORECASE,
)
_NEGATIVE_TOOL_PREFIX_RE = re.compile(
    r"(?:(?:不得|禁止|不要|不可)\s*(?:使用|调用)?"
    r"|(?:do\s+not|must\s+not|never)\s+(?:use|call)?\s*(?:the\s+)?)\s*$",
    re.IGNORECASE,
)
_TOOL_CAPTURE_STOPWORDS = {
    "a",
    "an",
    "any",
    "exactly",
    "only",
    "the",
    "these",
    "this",
    "those",
}
_PROCESS_MARKER_RE = re.compile(
    r"(?:ConductResearch|fetch_url|工具|tool|子任务|task|研究员|researcher|"
    r"搜索|search|来源|source|URL|域名|domain)",
    re.IGNORECASE,
)
_NORMATIVE_RE = re.compile(r"(?:必须|不得|禁止|只(?:能|创建|使用)?|must|do not|only|forbid)", re.IGNORECASE)
_REPORT_MARKER_RE = re.compile(r"(?:最终报告|报告必须|final report|report must)", re.IGNORECASE)


class ExecutionConstraints(BaseModel):
    """Recognized deterministic constraints extracted from the user message."""

    conduct_research_count: int | None = None
    required_tools: list[str] = Field(default_factory=list)
    forbidden_categories: list[str] = Field(default_factory=list)
    required_urls: list[str] = Field(default_factory=list)
    allowed_urls: list[str] = Field(default_factory=list)
    allowed_domains: list[str] = Field(default_factory=list)
    same_task_for_urls: bool = False
    unverifiable_clauses: list[str] = Field(default_factory=list)

    @property
    def applicable(self) -> bool:
        """Return whether the user supplied any execution constraint."""
        return bool(
            self.conduct_research_count is not None
            or self.required_tools
            or self.forbidden_categories
            or self.required_urls
            or self.allowed_urls
            or self.allowed_domains
            or self.same_task_for_urls
            or self.unverifiable_clauses
        )


class ExecutionComplianceResult(BaseModel):
    """Non-compensatory execution compliance outcome."""

    applicable: bool
    status: Literal["not_applicable", "passed", "failed", "evaluator_error"]
    score: float | None = None
    reason_codes: list[str] = Field(default_factory=list)
    details: dict[str, Any] = Field(default_factory=dict)
    constraints: ExecutionConstraints


def _normalize_url(raw_url: str) -> str:
    parsed = urlsplit(raw_url.rstrip("/.,;:"))
    path = parsed.path or "/"
    return urlunsplit(
        (
            parsed.scheme.lower(),
            parsed.netloc.lower(),
            path.rstrip("/") or "/",
            parsed.query,
            "",
        )
    )


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def _negative_tool_match(text: str, match_start: int) -> bool:
    prefix = text[max(0, match_start - 32) : match_start]
    return bool(_NEGATIVE_TOOL_PREFIX_RE.search(prefix))


def extract_execution_constraints(question: str) -> ExecutionConstraints:
    """Extract only workflow constraints that can be checked from the snapshot."""
    text = str(question or "")
    urls = _unique([_normalize_url(url) for url in _URL_RE.findall(text)])
    lower = text.lower()
    conduct_count: int | None = None
    if re.search(
        r"(?:只创建一个|仅创建一个|恰好一个|只从一个|仅从一个|"
        r"exactly one|only one)\s*ConductResearch",
        text,
        re.IGNORECASE,
    ):
        conduct_count = 1
    else:
        count_match = re.search(
            r"(?:创建|调用|use|create|call)\s*(\d+)\s*(?:个\s*)?ConductResearch",
            text,
            re.IGNORECASE,
        )
        if count_match:
            conduct_count = int(count_match.group(1))

    required_tools: list[str] = []
    for match in _NAMED_TOOL_RE.finditer(text):
        name = match.group(1).rstrip(".,;:!?")
        prefix = text[max(0, match.start() - 24) : match.start()]
        if (
            not name
            or name.lower() in _TOOL_CAPTURE_STOPWORDS
            or _negative_tool_match(text, match.start())
            or _REPORT_MARKER_RE.search(prefix)
        ):
            continue
        required_tools.append(name)

    no_search = bool(
        re.search(
            r"(?:不得|禁止|不要|不可)\s*(?:使用)?\s*(?:任何)?(?:搜索|search)|"
            r"(?:do not|must not|never)\s+(?:use\s+)?(?:web\s+)?search|no\s+search",
            text,
            re.IGNORECASE,
        )
    )
    same_task = bool(
        urls
        and (
            re.search(r"(?:同一|同一个).{0,24}(?:任务|task|研究员)", text, re.IGNORECASE)
            or re.search(r"(?:不得拆分|do not split|same (?:researcher|task))", text, re.IGNORECASE)
        )
    )
    source_restricted = bool(
        re.search(
            r"(?:不得|禁止|不要).{0,12}(?:二手|其他).{0,6}来源|"
            r"(?:only|仅|只).{0,24}(?:these|上述|指定).{0,8}(?:URLs?|来源)|"
            r"(?:no|without)\s+secondary\s+sources?",
            text,
            re.IGNORECASE,
        )
    )
    domains = _unique([domain.lower() for domain in _DOMAIN_RE.findall(text)])

    unverifiable: list[str] = []
    for clause in re.split(r"[。；;\n]+", text):
        compact = clause.strip()
        if not compact or _REPORT_MARKER_RE.search(compact):
            continue
        if not (_PROCESS_MARKER_RE.search(compact) and _NORMATIVE_RE.search(compact)):
            continue
        clause_known = bool(
            re.search(r"ConductResearch|fetch_url|不得拆分|do not split|same (?:researcher|task)", compact, re.IGNORECASE)
            or (no_search and re.search(r"搜索|search", compact, re.IGNORECASE))
            or (source_restricted and re.search(r"来源|source|URL", compact, re.IGNORECASE))
            or any(name.lower() in compact.lower() for name in required_tools)
            or any(domain in compact.lower() for domain in domains)
        )
        if not clause_known:
            unverifiable.append(compact[:300])

    return ExecutionConstraints(
        conduct_research_count=conduct_count,
        required_tools=_unique(required_tools),
        forbidden_categories=["search"] if no_search else [],
        required_urls=urls if (same_task or "fetch_url" in lower) else [],
        allowed_urls=urls if source_restricted and urls else [],
        allowed_domains=domains,
        same_task_for_urls=same_task,
        unverifiable_clauses=_unique(unverifiable),
    )


def is_execution_requirement(requirement: str) -> bool:
    """Return whether a checklist item describes workflow rather than report content."""
    text = str(requirement or "")
    if _REPORT_MARKER_RE.search(text):
        return False
    workflow_verb = re.search(
        r"(?:使用|调用|同一|拆分|use|call|same task|same researcher|split)",
        text,
        re.IGNORECASE,
    )
    return bool(
        _PROCESS_MARKER_RE.search(text)
        and (_NORMATIVE_RE.search(text) or workflow_verb)
    )


def content_coverage_requirements(requirements: list[str]) -> list[str]:
    """Remove execution-only constraints from a report completeness checklist."""
    return [item for item in requirements if not is_execution_requirement(item)]


def _trace_calls(trace: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    supervisor = [
        call
        for call in trace.get("supervisor_tool_calls", [])
        if isinstance(call, dict)
    ]
    researcher = [
        call
        for call in trace.get("researcher_tool_calls", [])
        if isinstance(call, dict)
    ]
    return supervisor, researcher


def evaluate_execution_compliance(
    question: str,
    tool_trace: dict[str, Any] | None,
    evidence_registry: list[dict[str, Any]] | None = None,
) -> ExecutionComplianceResult:
    """Evaluate recognized process constraints without calling an LLM."""
    constraints = extract_execution_constraints(question)
    if not constraints.applicable:
        return ExecutionComplianceResult(
            applicable=False,
            status="not_applicable",
            constraints=constraints,
        )
    if constraints.unverifiable_clauses:
        return ExecutionComplianceResult(
            applicable=True,
            status="evaluator_error",
            reason_codes=["unverifiable_execution_constraint"],
            details={"clauses": constraints.unverifiable_clauses},
            constraints=constraints,
        )
    if not isinstance(tool_trace, dict):
        return ExecutionComplianceResult(
            applicable=True,
            status="evaluator_error",
            reason_codes=["tool_trace_missing"],
            constraints=constraints,
        )

    supervisor_calls, researcher_calls = _trace_calls(tool_trace)
    requires_researcher_trace = bool(
        constraints.required_tools
        or constraints.forbidden_categories
        or constraints.required_urls
        or constraints.allowed_urls
        or constraints.allowed_domains
        or constraints.same_task_for_urls
    )
    availability = tool_trace.get("availability", {})
    if (
        requires_researcher_trace
        and not researcher_calls
        and not (
            isinstance(availability, dict)
            and availability.get("researcher_tool_names_retained") is True
        )
    ):
        return ExecutionComplianceResult(
            applicable=True,
            status="evaluator_error",
            reason_codes=["researcher_tool_trace_missing"],
            constraints=constraints,
        )

    failures: list[str] = []
    details: dict[str, Any] = {}
    all_calls = [*supervisor_calls, *researcher_calls]
    if constraints.conduct_research_count is not None:
        observed = sum(
            str(call.get("name", "")) == "ConductResearch"
            for call in supervisor_calls
        )
        details["conduct_research_count"] = observed
        if observed != constraints.conduct_research_count:
            failures.append("conduct_research_count_mismatch")

    observed_names = {str(call.get("name", "")) for call in all_calls}
    missing_tools = [
        name for name in constraints.required_tools if name not in observed_names
    ]
    if missing_tools:
        failures.append("required_tool_missing")
        details["missing_tools"] = missing_tools

    forbidden = [
        str(call.get("name", ""))
        for call in all_calls
        if classify_tool_name(str(call.get("name", "")))
        in constraints.forbidden_categories
    ]
    if forbidden:
        failures.append("forbidden_tool_used")
        details["forbidden_tools"] = forbidden

    url_tasks: dict[str, set[str]] = defaultdict(set)
    observed_urls: list[str] = []
    for call in researcher_calls:
        args = call.get("args", {})
        if not isinstance(args, dict):
            continue
        raw_url = args.get("url")
        if not isinstance(raw_url, str) or not raw_url.startswith(("http://", "https://")):
            continue
        normalized = _normalize_url(raw_url)
        observed_urls.append(normalized)
        url_tasks[normalized].add(str(call.get("task_id", "")))

    missing_urls = [url for url in constraints.required_urls if url not in url_tasks]
    if missing_urls:
        failures.append("required_url_not_read")
        details["missing_urls"] = missing_urls
    if constraints.same_task_for_urls and not missing_urls:
        common_tasks = set.intersection(
            *(url_tasks[url] for url in constraints.required_urls)
        ) if constraints.required_urls else set()
        common_tasks.discard("")
        if not common_tasks:
            failures.append("required_urls_not_read_by_same_task")

    evidence_urls = [
        _normalize_url(str(record.get("source_url")))
        for record in (evidence_registry or [])
        if isinstance(record, dict)
        and record.get("source_url")
        and str(record.get("security_status", "accepted")).lower()
        not in {"quarantined", "rejected", "blocked"}
    ]
    audited_urls = _unique([*observed_urls, *evidence_urls])
    if constraints.allowed_urls:
        outside = [url for url in audited_urls if url not in constraints.allowed_urls]
        if outside:
            failures.append("source_outside_allowed_urls")
            details["outside_allowed_urls"] = outside
    if constraints.allowed_domains:
        outside_domains = [
            url
            for url in audited_urls
            if (urlsplit(url).hostname or "").lower()
            not in constraints.allowed_domains
        ]
        if outside_domains:
            failures.append("source_outside_allowed_domains")
            details["outside_allowed_domains"] = outside_domains

    return ExecutionComplianceResult(
        applicable=True,
        status="failed" if failures else "passed",
        score=0.0 if failures else 1.0,
        reason_codes=_unique(failures),
        details=details,
        constraints=constraints,
    )
