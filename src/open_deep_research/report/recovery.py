"""Deterministic evidence-only report used after a quality-gate rejection."""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlsplit

from open_deep_research.security.content import sanitize_report_markdown

_MAX_EVIDENCE_RECORDS = 40
_MAX_CLAIM_CHARS = 800
_MAX_EXCERPT_CHARS = 1200
_MAX_LOCATOR_CHARS = 300
_MAX_SOURCE_TITLE_CHARS = 300
_MAX_GAP_CHARS = 500
_MAX_REJECTION_REASON_CHARS = 1500
_MAX_ARTIFACT_REFS = 50
_URL_RE = re.compile(r"https?://[^\s)\]}>]+", re.IGNORECASE)
_MARKDOWN_LINK_RE = re.compile(
    r"!?(?<!\\)\[((?:\\.|[^\]\\\r\n])*)\]\(([^)\r\n]*)\)",
)
_MARKDOWN_DESTINATION_RE = re.compile(
    r"!?(?<!\\)\[(?:\\.|[^\]\\\r\n])*\]\(([^)\r\n]*)\)",
)


def _safe_source_url(value: Any) -> str:
    raw = str(value or "").strip()
    parsed = urlsplit(raw)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return ""
    return raw


def _safe_markdown_link_label(value: Any) -> str:
    """Render source metadata without allowing it to terminate a link label."""
    label = " ".join(str(value or "").split())
    label = _URL_RE.sub("[URL omitted]", label)
    return (
        label.replace("\\", "\\\\")
        .replace("[", "\\[")
        .replace("]", "\\]")
    )


def _deduplicate_evidence(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique: list[dict[str, Any]] = []
    seen: set[tuple[str, ...]] = set()
    for record in records:
        identity = (
            str(record.get("evidence_id") or ""),
            str(record.get("claim") or ""),
            str(record.get("supporting_excerpt") or ""),
            str(record.get("source_url") or ""),
        )
        if identity in seen:
            continue
        seen.add(identity)
        unique.append(record)
    return unique


def _bounded_text(value: Any, max_chars: int) -> str:
    text = str(value or "")
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "…"


def _inert_text(value: Any) -> str:
    """Render untrusted prose without preserving active links."""
    text = str(value or "")
    text = _MARKDOWN_LINK_RE.sub(
        lambda match: f"{match.group(1)} [link omitted]",
        text,
    )
    text = _URL_RE.sub("[URL omitted]", text)
    return text.replace("<", "&lt;").replace(">", "&gt;")


def _bounded_inert_text(value: Any, max_chars: int) -> str:
    return _bounded_text(_inert_text(value), max_chars)


def _markdown_destination(value: str) -> str:
    destination = value.strip()
    if destination.startswith("<") and ">" in destination:
        return destination[1 : destination.index(">")]
    return destination.split(maxsplit=1)[0] if destination else ""


def _validate_report_urls(report: str, allowed_urls: set[str]) -> None:
    """Reject any final URL that is not an evidence record's source URL."""
    rendered_urls = set(_URL_RE.findall(report))
    markdown_destinations = {
        destination
        for value in _MARKDOWN_DESTINATION_RE.findall(report)
        if (destination := _markdown_destination(value))
    }
    if not rendered_urls.issubset(allowed_urls) or not (
        markdown_destinations.issubset(allowed_urls)
    ):
        raise ValueError("evidence_recovery_rendered_url_not_allowlisted")


def build_evidence_recovery_report(
    evidence_records: list[dict[str, Any]],
    *,
    gaps: list[str],
    rejection_reasons: list[str],
    artifact_refs: list[dict[str, Any]],
) -> str:
    """Render accepted structured fields without using rejected prose artifacts."""
    all_evidence = _deduplicate_evidence(evidence_records)
    evidence = all_evidence[:_MAX_EVIDENCE_RECORDS]
    lines = [
        "# 质量门禁未通过的证据恢复报告",
        "",
        "> 警告：研究阶段已取得通过安全检查且可追溯的结构化证据，但质量门禁未接纳完整研究交接。以下内容由系统从 SHA-256 校验后的 accepted 证据字段确定性生成，不包含被拒绝的压缩研究或原始笔记。",
        "",
        "## 恢复状态",
        "",
    ]
    normalized_gaps = list(
        dict.fromkeys(
            _bounded_inert_text(item, _MAX_GAP_CHARS)
            for item in gaps
            if item
        )
    )
    normalized_reasons = list(
        dict.fromkeys(
            _bounded_inert_text(item, _MAX_REJECTION_REASON_CHARS)
            for item in rejection_reasons
            if item
        )
    )
    if normalized_gaps:
        lines.extend(["缺失项：", ""])
        lines.extend(f"- {item}" for item in normalized_gaps)
        lines.append("")
    if normalized_reasons:
        lines.extend(["拒绝原因：", ""])
        lines.extend(f"- {item}" for item in normalized_reasons)
        lines.append("")

    lines.extend(["## 已恢复的可追溯证据", ""])
    for index, record in enumerate(evidence, 1):
        evidence_id = str(record.get("evidence_id") or f"evidence-{index}")
        claim = _bounded_inert_text(
            record.get("claim") or "未提供结构化主张",
            _MAX_CLAIM_CHARS,
        )
        excerpt = _bounded_inert_text(
            record.get("supporting_excerpt"),
            _MAX_EXCERPT_CHARS,
        )
        locator = _bounded_inert_text(
            record.get("locator") or "未提供定位信息",
            _MAX_LOCATOR_CHARS,
        )
        title = _bounded_text(
            record.get("source_title") or "来源",
            _MAX_SOURCE_TITLE_CHARS,
        )
        title = _safe_markdown_link_label(title) or "来源"
        source_url = _safe_source_url(record.get("source_url"))
        lines.extend(
            [
                f"### 证据 {index}：{evidence_id}",
                "",
                f"主张：{claim}",
                "",
                "证据摘录：",
                "",
            ]
        )
        if excerpt:
            lines.extend(f"> {line}" for line in excerpt.splitlines() or [excerpt])
        else:
            lines.append("> 未提供摘录。")
        lines.extend(["", f"定位：{locator}", ""])
        lines.append(
            f"来源：[{title}]({source_url})" if source_url else f"来源：{title}"
        )
        lines.append("")

    omitted_evidence = len(all_evidence) - len(evidence)
    if omitted_evidence:
        lines.extend(
            [
                (
                    f"> 为控制恢复报告大小，其余 {omitted_evidence} 条证据"
                    "保留在校验后的研究工件中。"
                ),
                "",
            ]
        )

    lines.extend(["## 可恢复研究工件", ""])
    for ref in artifact_refs[:_MAX_ARTIFACT_REFS]:
        task_id = str(ref.get("task_id") or "unknown-task")
        path = str(ref.get("path") or "")
        sha256 = str(ref.get("sha256") or "")
        lines.append(f"- `{task_id}`：`{path}`，SHA-256 `{sha256}`")
    omitted_refs = len(artifact_refs) - min(
        len(artifact_refs),
        _MAX_ARTIFACT_REFS,
    )
    if omitted_refs:
        lines.append(f"- 其余 {omitted_refs} 个工件引用已省略。")
    lines.extend(
        [
            "",
            "本报告状态为 partial。待质量门禁恢复后，可基于以上工件发起独立复评；本报告不把被拒绝或隔离的内容作为研究结论。",
        ]
    )
    report = sanitize_report_markdown("\n".join(lines).strip())
    allowed_urls = {
        source_url
        for record in evidence
        if (source_url := _safe_source_url(record.get("source_url")))
    }
    _validate_report_urls(report, allowed_urls)
    return report
