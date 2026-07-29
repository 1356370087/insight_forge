"""Deterministic evidence-only report used after a quality-gate rejection."""

from __future__ import annotations

from typing import Any
from urllib.parse import urlsplit

from open_deep_research.security.content import sanitize_report_markdown


def _safe_source_url(value: Any) -> str:
    raw = str(value or "").strip()
    parsed = urlsplit(raw)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return ""
    return raw


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


def build_evidence_recovery_report(
    evidence_records: list[dict[str, Any]],
    *,
    gaps: list[str],
    rejection_reasons: list[str],
    artifact_refs: list[dict[str, Any]],
) -> str:
    """Render accepted structured fields without using rejected prose artifacts."""
    evidence = _deduplicate_evidence(evidence_records)
    lines = [
        "# 质量门禁未通过的证据恢复报告",
        "",
        "> 警告：研究阶段已取得通过安全检查且可追溯的结构化证据，但质量门禁未接纳完整研究交接。以下内容由系统从 SHA-256 校验后的 accepted 证据字段确定性生成，不包含被拒绝的压缩研究或原始笔记。",
        "",
        "## 恢复状态",
        "",
    ]
    normalized_gaps = list(dict.fromkeys(str(item) for item in gaps if item))
    normalized_reasons = list(
        dict.fromkeys(str(item) for item in rejection_reasons if item)
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
        claim = str(record.get("claim") or "未提供结构化主张")
        excerpt = str(record.get("supporting_excerpt") or "")
        locator = str(record.get("locator") or "未提供定位信息")
        title = str(record.get("source_title") or "来源")
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

    lines.extend(["## 可恢复研究工件", ""])
    for ref in artifact_refs:
        task_id = str(ref.get("task_id") or "unknown-task")
        path = str(ref.get("path") or "")
        sha256 = str(ref.get("sha256") or "")
        lines.append(f"- `{task_id}`：`{path}`，SHA-256 `{sha256}`")
    lines.extend(
        [
            "",
            "本报告状态为 partial。待质量门禁恢复后，可基于以上工件发起独立复评；本报告不把被拒绝或隔离的内容作为研究结论。",
        ]
    )
    return sanitize_report_markdown("\n".join(lines).strip())
