"""Deterministic coverage checklist extraction for report planning and evaluation."""

from __future__ import annotations

import re

_TAG_RE = re.compile(r"<[^>]+>")
_SPACE_RE = re.compile(r"\s+")
_CLAUSE_RE = re.compile(r"[。；;\n]+|(?<=\S)，(?=(?:并|以及|同时|结合|区分|说明|提出|给出|评估|比较))")
_LIST_RE = re.compile(r"[、,，]+")
_LEADING_RE = re.compile(
    r"^(?:截至\S+?[，,]\s*)?(?:请|需要|应当|应如何|报告应)?(?:重点)?(?:覆盖|比较|评估|分析|说明|区分|提出|给出)?\s*"
)


def derive_coverage_checklist(text: str, *, max_items: int = 20) -> list[str]:
    """Extract explicit deliverables from a user question or research brief.

    The checklist is deliberately deterministic: it introduces no new factual
    requirements and can be reproduced by the writer and the Judge.
    """
    clean = _SPACE_RE.sub(" ", _TAG_RE.sub(" ", text or "")).strip()
    clean = re.sub(r"^截至[^，,。；;]+[，,]\s*", "", clean)
    if not clean:
        return []
    candidates: list[str] = []
    for clause in _CLAUSE_RE.split(clean):
        clause = _LEADING_RE.sub("", clause).strip(" ：:。. ")
        if not clause:
            continue
        parts = _LIST_RE.split(clause)
        candidates.extend(part.strip(" ：:。. ") for part in parts)

    checklist: list[str] = []
    seen: set[str] = set()
    for item in candidates:
        item = re.sub(r"^(?:以及|并且|同时|并|和|与)\s*", "", item).strip()
        key = re.sub(r"\W+", "", item).lower()
        if len(key) < 4 or key in seen:
            continue
        seen.add(key)
        checklist.append(item[:240])
        if len(checklist) >= max_items:
            break
    return checklist or [clean[:240]]


def render_coverage_checklist(items: list[str]) -> str:
    """Render checklist instructions for a report writer without exposing them."""
    if not items:
        return ""
    rows = "\n".join(f"COV-{index:02d}: {item}" for index, item in enumerate(items, 1))
    return (
        "<Coverage Checklist>\n"
        "Use this checklist internally before and after drafting. Address every item "
        "with evidence, or explicitly state that evidence is unavailable/uncertain. "
        "Do not silently omit an item and do not print checklist IDs in the report.\n"
        f"{rows}\n"
        "</Coverage Checklist>"
    )
