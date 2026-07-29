"""Deterministic coverage checklist extraction for report planning and evaluation."""

from __future__ import annotations

import re

_TAG_RE = re.compile(r"<[^>]+>")
_SPACE_RE = re.compile(r"\s+")
_CLAUSE_RE = re.compile(r"[。；;\n]+|(?<=\S)，(?=并|以及|同时|结合|区分|说明|提出|给出|评估|比较)")
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
        # Explicit list items such as "成本" and "安全性" are valid, compact
        # requirements.  Reject only one-character fragments, which are much
        # more likely to be punctuation/splitting noise.
        if len(key) < 2 or key in seen:
            continue
        seen.add(key)
        checklist.append(item[:240])
        if len(checklist) >= max_items:
            break
    return checklist or [clean[:240]]


def derive_state_coverage_checklist(
    state: dict,
    *,
    max_items: int = 20,
) -> list[str]:
    """Derive requirements from original user messages before the model brief."""
    source_texts: list[str] = []
    for message in state.get("messages", []):
        if isinstance(message, dict):
            role = str(message.get("role") or message.get("type") or "")
            content = message.get("content", "")
        else:
            role = str(getattr(message, "type", ""))
            content = getattr(message, "content", "")
        if role in {"user", "human"} and content:
            source_texts.append(str(content))
    if not source_texts and state.get("research_brief"):
        source_texts.append(str(state["research_brief"]))

    requirements: list[str] = []
    seen: set[str] = set()
    for text in source_texts:
        for requirement in derive_coverage_checklist(text, max_items=max_items):
            key = re.sub(r"\W+", "", requirement).lower()
            if not key or key in seen:
                continue
            seen.add(key)
            requirements.append(requirement)
            if len(requirements) >= max_items:
                return requirements
    return requirements


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
