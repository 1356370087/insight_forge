"""Deterministic coverage checklist extraction for report planning and evaluation."""

from __future__ import annotations

import re

_TAG_RE = re.compile(r"<[^>]+>")
_SPACE_RE = re.compile(r"\s+")
_CLAUSE_RE = re.compile(r"[。；;\n]+|(?<=\S)，(?=并|以及|同时|结合|区分|说明|提出|给出|评估|比较)")
_LIST_RE = re.compile(r"[、,，]+")
_FINAL_LIST_CONJUNCTION_RE = re.compile(r"\s*(?:以及|和|与)\s*(?=\S)")
_NUMBERED_ITEM_RE = re.compile(r"(?<!\w)(\d{1,2})[.)]\s+")
_GLOBAL_DIRECTIVE_RE = re.compile(
    r"(?<=[.!?。！？])\s+(?=(?:for all\b|finally\b|additionally\b|also\b|最后|此外|并(?:最终|另外)))",
    re.IGNORECASE,
)
_LEADING_RE = re.compile(
    r"^(?:截至\S+?[，,]\s*)?(?:请|需要|应当|应如何|报告应)?(?:重点)?(?:覆盖|比较|评估|分析|说明|区分|提出|给出)?\s*"
)


def _numbered_candidates(clean: str) -> list[str] | None:
    """Keep numbered deliverables atomic, including their internal commas."""
    markers = list(_NUMBERED_ITEM_RE.finditer(clean))
    ordinals = [int(marker.group(1)) for marker in markers]
    if (
        len(markers) < 2
        or ordinals[0] != 1
        or any(
            current != previous + 1
            for previous, current in zip(ordinals, ordinals[1:])
        )
    ):
        return None

    candidates: list[str] = []
    preamble = clean[: markers[0].start()].strip(" ：:。. ")
    if preamble:
        candidates.append(preamble)
    for index, marker in enumerate(markers):
        end = (
            markers[index + 1].start()
            if index + 1 < len(markers)
            else len(clean)
        )
        item = clean[marker.end() : end].strip(" ：:。. ")
        candidates.extend(
            part.strip(" ：:。. ")
            for part in _GLOBAL_DIRECTIVE_RE.split(item)
            if part.strip(" ：:。. ")
        )
    return candidates


def derive_coverage_checklist(text: str, *, max_items: int = 20) -> list[str]:
    """Extract explicit deliverables from a user question or research brief.

    The checklist is deliberately deterministic: it introduces no new factual
    requirements and can be reproduced by the writer and the Judge.
    """
    clean = _SPACE_RE.sub(" ", _TAG_RE.sub(" ", text or "")).strip()
    clean = re.sub(r"^截至[^，,。；;]+[，,]\s*", "", clean)
    if not clean:
        return []
    candidates = _numbered_candidates(clean)
    if candidates is None:
        candidates = []
        for clause in _CLAUSE_RE.split(clean):
            clause = _LEADING_RE.sub("", clause).strip(" ：:。. ")
            if not clause:
                continue
            parts = _LIST_RE.split(clause)
            # Chinese enumerations commonly use a delimiter for the first
            # items and a conjunction for the final pair (``A、B 和 C``).
            # Once an explicit list delimiter is present, keep the final item
            # atomic as well instead of merging two independently delegable
            # coverage requirements into one impossible Subagent contract.
            if len(parts) > 1:
                expanded_parts = [
                    nested
                    for part in parts
                    for nested in _FINAL_LIST_CONJUNCTION_RE.split(part)
                ]
            else:
                expanded_parts = parts
            candidates.extend(
                part.strip(" ：:。. ") for part in expanded_parts
            )

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
        checklist.append(item[:500])
        if len(checklist) >= max_items:
            break
    return checklist or [clean[:500]]


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
