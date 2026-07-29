"""Deterministic tool categories shared by telemetry and evaluation."""

from __future__ import annotations

import re
from typing import Literal

ToolCategory = Literal["search", "fetch", "control", "mcp"]

_CONTROL_TOOLS = {
    "researchcomplete",
    "think_tool",
    "readresearchartifact",
    "conductresearch",
    "startresearchtask",
    "checkresearchtask",
    "listresearchtasks",
    "waitforresearchupdates",
    "updateresearchtask",
    "cancelresearchtask",
}
_FETCH_TOOLS = {
    "fetch_url",
    "web_research",
    "open_url",
    "read_url",
    "browse_url",
}
_SEARCH_TOKEN_RE = re.compile(r"(?:^|[_.:-])search(?:$|[_.:-])")
_FETCH_TOKEN_RE = re.compile(r"(?:^|[_.:-])(?:fetch|browse|open|read)(?:$|[_.:-])")


def classify_tool_name(name: str) -> ToolCategory:
    """Classify exact/tokenized tool names without substring false positives."""
    normalized = str(name or "").strip().lower()
    if normalized in _CONTROL_TOOLS:
        return "control"
    if normalized in _FETCH_TOOLS:
        return "fetch"
    if _SEARCH_TOKEN_RE.search(normalized):
        return "search"
    if _FETCH_TOKEN_RE.search(normalized):
        return "fetch"
    return "mcp"
