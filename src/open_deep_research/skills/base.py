"""Agent Skills: pluggable domain context for research/report orchestration.

A *skill* is a lightweight, declarative pack that provides curated prompt
context for a specific domain (e.g. medical, legal, finance), shaping how the
researcher gathers evidence and how the report writer presents it. v1 ships
context-only builtins; :func:`load_skill_tools` is the single extension point
for future tool-contributing skills and mirrors the MCP loading path so they
reuse the same governance gate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional


@dataclass(frozen=True)
class SkillSpec:
    """A domain skill pack.

    Attributes:
        key: Stable identifier matching the ``skills`` config value.
        researcher_context: Guidance appended to the researcher system prompt.
        report_context: Guidance prepended to the report writer prompt.
        description: Human-readable summary.
    """

    key: str
    researcher_context: str = ""
    report_context: str = ""
    description: str = ""


def _collect_context(keys: Optional[Iterable[str]], attr: str) -> str:
    """Concatenate the named context field of each enabled builtin skill."""
    # Local import to avoid importing the registry at module load.
    from .registry import get_skill

    parts: List[str] = []
    for key in keys or []:
        skill = get_skill(key)
        if skill is None:
            continue
        val = getattr(skill, attr, "") or ""
        if val and val not in parts:
            parts.append(val)
    return "\n\n".join(parts)


def get_skill_researcher_context(keys: Optional[Iterable[str]]) -> str:
    """Researcher system-prompt guidance from all enabled skills."""
    return _collect_context(keys, "researcher_context")


def get_skill_report_context(keys: Optional[Iterable[str]]) -> str:
    """Report writer-prompt guidance from all enabled skills."""
    return _collect_context(keys, "report_context")


async def load_skill_tools(config, existing_tool_names: Optional[set] = None) -> list:
    """Load tools contributed by enabled skills.

    v1 skills are context-only and contribute no tools. This remains the single
    extension point: future tool-contributing skills must return project
    ``Tool`` objects with ``origin=ToolOrigin.SKILL``. External implementations
    are adapted by ``tools.utils.get_all_tools`` before governance.
    """
    return []
