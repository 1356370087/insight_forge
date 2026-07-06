"""Builtin domain skill packs (context-only in v1).

Each pack shapes researcher evidence-gathering and report presentation for a
domain. They are deliberately prompt-only — no code execution, no egress — so
they are safe to enable by configuration without governance review.
"""

from __future__ import annotations

from typing import Dict, Optional

from .base import SkillSpec

MEDICAL = SkillSpec(
    key="medical",
    researcher_context=(
        "Domain guidance (MEDICAL): Prioritize peer-reviewed and authoritative clinical "
        "sources (e.g. PubMed, Cochrane, WHO, CDC). Note study design, sample size, and "
        "whether evidence is preclinical or clinical. Distinguish scientific consensus "
        "from emerging or contested findings, and pair each claim with its evidence level."
    ),
    report_context=(
        "For key medical claims, include the evidence level/study design, and add a "
        "clear 'This is informational and not medical advice' disclaimer."
    ),
    description="Medical / clinical research context.",
)

LEGAL = SkillSpec(
    key="legal",
    researcher_context=(
        "Domain guidance (LEGAL): Prefer primary authority (statutes, regulations, "
        "binding case law) over secondary commentary. Always note jurisdiction and "
        "whether authority is binding or merely persuasive. Flag currency (recent "
        "amendments) and distinguish settled law from open questions."
    ),
    report_context=(
        "Cite legal authorities with their jurisdiction, and note that this is not "
        "legal advice."
    ),
    description="Legal research context.",
)

FINANCE = SkillSpec(
    key="finance",
    researcher_context=(
        "Domain guidance (FINANCE): Prefer primary filings and official statistics "
        "(regulator filings, central-bank data, audited financial statements). Note the "
        "reporting period, currency, and whether figures are nominal or real. Clearly "
        "flag forward-looking statements as estimates/forecasts."
    ),
    report_context=(
        "Add a disclaimer that financial figures are time-sensitive and this is not "
        "investment advice."
    ),
    description="Finance / economic research context.",
)


BUILTIN_SKILLS: Dict[str, SkillSpec] = {
    "medical": MEDICAL,
    "legal": LEGAL,
    "finance": FINANCE,
}


def get_skill(key: str) -> Optional[SkillSpec]:
    """Return a builtin skill by key, or ``None`` if unknown (never raises)."""
    return BUILTIN_SKILLS.get(key)
