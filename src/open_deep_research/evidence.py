"""Shared evidence eligibility rules used across the research lifecycle."""

from __future__ import annotations

from typing import Any, Iterable, cast


def is_evidence_eligible(record: object) -> bool:
    """Return whether one evidence record passed the security admission gate."""
    return (
        isinstance(record, dict)
        and record.get("security_status", "accepted") == "accepted"
    )


def eligible_evidence_records(records: Iterable[object]) -> list[dict[str, Any]]:
    """Return admitted evidence records while preserving their original order."""
    return [
        cast(dict[str, Any], record)
        for record in records
        if is_evidence_eligible(record)
    ]
