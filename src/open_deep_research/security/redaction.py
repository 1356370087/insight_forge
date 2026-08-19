"""Shared best-effort redaction for logs and observability payloads."""

from __future__ import annotations

import re

_SENSITIVE_TEXT_PATTERNS = (
    re.compile(r"(?i)(authorization\s*[:=]\s*['\"]?bearer\s+)[^\s,'\"}]+"),
    re.compile(
        r"(?i)((?:api[_-]?key|access[_-]?token|refresh[_-]?token|secret|password)"
        r"['\"]?\s*[:=]\s*['\"]?)[^\s,'\"}]+"
    ),
    re.compile(r"\b(?:sk|pk)-[A-Za-z0-9_-]{12,}\b"),
)


def redact_text(text: str) -> str:
    """Replace common credential shapes without raising on arbitrary text."""
    redacted = str(text)
    for pattern in _SENSITIVE_TEXT_PATTERNS:
        if pattern.groups:
            redacted = pattern.sub(r"\1[REDACTED]", redacted)
        else:
            redacted = pattern.sub("[REDACTED]", redacted)
    return redacted


__all__ = ["redact_text"]
