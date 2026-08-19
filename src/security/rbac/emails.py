"""Email normalization & validation helpers."""

from __future__ import annotations

import unicodedata

try:
    import email_validator as _email_validator
except ModuleNotFoundError:  # pragma: no cover - allows import without email_validator
    _email_validator = None  # type: ignore[assignment]


class InvalidEmail(ValueError):
    """Raised when an email address is not acceptable."""


def normalize_email(email: str) -> str:
    """Return the canonical normalized form of ``email``.

    Lower-cases and NFC-normalizes after stripping whitespace. This is the
    value stored in ``iam_users.email_normalized`` (unique) and used for login.
    """
    return unicodedata.normalize("NFC", (email or "").strip()).lower()


def validate_and_normalize(email: str) -> str:
    """Validate ``email`` with ``email_validator`` and return its normalized form.

    Fails closed when the validator is unavailable: without it we cannot reject
    addresses (e.g. containing newlines) that would corrupt downstream MIME
    headers, so no address is accepted at all.
    """
    raw = (email or "").strip()
    if not raw:
        raise InvalidEmail("email_required")
    if _email_validator is None:
        raise InvalidEmail("email_validator_unavailable")
    try:
        info = _email_validator.validate_email(raw, check_deliverability=False)
    except _email_validator.EmailNotValidError as exc:
        raise InvalidEmail(f"invalid_email:{exc}") from exc
    return normalize_email(info.normalized)
