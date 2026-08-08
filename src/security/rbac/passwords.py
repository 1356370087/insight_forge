"""Password hashing (Argon2id) and policy validation.

Uses ``pwdlib``'s Argon2 hasher — FastAPI's recommended choice — with the
OWASP/NIST-recommended policy: 15–128 Unicode characters, paste & password
manager friendly, no artificial composition rules, with screening for the most
common leaked passwords.
"""

from __future__ import annotations

import hmac
import logging
import unicodedata

from pwdlib import PasswordHash
from pwdlib.hashers.argon2 import Argon2Hasher

logger = logging.getLogger(__name__)

MIN_LENGTH = 15
MAX_LENGTH = 128

# Argon2id with OWASP-aligned parameters. ``memory_cost`` is in KiB.
_password_hasher = PasswordHash((
    Argon2Hasher(
        memory_cost=19456,   # 19 MiB
        time_cost=2,
        parallelism=1,
    ),
))

# A tiny built-in denylist of the most common passwords. This is intentionally
# small (a production deployment should layer a breached-password service); it
# still blocks the worst offenders (``123456789012345``, ``passwordpassword``).
COMMON_PASSWORDS: frozenset[str] = frozenset(
    p.lower()
    for p in (
        "123456789012345", "1234567890123456", "111111111111111", "000000000000000",
        "passwordpassword", "qwertyqwerty123", "asdfasdfasdf12", "aaaaaaaaaaaaaaa",
        "abcdefghijklmno", "abcabcabcabcab", "password1234567",
    )
)


class PasswordPolicyError(ValueError):
    """Raised when a password fails the length/composition policy."""


def _normalize(value: str) -> str:
    """Return NFC-normalized password text (NIST recommends Unicode normalization)."""
    return unicodedata.normalize("NFC", value)


def hash_password(password: str) -> str:
    """Return an Argon2id hash for ``password``."""
    return _password_hasher.hash(_normalize(password))


def verify_password(password: str, password_hash: str) -> bool:
    """Return whether ``password`` matches the stored ``password_hash``."""
    try:
        return _password_hasher.verify(_normalize(password), password_hash)
    except (ValueError, TypeError):
        return False


def needs_rehash(password_hash: str) -> bool:
    """Return whether ``password_hash`` should be upgraded on next successful login."""
    # pwdlib marks outdated parameters via the leading algorithm identifier.
    if not password_hash:
        return True
    return not password_hash.startswith("$argon2id$")


def validate_password(password: str) -> str:
    """Validate ``password`` against the policy and return the normalized form.

    Raises:
        PasswordPolicyError: if the password is too short/long or trivially common.
    """
    if password is None:
        raise PasswordPolicyError("password_required")
    normalized = _normalize(password)
    length = len(normalized)
    if length < MIN_LENGTH:
        raise PasswordPolicyError(f"password_too_short:min_{MIN_LENGTH}")
    if length > MAX_LENGTH:
        raise PasswordPolicyError(f"password_too_long:max_{MAX_LENGTH}")
    # Disallow control characters (everything else, incl. emoji & spaces, is fine).
    if any(unicodedata.category(ch).startswith("C") for ch in normalized):
        raise PasswordPolicyError("password_contains_control_chars")
    if normalized.lower() in COMMON_PASSWORDS:
        raise PasswordPolicyError("password_too_common")
    return normalized


def constant_time_verify(a: str, b: str) -> bool:
    """Compare two digests in constant time."""
    return hmac.compare_digest(a.encode("utf-8"), b.encode("utf-8"))


def dummy_hash() -> str:
    """Return a precomputed Argon2id hash to keep login timing uniform.

    When the supplied email does not exist, the login path performs one dummy
    Argon2 verification so response timing does not reveal account existence.
    """
    return _DUMMY_HASH


# Precompute a dummy hash lazily on import so timing is dominated by verify().
_DUMMY_HASH: str = _password_hasher.hash("dummy-password-for-timing-equality")


def verify_dummy(password: str) -> bool:  # pragma: no cover - timing helper
    """Always-false verify against the dummy hash (used for uniform timing)."""
    return verify_password(password, _DUMMY_HASH)


__all__ = [
    "MAX_LENGTH",
    "MIN_LENGTH",
    "PasswordPolicyError",
    "constant_time_verify",
    "dummy_hash",
    "hash_password",
    "needs_rehash",
    "validate_password",
    "verify_dummy",
    "verify_password",
]
