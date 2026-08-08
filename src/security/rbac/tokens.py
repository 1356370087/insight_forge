"""Opaque one-time tokens (email verification & password reset).

Tokens are 256-bit cryptographically-random URL-safe strings. Only an
HMAC-SHA256 digest is persisted, so a database leak never yields usable
tokens. Lookup is by digest; the raw token is shown to the user exactly once
(at creation time) and never stored.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets

from .settings import get_settings

# 32 bytes -> 256 bits of entropy -> ~43-char URL-safe token.
_TOKEN_BYTES = 32


def generate_token() -> str:
    """Return a fresh 256-bit URL-safe opaque token."""
    return secrets.token_urlsafe(_TOKEN_BYTES)


def _secret() -> bytes:
    """Return the HMAC secret, deriving a stable per-deployment key when unset.

    The secret mixes the configured JWT access signing key (or a fixed dev
    fallback) so digests are bound to the deployment. Digests are only ever
    compared against digests produced with the same key, so collision
    resistance against external attackers is preserved even with the fallback.
    """
    settings = get_settings()
    base = settings.token_digest_secret or settings.access_signing_key_pem
    if not base:
        if settings.is_production:
            raise RuntimeError("IAM_TOKEN_DIGEST_SECRET is required in production")
        base = "odr-dev-email-token-secret"
    return hashlib.sha256(base.encode("utf-8")).digest()


def digest_token(token: str, *, purpose: str) -> str:
    """Return the HMAC-SHA256 digest of ``token`` scoped to ``purpose``.

    The purpose is mixed into the HMAC so a digest from one flow (e.g. email
    verification) cannot be replayed in another (e.g. password reset).
    """
    msg = f"{purpose}:{token}".encode()
    return hmac.new(_secret(), msg, hashlib.sha256).hexdigest()


def constant_time_eq(a: str, b: str) -> bool:
    """Return whether two digest strings are equal in constant time."""
    return hmac.compare_digest(a.encode("utf-8"), b.encode("utf-8"))


def generate_token_with_digest(purpose: str) -> tuple[str, str]:
    """Generate a token and its digest as a ``(raw_token, digest)`` pair."""
    token = generate_token()
    return token, digest_token(token, purpose=purpose)
