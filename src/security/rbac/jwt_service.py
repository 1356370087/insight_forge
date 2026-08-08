"""JWT encoding/decoding for access & refresh tokens.

Both token kinds use EdDSA (Ed25519) but with *independent* keys, ``kid``,
``typ`` and ``aud`` claims, so an access key can never mint a refresh token
and vice versa. Decoding pins the algorithm and the trusted ``kid`` set, and
rejects any token whose header claims a different algorithm — the header is
never trusted for algorithm selection (RFC 8725).
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass
from typing import Literal
from uuid import uuid4

import jwt as pyjwt

from .keys import KeyKind, KeyMaterial, load_key_material
from .settings import (
    ACCESS_AUDIENCE,
    ACCESS_TOKEN_TYPE,
    JWT_ALGORITHM,
    REFRESH_AUDIENCE,
    REFRESH_TOKEN_TYPE,
    get_settings,
)

logger = logging.getLogger(__name__)

TokenKind = Literal["access", "refresh"]

_ACCESS_CLAIM = "odr-access+jwt"
_REFRESH_CLAIM = "odr-refresh+jwt"


@dataclass(frozen=True)
class DecodedToken:
    """A successfully verified token with its normalized claims."""

    claims: dict
    kind: TokenKind


class TokenError(Exception):
    """Raised when a token is malformed, expired, or fails verification."""


def _material(kind: KeyKind, settings) -> KeyMaterial:
    """Load (and cache) key material for ``kind``.

    Caching is keyed by an env signature so swapping JWT settings in tests
    invalidates the cache. Note that :class:`KeyMaterial` is frozen, so the
    signature is stored alongside the material in a ``(material, sig)`` tuple.
    """
    cache_key = f"__jwt_material_{kind}"
    cached = getattr(_material, cache_key, None)
    current_sig = _env_signature(settings)
    if cached is None or cached[1] != current_sig:
        material = load_key_material(kind, settings=settings)
        setattr(_material, cache_key, (material, current_sig))
        return material
    return cached[0]


def _env_signature(settings) -> str:
    """Return a cheap signature of the key-bearing settings for cache invalidation."""
    raw = "|".join(
        (
            settings.access_signing_kid,
            settings.refresh_signing_kid,
            settings.app_env,
            settings.access_signing_key_pem,
            settings.refresh_signing_key_pem,
            settings.access_verify_keys_json,
            settings.refresh_verify_keys_json,
        )
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _kind_params(kind: TokenKind) -> tuple[str, str]:
    """Return ``(audience, token_type)`` for ``kind``."""
    if kind == "access":
        return ACCESS_AUDIENCE, ACCESS_TOKEN_TYPE
    return REFRESH_AUDIENCE, REFRESH_TOKEN_TYPE


def encode_access_token(
    *,
    subject: str,
    session_id: str,
    authz_version: int,
    settings=None,
) -> tuple[str, str]:
    """Issue an access token; returns ``(jwt, jti)``.

    Lifetime is ``settings.access_token_ttl``. The ``authz_version`` claim lets
    the server reject tokens whose session was superseded by a security event.
    """
    settings = settings or get_settings()
    return _encode(
        kind="access",
        subject=subject,
        session_id=session_id,
        ttl=settings.access_token_ttl,
        extra={"authz_version": int(authz_version)},
        settings=settings,
    )


def encode_refresh_token(
    *,
    subject: str,
    session_id: str,
    jti: str | None = None,
    settings=None,
) -> tuple[str, str]:
    """Issue a refresh token; returns ``(jwt, jti)``.

    ``jti`` should normally be left ``None`` so a fresh id is generated. Callers
    pass an explicit ``jti`` only when persisting a pre-generated jti.
    """
    settings = settings or get_settings()
    return _encode(
        kind="refresh",
        subject=subject,
        session_id=session_id,
        ttl=settings.refresh_idle_ttl,
        extra={},
        settings=settings,
        jti=jti,
    )


def _encode(
    *,
    kind: TokenKind,
    subject: str,
    session_id: str,
    ttl: int,
    extra: dict,
    settings,
    jti: str | None = None,
) -> tuple[str, str]:
    audience, token_type = _kind_params(kind)
    material = _material("access" if kind == "access" else "refresh", settings)
    now = int(time.time())
    token_id = jti or uuid4().hex
    claims = {
        "iss": settings.issuer,
        "aud": audience,
        "sub": str(subject),
        "sid": str(session_id),
        "jti": token_id,
        "iat": now,
        "nbf": now,
        "exp": now + int(ttl),
        **extra,
    }
    headers = {"kid": material.signing_kid, "typ": token_type}
    token = pyjwt.encode(
        claims,
        material.private_pem(),
        algorithm=JWT_ALGORITHM,
        headers=headers,
    )
    return token, token_id


def decode_access_token(token: str, settings=None) -> DecodedToken:
    """Verify and decode an access token."""
    return _decode(token, kind="access", settings=settings or get_settings())


def decode_refresh_token(token: str, settings=None) -> DecodedToken:
    """Verify and decode a refresh token."""
    return _decode(token, kind="refresh", settings=settings or get_settings())


def _decode(token: str, *, kind: TokenKind, settings) -> DecodedToken:
    audience, token_type = _kind_params(kind)
    material = _material("access" if kind == "access" else "refresh", settings)

    # First inspect the unverified header to pin kid & typ before trusting key
    # material; reject alg/typ/kid mismatches explicitly.
    try:
        header = pyjwt.get_unverified_header(token)
    except pyjwt.PyJWTError as exc:
        raise TokenError("malformed_token") from exc
    if header.get("alg") != JWT_ALGORITHM:
        raise TokenError("unexpected_algorithm")
    if header.get("typ") != token_type:
        raise TokenError(f"unexpected_token_type:{header.get('typ')}")
    kid = header.get("kid")
    if not kid or kid not in material.verify_keys:
        raise TokenError("unknown_kid")
    public_pem = _public_pem_for_kid(material, kid)

    try:
        claims = pyjwt.decode(
            token,
            public_pem,
            algorithms=[JWT_ALGORITHM],
            audience=audience,
            issuer=settings.issuer,
            options={"require": ["iss", "aud", "sub", "sid", "jti", "iat", "nbf", "exp"]},
        )
    except pyjwt.PyJWTError as exc:
        # Map common failures to stable, non-leaky reasons.
        if isinstance(exc, pyjwt.ExpiredSignatureError):
            raise TokenError("expired") from exc
        if isinstance(exc, pyjwt.InvalidAudienceError):
            raise TokenError("invalid_audience") from exc
        if isinstance(exc, pyjwt.InvalidIssuerError):
            raise TokenError("invalid_issuer") from exc
        raise TokenError("invalid_signature") from exc
    return DecodedToken(claims=claims, kind=kind)


def _public_pem_for_kid(material: KeyMaterial, kid: str) -> str:
    """Return the PEM public key for ``kid`` (cached on the KeyMaterial)."""
    cache: dict[tuple[int, str], str] = getattr(_public_pem_for_kid, "_cache", {})
    cache_key = (id(material), kid)
    if cache_key in cache:
        return cache[cache_key]
    pub = material.verify_keys[kid]
    from .keys import _public_pem  # local import to avoid cycle at module load
    pem = _public_pem(pub)
    cache[cache_key] = pem
    _public_pem_for_kid._cache = cache  # type: ignore[attr-defined]
    return pem


def reset_cache_for_tests() -> None:
    """Clear cached key material (used when tests swap JWT settings)."""
    for attr in list(vars(_material)):
        if attr.startswith("__jwt_material_"):
            delattr(_material, attr)
    if hasattr(_public_pem_for_kid, "_cache"):
        delattr(_public_pem_for_kid, "_cache")
