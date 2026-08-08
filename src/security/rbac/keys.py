"""Ed25519 key material management for access & refresh JWTs.

Two independent key materials are used (access and refresh), each carrying:
the *current* signing private key + its ``kid``, and a ``kid -> public key``
verification map that may include a retired public key for smooth rotation.
Signing always uses the current key; verification is explicit and pinned to
``EdDSA`` — the algorithm in the token header is never trusted (RFC 8725).

In development (``APP_ENV != production``) keys may be auto-generated and held
ephemerally in memory; production refuses to start without explicit keys.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from typing import Literal

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from .settings import IAMSettings, get_settings

logger = logging.getLogger(__name__)

KeyKind = Literal["access", "refresh"]


class KeyError_(ValueError):
    """Raised when JWT key configuration is missing or invalid."""


@dataclass(frozen=True)
class KeyMaterial:
    """Signing + verification key material for one token kind."""

    signing_private: Ed25519PrivateKey
    signing_kid: str
    verify_keys: dict[str, Ed25519PublicKey]

    def private_pem(self) -> str:
        """Return the PEM-encoded current signing private key."""
        return _private_pem(self.signing_private)

    def public_pem(self) -> str:
        """Return the PEM-encoded current public key."""
        return _public_pem(self.signing_private.public_key())


def _private_pem(key: Ed25519PrivateKey) -> str:
    raw = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    return raw.decode("utf-8")


def _public_pem(key: Ed25519PublicKey) -> str:
    raw = key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    return raw.decode("utf-8")


def _derive_kid(public_pem: str) -> str:
    """Derive a short, stable ``kid`` from a public key PEM."""
    digest = hashlib.sha256(public_pem.encode("utf-8")).hexdigest()
    return digest[:16]


def _load_private_pem(pem: str, kind: KeyKind) -> Ed25519PrivateKey:
    """Parse a PEM private key, raising a descriptive error on failure."""
    try:
        key = serialization.load_pem_private_key(pem.encode("utf-8"), password=None)
    except Exception as exc:  # noqa: BLE001 - surface a single, clear error
        raise KeyError_(f"invalid_{kind}_signing_key: {exc}") from exc
    if not isinstance(key, Ed25519PrivateKey):
        raise KeyError_(f"invalid_{kind}_signing_key:not_ed25519")
    return key


def _load_public_pem(pem: str, kid: str, kind: KeyKind) -> Ed25519PublicKey:
    """Parse a PEM public key, raising a descriptive error on failure."""
    try:
        key = serialization.load_pem_public_key(pem.encode("utf-8"))
    except Exception as exc:  # noqa: BLE001 - surface a single, clear error
        raise KeyError_(f"invalid_{kind}_verify_key:{kid}:{exc}") from exc
    if not isinstance(key, Ed25519PublicKey):
        raise KeyError_(f"invalid_{kind}_verify_key:{kid}:not_ed25519")
    return key


def _parse_verify_map(verify_json: str) -> dict[str, str]:
    """Parse the ``{kid: public_pem}`` JSON verification map."""
    if not verify_json.strip():
        return {}
    try:
        parsed = json.loads(verify_json)
    except json.JSONDecodeError as exc:
        raise KeyError_(f"invalid_verify_keys_json:{exc}") from exc
    if not isinstance(parsed, dict):
        raise KeyError_("invalid_verify_keys_json:not_object")
    cleaned: dict[str, str] = {}
    for kid, pem in parsed.items():
        if not isinstance(kid, str) or not isinstance(pem, str):
            raise KeyError_("invalid_verify_keys_json:kid_or_pem_not_string")
        cleaned[kid] = pem
    return cleaned


def load_key_material(
    kind: KeyKind,
    *,
    settings: IAMSettings | None = None,
) -> KeyMaterial:
    """Build :class:`KeyMaterial` for ``kind`` from settings (or dev fallback).

    Args:
        kind: ``"access"`` or ``"refresh"``.
        settings: Optional resolved settings; defaults to a fresh snapshot.

    Raises:
        KeyError_: if production is missing/has-invalid keys.
    """
    settings = settings or get_settings()
    signing_pem, signing_kid, verify_json = _kind_settings(kind, settings)

    if signing_pem:
        private_key = _load_private_pem(signing_pem, kind)
        kid = signing_kid or _derive_kid(_public_pem(private_key.public_key()))
    elif settings.is_production:
        raise KeyError_(f"missing_{kind}_signing_key")
    else:
        # Development convenience: generate an ephemeral keypair.
        logger.warning("Generating ephemeral %s JWT key for development; set IAM_JWT_*_SIGNING_KEY in production.", kind)
        private_key = Ed25519PrivateKey.generate()
        kid = signing_kid or _derive_kid(_public_pem(private_key.public_key()))

    verify_map: dict[str, Ed25519PublicKey] = {}
    for entry_kid, entry_pem in _parse_verify_map(verify_json).items():
        verify_map[entry_kid] = _load_public_pem(entry_pem, entry_kid, kind)
    # The current signing key is always valid for verification under its kid.
    verify_map[kid] = private_key.public_key()

    return KeyMaterial(signing_private=private_key, signing_kid=kid, verify_keys=verify_map)


def _kind_settings(kind: KeyKind, settings: IAMSettings) -> tuple[str, str, str]:
    """Return ``(signing_pem, signing_kid, verify_json)`` for ``kind``."""
    if kind == "access":
        return (
            settings.access_signing_key_pem,
            settings.access_signing_kid,
            settings.access_verify_keys_json,
        )
    return (
        settings.refresh_signing_key_pem,
        settings.refresh_signing_kid,
        settings.refresh_verify_keys_json,
    )


def generate_signing_keypair() -> tuple[str, str]:
    """Generate a fresh keypair and return ``(private_pem, public_pem)``.

    Convenience for operators generating keys to put in environment variables.
    """
    private_key = Ed25519PrivateKey.generate()
    return _private_pem(private_key), _public_pem(private_key.public_key())
