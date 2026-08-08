"""Tests for JWT encode/decode, key isolation, and opaque token digests."""

from __future__ import annotations

import json

import pytest

from security.rbac import jwt_service, keys, tokens
from security.rbac.jwt_service import TokenError
from security.rbac.keys import KeyError_, generate_signing_keypair
from security.rbac.settings import IAMSettings


def _settings(**overrides):
    """Return dev IAMSettings with optional overrides."""
    base = {
        "app_env": "development",
        "access_signing_key_pem": "",
        "refresh_signing_key_pem": "",
    }
    base.update(overrides)
    return IAMSettings(**base)


@pytest.fixture(autouse=True)
def _reset_jwt_cache():
    """Clear JWT key-material cache between tests so settings changes take effect."""
    jwt_service.reset_cache_for_tests()
    yield
    jwt_service.reset_cache_for_tests()


class TestOpaqueTokens:
    """256-bit opaque token generation + HMAC digests."""

    def test_tokens_are_url_safe_and_distinct(self):
        """Two generated tokens differ and are URL-safe strings."""
        a, _ = tokens.generate_token_with_digest("email_verification")
        b, _ = tokens.generate_token_with_digest("email_verification")
        assert a != b
        assert all(ch not in a for ch in ' "\\')
        assert len(a) >= 40

    def test_digest_is_purpose_scoped(self):
        """The same raw token digests differently per purpose."""
        raw = tokens.generate_token()
        d1 = tokens.digest_token(raw, purpose="email_verification")
        d2 = tokens.digest_token(raw, purpose="password_reset")
        assert d1 != d2

    def test_constant_time_eq(self):
        """Constant-time comparison behaves correctly."""
        assert tokens.constant_time_eq("abc", "abc") is True
        assert tokens.constant_time_eq("abc", "abd") is False


class TestKeyMaterial:
    """Ed25519 key material load/generate and kid derivation."""

    def test_explicit_keys_round_trip(self):
        """Explicit signing keys produce stable kids and verify maps."""
        priv, pub = generate_signing_keypair()
        s = _settings(access_signing_key_pem=priv, access_signing_kid="my-kid")
        km = keys.load_key_material("access", settings=s)
        assert km.signing_kid == "my-kid"
        assert "my-kid" in km.verify_keys

    def test_production_requires_explicit_keys(self):
        """Production refuses to start without explicit signing keys."""
        s = _settings(app_env="production")
        with pytest.raises(KeyError_):
            keys.load_key_material("access", settings=s)

    def test_dev_generates_ephemeral_keys(self):
        """Development auto-generates an ephemeral keypair with a derived kid."""
        s = _settings(app_env="development")
        km = keys.load_key_material("refresh", settings=s)
        assert km.signing_kid
        assert km.signing_kid in km.verify_keys

    def test_verify_keys_json_can_add_retired_public_key(self):
        """A retired public key in the verify map is accepted for rotation."""
        priv1, pub1 = generate_signing_keypair()
        priv2, pub2 = generate_signing_keypair()
        verify_json = json.dumps({"old": pub1})
        s = _settings(
            access_signing_key_pem=priv2, access_signing_kid="new",
            access_verify_keys_json=verify_json,
        )
        km = keys.load_key_material("access", settings=s)
        assert set(km.verify_keys) >= {"new", "old"}


class TestJWTAccessTokens:
    """Access token claims, header pinning, and verification."""

    def test_access_token_roundtrip(self):
        """An access token decodes with the expected claims."""
        s = _settings()
        token, jti = jwt_service.encode_access_token(
            subject="user-1", session_id="sess-1", authz_version=7, settings=s,
        )
        decoded = jwt_service.decode_access_token(token, settings=s)
        assert decoded.kind == "access"
        assert decoded.claims["sub"] == "user-1"
        assert decoded.claims["sid"] == "sess-1"
        assert decoded.claims["authz_version"] == 7
        assert decoded.claims["aud"] == "odr-api"

    def test_header_pinned_algorithm_and_typ(self):
        """The header carries EdDSA + the access token type + the current kid."""
        s = _settings()
        token, _ = jwt_service.encode_access_token(
            subject="u", session_id="s", authz_version=1, settings=s,
        )
        header = jwt_service.pyjwt.get_unverified_header(token)
        assert header["alg"] == "EdDSA"
        assert header["typ"] == "odr-access+jwt"
        assert "kid" in header

    def test_tampered_signature_rejected(self):
        """A modified token fails signature verification."""
        s = _settings()
        token, _ = jwt_service.encode_access_token(
            subject="u", session_id="s", authz_version=1, settings=s,
        )
        with pytest.raises(TokenError):
            jwt_service.decode_access_token(token[:-3] + "abc", settings=s)


class TestJWTKeyIsolation:
    """Access and refresh use independent keys/types/audiences (SPEC §2)."""

    def test_refresh_token_uses_refresh_typ_and_aud(self):
        """A refresh token has the refresh type + odr-refresh audience."""
        s = _settings()
        token, _ = jwt_service.encode_refresh_token(subject="u", session_id="s", settings=s)
        header = jwt_service.pyjwt.get_unverified_header(token)
        assert header["typ"] == "odr-refresh+jwt"
        decoded = jwt_service.decode_refresh_token(token, settings=s)
        assert decoded.claims["aud"] == "odr-refresh"

    def test_access_key_cannot_mint_refresh(self):
        """An access token decoded as refresh is rejected (typ mismatch)."""
        s = _settings()
        access, _ = jwt_service.encode_access_token(
            subject="u", session_id="s", authz_version=1, settings=s,
        )
        with pytest.raises(TokenError):
            jwt_service.decode_refresh_token(access, settings=s)

    def test_refresh_key_cannot_mint_access(self):
        """A refresh token decoded as access is rejected (typ mismatch)."""
        s = _settings()
        refresh, _ = jwt_service.encode_refresh_token(subject="u", session_id="s", settings=s)
        with pytest.raises(TokenError):
            jwt_service.decode_access_token(refresh, settings=s)

    def test_cross_kind_keys_are_independent(self):
        """Access and refresh key materials use different kids."""
        s = _settings()
        a = keys.load_key_material("access", settings=s)
        r = keys.load_key_material("refresh", settings=s)
        assert a.signing_kid != r.signing_kid

    def test_access_signed_with_refresh_key_rejected(self):
        """A token signed by the refresh key cannot be verified as access."""
        s = _settings()
        # Sign a token using the refresh key but present it for access decode.
        refresh_token, _ = jwt_service.encode_refresh_token(
            subject="u", session_id="s", settings=s,
        )
        # Forge: take refresh claims, sign with refresh key, decode as access fails on typ/aud.
        with pytest.raises(TokenError):
            jwt_service.decode_access_token(refresh_token, settings=s)
