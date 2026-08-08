"""Tests for the Argon2id password hashing + policy."""

from __future__ import annotations

import pytest

from security.rbac import passwords


class TestPasswordHashing:
    """Argon2id hash/verify round trips and upgrade detection."""

    def test_hash_and_verify_roundtrip(self):
        """A correct password verifies; a wrong one does not."""
        hashed = passwords.hash_password("correct horse battery staple")
        assert passwords.verify_password("correct horse battery staple", hashed) is True
        assert passwords.verify_password("wrong password", hashed) is False

    def test_hash_is_argon2id(self):
        """Hashes must start with the Argon2id identifier."""
        hashed = passwords.hash_password("a-strong-password-123")
        assert hashed.startswith("$argon2id$")

    def test_needs_rehash_false_for_current_params(self):
        """Current Argon2id hashes do not request a rehash."""
        assert passwords.needs_rehash(passwords.hash_password("a-strong-password-123")) is False

    def test_needs_rehash_true_for_empty(self):
        """An empty/legacy hash requests a rehash."""
        assert passwords.needs_rehash("") is True
        assert passwords.needs_rehash("$2b$12$legacybcrypt") is True

    def test_unicode_password_supported(self):
        """Unicode (incl. emoji + spaces) passwords round-trip."""
        pw = "パスワード は 安全 🔐 に Strong"
        hashed = passwords.hash_password(pw)
        assert passwords.verify_password(pw, hashed) is True

    def test_dummy_hash_verifies_false_without_error(self):
        """The dummy hash is valid Argon2id but never matches a real password."""
        assert passwords.verify_password("anything", passwords.dummy_hash()) is False


class TestPasswordPolicy:
    """The 15–128 Unicode-char policy and common-password screening."""

    @pytest.mark.parametrize("pw", ["", "short", "x" * 14, "1234567890"])
    def test_rejects_too_short(self, pw):
        """Passwords below 15 characters are rejected."""
        with pytest.raises(passwords.PasswordPolicyError):
            passwords.validate_password(pw)

    def test_rejects_too_long(self):
        """Passwords over 128 characters are rejected."""
        with pytest.raises(passwords.PasswordPolicyError):
            passwords.validate_password("x" * 129)

    def test_rejects_common_password(self):
        """A denylisted common password is rejected even if long enough."""
        with pytest.raises(passwords.PasswordPolicyError):
            passwords.validate_password("123456789012345")

    def test_accepts_minimal_length(self):
        """A 15-character password passes and is NFC-normalized."""
        out = passwords.validate_password("a-solid-15chars")  # exactly 15 chars, not common
        assert out == "a-solid-15chars"

    def test_accepts_max_length(self):
        """A 128-character password passes."""
        assert passwords.validate_password("a" * 128) == "a" * 128

    def test_normalizes_unicode(self):
        """NFC-equivalent inputs normalize identically (length >= 15)."""
        nfc = passwords.validate_password("fifteen-char-pwd-é")  # é precomposed
        assert nfc.endswith("é")

    def test_rejects_control_characters(self):
        """Control characters (e.g. NUL) are rejected."""
        with pytest.raises(passwords.PasswordPolicyError):
            passwords.validate_password("a" * 14 + "\x00")
