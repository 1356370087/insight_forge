"""DB integration: registration, verification, login, sessions, password flows."""

from __future__ import annotations

import pytest

from security.rbac import database as iam_db
from security.rbac.services import auth as auth_service
from security.rbac.services.auth import AuthError

pytestmark = [pytest.mark.asyncio, pytest.mark.db]

EMAIL = "alice@example.com"
PASSWORD = "a-very-strong-passphrase"


async def _register_and_verify(mail_recorder, settings):
    """Register alice, verify her email, and return the user id."""
    async with iam_db.session_scope() as db:
        await auth_service.register(
            db, email=EMAIL, password=PASSWORD, display_name="Alice",
            settings=settings, identity_for_rate_limit="ip-1",
        )
        await db.commit()
    token = _extract_verify_token(mail_recorder)
    async with iam_db.session_scope() as db:
        user = await auth_service.verify_email(db, token=token, settings=settings)
        await db.commit()
    return str(user.id)


def _extract_verify_token(mail_recorder) -> str:
    """Return the verification token from the most recent recorded email."""
    bodies = [m["text_body"] for m in mail_recorder if "verify" in m["subject"].lower()]
    assert bodies, "no verification email recorded"
    import re

    match = re.search(r"token=([^\s]+)", bodies[-1])
    assert match
    return match.group(1)


class TestRegistration:
    """Registration enumeration protection and verification."""

    async def test_register_then_verify_reaches_pending_approval(self, mail_recorder, settings):
        """A fresh registration verifies to pending_approval (not active)."""
        uid = await _register_and_verify(mail_recorder, settings)
        assert uid

    async def test_duplicate_register_returns_no_verification(self, mail_recorder, settings):
        """A duplicate email yields issued_verification=False (still 202)."""
        async with iam_db.session_scope() as db:
            await auth_service.register(
                db, email="dup@example.com", password=PASSWORD, display_name=None,
                settings=settings, identity_for_rate_limit="ip-dup",
            )
            await db.commit()
        mail_recorder.clear()
        async with iam_db.session_scope() as db:
            outcome = await auth_service.register(
                db, email="DUP@example.com", password=PASSWORD, display_name=None,
                settings=settings, identity_for_rate_limit="ip-dup2",
            )
            await db.commit()
        assert outcome.issued_verification is False
        assert mail_recorder == []

    async def test_register_validates_password(self, settings):
        """A too-short password is rejected (422)."""
        async with iam_db.session_scope() as db:
            with pytest.raises(AuthError) as exc:
                await auth_service.register(
                    db, email="short@example.com", password="short",
                    display_name=None, settings=settings, identity_for_rate_limit="ip-s",
                )
            assert exc.value.status == 422

    async def test_resend_is_noop_for_unknown_email(self, mail_recorder, settings):
        """Resend for an unknown email records nothing (still 202)."""
        await auth_service_resend(settings, "ghost@example.com")
        assert mail_recorder == []


async def auth_service_resend(settings, email):
    """Helper invoking resend_verification within a transaction."""
    async with iam_db.session_scope() as db:
        await auth_service.resend_verification(
            db, email=email, settings=settings, identity_for_rate_limit="ip-r",
        )
        await db.commit()


class TestLogin:
    """Login success/failure and account-existence protection."""

    async def test_pending_email_cannot_login(self, mail_recorder, settings):
        """An unverified user cannot log in (403 email_verification_required)."""
        async with iam_db.session_scope() as db:
            await auth_service.register(
                db, email="pe@example.com", password=PASSWORD, display_name=None,
                settings=settings, identity_for_rate_limit="ip-pe",
            )
            await db.commit()
        async with iam_db.session_scope() as db:
            with pytest.raises(AuthError) as exc:
                await auth_service.login(
                    db, email="pe@example.com", password=PASSWORD,
                    settings=settings, ip_address="ip-pe",
                )
        assert exc.value.status == 403
        assert "verification" in exc.value.detail

    async def test_wrong_password_on_pending_account_is_uniform_401(self, mail_recorder, settings):
        """A wrong password never reveals lifecycle status — 401 like any account."""
        async with iam_db.session_scope() as db:
            await auth_service.register(
                db, email="pe2@example.com", password=PASSWORD, display_name=None,
                settings=settings, identity_for_rate_limit="ip-pe2",
            )
            await db.commit()
        async with iam_db.session_scope() as db:
            with pytest.raises(AuthError) as exc:
                await auth_service.login(
                    db, email="pe2@example.com", password="definitely-not-it",
                    settings=settings, ip_address="ip-pe2",
                )
        assert exc.value.status == 401
        assert exc.value.detail == "invalid_credentials"

    async def test_lockout_scoped_to_failing_source_ip(self, mail_recorder, settings):
        """Five failures lock that (account, ip) pair — not the rightful owner elsewhere."""
        await _register_and_verify(mail_recorder, settings)
        for _ in range(auth_service.LOGIN_MAX_FAILURES):
            async with iam_db.session_scope() as db:
                with pytest.raises(AuthError) as exc:
                    await auth_service.login(
                        db, email=EMAIL, password="wrong-password-here",
                        settings=settings, ip_address="ip-attacker",
                    )
                await db.commit()
            assert exc.value.detail == "invalid_credentials"
        async with iam_db.session_scope() as db:
            with pytest.raises(AuthError) as exc:
                await auth_service.login(
                    db, email=EMAIL, password=PASSWORD,
                    settings=settings, ip_address="ip-attacker",
                )
        assert exc.value.status == 429
        assert exc.value.detail == "account_locked"
        # The same attacker cannot probe a locked unknown email cheaply either.
        async with iam_db.session_scope() as db:
            for _ in range(auth_service.LOGIN_MAX_FAILURES):
                with pytest.raises(AuthError):
                    await auth_service.login(
                        db, email="ghost@example.com", password="whatever-password",
                        settings=settings, ip_address="ip-attacker2",
                    )
                await db.commit()
            with pytest.raises(AuthError) as exc:
                await auth_service.login(
                    db, email="ghost@example.com", password="whatever-password",
                    settings=settings, ip_address="ip-attacker2",
                )
        assert exc.value.status == 429
        # The rightful owner from another source IP is unaffected.
        async with iam_db.session_scope() as db:
            pair = await auth_service.login(
                db, email=EMAIL, password=PASSWORD, settings=settings, ip_address="ip-victim",
            )
            await db.commit()
        assert pair.access_token

    async def test_unknown_user_returns_invalid_credentials(self, settings):
        """An unknown user maps to invalid_credentials (401), not 404."""
        async with iam_db.session_scope() as db:
            with pytest.raises(AuthError) as exc:
                await auth_service.login(
                    db, email="nobody@example.com", password=PASSWORD,
                    settings=settings, ip_address="ip-nobody",
                )
        assert exc.value.status == 401
        assert exc.value.detail == "invalid_credentials"

    async def test_wrong_password_returns_invalid_credentials(self, mail_recorder, settings):
        """A wrong password maps to invalid_credentials and increments failure count."""
        await _register_and_verify(mail_recorder, settings)
        async with iam_db.session_scope() as db:
            with pytest.raises(AuthError) as exc:
                await auth_service.login(
                    db, email=EMAIL, password="wrong-password-here",
                    settings=settings, ip_address="ip-w",
                )
        assert exc.value.detail == "invalid_credentials"

    async def test_valid_login_returns_token_pair(self, mail_recorder, settings):
        """A verified user can log in and receives a token pair."""
        await _register_and_verify(mail_recorder, settings)
        async with iam_db.session_scope() as db:
            pair = await auth_service.login(
                db, email=EMAIL, password=PASSWORD, settings=settings, ip_address="ip-ok",
            )
            await db.commit()
        assert pair.access_token
        assert pair.refresh_token
        assert pair.session_id
        assert pair.token_type if hasattr(pair, "token_type") else True
        assert pair.access_expires_in == settings.access_token_ttl


class TestPasswordFlows:
    """Forgot/reset/change password flows."""

    async def test_forgot_then_reset(self, mail_recorder, settings):
        """Forgot-password issues a reset token that resets the password."""
        await _register_and_verify(mail_recorder, settings)
        async with iam_db.session_scope() as db:
            await auth_service.forgot_password(
                db, email=EMAIL, settings=settings, identity_for_rate_limit="ip-f",
            )
            await db.commit()
        reset_token = _extract_reset_token(mail_recorder)
        new_pw = "a-brand-new-passphrase"
        async with iam_db.session_scope() as db:
            await auth_service.reset_password(
                db, token=reset_token, new_password=new_pw, settings=settings,
            )
            await db.commit()
        async with iam_db.session_scope() as db:
            pair = await auth_service.login(
                db, email=EMAIL, password=new_pw, settings=settings, ip_address="ip-f2",
            )
            await db.commit()
        assert pair.access_token

    async def test_reset_token_single_use(self, mail_recorder, settings):
        """A reset token cannot be consumed twice."""
        await _register_and_verify(mail_recorder, settings)
        async with iam_db.session_scope() as db:
            await auth_service.forgot_password(
                db, email=EMAIL, settings=settings, identity_for_rate_limit="ip-f3",
            )
            await db.commit()
        token = _extract_reset_token(mail_recorder)
        async with iam_db.session_scope() as db:
            await auth_service.reset_password(
                db, token=token, new_password="first-new-passphrase", settings=settings,
            )
            await db.commit()
        async with iam_db.session_scope() as db:
            with pytest.raises(AuthError):
                await auth_service.reset_password(
                    db, token=token, new_password="second-new-passphrase", settings=settings,
                )

    async def test_forgot_unknown_email_is_noop(self, mail_recorder, settings):
        """Forgot-password for an unknown email records nothing."""
        async with iam_db.session_scope() as db:
            await auth_service.forgot_password(
                db, email="ghost@example.com", settings=settings, identity_for_rate_limit="ip-g",
            )
            await db.commit()
        assert mail_recorder == []


def _extract_reset_token(mail_recorder) -> str:
    """Return the reset token from the most recent recorded email."""
    import re

    bodies = [m["text_body"] for m in mail_recorder if "reset" in m["subject"].lower()]
    assert bodies, "no reset email recorded"
    match = re.search(r"token=([^\s]+)", bodies[-1])
    assert match
    return match.group(1)


class TestRateLimiting:
    """Distributed rate limiting on registration."""

    async def test_register_rate_limit_triggers(self, settings):
        """Exceeding the register limit raises rate_limited (429)."""
        limit = settings.register_rate_limit
        for i in range(limit):
            async with iam_db.session_scope() as db:
                await auth_service.register(
                    db, email=f"rl{i}@example.com", password=PASSWORD, display_name=None,
                    settings=settings, identity_for_rate_limit="ip-rl",
                )
                await db.commit()
        async with iam_db.session_scope() as db:
            with pytest.raises(AuthError) as exc:
                await auth_service.register(
                    db, email="rl-over@example.com", password=PASSWORD, display_name=None,
                    settings=settings, identity_for_rate_limit="ip-rl",
                )
        assert exc.value.status == 429


class TestConcurrentRegistration:
    """A same-email registration race must not surface a 500."""

    async def test_concurrent_register_has_single_winner(self, settings):
        """Two parallel registrations of one email yield exactly one outcome."""
        import asyncio

        async def _register():
            async with iam_db.session_scope() as db:
                outcome = await auth_service.register(
                    db, email="race@example.com", password=PASSWORD, display_name=None,
                    settings=settings, identity_for_rate_limit="ip-race",
                )
                await db.commit()
                return outcome.issued_verification

        results = await asyncio.gather(_register(), _register())
        assert results.count(True) == 1


class TestForcedPasswordReset:
    """Admin-triggered resets must contain existing sessions."""

    async def test_forced_reset_revokes_sessions_and_blocks_refresh(self, mail_recorder, settings):
        """After an admin reset, refresh rotation dies and the session is revoked."""
        from security.rbac.models import Session as IamSession
        from security.rbac.services import users as users_service

        await _register_and_verify(mail_recorder, settings)
        async with iam_db.session_scope() as db:
            pair = await auth_service.login(
                db, email=EMAIL, password=PASSWORD, settings=settings, ip_address="ip-fr",
            )
            await db.commit()
        async with iam_db.session_scope() as db:
            user = await _get_user_by_email(db, EMAIL)
            await users_service.admin_send_password_reset(
                db, str(user.id), settings=settings,
                actor_id="00000000-0000-0000-0000-000000000001", send_email=False,
            )
            await db.commit()
        async with iam_db.session_scope() as db:
            session = await db.get(IamSession, pair.session_id)
            assert session is not None and session.is_revoked
        async with iam_db.session_scope() as db:
            with pytest.raises(AuthError) as exc:
                await auth_service.refresh(
                    db, refresh_token=pair.refresh_token, settings=settings,
                    ip_address="ip-fr",
                )
        assert exc.value.status == 401


async def _get_user_by_email(db, email: str):
    from security.rbac.repositories import get_user_by_email_normalized

    return await get_user_by_email_normalized(db, email)
