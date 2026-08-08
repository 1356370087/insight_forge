"""DB integration: refresh-token rotation, reuse detection, session revocation."""

from __future__ import annotations

import dataclasses

import pytest
from fastapi import HTTPException

from security.rbac import database as iam_db
from security.rbac.dependencies import build_principal
from security.rbac.services import auth as auth_service
from security.rbac.services import sessions as sessions_service
from security.rbac.services.sessions import (
    InvalidRefreshToken,
    RefreshInProgress,
    RefreshReuseDetected,
)

pytestmark = [pytest.mark.asyncio, pytest.mark.db]

EMAIL = "bob@example.com"
PASSWORD = "a-very-strong-passphrase"


async def _login_verified(mail_recorder, settings):
    """Register+verify+login bob and return the TokenPairRecord."""
    async with iam_db.session_scope() as db:
        await auth_service.register(
            db, email=EMAIL, password=PASSWORD, display_name="Bob",
            settings=settings, identity_for_rate_limit="ip-bob",
        )
        await db.commit()
    import re

    body = [m["text_body"] for m in mail_recorder if "verify" in m["subject"].lower()][0]
    token = re.search(r"token=([^\s]+)", body).group(1)
    async with iam_db.session_scope() as db:
        await auth_service.verify_email(db, token=token, settings=settings)
        await db.commit()
    async with iam_db.session_scope() as db:
        pair = await auth_service.login(
            db, email=EMAIL, password=PASSWORD, settings=settings, ip_address="ip-bob",
        )
        await db.commit()
    return pair


class TestRotation:
    """Normal rotation issues a fresh chain link and retires the old token."""

    async def test_rotate_issues_new_token(self, mail_recorder, settings):
        """A valid refresh rotation returns a new access+refresh pair."""
        pair = await _login_verified(mail_recorder, settings)
        async with iam_db.session_scope() as db:
            new_pair = await sessions_service.rotate(
                db, pair.refresh_token, settings=settings, ip_address="ip-bob",
            )
            await db.commit()
        assert new_pair.refresh_token != pair.refresh_token
        assert new_pair.access_token != pair.access_token
        assert new_pair.session_id == pair.session_id


class TestReuseDetection:
    """Replay of a used refresh token is detected and (outside grace) revokes."""

    async def test_reuse_within_grace_returns_in_progress(self, mail_recorder, settings):
        """Immediate double-submit inside the grace window is 409, no revoke."""
        pair = await _login_verified(mail_recorder, settings)
        async with iam_db.session_scope() as db:
            await sessions_service.rotate(db, pair.refresh_token, settings=settings, ip_address="ip")
            await db.commit()
        async with iam_db.session_scope() as db:
            with pytest.raises(RefreshInProgress):
                await sessions_service.rotate(db, pair.refresh_token, settings=settings, ip_address="ip")

    async def test_reuse_after_grace_revokes_session(self, mail_recorder, settings):
        """Replay outside the grace window revokes the session family."""
        pair = await _login_verified(mail_recorder, settings)
        no_grace = dataclasses.replace(settings, refresh_reuse_grace=0)
        async with iam_db.session_scope() as db:
            new_pair = await sessions_service.rotate(db, pair.refresh_token, settings=no_grace, ip_address="ip")
            await db.commit()
        async with iam_db.session_scope() as db:
            with pytest.raises(RefreshReuseDetected):
                await sessions_service.rotate(db, pair.refresh_token, settings=no_grace, ip_address="ip")
            await db.commit()
        # The whole session family is revoked: the freshly rotated token fails too.
        async with iam_db.session_scope() as db:
            with pytest.raises((InvalidRefreshToken, RefreshReuseDetected)):
                await sessions_service.rotate(db, new_pair.refresh_token, settings=no_grace, ip_address="ip")


class TestRevocationAndAuthzVersion:
    """Disabling / password-change / logout revoke sessions and access tokens."""

    async def test_revoked_session_invalidates_access_token(self, mail_recorder, settings):
        """After logout, the access token is rejected (session_revoked)."""
        pair = await _login_verified(mail_recorder, settings)
        async with iam_db.session_scope() as db:
            await sessions_service.revoke_session(db, pair.session_id, reason="logout")
            await db.commit()
        async with iam_db.session_scope() as db:
            with pytest.raises(HTTPException) as exc:
                await build_principal(db, pair.access_token)
        assert exc.value.status_code == 401

    async def test_password_change_revokes_sessions(self, mail_recorder, settings):
        """Changing the password bumps authz_version and invalidates the access token."""
        pair = await _login_verified(mail_recorder, settings)
        async with iam_db.session_scope() as db:
            await auth_service.change_password(
                db, user_id=_user_id_from_login(pair, settings, mail_recorder),
                current_password=PASSWORD, new_password="a-different-passphrase", settings=settings,
            )
            await db.commit()
        async with iam_db.session_scope() as db:
            with pytest.raises(HTTPException) as exc:
                await build_principal(db, pair.access_token)
        assert exc.value.status_code in (401,)


def _user_id_from_login(pair, settings, mail_recorder):
    """Decode the access token's subject to recover the user id (for change_password)."""
    from security.rbac.jwt_service import decode_access_token

    decoded = decode_access_token(pair.access_token, settings=settings)
    return decoded.claims["sub"]


class TestMultiDevice:
    """Multiple device sessions are independent."""

    async def test_revoking_one_session_keeps_others(self, mail_recorder, settings):
        """Revoking session A does not affect a concurrently-issued session B."""
        pair_a = await _login_verified(mail_recorder, settings)
        # Second login on another "device".
        async with iam_db.session_scope() as db:
            pair_b = await auth_service.login(
                db, email=EMAIL, password=PASSWORD, settings=settings, ip_address="ip-b",
            )
            await db.commit()
        assert pair_a.session_id != pair_b.session_id
        async with iam_db.session_scope() as db:
            await sessions_service.revoke_session(db, pair_a.session_id, reason="logout")
            await db.commit()
        # B's access token still works.
        async with iam_db.session_scope() as db:
            principal = await build_principal(db, pair_b.access_token)
        assert principal.user_id
