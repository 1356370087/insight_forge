"""DB integration: HTTP layer via AsyncClient (Cache-Control: no-store, routers)."""

from __future__ import annotations

import asyncio
import re
from datetime import timedelta

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from security.rbac import database as iam_db
from security.rbac.jwt_service import decode_refresh_token
from security.rbac.models import RefreshToken, Session, utcnow
from security.rbac.repositories import get_user_by_email_normalized
from security.rbac.routers import admin_router, auth_router
from security.rbac.services import auth as auth_service
from security.rbac.settings import get_settings

pytestmark = [pytest.mark.asyncio, pytest.mark.db]


@pytest.fixture
def app():
    """Build a minimal FastAPI app with the IAM routers mounted."""
    application = FastAPI()
    application.include_router(auth_router)
    application.include_router(admin_router)
    return application


async def _register_verified(mail_recorder):
    """Register zoe and verify her email (reaching pending_approval)."""
    settings = get_settings()
    async with iam_db.session_scope() as db:
        await auth_service.register(
            db, email="zoe@example.com", password="a-very-strong-passphrase", display_name="Zoe",
            settings=settings, identity_for_rate_limit="ip-zoe",
        )
        await db.commit()
    body = [m["text_body"] for m in mail_recorder if "verify" in m["subject"].lower()][-1]
    token = re.search(r"token=([^\s]+)", body).group(1)
    async with iam_db.session_scope() as db:
        await auth_service.verify_email(db, token=token, settings=settings)
        await db.commit()


async def _login(app) -> dict:
    """Log in as zoe via the HTTP API and return the token pair."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        resp = await ac.post("/auth/login", json={"email": "zoe@example.com", "password": "a-very-strong-passphrase"})
    assert resp.status_code == 200, resp.text
    return resp.json()


class TestCacheControlAndShape:
    """Auth responses carry Cache-Control: no-store and the TokenPair shape."""

    async def test_login_response_has_no_store_header(self, app, mail_recorder, settings):
        """A successful login response sets Cache-Control: no-store."""
        await _register_verified(mail_recorder)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            resp = await ac.post(
                "/auth/login", json={"email": "zoe@example.com", "password": "a-very-strong-passphrase"},
            )
        assert resp.status_code == 200
        assert resp.headers.get("cache-control") == "no-store"
        body = resp.json()
        assert body["token_type"] == "bearer"
        assert body["access_token"]
        assert body["refresh_token"]
        assert body["expires_in"] == settings.access_token_ttl
        assert body["session_id"]

    async def test_refresh_response_has_no_store_header(self, app, mail_recorder):
        """A successful refresh response sets Cache-Control: no-store."""
        await _register_verified(mail_recorder)
        tokens = await _login(app)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            resp = await ac.post("/auth/refresh", json={"refresh_token": tokens["refresh_token"]})
        assert resp.status_code == 200
        assert resp.headers.get("cache-control") == "no-store"

    async def test_me_requires_authorization(self, app):
        """GET /auth/me without a token returns 401."""
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            resp = await ac.get("/auth/me")
        assert resp.status_code == 401

    async def test_me_returns_profile(self, app, mail_recorder):
        """GET /auth/me with a valid token returns the profile + status."""
        await _register_verified(mail_recorder)
        tokens = await _login(app)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            resp = await ac.get("/auth/me", headers={"Authorization": f"Bearer {tokens['access_token']}"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["email"] == "zoe@example.com"
        assert body["status"] == "pending_approval"

    async def test_logout_revokes_session(self, app, mail_recorder):
        """Logout revokes the session; /auth/me then returns 401."""
        await _register_verified(mail_recorder)
        tokens = await _login(app)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            resp = await ac.post("/auth/logout", headers={"Authorization": f"Bearer {tokens['access_token']}"})
            assert resp.status_code == 200
            assert resp.json()["revoked"] is True
            after = await ac.get("/auth/me", headers={"Authorization": f"Bearer {tokens['access_token']}"})
        assert after.status_code == 401

    async def test_admin_requires_permission(self, app, mail_recorder):
        """A pending-approval (non-admin) user cannot list admin users (403)."""
        await _register_verified(mail_recorder)
        tokens = await _login(app)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            resp = await ac.get("/admin/users", headers={"Authorization": f"Bearer {tokens['access_token']}"})
        assert resp.status_code == 403

    async def test_register_returns_202(self, app, monkeypatch):
        """Open registration answers 202 without disclosing account state."""
        monkeypatch.setenv("IAM_OPEN_REGISTRATION", "true")
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            resp = await ac.post(
                "/auth/register", json={"email": "new@example.com", "password": "a-very-strong-passphrase"},
            )
        assert resp.status_code == 202

    async def test_register_honors_closed_registration(self, app, monkeypatch):
        """The public registration switch is enforced at the HTTP boundary."""
        monkeypatch.setenv("IAM_OPEN_REGISTRATION", "false")
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            resp = await ac.post(
                "/auth/register",
                json={"email": "closed@example.com", "password": "a-very-strong-passphrase"},
            )
        assert resp.status_code == 403
        assert resp.json()["detail"] == "registration_closed"

    async def test_failed_login_counter_survives_http_error(self, app, mail_recorder):
        """Security counters commit even though the response is a 401."""
        await _register_verified(mail_recorder)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            resp = await ac.post(
                "/auth/login",
                json={"email": "zoe@example.com", "password": "definitely-the-wrong-password"},
            )
        assert resp.status_code == 401
        async with iam_db.session_scope() as db:
            user = await get_user_by_email_normalized(db, "zoe@example.com")
            assert user is not None
            assert user.failed_login_count == 1

    async def test_concurrent_refresh_has_single_winner(self, app, mail_recorder):
        """A row lock prevents two valid branches from one refresh token."""
        await _register_verified(mail_recorder)
        tokens = await _login(app)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            first, second = await asyncio.gather(
                ac.post("/auth/refresh", json={"refresh_token": tokens["refresh_token"]}),
                ac.post("/auth/refresh", json={"refresh_token": tokens["refresh_token"]}),
            )
        assert sorted((first.status_code, second.status_code)) == [200, 409]

    async def test_refresh_reuse_revocation_survives_401(self, app, mail_recorder):
        """Post-grace replay revokes the session and commits before returning 401."""
        await _register_verified(mail_recorder)
        tokens = await _login(app)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            rotated = await ac.post("/auth/refresh", json={"refresh_token": tokens["refresh_token"]})
            assert rotated.status_code == 200
        jti = str(decode_refresh_token(tokens["refresh_token"]).claims["jti"])
        async with iam_db.session_scope() as db:
            record = await db.get(RefreshToken, jti)
            assert record is not None
            record.used_at = utcnow() - timedelta(seconds=30)
            await db.commit()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            replay = await ac.post("/auth/refresh", json={"refresh_token": tokens["refresh_token"]})
        assert replay.status_code == 401
        async with iam_db.session_scope() as db:
            session = await db.get(Session, tokens["session_id"])
            assert session is not None
            assert session.is_revoked

    async def test_concurrent_password_reset_single_winner(self, app, mail_recorder):
        """Two parallel presentations of one reset token consume it exactly once."""
        await _register_verified(mail_recorder)
        settings = get_settings()
        async with iam_db.session_scope() as db:
            await auth_service.forgot_password(
                db, email="zoe@example.com", settings=settings, identity_for_rate_limit="ip-rst",
            )
            await db.commit()
        body = [m["text_body"] for m in mail_recorder if "reset" in m["subject"].lower()][-1]
        token = re.search(r"token=([^\s]+)", body).group(1)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            first, second = await asyncio.gather(
                ac.post("/auth/password/reset", json={"token": token, "password": "first-new-passphrase"}),
                ac.post("/auth/password/reset", json={"token": token, "password": "second-new-passphrase"}),
            )
        assert sorted((first.status_code, second.status_code)) == [200, 400]
