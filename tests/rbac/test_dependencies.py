"""Tests for email normalization and the RBAC FastAPI dependencies."""

from __future__ import annotations

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from security.rbac.dependencies import (
    apply_principal_to_config,
    get_current_principal,
    require_active_user,
    require_permissions,
)
from security.rbac.emails import InvalidEmail, normalize_email, validate_and_normalize
from security.rbac.principal import Principal


@pytest.fixture(autouse=True)
def _no_dev_bypass(monkeypatch):
    """Ensure the local-dev bypass never interferes with dependency tests."""
    monkeypatch.delenv("LOCAL_DEV_AUTH_BYPASS", raising=False)


class TestEmailNormalization:
    """normalize_email lower-cases + NFC + strips; validate rejects junk."""

    def test_lowercases_and_strips(self):
        """Whitespace is stripped and the result is lower-case."""
        assert normalize_email("  User@Example.COM ") == "user@example.com"

    def test_validate_accepts_valid(self):
        """A valid email normalizes successfully."""
        assert validate_and_normalize("Alice@Example.com") == "alice@example.com"

    def test_validate_rejects_empty(self):
        """Empty emails are rejected."""
        with pytest.raises(InvalidEmail):
            validate_and_normalize("")

    def test_validate_rejects_malformed(self):
        """Malformed emails are rejected."""
        with pytest.raises(InvalidEmail):
            validate_and_normalize("not-an-email")


def _make_principal(status="active", perms=("research.run.create",)):
    """Build a Principal with the given status/permissions."""
    return Principal(
        user_id="u-1", email="u@example.com", status=status, session_id="s-1",
        roles=frozenset({"researcher"}), permissions=frozenset(perms), authz_version=1,
    )


def _build_app() -> FastAPI:
    """Build a tiny app exercising the permission dependencies."""
    app = FastAPI()

    @app.get("/open")
    async def open_endpoint(principal: Principal = Depends(get_current_principal)):
        return {"id": principal.user_id}

    @app.get("/active")
    async def active_endpoint(principal: Principal = Depends(require_active_user)):
        return {"id": principal.user_id}

    @app.get("/create")
    async def create_endpoint(principal: Principal = Depends(require_permissions("research.run.create"))):
        return {"id": principal.user_id}

    @app.get("/multi")
    async def multi_endpoint(
        principal: Principal = Depends(
            require_permissions("iam.users.read", "iam.users.write", mode="all"),
        ),
    ):
        return {"id": principal.user_id}

    return app


class TestPermissionDependencies:
    """require_permissions / require_active_user via TestClient + overrides."""

    def test_missing_token_returns_401(self):
        """No Authorization header yields 401."""
        client = TestClient(_build_app())
        resp = client.get("/open")
        assert resp.status_code == 401

    def test_active_principal_passes_active(self):
        """An active principal satisfies require_active_user."""
        app = _build_app()
        app.dependency_overrides[get_current_principal] = lambda: _make_principal()
        resp = TestClient(app).get("/active")
        assert resp.status_code == 200

    def test_pending_principal_blocked_from_active(self):
        """A pending-approval principal is rejected by require_active_user (403)."""
        app = _build_app()
        app.dependency_overrides[get_current_principal] = lambda: _make_principal(status="pending_approval")
        resp = TestClient(app).get("/active")
        assert resp.status_code == 403
        assert resp.json()["detail"] == "pending_approval"

    def test_permission_present_passes(self):
        """Holding the required permission yields 200."""
        app = _build_app()
        app.dependency_overrides[get_current_principal] = lambda: _make_principal()
        resp = TestClient(app).get("/create")
        assert resp.status_code == 200

    def test_permission_absent_blocked(self):
        """Missing the required permission yields 403."""
        app = _build_app()
        app.dependency_overrides[get_current_principal] = lambda: _make_principal(perms=())
        resp = TestClient(app).get("/create")
        assert resp.status_code == 403
        assert resp.json()["detail"] == "insufficient_permissions"

    def test_permission_mode_all_requires_every(self):
        """mode='all' requires all listed permissions."""
        app = _build_app()
        app.dependency_overrides[get_current_principal] = lambda: _make_principal(perms=("iam.users.read",))
        assert TestClient(app).get("/multi").status_code == 403
        app.dependency_overrides[get_current_principal] = lambda: _make_principal(
            perms=("iam.users.read", "iam.users.write"),
        )
        assert TestClient(app).get("/multi").status_code == 200

    def test_require_permissions_factory_validates_codes(self):
        """The factory requires at least one permission code."""
        with pytest.raises(ValueError):
            require_permissions()


class TestApplyPrincipalToConfig:
    """apply_principal_to_config writes owner + roles/permissions into config."""

    def test_sets_owner_and_auth_user(self):
        """The principal's user id is the run owner; roles/permissions split."""
        principal = _make_principal(perms=("research.run.create",))
        cfg = apply_principal_to_config({"configurable": {}}, principal)
        assert cfg["metadata"]["owner"] == "u-1"
        auth = cfg["configurable"]["langgraph_auth_user"]
        assert auth["identity"] == "u-1"
        assert auth["roles"] == ["researcher"]
        assert "research.run.create" in auth["effective_permissions"]

    def test_idempotent_existing_config(self):
        """Existing configurable/metadata dicts are extended, not replaced."""
        principal = _make_principal()
        cfg = apply_principal_to_config({"configurable": {"keep": 1}, "metadata": {"k": "v"}}, principal)
        assert cfg["configurable"]["keep"] == 1
        assert cfg["metadata"]["k"] == "v"
