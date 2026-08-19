"""Trusted-proxy client IP resolution tests."""

from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from starlette.requests import Request

from security.rbac.routers import auth as auth_router
from security.rbac.schemas import LoginRequest
from security.rbac.services.auth import AuthError
from security.rbac.settings import get_settings


def _request(*, forwarded_for: str | None, peer: str = "10.0.0.9") -> Request:
    headers = []
    if forwarded_for is not None:
        headers.append((b"x-forwarded-for", forwarded_for.encode()))
    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/auth/login",
            "headers": headers,
            "client": (peer, 12345),
        }
    )


def test_default_trusted_proxy_uses_rightmost_forwarded_address(monkeypatch) -> None:
    """A caller-controlled first hop cannot change the default bucket identity."""
    monkeypatch.delenv("IAM_TRUSTED_PROXY_COUNT", raising=False)

    first = auth_router._client_ip(
        _request(forwarded_for="fake-a, 198.51.100.7")
    )
    second = auth_router._client_ip(
        _request(forwarded_for="fake-b, 198.51.100.7")
    )

    assert first == second == "198.51.100.7"


def test_missing_forwarded_hops_falls_back_to_peer(monkeypatch) -> None:
    """An incomplete chain is not trusted."""
    monkeypatch.setenv("IAM_TRUSTED_PROXY_COUNT", "2")

    assert auth_router._client_ip(
        _request(forwarded_for="198.51.100.7")
    ) == "10.0.0.9"
    assert auth_router._client_ip(_request(forwarded_for=None)) == "10.0.0.9"


def test_zero_trusted_proxies_ignores_forwarded_header(monkeypatch) -> None:
    """Direct deployments always use the socket peer."""
    monkeypatch.setenv("IAM_TRUSTED_PROXY_COUNT", "0")

    assert auth_router._client_ip(
        _request(forwarded_for="203.0.113.99")
    ) == "10.0.0.9"


def test_negative_trusted_proxy_count_is_clamped(monkeypatch) -> None:
    """Invalid negative topology values behave like a direct deployment."""
    monkeypatch.setenv("IAM_TRUSTED_PROXY_COUNT", "-3")

    assert get_settings().trusted_proxy_count == 0
    assert auth_router._client_ip(
        _request(forwarded_for="203.0.113.99")
    ) == "10.0.0.9"


@pytest.mark.asyncio
async def test_forged_first_hop_still_hits_same_login_bucket(monkeypatch) -> None:
    """Changing the untrusted XFF prefix cannot bypass a login 429."""
    monkeypatch.setenv("IAM_TRUSTED_PROXY_COUNT", "1")
    buckets: dict[str, int] = {}

    async def fake_login(_db, *, ip_address, **_kwargs):
        buckets[ip_address] = buckets.get(ip_address, 0) + 1
        if buckets[ip_address] > 1:
            raise AuthError("rate_limited", status=429)
        return SimpleNamespace(
            access_token="access",
            refresh_token="refresh",
            access_expires_in=900,
            refresh_expires_in=3600,
            session_id="session-1",
        )

    class FakeDB:
        async def commit(self):
            return None

    monkeypatch.setattr(auth_router.auth_service, "login", fake_login)
    body = LoginRequest(email="user@example.com", password="long-enough-password")

    await auth_router.login(
        body,
        _request(forwarded_for="fake-a, 198.51.100.7"),
        FakeDB(),
    )
    with pytest.raises(HTTPException) as exc_info:
        await auth_router.login(
            body,
            _request(forwarded_for="fake-b, 198.51.100.7"),
            FakeDB(),
        )

    assert exc_info.value.status_code == 429
    assert buckets == {"198.51.100.7": 2}
