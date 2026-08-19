"""MCP OAuth failures must never log bearer credentials."""

from __future__ import annotations

import logging

import pytest

from open_deep_research.observability.core import _redact_text
from open_deep_research.security.redaction import redact_text
from open_deep_research.tools.mcp import oauth


class _Response:
    status = 400

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    async def text(self):
        return (
            "Authorization: Bearer bearer-secret-value "
            "api_key=sk-12345678901234567890"
        )


class _Session:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    def post(self, *_args, **_kwargs):
        return _Response()


@pytest.mark.asyncio
async def test_token_exchange_response_is_redacted(caplog, monkeypatch) -> None:
    monkeypatch.setattr(oauth.aiohttp, "ClientSession", _Session)

    with caplog.at_level(logging.WARNING, logger=oauth.__name__):
        result = await oauth.exchange_mcp_subject_token(
            "subject-secret",
            "https://mcp.example",
        )

    assert result is None
    assert "bearer-secret-value" not in caplog.text
    assert "sk-12345678901234567890" not in caplog.text
    assert "[REDACTED]" in caplog.text
    assert "status=400" in caplog.text


@pytest.mark.asyncio
async def test_token_exchange_exception_is_redacted(caplog, monkeypatch) -> None:
    class BrokenSession:
        async def __aenter__(self):
            raise RuntimeError("Authorization: Bearer exception-secret-value")

        async def __aexit__(self, *_args):
            return None

    monkeypatch.setattr(oauth.aiohttp, "ClientSession", BrokenSession)

    with caplog.at_level(logging.WARNING, logger=oauth.__name__):
        result = await oauth.exchange_mcp_subject_token(
            "subject-secret",
            "https://mcp.example",
        )

    assert result is None
    assert "exception-secret-value" not in caplog.text
    assert "[REDACTED]" in caplog.text


def test_observability_redaction_compatibility_export() -> None:
    text = "Authorization: Bearer token-value api_key=sk-12345678901234567890"

    assert _redact_text(text) == redact_text(text)
