"""OAuth token exchange, caching, and error translation for MCP tools."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import aiohttp
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import StructuredTool, ToolException
from mcp import McpError

from open_deep_research.security.redaction import redact_text
from open_deep_research.tools.token_store import get_token_store

logger = logging.getLogger(__name__)


async def exchange_mcp_subject_token(
    subject_token: str,
    base_mcp_url: str,
) -> dict[str, Any] | None:
    """Exchange a trusted server-side subject token for an MCP access token."""
    form_data = {
        "client_id": "mcp_default",
        "subject_token": subject_token,
        "grant_type": "urn:ietf:params:oauth:grant-type:token-exchange",
        "resource": base_mcp_url.rstrip("/") + "/mcp",
        "subject_token_type": "urn:ietf:params:oauth:token-type:access_token",
    }
    try:
        async with aiohttp.ClientSession() as session:
            token_url = base_mcp_url.rstrip("/") + "/oauth/token"
            async with session.post(
                token_url,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                data=form_data,
            ) as response:
                if response.status == 200:
                    return await response.json()
                response_body = redact_text(await response.text())[:512]
                logger.warning(
                    "MCP token exchange failed status=%s response=%s",
                    response.status,
                    response_body,
                )
    except Exception as exc:
        logger.warning(
            "MCP token exchange error: %s",
            redact_text(str(exc))[:512],
        )
    return None


async def get_tokens(config: RunnableConfig) -> dict[str, Any] | None:
    """Retrieve a user's cached MCP tokens when they have not expired."""
    if not config.get("configurable", {}).get("thread_id"):
        return None
    user_id = config.get("metadata", {}).get("owner")
    if not user_id:
        return None
    store = get_token_store()
    tokens = await store.get(str(user_id))
    if not tokens:
        return None
    expires_in = tokens.value.get("expires_in")
    if not expires_in:
        return tokens.value
    expiration_time = tokens.created_at + timedelta(seconds=expires_in)
    if datetime.now(timezone.utc) > expiration_time:
        await store.delete(str(user_id))
        return None
    return tokens.value


async def set_tokens(config: RunnableConfig, tokens: dict[str, Any]) -> None:
    """Store MCP tokens in the configured per-user token store."""
    if not config.get("configurable", {}).get("thread_id"):
        return
    user_id = config.get("metadata", {}).get("owner")
    if user_id:
        await get_token_store().set(str(user_id), tokens)


async def fetch_tokens(config: RunnableConfig) -> dict[str, Any] | None:
    """Return cached tokens or perform RFC 8693 exchange when configured."""
    current_tokens = await get_tokens(config)
    if current_tokens:
        return current_tokens
    subject_token = config.get("configurable", {}).get("mcp_subject_token")
    mcp_config = config.get("configurable", {}).get("mcp_config")
    if not subject_token or not mcp_config or not mcp_config.get("url"):
        return None
    tokens = await exchange_mcp_subject_token(subject_token, mcp_config["url"])
    if tokens:
        await set_tokens(config, tokens)
    return tokens


def _find_mcp_error(exc: BaseException) -> McpError | None:
    if isinstance(exc, McpError):
        return exc
    for nested in getattr(exc, "exceptions", ()):
        if found := _find_mcp_error(nested):
            return found
    return None


def wrap_mcp_authenticate_tool(tool: StructuredTool) -> StructuredTool:
    """Translate MCP interaction-required failures into ToolException."""
    original_coroutine = tool.coroutine

    async def authentication_wrapper(**kwargs):
        try:
            return await original_coroutine(**kwargs)
        except BaseException as original_error:
            mcp_error = _find_mcp_error(original_error)
            if mcp_error is None:
                raise
            error_details = mcp_error.error
            error_data = getattr(error_details, "data", None) or {}
            if getattr(error_details, "code", None) == -32003:
                message_payload = error_data.get("message", {})
                error_message = "Required interaction"
                if isinstance(message_payload, dict):
                    error_message = message_payload.get("text") or error_message
                if url := error_data.get("url"):
                    error_message = f"{error_message} {url}"
                raise ToolException(error_message) from original_error
            raise

    tool.coroutine = authentication_wrapper
    return tool

__all__ = [
    "exchange_mcp_subject_token",
    "fetch_tokens",
    "get_tokens",
    "set_tokens",
    "wrap_mcp_authenticate_tool",
]
