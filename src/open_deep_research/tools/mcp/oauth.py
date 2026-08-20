"""OAuth token exchange, caching, and error translation for MCP tools."""

from __future__ import annotations

import ipaddress
import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import aiohttp
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import StructuredTool, ToolException
from mcp import MCPError
from mcp.types import URL_ELICITATION_REQUIRED

from open_deep_research.security.redaction import redact_text
from open_deep_research.tools.token_store import get_token_store

logger = logging.getLogger(__name__)

_LEGACY_INTERACTION_REQUIRED = -32003
"""Pre-v2 server convention for "visit this URL to interact"; kept for older servers."""


class MCPInteractionRequired(ToolException):
    """Structured MCP interaction request forwarded to the approval layer."""

    def __init__(self, message: str, interaction_url: str | None = None) -> None:
        """Store a bounded message and validated interaction URL."""
        super().__init__(message)
        self.interaction_url = interaction_url


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
    metadata = config.get("metadata", {})
    if metadata.get("sandbox_gateway_physical"):
        vault = config.get("configurable", {}).get("_sandbox_credential_vault")
        if not isinstance(vault, dict):
            return None
        tokens = vault.get("mcp_tokens")
        if not isinstance(tokens, dict):
            return None
        expires_at = tokens.get("_expires_at")
        if expires_at is not None and float(expires_at) <= datetime.now(
            timezone.utc
        ).timestamp():
            vault.pop("mcp_tokens", None)
            return None
        return {key: value for key, value in tokens.items() if key != "_expires_at"}
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
    metadata = config.get("metadata", {})
    if metadata.get("sandbox_gateway_physical"):
        configurable = config.setdefault("configurable", {})
        vault = configurable.setdefault("_sandbox_credential_vault", {})
        if not isinstance(vault, dict):
            raise RuntimeError("sandbox_gateway_credential_vault_invalid")
        value = dict(tokens)
        if tokens.get("expires_in"):
            value["_expires_at"] = datetime.now(timezone.utc).timestamp() + float(
                tokens["expires_in"]
            )
        vault["mcp_tokens"] = value
        return
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


def _find_mcp_error(exc: BaseException) -> MCPError | None:
    if isinstance(exc, MCPError):
        return exc
    for nested in getattr(exc, "exceptions", ()):
        if found := _find_mcp_error(nested):
            return found
    return None


def _interaction_required_message(code: int, error_data: Any) -> str | None:
    """Build the HITL message for an interaction-required error, if it is one."""
    data = error_data if isinstance(error_data, dict) else {}
    if code == URL_ELICITATION_REQUIRED:
        parts: list[str] = []
        for item in data.get("elicitations") or []:
            if not isinstance(item, dict):
                continue
            message = str(item.get("message") or "Required interaction")
            if url := item.get("url"):
                message = f"{message} {url}"
            parts.append(message)
        return "\n".join(parts) if parts else "Required interaction"
    if code == _LEGACY_INTERACTION_REQUIRED:
        message_payload = data.get("message", {})
        error_message = "Required interaction"
        if isinstance(message_payload, dict):
            error_message = message_payload.get("text") or error_message
        if url := data.get("url"):
            error_message = f"{error_message} {url}"
        return error_message
    return None


def _validated_interaction_url(value: str | None) -> str | None:
    """Accept a bounded HTTPS OAuth URL without user-info or private IP literals."""
    if not value or len(value) > 4096:
        return None
    try:
        parsed = urlsplit(value)
        host = parsed.hostname
        if (
            parsed.scheme.lower() not in {"http", "https"}
            or not host
            or parsed.username is not None
            or parsed.password is not None
        ):
            return None
        if parsed.scheme.lower() == "http" and host.lower() not in {
            "localhost",
            "127.0.0.1",
            "::1",
        }:
            return None
        try:
            address = ipaddress.ip_address(host)
        except ValueError:
            address = None
        if address is not None and not address.is_global and not address.is_loopback:
            return None
        netloc = host.lower()
        if parsed.port:
            netloc += f":{parsed.port}"
        return urlunsplit(
            (parsed.scheme.lower(), netloc, parsed.path or "/", parsed.query, "")
        )
    except (TypeError, ValueError):
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
            error_message = _interaction_required_message(
                getattr(error_details, "code", None), error_data
            )
            if error_message is not None:
                match = re.search(r"https?://[^\s]+", error_message)
                interaction_url = _validated_interaction_url(
                    match.group(0).rstrip(".,)") if match else None
                )
                raise MCPInteractionRequired(
                    error_message,
                    interaction_url,
                ) from original_error
            raise

    tool.coroutine = authentication_wrapper
    return tool

__all__ = [
    "exchange_mcp_subject_token",
    "fetch_tokens",
    "get_tokens",
    "set_tokens",
    "MCPInteractionRequired",
    "wrap_mcp_authenticate_tool",
]
