"""Token storage abstraction used after removing LangGraph Store."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Protocol


@dataclass
class TokenRecord:
    """Stored token payload plus creation time."""

    value: dict[str, Any]
    created_at: datetime


class TokenStore(Protocol):
    """Minimal async token store contract."""

    async def get(self, user_id: str) -> TokenRecord | None:
        """Return a token record for a user, if present."""

    async def set(self, user_id: str, tokens: dict[str, Any]) -> None:
        """Store tokens for a user."""

    async def delete(self, user_id: str) -> None:
        """Delete tokens for a user."""


class MemoryTokenStore:
    """Process-local token store for local runs and tests."""

    def __init__(self) -> None:
        """Create an empty in-memory token store."""
        self._tokens: dict[str, TokenRecord] = {}

    async def get(self, user_id: str) -> TokenRecord | None:
        """Return a token record for a user."""
        return self._tokens.get(user_id)

    async def set(self, user_id: str, tokens: dict[str, Any]) -> None:
        """Store tokens for a user."""
        self._tokens[user_id] = TokenRecord(
            value=tokens,
            created_at=datetime.now(timezone.utc),
        )

    async def delete(self, user_id: str) -> None:
        """Delete tokens for a user."""
        self._tokens.pop(user_id, None)


_DEFAULT_TOKEN_STORE = MemoryTokenStore()


def get_token_store() -> TokenStore:
    """Return the configured token store.

    The first LangGraph-free implementation is in-memory. The interface is kept
    deliberately small so alternative durable stores can be added without changing
    OAuth call sites.
    """
    return _DEFAULT_TOKEN_STORE
