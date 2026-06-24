"""FastAPI authentication helpers (Supabase JWT)."""

from __future__ import annotations

import asyncio
import os
from typing import Any, Optional

try:
    from fastapi import Header, HTTPException
except ModuleNotFoundError:  # pragma: no cover - lets unit tests import role helpers before uv sync
    def Header(default=None):
        """Fallback replacement for FastAPI Header."""
        return default

    class HTTPException(Exception):
        """Fallback replacement for FastAPI HTTPException."""

        def __init__(self, status_code: int, detail: str):
            """Create a fallback HTTP exception."""
            super().__init__(detail)
            self.status_code = status_code
            self.detail = detail
from supabase import Client, create_client


def _extract_roles(user: Any) -> list[str]:
    """Extract user roles from Supabase ``app_metadata``."""
    app_metadata = getattr(user, "app_metadata", None) or {}
    if not isinstance(app_metadata, dict):
        return []
    roles: list[str] = []
    seen: set[str] = set()
    single = app_metadata.get("role")
    if isinstance(single, str) and single and single not in seen:
        roles.append(single)
        seen.add(single)
    multiple = app_metadata.get("roles")
    if isinstance(multiple, list):
        for role in multiple:
            if isinstance(role, str) and role and role not in seen:
                roles.append(role)
                seen.add(role)
    return roles


supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_KEY")
supabase: Optional[Client] = None

if supabase_url and supabase_key:
    supabase = create_client(supabase_url, supabase_key)


async def get_current_user(authorization: str | None = Header(default=None)) -> dict[str, Any]:
    """Validate a Supabase Bearer token and return runtime user metadata."""
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header missing")

    try:
        scheme, token = authorization.split()
        assert scheme.lower() == "bearer"
    except (ValueError, AssertionError) as exc:
        raise HTTPException(status_code=401, detail="Invalid authorization header format") from exc

    if not supabase:
        raise HTTPException(status_code=500, detail="Supabase client not initialized")

    try:
        response = await asyncio.to_thread(supabase.auth.get_user, token)
        user = response.user
    except Exception as exc:  # noqa: BLE001 - normalize provider errors to HTTP
        raise HTTPException(status_code=401, detail=f"Authentication error: {exc}") from exc

    if not user:
        raise HTTPException(status_code=401, detail="Invalid token or user not found")

    roles = _extract_roles(user)
    return {
        "identity": user.id,
        "permissions": roles,
        "is_authenticated": True,
    }


def apply_user_to_config(config: dict[str, Any], user: dict[str, Any]) -> dict[str, Any]:
    """Propagate authenticated user details into runtime config."""
    configurable = config.setdefault("configurable", {})
    metadata = config.setdefault("metadata", {})
    configurable["langgraph_auth_user"] = user
    metadata["owner"] = user.get("identity")
    return config