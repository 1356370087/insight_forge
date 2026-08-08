"""FastAPI dependencies: principal extraction, status & permission gates.

The canonical dependency is :func:`get_current_principal`, which validates the
access JWT, loads the live user/session/RBAC state from PostgreSQL and returns
a :class:`Principal`. ``require_active_user`` and ``require_permissions`` layer
status and capability checks on top.

``require_run_owner`` / ``require_task_owner`` combine a permission check with
an ownership verification. The ownership verifier is registered by the
application (server.py) at startup via :func:`register_ownership_checker`,
keeping this module decoupled from the run/task storage layer.
"""

from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable, Literal

from fastapi import Depends, Header, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from .database import session_scope
from .jwt_service import TokenError, decode_access_token
from .models import Session, User, UserStatus
from .principal import Principal
from .repositories import collect_user_permissions, get_session, get_user_by_id
from .settings import get_settings, local_dev_bypass_enabled

logger = logging.getLogger(__name__)

PermissionMode = Literal["all", "any"]

# An ownership checker is registered per kind: "run" and "task".
# Signature: async fn(db, principal, resource_key) -> bool
OwnershipChecker = Callable[[AsyncSession, Principal, Any], Awaitable[bool]]
_ownership_checkers: dict[str, OwnershipChecker] = {}


def register_ownership_checker(kind: str, checker: OwnershipChecker) -> None:
    """Register the run/task ownership verifier for ``kind``.

    The application calls this at startup so ``require_run_owner`` /
    ``require_task_owner`` can delegate to the real run store without this
    module importing it.
    """
    _ownership_checkers[kind] = checker


def _extract_bearer(authorization: str | None) -> str:
    """Return the token from a ``Bearer <token>`` header or raise 401."""
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header missing")
    try:
        scheme, token = authorization.split()
    except ValueError as exc:
        raise HTTPException(status_code=401, detail="Invalid authorization header format") from exc
    if scheme.lower() != "bearer" or not token:
        raise HTTPException(status_code=401, detail="Invalid authorization header format")
    return token


async def build_principal(db: AsyncSession, access_token: str) -> Principal:
    """Validate ``access_token`` and assemble the live :class:`Principal`.

    Raises ``HTTPException`` (401/403) on any failure so it can be used
    directly inside a dependency or a periodic SSE re-auth check.
    """
    settings = get_settings()
    try:
        decoded = decode_access_token(access_token, settings=settings)
    except TokenError as exc:
        raise HTTPException(status_code=401, detail=f"invalid_token:{exc}") from exc

    claims = decoded.claims
    user_id = str(claims["sub"])
    session_id = str(claims["sid"])
    token_authz_version = int(claims.get("authz_version", 0))

    user = await get_user_by_id(db, user_id)
    if user is None:
        raise HTTPException(status_code=401, detail="invalid_token:user_not_found")

    if user.status == UserStatus.DISABLED:
        raise HTTPException(status_code=403, detail="account_disabled")
    if user.authz_version != token_authz_version:
        # A security event (password change, role change, disable) bumped the
        # version after this token was issued — treat the token as revoked.
        raise HTTPException(status_code=401, detail="session_superseded")

    session = await get_session(db, session_id)
    if session is None or session.is_revoked:
        raise HTTPException(status_code=401, detail="session_revoked")

    role_codes, permission_codes = await collect_user_permissions(db, user_id)
    return _assemble(user, session, role_codes, permission_codes)


def _assemble(
    user: User,
    session: Session,
    role_codes: set[str],
    permission_codes: set[str],
) -> Principal:
    """Build a :class:`Principal` from freshly-loaded DB state."""
    return Principal(
        user_id=str(user.id),
        email=user.email,
        status=user.status,
        session_id=str(session.id),
        roles=frozenset(role_codes),
        permissions=frozenset(permission_codes),
        authz_version=int(user.authz_version),
    )


async def get_current_principal(authorization: str | None = Header(default=None)) -> Principal:
    """Return the authenticated :class:`Principal` for the request.

    Honors ``LOCAL_DEV_AUTH_BYPASS`` (development only) by returning a
    synthetic researcher/developer principal. Otherwise validates the access
    JWT against the live database. Opens its own short-lived session so the
    dependency remains usable on routes that don't otherwise touch the DB.
    """
    if local_dev_bypass_enabled():
        from .principal import synthetic_dev_principal

        return synthetic_dev_principal()

    token = _extract_bearer(authorization)
    async with session_scope() as db:
        return await build_principal(db, token)


async def require_active_user(
    principal: Principal = Depends(get_current_principal),
) -> Principal:
    """Return the principal only when the account is fully ``active``.

    Pending-approval users get ``403 pending_approval``; disabled users are
    rejected upstream by :func:`get_current_principal`.
    """
    if not principal.is_active:
        raise HTTPException(status_code=403, detail="pending_approval")
    return principal


def require_permissions(*codes: str, mode: PermissionMode = "all") -> Callable[..., Any]:
    """Return a dependency that enforces the given permission(s).

    Args:
        *codes: Permission codes that must be present.
        mode: ``"all"`` (default) requires every code; ``"any"`` requires one.

    The dependency also implicitly requires an active user; a pending-approval
    or disabled account never satisfies a permission check.
    """
    required = list(codes)
    if not required:
        raise ValueError("require_permissions requires at least one permission code")

    async def _dependency(
        principal: Principal = Depends(get_current_principal),
    ) -> Principal:
        if not principal.is_active:
            raise HTTPException(status_code=403, detail="pending_approval")
        if mode == "any":
            satisfied = principal.has_any(required)
        else:
            satisfied = principal.has_all(required)
        if not satisfied:
            raise HTTPException(status_code=403, detail="insufficient_permissions")
        return principal

    return _dependency


def require_run_owner(permission: str) -> Callable[..., Any]:
    """Return a dependency that checks ``permission`` *and* run ownership.

    The run id is read from the ``run_id`` path parameter. Ownership is
    verified through the ``"run"`` ownership checker registered by the app.
    Non-owned or nonexistent runs return ``404`` to avoid leaking existence.
    """
    permission_dep = require_permissions(permission)

    async def _dependency(
        run_id: str,
        principal: Principal = Depends(get_current_principal),
    ) -> Principal:
        principal = await permission_dep(principal)
        checker = _ownership_checkers.get("run")
        if checker is None:
            logger.error("Run ownership checker is not registered")
            raise HTTPException(status_code=503, detail="authorization_unavailable")
        async with session_scope() as db:
            is_owner = await checker(db, principal, run_id)
        if not is_owner:
            raise HTTPException(status_code=404, detail="Run not found")
        return principal

    return _dependency


def require_task_owner(permission: str) -> Callable[..., Any]:
    """Return a dependency that checks ``permission`` *and* task ownership.

    The run id and task id are read from the ``run_id`` / ``task_id`` path
    parameters; ownership is verified through the ``"task"`` checker, which
    receives the ``(run_id, task_id)`` tuple.
    """
    permission_dep = require_permissions(permission)

    async def _dependency(
        run_id: str,
        task_id: str,
        principal: Principal = Depends(get_current_principal),
    ) -> Principal:
        principal = await permission_dep(principal)
        checker = _ownership_checkers.get("task")
        if checker is None:
            logger.error("Task ownership checker is not registered")
            raise HTTPException(status_code=503, detail="authorization_unavailable")
        async with session_scope() as db:
            is_owner = await checker(db, principal, (run_id, task_id))
        if not is_owner:
            raise HTTPException(status_code=404, detail="Task not found")
        return principal

    return _dependency


async def reauthorize_session(db: AsyncSession, principal: Principal) -> Principal | None:
    """Re-validate a live principal for long-lived SSE connections.

    Called every ``sse_reauth_interval`` seconds. Returns a refreshed principal
    when the session/user is still valid, or ``None`` when the session has been
    revoked, the user disabled, or the ``authz_version`` bumped — in which case
    the caller must close the stream.
    """
    user = await get_user_by_id(db, principal.user_id)
    if user is None or user.status == UserStatus.DISABLED:
        return None
    if int(user.authz_version) != principal.authz_version:
        return None
    if principal.session_id is None:
        return None
    session = await get_session(db, principal.session_id)
    if session is None or session.is_revoked:
        return None
    role_codes, permission_codes = await collect_user_permissions(db, principal.user_id)
    return _assemble(user, session, role_codes, permission_codes)


def apply_principal_to_config(config: dict[str, Any], principal: Principal) -> dict[str, Any]:
    """Propagate an authenticated principal into runtime config.

    Mirrors the legacy ``apply_user_to_config`` but uses the Principal's
    distinct ``roles`` and ``permissions`` sets. ``metadata.owner`` keeps the
    user UUID so the existing run-ownership model is unchanged.
    """
    configurable = config.setdefault("configurable", {})
    metadata = config.setdefault("metadata", {})
    configurable["langgraph_auth_user"] = principal.to_runtime_dict()
    metadata["owner"] = principal.user_id
    return config


__all__ = [
    "PermissionMode",
    "apply_principal_to_config",
    "build_principal",
    "get_current_principal",
    "register_ownership_checker",
    "reauthorize_session",
    "require_active_user",
    "require_permissions",
    "require_run_owner",
    "require_task_owner",
]
