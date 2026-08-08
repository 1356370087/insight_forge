"""User management: approval, role assignment, disable, last-admin guard."""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from ..models import User, UserRole, UserStatus
from ..repositories import collect_user_permissions, count_active_admins
from ..roles import ADMIN_CODE
from . import audit, email_tokens
from .roles import RoleError, get_role_by_code
from .sessions import bump_authz_version


class UserError(ValueError):
    """Base error for user-management failures."""


class LastAdminError(UserError):
    """Raised when an action would remove the last effective administrator."""


@dataclass(frozen=True)
class UserView:
    """A read model for a user (no credentials, no research content)."""

    id: str
    email: str
    display_name: str | None
    status: str
    email_verified_at: str | None
    authz_version: int
    role_codes: tuple[str, ...]
    created_at: str


def _to_view(user: User, role_codes: tuple[str, ...]) -> UserView:
    """Convert a User ORM object to a :class:`UserView`."""
    verified = user.email_verified_at
    return UserView(
        id=str(user.id),
        email=user.email,
        display_name=user.display_name,
        status=user.status,
        email_verified_at=verified.isoformat() if verified else None,
        authz_version=int(user.authz_version),
        role_codes=role_codes,
        created_at=user.created_at.isoformat(),
    )


async def _user_role_codes(db: AsyncSession, user_id: str) -> tuple[str, ...]:
    """Return the sorted role codes assigned to a user."""
    role_codes, _ = await collect_user_permissions(db, user_id)
    return tuple(sorted(role_codes))


async def list_users(
    db: AsyncSession,
    *,
    status: str | None = None,
    search: str | None = None,
    limit: int = 50,
) -> list[UserView]:
    """Return users (optionally filtered) as read models newest first."""
    query = select(User).order_by(User.created_at.desc()).limit(min(max(limit, 1), 200))
    if status:
        query = query.where(User.status == status)
    if search:
        query = query.where(User.email.ilike(f"%{search}%"))
    result = await db.execute(query)
    users = result.scalars().all()
    views: list[UserView] = []
    for user in users:
        views.append(_to_view(user, await _user_role_codes(db, str(user.id))))
    return views


async def get_user(db: AsyncSession, user_id: str) -> UserView | None:
    """Return one user as a read model or ``None``."""
    user = await db.get(User, str(user_id))
    if user is None:
        return None
    return _to_view(user, await _user_role_codes(db, str(user.id)))


async def patch_user(
    db: AsyncSession,
    user_id: str,
    *,
    display_name: str | None = None,
    actor_id: str | None = None,
) -> UserView:
    """Update editable user fields (display name). Status is changed via approve/disable."""
    user = await db.get(User, str(user_id))
    if user is None:
        raise UserError("user_not_found")
    if display_name is not None:
        user.display_name = display_name
    await db.flush()
    await audit.record(
        db, action="user.updated", actor_id=actor_id, target_user_id=str(user.id),
        detail={"display_name": display_name},
    )
    return _to_view(user, await _user_role_codes(db, str(user.id)))


async def _assign_role_codes(
    db: AsyncSession, user: User, codes: list[str], *, actor_id: str | None,
) -> tuple[str, ...]:
    """Replace a user's role assignments with ``codes`` (validated to exist)."""
    codes = [(c or "").strip().lower() for c in codes]
    resolved_ids: list[str] = []
    for code in dict.fromkeys(codes):  # de-dup, preserve order
        if not code:
            continue
        role = await get_role_by_code(db, code)
        if role is None:
            raise RoleError(f"unknown_role:{code}")
        resolved_ids.append(str(role.id))
    await db.execute(delete(UserRole).where(UserRole.user_id == user.id))
    for role_id in resolved_ids:
        db.add(UserRole(user_id=user.id, role_id=role_id, assigned_by=actor_id))
    await db.flush()
    return tuple(sorted(dict.fromkeys(codes)))


async def approve_user(
    db: AsyncSession,
    user_id: str,
    *,
    role_codes: list[str],
    actor_id: str | None = None,
) -> UserView:
    """Approve a pending user, assigning at least one non-admin role."""
    user = await db.get(User, str(user_id))
    if user is None:
        raise UserError("user_not_found")
    if user.status not in (UserStatus.PENDING_APPROVAL, UserStatus.ACTIVE, UserStatus.DISABLED):
        raise UserError(f"user_not_approvable:{user.status}")
    if not any(c != ADMIN_CODE for c in role_codes):
        raise UserError("approval_requires_non_admin_role")
    assigned = await _assign_role_codes(db, user, role_codes, actor_id=actor_id)
    user.status = UserStatus.ACTIVE
    # Approval is itself a security event: rotate sessions.
    await bump_authz_version(db, str(user.id), reason="approval")
    await db.flush()
    await audit.record(
        db, action="user.approved", actor_id=actor_id, target_user_id=str(user.id),
        detail={"roles": list(assigned)},
    )
    return _to_view(user, assigned)


async def assign_roles(
    db: AsyncSession,
    user_id: str,
    role_codes: list[str],
    *,
    actor_id: str | None = None,
) -> UserView:
    """Replace a user's roles, enforcing the last-admin guard."""
    await _guard_can_lose_admin(db, user_id, prospective_codes=set(role_codes))
    user = await db.get(User, str(user_id))
    if user is None:
        raise UserError("user_not_found")
    assigned = await _assign_role_codes(db, user, role_codes, actor_id=actor_id)
    await bump_authz_version(db, str(user.id), reason="role_change")
    await db.flush()
    await audit.record(
        db, action="user.roles_updated", actor_id=actor_id, target_user_id=str(user.id),
        detail={"roles": list(assigned)},
    )
    return _to_view(user, assigned)


async def disable_user(db: AsyncSession, user_id: str, *, actor_id: str | None = None) -> UserView:
    """Disable a user, revoking sessions. Blocks the last effective admin."""
    await _guard_last_admin(db, user_id)
    user = await db.get(User, str(user_id))
    if user is None:
        raise UserError("user_not_found")
    user.status = UserStatus.DISABLED
    await bump_authz_version(db, str(user.id), reason="account_disabled")
    await db.flush()
    await audit.record(
        db, action="user.disabled", actor_id=actor_id, target_user_id=str(user.id),
    )
    return _to_view(user, await _user_role_codes(db, str(user.id)))


async def enable_user(db: AsyncSession, user_id: str, *, actor_id: str | None = None) -> UserView:
    """Re-activate a previously disabled user (keeps pending-approval otherwise)."""
    user = await db.get(User, str(user_id))
    if user is None:
        raise UserError("user_not_found")
    if user.email_verified_at is None:
        user.status = UserStatus.PENDING_APPROVAL
    else:
        user.status = UserStatus.ACTIVE
    await db.flush()
    await audit.record(
        db, action="user.enabled", actor_id=actor_id, target_user_id=str(user.id),
    )
    return _to_view(user, await _user_role_codes(db, str(user.id)))


async def admin_send_password_reset(
    db: AsyncSession,
    user_id: str,
    *,
    settings,
    actor_id: str | None = None,
    send_email: bool = True,
) -> str | None:
    """Issue a password-reset token for a user (admin-triggered).

    Returns the raw token when ``send_email`` is False (for batch tooling);
    otherwise sends the email and returns ``None``. The user is marked
    ``password_reset_required``.
    """
    user = await db.get(User, str(user_id))
    if user is None:
        raise UserError("user_not_found")
    raw = await email_tokens.issue(
        db, user_id=str(user.id), purpose="password_reset", settings=settings,
    )
    user.status = UserStatus.PASSWORD_RESET_REQUIRED
    await db.flush()
    await audit.record(
        db, action="user.password_reset_sent", actor_id=actor_id, target_user_id=str(user.id),
    )
    if send_email:
        from ..email import send_password_reset_email

        await send_password_reset_email(to=user.email, token=raw, settings=settings)
        return None
    return raw


async def _guard_last_admin(db: AsyncSession, user_id: str) -> None:
    """Raise if disabling ``user_id`` would remove the last effective admin."""
    user = await db.get(User, str(user_id))
    if user is None or user.status != UserStatus.ACTIVE:
        return
    role_codes, _ = await collect_user_permissions(db, str(user_id))
    if ADMIN_CODE not in role_codes:
        return
    if await count_active_admins(db) <= 1:
        raise LastAdminError("last_admin_protected")


async def _guard_can_lose_admin(db: AsyncSession, user_id: str, *, prospective_codes: set[str]) -> None:
    """Raise if replacing the user's roles would remove the last admin."""
    user = await db.get(User, str(user_id))
    if user is None or user.status != UserStatus.ACTIVE:
        return
    role_codes, _ = await collect_user_permissions(db, str(user_id))
    if ADMIN_CODE not in role_codes:
        return  # not currently an effective admin
    if ADMIN_CODE in prospective_codes:
        return  # will remain an admin
    if await count_active_admins(db) <= 1:
        raise LastAdminError("last_admin_protected")


__all__ = [
    "LastAdminError",
    "UserError",
    "UserView",
    "admin_send_password_reset",
    "approve_user",
    "assign_roles",
    "disable_user",
    "enable_user",
    "get_user",
    "list_users",
    "patch_user",
]
