"""Read-side queries used to assemble a :class:`Principal` from the database.

Kept separate from the dependency layer so the same queries power both the
request-time dependency and the periodic SSE re-authorization check.
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from .models import Permission, Role, Session, User, UserRole


async def get_user_by_id(db: AsyncSession, user_id: str) -> User | None:
    """Return the user with ``user_id`` or ``None``."""
    result = await db.get(User, str(user_id))
    return result


async def get_user_by_email_normalized(db: AsyncSession, email_normalized: str) -> User | None:
    """Return the user matching the normalized email or ``None``."""
    result = await db.execute(select(User).where(User.email_normalized == email_normalized))
    return result.scalar_one_or_none()


async def get_session(db: AsyncSession, session_id: str) -> Session | None:
    """Return the session with ``session_id`` or ``None``."""
    return await db.get(Session, str(session_id))


async def get_refresh_token(db: AsyncSession, jti: str):
    """Return the refresh-token record for ``jti`` or ``None``."""
    from .models import RefreshToken

    return await db.get(RefreshToken, jti)


async def collect_user_permissions(db: AsyncSession, user_id: str) -> tuple[set[str], set[str]]:
    """Return ``(role_codes, permission_codes)`` for the user's assigned roles.

    System + custom roles are both included. Permissions are the union across
    all roles; role codes are the distinct set of assigned role codes.
    """
    user_result = await db.execute(
        select(User).where(User.id == str(user_id))
    )
    user = user_result.scalar_one_or_none()
    if user is None:
        return set(), set()

    role_ids = [ur.role_id for ur in user.roles]
    if not role_ids:
        return set(), set()

    roles_result = await db.execute(select(Role).where(Role.id.in_(role_ids)))
    roles = roles_result.scalars().all()
    role_codes = {r.code for r in roles}

    permission_ids = {rp.permission_id for r in roles for rp in r.permissions}
    if not permission_ids:
        return role_codes, set()

    perms_result = await db.execute(select(Permission).where(Permission.id.in_(permission_ids)))
    permission_codes = {p.code for p in perms_result.scalars().all()}
    return role_codes, permission_codes


async def count_active_admins(db: AsyncSession) -> int:
    """Return the number of active users holding the ``admin`` role.

    Used by the last-admin guard: at least one must always remain.
    """
    from .roles import ADMIN_CODE

    admin_role = await db.execute(select(Role).where(Role.code == ADMIN_CODE))
    admin_role_obj = admin_role.scalar_one_or_none()
    if admin_role_obj is None:
        return 0
    result = await db.execute(
        select(UserRole.user_id)
        .join(User, User.id == UserRole.user_id)
        .where(UserRole.role_id == admin_role_obj.id, User.status == "active")
        .distinct()
    )
    return len(result.all())
