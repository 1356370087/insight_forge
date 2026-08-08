"""Bootstrap the first administrator.

Creates the first user with both ``admin`` and ``researcher`` roles (active,
email-verified) when no administrator exists yet. Idempotent: a no-op once an
admin is present. Invoked by the ``bootstrap-admin`` CLI and may also be wired
into a post-migration step.
"""

from __future__ import annotations

import logging

from sqlalchemy.ext.asyncio import AsyncSession

from ..emails import validate_and_normalize
from ..models import User, UserRole, UserStatus, utcnow
from ..passwords import hash_password, validate_password
from ..repositories import count_active_admins, get_user_by_email_normalized
from ..roles import ADMIN_CODE, RESEARCHER_CODE
from .roles import get_role_by_code

logger = logging.getLogger(__name__)


async def bootstrap_admin(
    db: AsyncSession,
    *,
    email: str,
    password: str,
    display_name: str | None = None,
) -> User | None:
    """Create the first admin if none exists; return the user or ``None``.

    Returns ``None`` (no-op) when an administrator already exists, so the
    operation is safe to run repeatedly.
    """
    if await count_active_admins(db) > 0:
        logger.info("bootstrap_admin: an admin already exists; skipping.")
        return None

    normalized = validate_and_normalize(email)
    validated = validate_password(password)
    existing = await get_user_by_email_normalized(db, normalized)
    if existing is not None:
        logger.warning("bootstrap_admin: email %s already in use; skipping.", normalized)
        return None

    user = User(
        email=normalized,
        email_normalized=normalized,
        password_hash=hash_password(validated),
        display_name=display_name or "Administrator",
        status=UserStatus.ACTIVE,
        email_verified_at=utcnow(),
    )
    db.add(user)
    await db.flush()
    for code in (ADMIN_CODE, RESEARCHER_CODE):
        role = await get_role_by_code(db, code)
        if role is None:
            raise RuntimeError(f"system_role_missing:{code}")
        db.add(UserRole(user_id=user.id, role_id=role.id))
    await db.flush()
    logger.info("bootstrap_admin: created initial administrator %s.", normalized)
    return user
