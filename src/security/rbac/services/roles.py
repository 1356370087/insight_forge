"""Role management: CRUD, permission grants, system-role protection."""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from ..models import Permission, Role, RolePermission
from ..permissions import validate_permission_codes
from ..roles import is_system_role_code
from . import audit


class RoleError(ValueError):
    """Base error for role-management failures."""


@dataclass(frozen=True)
class RoleView:
    """A read model for a role and its permission codes."""

    id: str
    code: str
    name: str
    description: str | None
    is_system: bool
    permission_codes: tuple[str, ...]


def _view(role: Role, codes: set[str]) -> RoleView:
    """Build a :class:`RoleView` from a Role + a pre-fetched code set."""
    return RoleView(
        id=str(role.id),
        code=role.code,
        name=role.name,
        description=role.description,
        is_system=bool(role.is_system),
        permission_codes=tuple(sorted(codes)),
    )


async def _permission_codes_for_role(db: AsyncSession, role_id: str) -> set[str]:
    """Return the permission codes granted to ``role_id`` via an explicit join."""
    result = await db.execute(
        select(Permission.code)
        .join(RolePermission, RolePermission.permission_id == Permission.id)
        .where(RolePermission.role_id == str(role_id))
    )
    return {row[0] for row in result.all()}


async def list_roles(db: AsyncSession) -> list[RoleView]:
    """Return all roles with their permission codes (single joined query)."""
    result = await db.execute(
        select(Role, Permission.code)
        .outerjoin(RolePermission, RolePermission.role_id == Role.id)
        .outerjoin(Permission, Permission.id == RolePermission.permission_id)
        .order_by(Role.is_system.desc(), Role.code.asc())
    )
    by_role: dict[str, tuple[Role, set[str]]] = {}
    order: list[str] = []
    for role, code in result.all():
        if role.id not in by_role:
            by_role[role.id] = (role, set())
            order.append(role.id)
        if code:
            by_role[role.id][1].add(code)
    return [_view(by_role[rid][0], by_role[rid][1]) for rid in order]


async def get_role(db: AsyncSession, role_id: str) -> RoleView | None:
    """Return one role by id or ``None``."""
    role = await db.get(Role, str(role_id))
    if role is None:
        return None
    codes = await _permission_codes_for_role(db, str(role.id))
    return _view(role, codes)


async def get_role_by_code(db: AsyncSession, code: str) -> Role | None:
    """Return the Role ORM object for ``code`` or ``None``."""
    result = await db.execute(select(Role).where(Role.code == code))
    return result.scalar_one_or_none()


async def create_role(
    db: AsyncSession,
    *,
    code: str,
    name: str,
    description: str | None,
    permission_codes: list[str],
    actor_id: str | None = None,
) -> RoleView:
    """Create a custom role combining catalog permissions.

    Raises :class:`RoleError` for duplicate/system codes or unknown permissions.
    """
    code = (code or "").strip().lower()
    if not code:
        raise RoleError("role_code_required")
    if is_system_role_code(code):
        raise RoleError("role_code_is_system")
    if await get_role_by_code(db, code) is not None:
        raise RoleError("role_code_in_use")
    codes = validate_permission_codes(list(permission_codes))
    permission_ids = await _permission_ids_for(db, codes)
    role = Role(code=code, name=name, description=description, is_system=False)
    db.add(role)
    await db.flush()
    for perm_id in permission_ids:
        db.add(RolePermission(role_id=role.id, permission_id=perm_id))
    await db.flush()
    await audit.record(
        db, action="role.created", actor_id=actor_id, target_user_id=None,
        detail={"role_id": str(role.id), "code": code, "permissions": list(codes)},
    )
    return _view(role, set(codes))


async def update_role(
    db: AsyncSession,
    role_id: str,
    *,
    name: str | None = None,
    description: str | None = None,
    actor_id: str | None = None,
) -> RoleView:
    """Update a role's display fields. Codes are immutable."""
    role = await db.get(Role, str(role_id))
    if role is None:
        raise RoleError("role_not_found")
    if name is not None:
        role.name = name
    if description is not None:
        role.description = description
    await db.flush()
    await audit.record(
        db, action="role.updated", actor_id=actor_id, detail={"role_id": str(role.id)},
    )
    codes = await _permission_codes_for_role(db, str(role.id))
    return _view(role, codes)


async def set_role_permissions(
    db: AsyncSession,
    role_id: str,
    permission_codes: list[str],
    *,
    actor_id: str | None = None,
) -> RoleView:
    """Replace a role's permission set. System roles are immutable here."""
    role = await db.get(Role, str(role_id))
    if role is None:
        raise RoleError("role_not_found")
    if role.is_system:
        raise RoleError("system_role_permissions_immutable")
    codes = validate_permission_codes(list(permission_codes))
    permission_ids = set(await _permission_ids_for(db, codes))
    await db.execute(delete(RolePermission).where(RolePermission.role_id == role.id))
    for perm_id in permission_ids:
        db.add(RolePermission(role_id=role.id, permission_id=perm_id))
    await db.flush()
    await audit.record(
        db, action="role.permissions_updated", actor_id=actor_id,
        detail={"role_id": str(role.id), "permissions": list(codes)},
    )
    return _view(role, set(codes))


async def delete_role(
    db: AsyncSession,
    role_id: str,
    *,
    actor_id: str | None = None,
) -> None:
    """Delete a custom role. System roles cannot be deleted."""
    role = await db.get(Role, str(role_id))
    if role is None:
        raise RoleError("role_not_found")
    if role.is_system:
        raise RoleError("system_role_not_deletable")
    await db.delete(role)
    await db.flush()
    await audit.record(
        db, action="role.deleted", actor_id=actor_id, detail={"role_id": str(role_id)},
    )


async def copy_role(
    db: AsyncSession,
    role_id: str,
    *,
    new_code: str,
    new_name: str | None = None,
    actor_id: str | None = None,
) -> RoleView:
    """Copy a role (system or custom) into a new custom role."""
    codes = await _permission_codes_for_role(db, str(role_id))
    source = await db.get(Role, str(role_id))
    if source is None:
        raise RoleError("role_not_found")
    return await create_role(
        db,
        code=new_code,
        name=new_name or f"{source.name} (copy)",
        description=source.description,
        permission_codes=sorted(codes),
        actor_id=actor_id,
    )


async def list_permissions(db: AsyncSession) -> list[Permission]:
    """Return the full permission catalog."""
    result = await db.execute(select(Permission).order_by(Permission.domain, Permission.code))
    return list(result.scalars().all())


async def _permission_ids_for(db: AsyncSession, codes: list[str]) -> list[str]:
    """Return the permission row ids for the given catalog codes."""
    if not codes:
        return []
    result = await db.execute(select(Permission).where(Permission.code.in_(codes)))
    rows = result.scalars().all()
    if len(rows) != len(codes):
        found = {r.code for r in rows}
        missing = [c for c in codes if c not in found]
        raise RoleError(f"unknown_permission:{missing[0]}")
    return [str(r.id) for r in rows]


__all__ = ["RoleError", "RoleView", "copy_role", "create_role", "delete_role", "get_role", "get_role_by_code", "list_permissions", "list_roles", "set_role_permissions", "update_role"]
