"""``/admin/*`` router: user, role, permission and audit management."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import get_transactional_db
from ..dependencies import Principal, require_permissions
from ..models import AuditEvent
from ..permissions import (
    IAM_AUDIT_READ,
    IAM_PERMISSIONS_READ,
    IAM_ROLES_READ,
    IAM_ROLES_WRITE,
    IAM_USERS_READ,
    IAM_USERS_WRITE,
)
from ..schemas import (
    ApproveRequest,
    AuditEventOut,
    CopyRoleRequest,
    CreateRoleRequest,
    PatchUserRequest,
    PermissionOut,
    RoleOut,
    SetPermissionsRequest,
    SetRolesRequest,
    UpdateRoleRequest,
    UserOut,
    role_to_dict,
    user_to_dict,
)
from ..services import roles as roles_service
from ..services import users as users_service
from ..settings import get_settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin", tags=["admin"])

# Compound admin capabilities that always imply a read of the same domain.
_USERS_ADMIN = require_permissions(IAM_USERS_WRITE.code, IAM_USERS_READ.code)
_ROLES_ADMIN = require_permissions(IAM_ROLES_WRITE.code, IAM_ROLES_READ.code)


def _role_out(role) -> RoleOut:
    """Serialize a service :class:`RoleView` via :class:`RoleOut`."""
    return RoleOut(**role_to_dict(role))


def _user_out(user) -> UserOut:
    """Serialize a service :class:`UserView` via :class:`UserOut`."""
    return UserOut(**user_to_dict(user))


# --------------------------------------------------------------------------- users

@router.get("/users", response_model=list[UserOut], dependencies=[Depends(require_permissions(IAM_USERS_READ.code))])
async def list_users(
    status_filter: str | None = Query(default=None, alias="status"),
    search: str | None = None,
    limit: int = Query(default=50, ge=1, le=200),
    db: AsyncSession = Depends(get_transactional_db),
) -> list[UserOut]:
    """List users (optionally filtered by status/email search)."""
    users = await users_service.list_users(db, status=status_filter, search=search, limit=limit)
    return [_user_out(u) for u in users]


@router.get("/users/{user_id}", response_model=UserOut, dependencies=[Depends(require_permissions(IAM_USERS_READ.code))])
async def get_user(user_id: str, db: AsyncSession = Depends(get_transactional_db)) -> UserOut:
    """Return one user by id."""
    user = await users_service.get_user(db, user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return _user_out(user)


@router.patch("/users/{user_id}", response_model=UserOut)
async def patch_user(
    user_id: str,
    request_body: PatchUserRequest,
    principal: Principal = Depends(_USERS_ADMIN),
    db: AsyncSession = Depends(get_transactional_db),
) -> UserOut:
    """Update editable user fields (display name)."""
    user = await users_service.patch_user(db, user_id, display_name=request_body.display_name, actor_id=principal.user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return _user_out(user)


@router.post("/users/{user_id}/approve", response_model=UserOut)
async def approve_user(
    user_id: str,
    request_body: ApproveRequest,
    principal: Principal = Depends(_USERS_ADMIN),
    db: AsyncSession = Depends(get_transactional_db),
) -> UserOut:
    """Approve a pending user and assign at least one non-admin role."""
    try:
        user = await users_service.approve_user(
            db, user_id, role_codes=request_body.role_codes, actor_id=principal.user_id,
        )
    except users_service.LastAdminError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except users_service.UserError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except roles_service.RoleError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _user_out(user)


@router.put("/users/{user_id}/roles", response_model=UserOut)
async def set_user_roles(
    user_id: str,
    request_body: SetRolesRequest,
    principal: Principal = Depends(_USERS_ADMIN),
    db: AsyncSession = Depends(get_transactional_db),
) -> UserOut:
    """Replace a user's role assignments (last-admin guard enforced)."""
    try:
        user = await users_service.assign_roles(
            db, user_id, request_body.role_codes, actor_id=principal.user_id,
        )
    except users_service.LastAdminError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except roles_service.RoleError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _user_out(user)


@router.post("/users/{user_id}/disable", response_model=UserOut)
async def disable_user(
    user_id: str,
    principal: Principal = Depends(_USERS_ADMIN),
    db: AsyncSession = Depends(get_transactional_db),
) -> UserOut:
    """Disable a user and revoke all of their active sessions."""
    try:
        user = await users_service.disable_user(db, user_id, actor_id=principal.user_id)
    except users_service.LastAdminError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except users_service.UserError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return _user_out(user)


@router.post("/users/{user_id}/enable", response_model=UserOut)
async def enable_user(
    user_id: str,
    principal: Principal = Depends(_USERS_ADMIN),
    db: AsyncSession = Depends(get_transactional_db),
) -> UserOut:
    """Enable a disabled user, preserving the approval lifecycle."""
    try:
        user = await users_service.enable_user(db, user_id, actor_id=principal.user_id)
    except users_service.UserError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return _user_out(user)


@router.post(
    "/users/{user_id}/send-password-reset",
    response_model=dict,
)
async def send_password_reset(
    user_id: str,
    principal: Principal = Depends(_USERS_ADMIN),
    db: AsyncSession = Depends(get_transactional_db),
) -> dict:
    """Issue a password-reset token for a user and email it."""
    settings = get_settings()
    try:
        await users_service.admin_send_password_reset(
            db, user_id, settings=settings, actor_id=principal.user_id, send_email=True,
        )
    except users_service.UserError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"message": "Password reset email sent."}


# --------------------------------------------------------------------------- roles

@router.get("/roles", response_model=list[RoleOut], dependencies=[Depends(require_permissions(IAM_ROLES_READ.code))])
async def list_roles(db: AsyncSession = Depends(get_transactional_db)) -> list[RoleOut]:
    """List all roles with their permission codes."""
    roles = await roles_service.list_roles(db)
    return [_role_out(r) for r in roles]


@router.post("/roles", response_model=RoleOut, status_code=status.HTTP_201_CREATED)
async def create_role(
    request_body: CreateRoleRequest,
    principal: Principal = Depends(_ROLES_ADMIN),
    db: AsyncSession = Depends(get_transactional_db),
) -> RoleOut:
    """Create a custom role combining catalog permissions."""
    try:
        role = await roles_service.create_role(
            db,
            code=request_body.code,
            name=request_body.name,
            description=request_body.description,
            permission_codes=request_body.permission_codes,
            actor_id=principal.user_id,
        )
    except roles_service.RoleError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _role_out(role)


@router.patch("/roles/{role_id}", response_model=RoleOut)
async def update_role(
    role_id: str,
    request_body: UpdateRoleRequest,
    principal: Principal = Depends(_ROLES_ADMIN),
    db: AsyncSession = Depends(get_transactional_db),
) -> RoleOut:
    """Update a role's display fields (codes are immutable)."""
    try:
        role = await roles_service.update_role(
            db, role_id, name=request_body.name, description=request_body.description, actor_id=principal.user_id,
        )
    except roles_service.RoleError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return _role_out(role)


@router.delete("/roles/{role_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_role(
    role_id: str,
    principal: Principal = Depends(_ROLES_ADMIN),
    db: AsyncSession = Depends(get_transactional_db),
) -> None:
    """Delete a custom role (system roles are protected)."""
    try:
        await roles_service.delete_role(db, role_id, actor_id=principal.user_id)
    except roles_service.RoleError as exc:
        if "not_found" in str(exc):
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/roles/{role_id}/copy", response_model=RoleOut, status_code=status.HTTP_201_CREATED)
async def copy_role(
    role_id: str,
    request_body: CopyRoleRequest,
    principal: Principal = Depends(_ROLES_ADMIN),
    db: AsyncSession = Depends(get_transactional_db),
) -> RoleOut:
    """Copy a role (system or custom) into a new custom role."""
    try:
        role = await roles_service.copy_role(
            db, role_id, new_code=request_body.new_code, new_name=request_body.new_name,
            actor_id=principal.user_id,
        )
    except roles_service.RoleError as exc:
        if "not_found" in str(exc):
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _role_out(role)


@router.put("/roles/{role_id}/permissions", response_model=RoleOut)
async def set_role_permissions(
    role_id: str,
    request_body: SetPermissionsRequest,
    principal: Principal = Depends(_ROLES_ADMIN),
    db: AsyncSession = Depends(get_transactional_db),
) -> RoleOut:
    """Replace a role's permission set (system roles are immutable)."""
    try:
        role = await roles_service.set_role_permissions(
            db, role_id, request_body.permission_codes, actor_id=principal.user_id,
        )
    except roles_service.RoleError as exc:
        if "not_found" in str(exc):
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _role_out(role)


@router.get(
    "/permissions",
    response_model=list[PermissionOut],
    dependencies=[Depends(require_permissions(IAM_PERMISSIONS_READ.code))],
)
async def list_permissions(db: AsyncSession = Depends(get_transactional_db)) -> list[PermissionOut]:
    """Return the fixed permission catalog."""
    perms = await roles_service.list_permissions(db)
    return [
        PermissionOut(code=p.code, name=p.name, description=p.description, domain=p.domain) for p in perms
    ]


@router.get("/audit-events", response_model=list[AuditEventOut], dependencies=[Depends(require_permissions(IAM_AUDIT_READ.code))])
async def list_audit_events(
    action: str | None = None,
    target_user_id: str | None = None,
    limit: int = Query(default=100, ge=1, le=500),
    db: AsyncSession = Depends(get_transactional_db),
) -> list[AuditEventOut]:
    """Query the audit log newest first."""
    query = select(AuditEvent).order_by(AuditEvent.created_at.desc()).limit(limit)
    if action:
        query = query.where(AuditEvent.action == action)
    if target_user_id:
        query = query.where(AuditEvent.target_user_id == str(target_user_id))
    result = await db.execute(query)
    return [
        AuditEventOut(
            id=str(event.id),
            actor_id=event.actor_id,
            actor_email=event.actor_email,
            target_user_id=event.target_user_id,
            action=event.action,
            detail=event.detail,
            ip_address=event.ip_address,
            created_at=event.created_at.isoformat(),
        )
        for event in result.scalars().all()
    ]
