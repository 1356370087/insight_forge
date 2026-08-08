"""Pydantic request/response schemas for the ``/auth/*`` and ``/admin/*`` APIs."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, EmailStr, Field

EMAIL_FIELD = Field(max_length=320)


class RegisterRequest(BaseModel):
    """Registration request body."""

    model_config = ConfigDict(extra="forbid")

    email: EmailStr = EMAIL_FIELD
    password: str = Field(min_length=1, max_length=200)
    display_name: str | None = Field(default=None, max_length=160)


class VerifyEmailRequest(BaseModel):
    """Email-verification request body."""

    model_config = ConfigDict(extra="forbid")

    token: str = Field(min_length=1, max_length=256)


class ResendVerificationRequest(BaseModel):
    """Re-send verification email request body."""

    model_config = ConfigDict(extra="forbid")

    email: EmailStr = EMAIL_FIELD


class LoginRequest(BaseModel):
    """Login request body."""

    model_config = ConfigDict(extra="forbid")

    email: EmailStr = EMAIL_FIELD
    password: str = Field(min_length=1, max_length=200)


class RefreshRequest(BaseModel):
    """Refresh-token rotation request body."""

    model_config = ConfigDict(extra="forbid")

    refresh_token: str = Field(min_length=1, max_length=4096)


class ForgotPasswordRequest(BaseModel):
    """Forgot-password request body."""

    model_config = ConfigDict(extra="forbid")

    email: EmailStr = EMAIL_FIELD


class ResetPasswordRequest(BaseModel):
    """Password-reset request body."""

    model_config = ConfigDict(extra="forbid")

    token: str = Field(min_length=1, max_length=256)
    password: str = Field(min_length=1, max_length=200)


class ChangePasswordRequest(BaseModel):
    """Change-password request body."""

    model_config = ConfigDict(extra="forbid")

    current_password: str = Field(min_length=1, max_length=200)
    new_password: str = Field(min_length=1, max_length=200)


class TokenPair(BaseModel):
    """Standard access + refresh token pair for non-browser clients."""

    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    refresh_expires_in: int
    session_id: str


class UserMe(BaseModel):
    """The authenticated user's own profile + effective RBAC."""

    id: str
    email: str
    display_name: str | None
    status: str
    roles: list[str]
    permissions: list[str]
    authz_version: int
    session_id: str | None = None


class SessionItem(BaseModel):
    """A device session in the sessions list."""

    id: str
    created_at: str
    last_activity_at: str
    user_agent: str | None
    ip_address: str | None
    is_current: bool
    is_revoked: bool


class AcceptedResponse(BaseModel):
    """Generic ``202 Accepted`` / success body."""

    message: str = "accepted"


class LogoutResponse(BaseModel):
    """Logout result."""

    revoked: bool = True


class LogoutAllResponse(BaseModel):
    """Logout-all result."""

    revoked_count: int


class ApproveRequest(BaseModel):
    """Admin user-approval request body."""

    model_config = ConfigDict(extra="forbid")

    role_codes: list[str] = Field(min_length=1)


class SetRolesRequest(BaseModel):
    """Admin role-assignment request body (replaces the set)."""

    model_config = ConfigDict(extra="forbid")

    role_codes: list[str] = Field(default_factory=list)


class CreateRoleRequest(BaseModel):
    """Custom-role creation request body."""

    model_config = ConfigDict(extra="forbid")

    code: str = Field(min_length=1, max_length=64)
    name: str = Field(min_length=1, max_length=160)
    description: str | None = Field(default=None, max_length=2000)
    permission_codes: list[str] = Field(default_factory=list)


class UpdateRoleRequest(BaseModel):
    """Role display-field update request body."""

    model_config = ConfigDict(extra="forbid")

    name: str | None = Field(default=None, max_length=160)
    description: str | None = Field(default=None, max_length=2000)


class SetPermissionsRequest(BaseModel):
    """Role permission-replacement request body."""

    model_config = ConfigDict(extra="forbid")

    permission_codes: list[str] = Field(default_factory=list)


class PatchUserRequest(BaseModel):
    """Admin user patch request body."""

    model_config = ConfigDict(extra="forbid")

    display_name: str | None = Field(default=None, max_length=160)


class CopyRoleRequest(BaseModel):
    """Role-copy request body."""

    model_config = ConfigDict(extra="forbid")

    new_code: str = Field(min_length=1, max_length=64)
    new_name: str | None = Field(default=None, max_length=160)


class RoleOut(BaseModel):
    """Role read model."""

    id: str
    code: str
    name: str
    description: str | None
    is_system: bool
    permission_codes: list[str]


class PermissionOut(BaseModel):
    """Permission catalog entry read model."""

    code: str
    name: str
    description: str | None
    domain: str


class UserOut(BaseModel):
    """User read model (no credentials, no research content)."""

    id: str
    email: str
    display_name: str | None
    status: str
    email_verified_at: str | None
    authz_version: int
    role_codes: list[str]
    created_at: str


class AuditEventOut(BaseModel):
    """Audit-event read model."""

    id: str
    actor_id: str | None
    actor_email: str | None
    target_user_id: str | None
    action: str
    detail: dict[str, Any] | None
    ip_address: str | None
    created_at: str


def role_to_dict(role) -> dict[str, Any]:
    """Serialize a service :class:`RoleView` to a plain dict."""
    return {
        "id": role.id,
        "code": role.code,
        "name": role.name,
        "description": role.description,
        "is_system": role.is_system,
        "permission_codes": list(role.permission_codes),
    }


def user_to_dict(user) -> dict[str, Any]:
    """Serialize a service :class:`UserView` to a plain dict."""
    return {
        "id": user.id,
        "email": user.email,
        "display_name": user.display_name,
        "status": user.status,
        "email_verified_at": user.email_verified_at,
        "authz_version": user.authz_version,
        "role_codes": list(user.role_codes),
        "created_at": user.created_at,
    }
