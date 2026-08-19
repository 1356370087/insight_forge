"""SQLAlchemy 2 (async) ORM models for the identity & RBAC subsystem.

Tables are prefixed ``iam_`` to keep the schema self-contained and avoid
clashes with any future application tables. All timestamps are timezone-aware
UTC. UUIDs are stored as ``UUID`` via asyncpg's native support.

The model intentionally mirrors the SPEC's section 2 table list:

* ``iam_users``            — identity, credentials, status, authz version.
* ``iam_roles``            — fixed/system and custom roles.
* ``iam_permissions``      — closed permission catalog.
* ``iam_user_roles``       — user <-> role assignments (audited).
* ``iam_role_permissions`` — role <-> permission grants.
* ``iam_sessions``         — device sessions with absolute expiry & revocation.
* ``iam_refresh_tokens``   — rotating refresh JWT jti chain.
* ``iam_email_tokens``     — one-time email-verify / password-reset digests.
* ``iam_audit_events``     — append-only audit trail.
* ``iam_rate_limits``      — distributed login/register/reset rate buckets.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utcnow() -> datetime:
    """Return the current timezone-aware UTC datetime."""
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    """Declarative base for all IAM models."""


class UserStatus:
    """User lifecycle states (see SPEC section 2)."""

    PENDING_EMAIL = "pending_email"
    PENDING_APPROVAL = "pending_approval"
    ACTIVE = "active"
    DISABLED = "disabled"
    PASSWORD_RESET_REQUIRED = "password_reset_required"

    ALL: tuple[str, ...] = (
        PENDING_EMAIL,
        PENDING_APPROVAL,
        ACTIVE,
        DISABLED,
        PASSWORD_RESET_REQUIRED,
    )


class EmailTokenPurpose:
    """One-time email token purposes."""

    EMAIL_VERIFICATION = "email_verification"
    PASSWORD_RESET = "password_reset"


class SessionRevocationReason:
    """Reason codes recorded when a session is revoked."""

    LOGOUT = "logout"
    LOGOUT_ALL = "logout_all"
    PASSWORD_CHANGE = "password_change"
    FORCED_PASSWORD_RESET = "forced_password_reset"
    DISABLED = "account_disabled"
    REPLAY = "refresh_token_reuse"
    REVOKED = "revoked"
    EXPIRED = "expired"


class User(Base):
    """An identity record. Never hard-deleted (preserves run ``owner_id``)."""

    __tablename__ = "iam_users"

    id: Mapped[str] = mapped_column(PG_UUID(as_uuid=False), primary_key=True, server_default=text("gen_random_uuid()"))
    email: Mapped[str] = mapped_column(String(320), nullable=False, unique=True)
    email_normalized: Mapped[str] = mapped_column(String(320), nullable=False, unique=True, index=True)
    password_hash: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    display_name: Mapped[Optional[str]] = mapped_column(String(160), nullable=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default=UserStatus.PENDING_EMAIL, index=True)
    email_verified_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    # Bumped on disable, password change and security events; encoded in access
    # tokens so that server-side comparisons revoke stale sessions immediately.
    authz_version: Mapped[int] = mapped_column(BigInteger, nullable=False, default=1)

    failed_login_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    locked_until: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utcnow, onupdate=utcnow)

    roles: Mapped[list[UserRole]] = relationship(
        "UserRole",
        back_populates="user",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    __table_args__ = (
        CheckConstraint(
            f"status IN ({', '.join(repr(s) for s in UserStatus.ALL)})",
            name="iam_users_status_check",
        ),
    )


class Role(Base):
    """A role (system or custom). System roles are immutable."""

    __tablename__ = "iam_roles"

    id: Mapped[str] = mapped_column(PG_UUID(as_uuid=False), primary_key=True, server_default=text("gen_random_uuid()"))
    code: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    name: Mapped[str] = mapped_column(String(160), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    is_system: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utcnow, onupdate=utcnow)

    permissions: Mapped[list[RolePermission]] = relationship(
        "RolePermission",
        back_populates="role",
        cascade="all, delete-orphan",
        lazy="selectin",
    )


class Permission(Base):
    """A fixed permission catalog entry, managed by code + Alembic."""

    __tablename__ = "iam_permissions"

    id: Mapped[str] = mapped_column(PG_UUID(as_uuid=False), primary_key=True, server_default=text("gen_random_uuid()"))
    code: Mapped[str] = mapped_column(String(96), nullable=False, unique=True)
    name: Mapped[str] = mapped_column(String(160), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    domain: Mapped[str] = mapped_column(String(32), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utcnow)


class UserRole(Base):
    """Association between a user and a role, with assignment audit data."""

    __tablename__ = "iam_user_roles"

    user_id: Mapped[str] = mapped_column(
        PG_UUID(as_uuid=False), ForeignKey("iam_users.id", ondelete="CASCADE"), primary_key=True,
    )
    role_id: Mapped[str] = mapped_column(
        PG_UUID(as_uuid=False), ForeignKey("iam_roles.id", ondelete="CASCADE"), primary_key=True,
    )
    assigned_by: Mapped[Optional[str]] = mapped_column(PG_UUID(as_uuid=False), nullable=True)
    assigned_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utcnow)

    user: Mapped[User] = relationship("User", back_populates="roles")
    role: Mapped[Role] = relationship("Role", lazy="joined")


class RolePermission(Base):
    """Association between a role and a permission."""

    __tablename__ = "iam_role_permissions"

    role_id: Mapped[str] = mapped_column(
        PG_UUID(as_uuid=False), ForeignKey("iam_roles.id", ondelete="CASCADE"), primary_key=True,
    )
    permission_id: Mapped[str] = mapped_column(
        PG_UUID(as_uuid=False), ForeignKey("iam_permissions.id", ondelete="CASCADE"), primary_key=True,
    )

    role: Mapped[Role] = relationship("Role", back_populates="permissions")
    permission: Mapped[Permission] = relationship("Permission", lazy="joined")

    __table_args__ = (UniqueConstraint("role_id", "permission_id", name="iam_role_permissions_unique"),)


class Session(Base):
    """A user device session owning a refresh-token family."""

    __tablename__ = "iam_sessions"

    id: Mapped[str] = mapped_column(PG_UUID(as_uuid=False), primary_key=True, server_default=text("gen_random_uuid()"))
    user_id: Mapped[str] = mapped_column(
        PG_UUID(as_uuid=False), ForeignKey("iam_users.id", ondelete="CASCADE"), nullable=False, index=True,
    )
    user_agent: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    ip_address: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    absolute_expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    last_activity_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utcnow)
    revoked_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    revocation_reason: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utcnow)

    refresh_tokens: Mapped[list[RefreshToken]] = relationship(
        "RefreshToken", back_populates="session", cascade="all, delete-orphan", lazy="selectin",
    )

    @property
    def is_revoked(self) -> bool:
        """Return whether this session has been revoked."""
        return self.revoked_at is not None


class RefreshToken(Base):
    """A refresh-token jti record in a rotating chain (one per session family)."""

    __tablename__ = "iam_refresh_tokens"

    jti: Mapped[str] = mapped_column(String(64), primary_key=True)
    session_id: Mapped[str] = mapped_column(
        PG_UUID(as_uuid=False), ForeignKey("iam_sessions.id", ondelete="CASCADE"), nullable=False, index=True,
    )
    digest: Mapped[str] = mapped_column(String(128), nullable=False)
    previous_jti: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)
    used_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utcnow)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    session: Mapped[Session] = relationship("Session", back_populates="refresh_tokens")

    __table_args__ = (
        Index("iam_refresh_tokens_session_idx", "session_id", "used_at"),
    )


class EmailToken(Base):
    """A one-time email-verification or password-reset token digest."""

    __tablename__ = "iam_email_tokens"

    id: Mapped[str] = mapped_column(PG_UUID(as_uuid=False), primary_key=True, server_default=text("gen_random_uuid()"))
    user_id: Mapped[str] = mapped_column(
        PG_UUID(as_uuid=False), ForeignKey("iam_users.id", ondelete="CASCADE"), nullable=False, index=True,
    )
    purpose: Mapped[str] = mapped_column(String(32), nullable=False)
    digest: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utcnow)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    used_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        CheckConstraint(
            f"purpose IN ({repr(EmailTokenPurpose.EMAIL_VERIFICATION)}, {repr(EmailTokenPurpose.PASSWORD_RESET)})",
            name="iam_email_tokens_purpose_check",
        ),
        Index("iam_email_tokens_user_purpose_idx", "user_id", "purpose"),
    )


class AuditEvent(Base):
    """An append-only audit record for identity & RBAC operations."""

    __tablename__ = "iam_audit_events"

    id: Mapped[str] = mapped_column(PG_UUID(as_uuid=False), primary_key=True, server_default=text("gen_random_uuid()"))
    actor_id: Mapped[Optional[str]] = mapped_column(PG_UUID(as_uuid=False), nullable=True, index=True)
    actor_email: Mapped[Optional[str]] = mapped_column(String(320), nullable=True)
    target_user_id: Mapped[Optional[str]] = mapped_column(PG_UUID(as_uuid=False), nullable=True, index=True)
    action: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    detail: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    ip_address: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    user_agent: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utcnow, index=True)


class RateLimit(Base):
    """A distributed rate-limit bucket keyed by (bucket, identity)."""

    __tablename__ = "iam_rate_limits"

    bucket: Mapped[str] = mapped_column(String(32), nullable=False, primary_key=True)
    identity: Mapped[str] = mapped_column(String(320), nullable=False, primary_key=True)
    count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    window_started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utcnow, onupdate=utcnow)
