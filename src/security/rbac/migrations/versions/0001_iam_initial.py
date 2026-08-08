"""IAM/RBAC initial schema + seed permissions and system roles.

Creates all ``iam_*`` tables and seeds the closed permission catalog plus the
four system roles (viewer, researcher, developer, admin) with their fixed
permission grants. The seed reads the same definitions the runtime uses, so
the migration and code can never drift.

Revision ID: 0001_iam_initial
Revises:
Create Date: 2026-08-08
"""

from __future__ import annotations

# Make the IAM package importable for the seed definitions.
import os
import sys
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "src"))

from security.rbac.models import (  # noqa: E402
    EmailTokenPurpose,
    Permission,
    Role,
    RolePermission,
    UserStatus,
)
from security.rbac.permissions import PERMISSIONS  # noqa: E402
from security.rbac.roles import SYSTEM_ROLES  # noqa: E402

# revision identifiers, used by Alembic.
revision: str = "0001_iam_initial"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _uuid() -> sa.text:
    """Return the server-side default UUID generator expression."""
    return sa.text("gen_random_uuid()")


def upgrade() -> None:
    """Create the IAM schema and seed the catalog + system roles."""
    # gen_random_uuid() is core in PG13+; ensure availability on older versions.
    op.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto")

    op.create_table(
        "iam_permissions",
        sa.Column("id", postgresql.UUID(as_uuid=False), primary_key=True, server_default=_uuid()),
        sa.Column("code", sa.String(96), nullable=False, unique=True),
        sa.Column("name", sa.String(160), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("domain", sa.String(32), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
    )

    op.create_table(
        "iam_roles",
        sa.Column("id", postgresql.UUID(as_uuid=False), primary_key=True, server_default=_uuid()),
        sa.Column("code", sa.String(64), nullable=False, unique=True),
        sa.Column("name", sa.String(160), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("is_system", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
    )

    op.create_table(
        "iam_users",
        sa.Column("id", postgresql.UUID(as_uuid=False), primary_key=True, server_default=_uuid()),
        sa.Column("email", sa.String(320), nullable=False, unique=True),
        sa.Column("email_normalized", sa.String(320), nullable=False, unique=True),
        sa.Column("password_hash", sa.Text(), nullable=True),
        sa.Column("display_name", sa.String(160), nullable=True),
        sa.Column("status", sa.String(32), nullable=False, server_default=sa.text("'pending_email'")),
        sa.Column("email_verified_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("authz_version", sa.BigInteger(), nullable=False, server_default=sa.text("1")),
        sa.Column("failed_login_count", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("locked_until", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.CheckConstraint(
            f"status IN ({', '.join(repr(s) for s in UserStatus.ALL)})",
            name="iam_users_status_check",
        ),
    )
    op.create_index("ix_iam_users_status", "iam_users", ["status"])
    op.create_index("ix_iam_users_email_normalized", "iam_users", ["email_normalized"])

    op.create_table(
        "iam_user_roles",
        sa.Column("user_id", postgresql.UUID(as_uuid=False),
                  sa.ForeignKey("iam_users.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("role_id", postgresql.UUID(as_uuid=False),
                  sa.ForeignKey("iam_roles.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("assigned_by", postgresql.UUID(as_uuid=False), nullable=True),
        sa.Column("assigned_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
    )

    op.create_table(
        "iam_role_permissions",
        sa.Column("role_id", postgresql.UUID(as_uuid=False),
                  sa.ForeignKey("iam_roles.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("permission_id", postgresql.UUID(as_uuid=False),
                  sa.ForeignKey("iam_permissions.id", ondelete="CASCADE"), primary_key=True),
        sa.UniqueConstraint("role_id", "permission_id", name="iam_role_permissions_unique"),
    )

    op.create_table(
        "iam_sessions",
        sa.Column("id", postgresql.UUID(as_uuid=False), primary_key=True, server_default=_uuid()),
        sa.Column("user_id", postgresql.UUID(as_uuid=False),
                  sa.ForeignKey("iam_users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("user_agent", sa.Text(), nullable=True),
        sa.Column("ip_address", sa.String(64), nullable=True),
        sa.Column("absolute_expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_activity_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("revocation_reason", sa.String(64), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
    )
    op.create_index("ix_iam_sessions_user_id", "iam_sessions", ["user_id"])

    op.create_table(
        "iam_refresh_tokens",
        sa.Column("jti", sa.String(64), primary_key=True),
        sa.Column("session_id", postgresql.UUID(as_uuid=False),
                  sa.ForeignKey("iam_sessions.id", ondelete="CASCADE"), nullable=False),
        sa.Column("digest", sa.String(128), nullable=False),
        sa.Column("previous_jti", sa.String(64), nullable=True),
        sa.Column("used_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_iam_refresh_tokens_session_id", "iam_refresh_tokens", ["session_id"])
    op.create_index("ix_iam_refresh_tokens_previous_jti", "iam_refresh_tokens", ["previous_jti"])
    op.create_index("iam_refresh_tokens_session_idx", "iam_refresh_tokens", ["session_id", "used_at"])

    op.create_table(
        "iam_email_tokens",
        sa.Column("id", postgresql.UUID(as_uuid=False), primary_key=True, server_default=_uuid()),
        sa.Column("user_id", postgresql.UUID(as_uuid=False),
                  sa.ForeignKey("iam_users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("purpose", sa.String(32), nullable=False),
        sa.Column("digest", sa.String(128), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("used_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            f"purpose IN ({repr(EmailTokenPurpose.EMAIL_VERIFICATION)}, {repr(EmailTokenPurpose.PASSWORD_RESET)})",
            name="iam_email_tokens_purpose_check",
        ),
    )
    op.create_index("ix_iam_email_tokens_user_id", "iam_email_tokens", ["user_id"])
    op.create_index("ix_iam_email_tokens_digest", "iam_email_tokens", ["digest"])
    op.create_index("iam_email_tokens_user_purpose_idx", "iam_email_tokens", ["user_id", "purpose"])

    op.create_table(
        "iam_audit_events",
        sa.Column("id", postgresql.UUID(as_uuid=False), primary_key=True, server_default=_uuid()),
        sa.Column("actor_id", postgresql.UUID(as_uuid=False), nullable=True),
        sa.Column("actor_email", sa.String(320), nullable=True),
        sa.Column("target_user_id", postgresql.UUID(as_uuid=False), nullable=True),
        sa.Column("action", sa.String(64), nullable=False),
        sa.Column("detail", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("ip_address", sa.String(64), nullable=True),
        sa.Column("user_agent", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
    )
    op.create_index("ix_iam_audit_events_actor_id", "iam_audit_events", ["actor_id"])
    op.create_index("ix_iam_audit_events_target_user_id", "iam_audit_events", ["target_user_id"])
    op.create_index("ix_iam_audit_events_action", "iam_audit_events", ["action"])
    op.create_index("ix_iam_audit_events_created_at", "iam_audit_events", ["created_at"])

    op.create_table(
        "iam_rate_limits",
        sa.Column("bucket", sa.String(32), primary_key=True),
        sa.Column("identity", sa.String(320), primary_key=True),
        sa.Column("count", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("window_started_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
    )

    _seed_catalog_and_roles()


def _seed_catalog_and_roles() -> None:
    """Insert the fixed permission catalog, system roles and their grants."""
    bind = op.get_bind()
    permission_table = Permission.__table__
    role_table = Role.__table__
    role_permission_table = RolePermission.__table__

    op.bulk_insert(
        permission_table,
        [
            {"code": p.code, "name": p.name, "description": p.description, "domain": p.domain}
            for p in PERMISSIONS
        ],
    )
    op.bulk_insert(
        role_table,
        [
            {"code": r.code, "name": r.name, "description": r.description, "is_system": True}
            for r in SYSTEM_ROLES
        ],
    )

    perm_ids = dict(bind.execute(sa.text("SELECT code, id FROM iam_permissions")).fetchall())
    role_ids = dict(bind.execute(sa.text("SELECT code, id FROM iam_roles")).fetchall())
    grants = [
        {"role_id": role_ids[role.code], "permission_id": perm_ids[perm_code]}
        for role in SYSTEM_ROLES
        for perm_code in role.permissions
    ]
    if grants:
        op.bulk_insert(role_permission_table, grants)


def downgrade() -> None:
    """Drop all IAM tables in reverse dependency order."""
    for table in (
        "iam_rate_limits",
        "iam_audit_events",
        "iam_email_tokens",
        "iam_refresh_tokens",
        "iam_sessions",
        "iam_role_permissions",
        "iam_user_roles",
        "iam_users",
        "iam_permissions",
        "iam_roles",
    ):
        op.drop_table(table)
