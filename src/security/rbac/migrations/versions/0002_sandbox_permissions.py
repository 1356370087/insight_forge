"""Add sandbox approval and developer tool permissions.

Revision ID: 0002_sandbox_permissions
Revises: 0001_iam_initial
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

from security.rbac.permissions import PERMISSIONS
from security.rbac.roles import SYSTEM_ROLES

revision: str = "0002_sandbox_permissions"
down_revision: str | None = "0001_iam_initial"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_CODES = {
    "research.security_approval.read_own",
    "research.security_approval.resolve_own",
    "research.security_approval.read_any",
    "research.security_approval.resolve_any",
    "research.tool.shell.execute",
    "research.tool.file.read",
    "research.tool.file.write",
}


def upgrade() -> None:
    """Insert the V7 sandbox permission catalog and system-role grants."""
    bind = op.get_bind()
    permissions = sa.table(
        "iam_permissions",
        sa.column("id", sa.String),
        sa.column("code", sa.String),
        sa.column("name", sa.String),
        sa.column("description", sa.String),
        sa.column("domain", sa.String),
    )
    role_permissions = sa.table(
        "iam_role_permissions",
        sa.column("role_id", sa.String),
        sa.column("permission_id", sa.String),
    )
    entries = [permission for permission in PERMISSIONS if permission.code in _CODES]
    op.bulk_insert(
        permissions,
        [
            {
                "code": item.code,
                "name": item.name,
                "description": item.description,
                "domain": item.domain,
            }
            for item in entries
        ],
    )
    permission_ids = dict(
        bind.execute(
            sa.text("SELECT code, id FROM iam_permissions WHERE code = ANY(:codes)"),
            {"codes": list(_CODES)},
        ).fetchall()
    )
    role_ids = dict(bind.execute(sa.text("SELECT code, id FROM iam_roles")).fetchall())
    grants = []
    for role in SYSTEM_ROLES:
        for code in role.permissions:
            if code in _CODES:
                grants.append(
                    {"role_id": role_ids[role.code], "permission_id": permission_ids[code]}
                )
    if grants:
        op.bulk_insert(role_permissions, grants)


def downgrade() -> None:
    """Remove V7 sandbox grants and permission catalog rows."""
    bind = op.get_bind()
    bind.execute(
        sa.text(
            "DELETE FROM iam_role_permissions WHERE permission_id IN "
            "(SELECT id FROM iam_permissions WHERE code = ANY(:codes))"
        ),
        {"codes": list(_CODES)},
    )
    bind.execute(
        sa.text("DELETE FROM iam_permissions WHERE code = ANY(:codes)"),
        {"codes": list(_CODES)},
    )
