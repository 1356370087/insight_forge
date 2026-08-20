"""Built-in (system) roles and their fixed permission grants.

System roles are seeded by Alembic and flagged ``is_system=True``: they cannot
be deleted and their permission set is immutable. Administrators may copy a
system role into a custom role and then edit the copy.

Permission composition mirrors the SPEC:

* ``viewer``     — read own runs / reports / events / task activity.
* ``researcher`` — viewer + create / control / interact + search & native tools.
* ``developer``  — researcher + own observability + diagnostics + MCP/browser/skill tools.
* ``admin``      — IAM management only; never grants cross-user run access.

The bootstrap administrator is granted *both* ``admin`` and ``researcher``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import permissions as P

# Stable role codes for the system roles.
VIEWER_CODE = "viewer"
RESEARCHER_CODE = "researcher"
DEVELOPER_CODE = "developer"
ADMIN_CODE = "admin"

SYSTEM_ROLE_CODES: frozenset[str] = frozenset({VIEWER_CODE, RESEARCHER_CODE, DEVELOPER_CODE, ADMIN_CODE})


@dataclass(frozen=True)
class RoleDefinition:
    """A system role definition used for seeding."""

    code: str
    name: str
    description: str
    is_system: bool = True
    permissions: tuple[str, ...] = field(default_factory=tuple)


VIEWER = RoleDefinition(
    VIEWER_CODE, "Viewer",
    "Read own runs, reports, events and task activity.",
    permissions=(
        P.RESEARCH_RUN_READ_OWN.code,
        P.RESEARCH_TASK_ACTIVITY_READ_OWN.code,
    ),
)

RESEARCHER = RoleDefinition(
    RESEARCHER_CODE, "Researcher",
    "Create, control and interact with own research; use search & native tools.",
    permissions=(
        P.RESEARCH_RUN_READ_OWN.code,
        P.RESEARCH_TASK_ACTIVITY_READ_OWN.code,
        P.RESEARCH_RUN_CREATE.code,
        P.RESEARCH_RUN_CONTROL_OWN.code,
        P.RESEARCH_RUN_INTERACT_OWN.code,
        P.RESEARCH_TOOL_SEARCH.code,
        P.RESEARCH_TOOL_PROVIDER_NATIVE.code,
        P.RESEARCH_SECURITY_APPROVAL_READ_OWN.code,
        P.RESEARCH_SECURITY_APPROVAL_RESOLVE_OWN.code,
    ),
)

DEVELOPER = RoleDefinition(
    DEVELOPER_CODE, "Developer",
    "Researcher capabilities plus own observability, diagnostics and advanced tools.",
    permissions=(
        P.RESEARCH_RUN_READ_OWN.code,
        P.RESEARCH_TASK_ACTIVITY_READ_OWN.code,
        P.RESEARCH_RUN_CREATE.code,
        P.RESEARCH_RUN_CONTROL_OWN.code,
        P.RESEARCH_RUN_INTERACT_OWN.code,
        P.RESEARCH_TOOL_SEARCH.code,
        P.RESEARCH_TOOL_PROVIDER_NATIVE.code,
        P.RESEARCH_OBSERVABILITY_READ_OWN.code,
        P.RESEARCH_DIAGNOSTICS_PREVIEW.code,
        P.RESEARCH_TOOL_MCP.code,
        P.RESEARCH_TOOL_BROWSER.code,
        P.RESEARCH_TOOL_SKILL.code,
        P.RESEARCH_SECURITY_APPROVAL_READ_OWN.code,
        P.RESEARCH_SECURITY_APPROVAL_RESOLVE_OWN.code,
        P.RESEARCH_TOOL_SHELL_EXECUTE.code,
        P.RESEARCH_TOOL_FILE_READ.code,
        P.RESEARCH_TOOL_FILE_WRITE.code,
    ),
)

ADMIN = RoleDefinition(
    ADMIN_CODE, "Administrator",
    "Manage users, roles, permissions and audit. No cross-user run access.",
    permissions=(
        P.IAM_USERS_READ.code,
        P.IAM_USERS_WRITE.code,
        P.IAM_ROLES_READ.code,
        P.IAM_ROLES_WRITE.code,
        P.IAM_PERMISSIONS_READ.code,
        P.IAM_AUDIT_READ.code,
        P.RESEARCH_SECURITY_APPROVAL_READ_ANY.code,
        P.RESEARCH_SECURITY_APPROVAL_RESOLVE_ANY.code,
    ),
)


SYSTEM_ROLES: tuple[RoleDefinition, ...] = (VIEWER, RESEARCHER, DEVELOPER, ADMIN)


def is_system_role_code(code: str) -> bool:
    """Return whether ``code`` identifies an immutable system role."""
    return code in SYSTEM_ROLE_CODES
