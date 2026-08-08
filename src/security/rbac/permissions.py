"""Fixed permission catalog for the self-hosted RBAC system.

The permission set is closed: it is defined here and seeded by Alembic, and
administrators can only *combine* existing permissions into custom roles.
Adding a new capability is a code change (plus a migration), never a runtime
operation.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Permission:
    """A single fixed capability in the permission catalog."""

    code: str
    name: str
    description: str
    domain: str


# --- Research capabilities (always scoped to the requesting user's own data) ---

RESEARCH_RUN_READ_OWN = Permission(
    "research.run.read_own", "Read own runs",
    "List and read the caller's own runs, reports and public events.", "research",
)
RESEARCH_RUN_CREATE = Permission(
    "research.run.create", "Create runs",
    "Start a new research run and stream its events.", "research",
)
RESEARCH_RUN_CONTROL_OWN = Permission(
    "research.run.control_own", "Control own runs",
    "Cancel or resume the caller's own runs.", "research",
)
RESEARCH_RUN_INTERACT_OWN = Permission(
    "research.run.interact_own", "Interact with own runs",
    "Submit feedback and resolve human-in-the-loop actions on own runs.", "research",
)
RESEARCH_TASK_ACTIVITY_READ_OWN = Permission(
    "research.task_activity.read_own", "Read own task activity",
    "Read the task activity stream of the caller's own runs.", "research",
)
RESEARCH_OBSERVABILITY_READ_OWN = Permission(
    "research.observability.read_own", "Read own observability",
    "Read persisted traces, spans and usage for the caller's own runs.", "research",
)
RESEARCH_DIAGNOSTICS_PREVIEW = Permission(
    "research.diagnostics.preview", "Diagnostics preview",
    "View bounded diagnostic previews (e.g. task activity previews).", "research",
)

# --- Tool-source gates (compose with existing agent tool governance) ---

RESEARCH_TOOL_SEARCH = Permission(
    "research.tool.search", "Search tools",
    "Use search-tool sources.", "research",
)
RESEARCH_TOOL_PROVIDER_NATIVE = Permission(
    "research.tool.provider_native", "Provider-native tools",
    "Use provider-native tool sources.", "research",
)
RESEARCH_TOOL_MCP = Permission(
    "research.tool.mcp", "MCP tools",
    "Use MCP tool sources.", "research",
)
RESEARCH_TOOL_BROWSER = Permission(
    "research.tool.browser", "Browser tools",
    "Use browser-automation tool sources.", "research",
)
RESEARCH_TOOL_SKILL = Permission(
    "research.tool.skill", "Skill tools",
    "Use skill tool sources.", "research",
)

# --- Identity & access management (never grants cross-user research access) ---

IAM_USERS_READ = Permission(
    "iam.users.read", "Read users",
    "List and read identity user records (no research content).", "iam",
)
IAM_USERS_WRITE = Permission(
    "iam.users.write", "Manage users",
    "Approve, disable, assign roles and trigger password resets for users.", "iam",
)
IAM_ROLES_READ = Permission(
    "iam.roles.read", "Read roles",
    "List roles and the fixed permission catalog.", "iam",
)
IAM_ROLES_WRITE = Permission(
    "iam.roles.write", "Manage roles",
    "Create, update and delete custom roles and edit their permissions.", "iam",
)
IAM_PERMISSIONS_READ = Permission(
    "iam.permissions.read", "Read permission catalog",
    "Read the fixed permission catalog.", "iam",
)
IAM_AUDIT_READ = Permission(
    "iam.audit.read", "Read audit events",
    "Query the identity audit log.", "iam",
)


# The ordered, closed catalog. Keep stable; migrations depend on these codes.
PERMISSIONS: tuple[Permission, ...] = (
    RESEARCH_RUN_READ_OWN,
    RESEARCH_RUN_CREATE,
    RESEARCH_RUN_CONTROL_OWN,
    RESEARCH_RUN_INTERACT_OWN,
    RESEARCH_TASK_ACTIVITY_READ_OWN,
    RESEARCH_OBSERVABILITY_READ_OWN,
    RESEARCH_DIAGNOSTICS_PREVIEW,
    RESEARCH_TOOL_SEARCH,
    RESEARCH_TOOL_PROVIDER_NATIVE,
    RESEARCH_TOOL_MCP,
    RESEARCH_TOOL_BROWSER,
    RESEARCH_TOOL_SKILL,
    IAM_USERS_READ,
    IAM_USERS_WRITE,
    IAM_ROLES_READ,
    IAM_ROLES_WRITE,
    IAM_PERMISSIONS_READ,
    IAM_AUDIT_READ,
)

PERMISSION_CODES: frozenset[str] = frozenset(p.code for p in PERMISSIONS)
_PERMISSION_INDEX: dict[str, Permission] = {p.code: p for p in PERMISSIONS}


def permission_for(code: str) -> Permission | None:
    """Return the catalog entry for ``code`` or ``None`` if unknown."""
    return _PERMISSION_INDEX.get(code)


def is_known_permission(code: str) -> bool:
    """Return whether ``code`` is part of the closed permission catalog."""
    return code in _PERMISSION_INDEX


def validate_permission_codes(codes: list[str]) -> list[str]:
    """Return the deduplicated, valid subset of ``codes``.

    Raises:
        ValueError: if any code is not part of the fixed catalog.
    """
    seen: list[str] = []
    seen_set: set[str] = set()
    for code in codes:
        if not is_known_permission(code):
            raise ValueError(f"unknown_permission:{code}")
        if code not in seen_set:
            seen.append(code)
            seen_set.add(code)
    return seen
