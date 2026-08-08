"""Test principals for protected research API routes."""

from security.rbac.models import UserStatus
from security.rbac.permissions import (
    RESEARCH_DIAGNOSTICS_PREVIEW,
    RESEARCH_OBSERVABILITY_READ_OWN,
    RESEARCH_RUN_CONTROL_OWN,
    RESEARCH_RUN_CREATE,
    RESEARCH_RUN_INTERACT_OWN,
    RESEARCH_RUN_READ_OWN,
    RESEARCH_TASK_ACTIVITY_READ_OWN,
)
from security.rbac.principal import Principal


def research_principal(user_id: str = "user-1") -> Principal:
    """Return an active developer-like principal for API contract tests."""
    return Principal(
        user_id=user_id,
        email=f"{user_id}@example.com",
        status=UserStatus.ACTIVE,
        session_id=None,
        roles=frozenset({"developer"}),
        permissions=frozenset({
            RESEARCH_RUN_READ_OWN.code,
            RESEARCH_RUN_CREATE.code,
            RESEARCH_RUN_CONTROL_OWN.code,
            RESEARCH_RUN_INTERACT_OWN.code,
            RESEARCH_TASK_ACTIVITY_READ_OWN.code,
            RESEARCH_OBSERVABILITY_READ_OWN.code,
            RESEARCH_DIAGNOSTICS_PREVIEW.code,
        }),
        authz_version=1,
    )
