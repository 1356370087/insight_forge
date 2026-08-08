"""The unified identity DTO carried through every protected request.

A :class:`Principal` is the only thing application code should need: the user
id, lifecycle status, active session id, role codes, effective permission set
and the ``authz_version`` snapshot used to detect revoked sessions. Roles are
*not* an authorization source on their own — callers must check permissions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .models import UserStatus


@dataclass(frozen=True)
class Principal:
    """The authenticated identity for the current request/task."""

    user_id: str
    email: str
    status: str
    session_id: str | None
    roles: frozenset[str]
    permissions: frozenset[str]
    authz_version: int

    @property
    def is_active(self) -> bool:
        """Return whether the principal may access research APIs."""
        return self.status == UserStatus.ACTIVE

    @property
    def is_pending_approval(self) -> bool:
        """Return whether the principal may access approval/auth APIs only."""
        return self.status == UserStatus.PENDING_APPROVAL

    @property
    def identity(self) -> str:
        """Return the owner identity string used by run-ownership checks."""
        return self.user_id

    def has(self, code: str) -> bool:
        """Return whether the principal holds a single permission."""
        return code in self.permissions

    def has_all(self, codes: list[str]) -> bool:
        """Return whether the principal holds every permission in ``codes``."""
        return all(code in self.permissions for code in codes)

    def has_any(self, codes: list[str]) -> bool:
        """Return whether the principal holds at least one permission in ``codes``."""
        return any(code in self.permissions for code in codes)

    def to_runtime_dict(self) -> dict[str, Any]:
        """Return a backward-compatible dict for legacy consumers.

        Existing code (e.g. tool governance) reads ``identity`` and
        ``permissions``; this shape lets a Principal flow through unchanged.
        """
        return {
            "identity": self.user_id,
            "id": self.user_id,
            "email": self.email,
            "status": self.status,
            "session_id": self.session_id,
            "roles": sorted(self.roles),
            "permissions": sorted(self.permissions),
            "effective_permissions": sorted(self.permissions),
            "authz_version": self.authz_version,
            "is_authenticated": True,
        }


def synthetic_dev_principal() -> Principal:
    """Return the synthetic researcher/developer principal for local-dev bypass.

    Per the SPEC, ``LOCAL_DEV_AUTH_BYPASS`` is only honored in development and
    grants researcher + developer capabilities — never IAM-admin permissions.
    """
    from .roles import DEVELOPER_CODE, RESEARCHER_CODE

    return Principal(
        user_id="local-dev-user",
        email="local-dev@example.invalid",
        status=UserStatus.ACTIVE,
        session_id=None,
        roles=frozenset({RESEARCHER_CODE, DEVELOPER_CODE}),
        permissions=dev_permissions(),
        authz_version=1,
    )


def dev_permissions() -> frozenset[str]:
    """Return the permission union of the researcher + developer system roles."""
    from .roles import DEVELOPER, RESEARCHER

    return frozenset(set(RESEARCHER.permissions) | set(DEVELOPER.permissions))
