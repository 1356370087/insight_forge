"""Audit event logging for identity & RBAC operations."""

from __future__ import annotations

import logging
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from ..models import AuditEvent

logger = logging.getLogger(__name__)


async def record(
    db: AsyncSession,
    *,
    action: str,
    actor_id: str | None = None,
    actor_email: str | None = None,
    target_user_id: str | None = None,
    detail: dict[str, Any] | None = None,
    ip_address: str | None = None,
    user_agent: str | None = None,
) -> None:
    """Append an audit event. Safe to call inside a larger transaction.

    Failures to write audit records are logged but never propagate: an audit
    problem must not block the user-facing operation that triggered it.
    """
    event = AuditEvent(
        action=action,
        actor_id=actor_id,
        actor_email=actor_email,
        target_user_id=target_user_id,
        detail=detail,
        ip_address=ip_address,
        user_agent=user_agent,
    )
    try:
        db.add(event)
        await db.flush()
    except Exception:  # noqa: BLE001 - never let audit break the request
        logger.exception("audit_record_failed action=%s", action)


def audit_meta(principal) -> dict[str, Any]:
    """Return common audit fields derived from a principal."""
    if principal is None:
        return {}
    return {
        "actor_id": principal.user_id,
        "actor_email": principal.email,
    }
