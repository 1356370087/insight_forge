"""Session lifecycle and refresh-token rotation with reuse detection.

Refresh rotation is the security core of the system:

* Every refresh issues a brand-new refresh token and marks the old one used.
* Presenting an *already-used* token is replay. Within a short concurrency
  grace window it returns ``409 refresh_in_progress`` (legitimate double-submit
  from one client). Outside the window it indicates theft — the whole session
  family is revoked and ``refresh_token_reuse`` is audited.
* Disabling the user, changing the password, or logout-all bumps
  ``authz_version`` and revokes sessions, so outstanding access tokens are
  rejected on their next server-side check.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import timedelta

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from ..jwt_service import (
    TokenError,
    decode_refresh_token,
    encode_access_token,
    encode_refresh_token,
)
from ..models import (
    RefreshToken,
    Session,
    SessionRevocationReason,
    User,
    UserStatus,
    utcnow,
)
from ..settings import IAMSettings
from ..tokens import constant_time_eq, digest_token
from . import audit

logger = logging.getLogger(__name__)

REFRESH_PURPOSE = "refresh"


class RefreshError(Exception):
    """Base error for refresh-token failures."""


class InvalidRefreshToken(RefreshError):
    """The refresh token is malformed, expired, or unknown."""


class RefreshInProgress(RefreshError):
    """A concurrent refresh for the same token is in flight (grace window)."""


class RefreshReuseDetected(RefreshError):
    """A used refresh token was presented again outside the grace window."""


@dataclass(frozen=True)
class TokenPairRecord:
    """A freshly issued access + refresh token pair with its session id."""

    session_id: str
    access_token: str
    refresh_token: str
    access_expires_in: int
    refresh_expires_in: int


async def create_session(
    db: AsyncSession,
    user: User,
    *,
    settings: IAMSettings,
    user_agent: str | None = None,
    ip_address: str | None = None,
) -> TokenPairRecord:
    """Create a session + first refresh token and return a token pair."""
    now = utcnow()
    session = Session(
        user_id=str(user.id),
        user_agent=user_agent,
        ip_address=ip_address,
        absolute_expires_at=now + timedelta(seconds=settings.session_absolute_ttl),
        last_activity_at=now,
    )
    db.add(session)
    await db.flush()
    return await _issue_pair(db, user=user, session=session, settings=settings, previous_jti=None)


async def rotate(
    db: AsyncSession,
    raw_refresh_token: str,
    *,
    settings: IAMSettings,
    actor_id: str | None = None,
    ip_address: str | None = None,
    user_agent: str | None = None,
) -> TokenPairRecord:
    """Rotate a refresh token, returning a fresh token pair.

    Raises :class:`InvalidRefreshToken`, :class:`RefreshInProgress`, or
    :class:`RefreshReuseDetected`. Reuse detection revokes the session family
    and records an audit event before raising.
    """
    try:
        decoded = decode_refresh_token(raw_refresh_token, settings=settings)
    except TokenError as exc:
        raise InvalidRefreshToken(str(exc)) from exc

    claims = decoded.claims
    jti = str(claims["jti"])
    session_id = str(claims["sid"])
    presented_digest = digest_token(raw_refresh_token, purpose=REFRESH_PURPOSE)

    session_result = await db.execute(
        select(Session).where(Session.id == session_id).with_for_update()
    )
    session = session_result.scalar_one_or_none()
    if session is None or session.is_revoked:
        raise InvalidRefreshToken("session_revoked")

    record_result = await db.execute(
        select(RefreshToken).where(RefreshToken.jti == jti).with_for_update()
    )
    record = record_result.scalar_one_or_none()
    if record is None or record.session_id != session_id:
        # A validly-signed refresh whose jti we have no record of, or that
        # belongs to another session — treat as reuse and revoke the family.
        await _revoke_family(db, session, reason=SessionRevocationReason.REPLAY, actor_id=actor_id)
        await audit.record(
            db,
            action="refresh_token_reuse",
            actor_id=actor_id,
            detail={"session_id": session_id, "jti": jti, "stage": "unknown_jti"},
            ip_address=ip_address,
            user_agent=user_agent,
        )
        raise RefreshReuseDetected("unknown_jti")

    if not constant_time_eq(record.digest, presented_digest):
        await _revoke_family(db, session, reason=SessionRevocationReason.REPLAY, actor_id=actor_id)
        await audit.record(
            db,
            action="refresh_token_reuse",
            actor_id=actor_id,
            detail={"session_id": session_id, "jti": jti, "stage": "digest_mismatch"},
            ip_address=ip_address,
            user_agent=user_agent,
        )
        raise RefreshReuseDetected("digest_mismatch")

    now = utcnow()
    if record.expires_at <= now or session.absolute_expires_at <= now:
        await _revoke_family(db, session, reason=SessionRevocationReason.EXPIRED, actor_id=actor_id)
        raise InvalidRefreshToken("expired")

    if record.used_at is not None:
        # Replay of a used token.
        age = (now - record.used_at).total_seconds()
        if age <= settings.refresh_reuse_grace:
            # Legitimate double-submit inside the concurrency window.
            raise RefreshInProgress()
        # Outside the grace window -> theft. Revoke the whole family.
        await _revoke_family(db, session, reason=SessionRevocationReason.REPLAY, actor_id=actor_id)
        await audit.record(
            db,
            action="refresh_token_reuse",
            actor_id=actor_id,
            detail={"session_id": session_id, "jti": jti, "stage": "post_grace_replay"},
            ip_address=ip_address,
            user_agent=user_agent,
        )
        raise RefreshReuseDetected("replay_after_grace")

    # Valid rotation: retire the presented token and issue the next one.
    record.used_at = now
    session.last_activity_at = now
    await db.flush()

    user = await db.get(User, str(claims["sub"]))
    if user is None or str(session.user_id) != str(claims["sub"]) or user.status == UserStatus.DISABLED:
        await _revoke_family(db, session, reason=SessionRevocationReason.DISABLED, actor_id=actor_id)
        raise InvalidRefreshToken("user_unavailable")

    pair = await _issue_pair(
        db, user=user, session=session, settings=settings, previous_jti=jti,
    )
    await audit.record(
        db,
        action="refresh",
        actor_id=actor_id or str(user.id),
        detail={"session_id": session_id, "previous_jti": jti},
        ip_address=ip_address,
        user_agent=user_agent,
    )
    return pair


async def revoke_session(
    db: AsyncSession,
    session_id: str,
    *,
    reason: str = SessionRevocationReason.LOGOUT,
    actor_id: str | None = None,
) -> bool:
    """Revoke a single session by id; return whether it existed & was active."""
    session = await db.get(Session, str(session_id))
    if session is None or session.is_revoked:
        return False
    session.revoked_at = utcnow()
    session.revocation_reason = reason
    await db.flush()
    return True


async def revoke_all_user_sessions(
    db: AsyncSession,
    user_id: str,
    *,
    reason: str = SessionRevocationReason.LOGOUT_ALL,
    except_session_id: str | None = None,
    actor_id: str | None = None,
) -> int:
    """Revoke every active session for ``user_id``; return the count revoked."""
    now = utcnow()
    result = await db.execute(
        select(Session).where(
            Session.user_id == str(user_id),
            Session.revoked_at.is_(None),
        )
    )
    sessions = result.scalars().all()
    count = 0
    for session in sessions:
        if except_session_id is not None and str(session.id) == str(except_session_id):
            continue
        session.revoked_at = now
        session.revocation_reason = reason
        count += 1
    await db.flush()
    return count


async def bump_authz_version(
    db: AsyncSession,
    user_id: str,
    *,
    revoke_sessions: bool = True,
    reason: str = SessionRevocationReason.REVOKED,
) -> None:
    """Increment ``authz_version`` and (optionally) revoke all sessions.

    Called on password change, disable and role change so outstanding access
    tokens are rejected on their next server-side check.
    """
    user = await db.get(User, str(user_id))
    if user is None:
        return
    user.authz_version = int(user.authz_version) + 1
    await db.flush()
    if revoke_sessions:
        await revoke_all_user_sessions(db, user_id, reason=reason, actor_id=user_id)


async def _issue_pair(
    db: AsyncSession,
    *,
    user: User,
    session: Session,
    settings: IAMSettings,
    previous_jti: str | None,
) -> TokenPairRecord:
    """Mint a new access + refresh token pair and persist the refresh record."""
    access_token, _ = encode_access_token(
        subject=str(user.id),
        session_id=str(session.id),
        authz_version=int(user.authz_version),
        settings=settings,
    )
    # Pre-generate the jti so we can persist the digest before signing.
    from uuid import uuid4

    jti = uuid4().hex
    now = utcnow()
    remaining = max(0, int((session.absolute_expires_at - now).total_seconds()))
    refresh_ttl = min(settings.refresh_idle_ttl, remaining)
    if refresh_ttl <= 0:
        raise InvalidRefreshToken("session_expired")
    refresh_token = _encode_refresh_with_jti(
        subject=str(user.id), session_id=str(session.id), jti=jti, settings=settings,
        ttl=refresh_ttl,
    )
    db.add(
        RefreshToken(
            jti=jti,
            session_id=str(session.id),
            digest=digest_token(refresh_token, purpose=REFRESH_PURPOSE),
            previous_jti=previous_jti,
            expires_at=now + timedelta(seconds=refresh_ttl),
        )
    )
    await db.flush()
    return TokenPairRecord(
        session_id=str(session.id),
        access_token=access_token,
        refresh_token=refresh_token,
        access_expires_in=settings.access_token_ttl,
        refresh_expires_in=refresh_ttl,
    )


def _encode_refresh_with_jti(
    *, subject: str, session_id: str, jti: str, settings: IAMSettings, ttl: int
) -> str:
    """Encode a refresh token with a caller-supplied jti; return the token."""
    from dataclasses import replace

    token, _ = encode_refresh_token(
        subject=subject,
        session_id=session_id,
        jti=jti,
        settings=replace(settings, refresh_idle_ttl=ttl),
    )
    return token


async def _revoke_family(
    db: AsyncSession, session: Session, *, reason: str, actor_id: str | None,
) -> None:
    """Revoke a session (and thus its whole refresh family)."""
    session.revoked_at = utcnow()
    session.revocation_reason = reason
    await db.flush()


async def list_user_sessions(db: AsyncSession, user_id: str) -> list[Session]:
    """Return all sessions for ``user_id`` newest first."""
    result = await db.execute(
        select(Session)
        .where(Session.user_id == str(user_id))
        .order_by(Session.created_at.desc())
    )
    return list(result.scalars().all())


async def touch_session_activity(db: AsyncSession, session_id: str) -> None:
    """Update ``last_activity_at`` for a session (used by SSE heartbeats)."""
    await db.execute(
        update(Session).where(Session.id == str(session_id)).values(last_activity_at=utcnow())
    )
