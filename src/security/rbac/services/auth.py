"""High-level authentication service: registration, login, refresh, password flows.

Orchestrates :mod:`email_tokens`, :mod:`sessions`, :mod:`rate_limit` and
:mod:`audit` behind a clean API consumed by the ``/auth/*`` router. All public
functions operate within the caller's transaction (the router opens one
session per request and commits on success).
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from datetime import timedelta

from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from ..emails import InvalidEmail, validate_and_normalize
from ..models import User, UserStatus, utcnow
from ..passwords import (
    PasswordPolicyError,
    dummy_hash,
    hash_password,
    needs_rehash,
    validate_password,
    verify_password,
)
from ..repositories import get_user_by_email_normalized, get_user_by_id
from ..settings import IAMSettings
from . import audit, email_tokens, rate_limit, sessions
from .rate_limit import RateLimited
from .sessions import (
    InvalidRefreshToken,
    RefreshInProgress,
    RefreshReuseDetected,
    TokenPairRecord,
)

logger = logging.getLogger(__name__)

LOGIN_MAX_FAILURES = 5
LOGIN_LOCK_MINUTES = 15
# Lockout window == lock duration: five failures within the window lock that
# (account, source-IP) pair for the window's remainder.
_LOGIN_LOCK_WINDOW = LOGIN_LOCK_MINUTES * 60
_LOGIN_LOCK_BUCKET = "login-lock"


class AuthError(Exception):
    """Base error for authentication-flow failures (mapped to HTTP in the router)."""

    def __init__(self, detail: str, status: int = 400) -> None:
        """Store an HTTP-friendly detail and status code."""
        super().__init__(detail)
        self.detail = detail
        self.status = status


@dataclass(frozen=True)
class RegisterOutcome:
    """Result of a registration attempt.

    ``issued_verification`` is False when the email was already in use — the
    router still returns ``202`` either way to prevent account enumeration.
    """

    issued_verification: bool
    email: str


async def register(
    db: AsyncSession,
    *,
    email: str,
    password: str,
    display_name: str | None,
    settings: IAMSettings,
    identity_for_rate_limit: str,
) -> RegisterOutcome:
    """Register a new pending-email user and send a verification link.

    Always returns successfully: a duplicate email returns
    ``issued_verification=False`` so the caller can still answer ``202`` and
    avoid revealing which emails are registered.
    """
    try:
        await rate_limit.check_and_consume(
            db,
            bucket="register",
            identity=identity_for_rate_limit,
            limit=settings.register_rate_limit,
            window_seconds=settings.rate_limit_window,
        )
    except RateLimited as exc:
        raise AuthError("rate_limited", status=429) from exc

    try:
        normalized = validate_and_normalize(email)
    except InvalidEmail as exc:
        raise AuthError(str(exc), status=422) from exc
    try:
        validated = validate_password(password)
    except PasswordPolicyError as exc:
        raise AuthError(str(exc), status=422) from exc

    existing = await get_user_by_email_normalized(db, normalized)
    if existing is not None:
        # Duplicate registration: respond identically to success to avoid
        # enumeration. Do not issue or send anything.
        return RegisterOutcome(issued_verification=False, email=normalized)

    user = User(
        email=normalized,
        email_normalized=normalized,
        password_hash=hash_password(validated),
        display_name=(display_name or "").strip() or None,
        status=UserStatus.PENDING_EMAIL,
    )
    try:
        # Savepoint so a concurrent registration of the same email (unique
        # constraint) only rolls back the insert — the consumed rate-limit
        # slot stays counted, and the caller still sees the 202-shaped outcome.
        async with db.begin_nested():
            db.add(user)
            await db.flush()
    except IntegrityError:
        return RegisterOutcome(issued_verification=False, email=normalized)
    raw_token = await email_tokens.issue(
        db, user_id=str(user.id), purpose=email_tokens.EmailTokenPurpose.EMAIL_VERIFICATION, settings=settings,
    )
    from ..email import send_verification_email

    await send_verification_email(to=user.email, token=raw_token, settings=settings)
    await audit.record(db, action="user.registered", actor_id=str(user.id), detail={"email": normalized})
    return RegisterOutcome(issued_verification=True, email=normalized)


async def verify_email(db: AsyncSession, *, token: str, settings: IAMSettings) -> User:
    """Verify an email via a one-time token; returns the verified user."""
    try:
        user = await email_tokens.verify_email_with_token(db, raw_token=token, settings=settings)
    except email_tokens.TokenNotFound as exc:
        raise AuthError("invalid_or_expired_token", status=400) from exc
    await audit.record(db, action="user.email_verified", actor_id=str(user.id))
    return user


async def resend_verification(
    db: AsyncSession, *, email: str, settings: IAMSettings, identity_for_rate_limit: str,
) -> None:
    """Re-send a verification email if the user is still ``pending_email``.

    Always returns successfully (``202``) to avoid enumeration.
    """
    try:
        await rate_limit.check_and_consume(
            db, bucket="resend", identity=identity_for_rate_limit,
            limit=settings.resend_rate_limit, window_seconds=settings.rate_limit_window,
        )
    except RateLimited as exc:
        raise AuthError("rate_limited", status=429) from exc
    try:
        normalized = validate_and_normalize(email)
    except InvalidEmail:
        return
    user = await get_user_by_email_normalized(db, normalized)
    if user is None or user.status != UserStatus.PENDING_EMAIL:
        return
    raw_token = await email_tokens.issue(
        db, user_id=str(user.id), purpose=email_tokens.EmailTokenPurpose.EMAIL_VERIFICATION, settings=settings,
    )
    from ..email import send_verification_email

    await send_verification_email(to=user.email, token=raw_token, settings=settings)


def _lock_identity(subject: str, ip_address: str | None) -> str:
    """Return the hashed (subject, ip) lockout key (fits the identity column)."""
    raw = f"{subject}|{ip_address or 'unknown'}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


async def _login_lock_count(db: AsyncSession, subject: str, ip_address: str | None) -> int:
    """Return in-window failures recorded against (subject, ip)."""
    return await rate_limit.current_count(
        db,
        bucket=_LOGIN_LOCK_BUCKET,
        identity=_lock_identity(subject, ip_address),
        window_seconds=_LOGIN_LOCK_WINDOW,
    )


async def _record_login_failure(db: AsyncSession, subject: str, ip_address: str | None) -> None:
    """Count a failed attempt against (subject, ip); a locked pair stays put."""
    try:
        await rate_limit.check_and_consume(
            db,
            bucket=_LOGIN_LOCK_BUCKET,
            identity=_lock_identity(subject, ip_address),
            limit=LOGIN_MAX_FAILURES,
            window_seconds=_LOGIN_LOCK_WINDOW,
        )
    except RateLimited:
        pass


async def login(
    db: AsyncSession,
    *,
    email: str,
    password: str,
    settings: IAMSettings,
    user_agent: str | None = None,
    ip_address: str | None = None,
) -> TokenPairRecord:
    """Authenticate and start a session; returns a fresh token pair.

    Account-existence protection: every wrong-password path answers a uniform
    ``invalid_credentials``, and lifecycle-status errors are only reported
    *after* the password verifies. Lockout is scoped to the failing
    ``(account, source IP)`` pair, so an attacker cannot lock the account for
    its rightful owner (who signs in from elsewhere).
    """
    identity_for_rate_limit = (ip_address or (email or "").lower() or "anon")
    try:
        await rate_limit.check_and_consume(
            db, bucket="login", identity=identity_for_rate_limit,
            limit=settings.login_rate_limit, window_seconds=settings.rate_limit_window,
        )
    except RateLimited as exc:
        raise AuthError("rate_limited", status=429) from exc

    normalized: str | None = None
    user = None
    try:
        normalized = validate_and_normalize(email)
        user = await get_user_by_email_normalized(db, normalized)
    except InvalidEmail:
        user = None

    # Lockout subject: the user id when known, else the attempted email — both
    # record and lock identically so the responses stay indistinguishable.
    lock_subject = (
        str(user.id)
        if user is not None
        else f"email:{normalized or (email or '').strip().lower() or 'unknown'}"
    )
    if await _login_lock_count(db, lock_subject, ip_address) >= LOGIN_MAX_FAILURES:
        # Uniform timing: still pay the Argon2 cost before rejecting.
        verify_password(password or "", dummy_hash())
        raise AuthError("account_locked", status=429)

    if user is None:
        # Uniform timing for unknown emails.
        verify_password(password or "", dummy_hash())
        await _record_login_failure(db, lock_subject, ip_address)
        raise AuthError("invalid_credentials", status=401)

    stored_hash = user.password_hash or dummy_hash()
    if not verify_password(password or "", stored_hash):
        await _record_login_failure(db, lock_subject, ip_address)
        user.failed_login_count = int(user.failed_login_count) + 1
        if user.failed_login_count >= LOGIN_MAX_FAILURES:
            user.locked_until = utcnow() + timedelta(minutes=LOGIN_LOCK_MINUTES)
            user.failed_login_count = 0
            await audit.record(db, action="login.locked", actor_id=str(user.id), ip_address=ip_address)
        await db.flush()
        raise AuthError("invalid_credentials", status=401)

    # Credentials verified — status errors from here on cannot leak account
    # existence to an attacker without the password.
    if user.status == UserStatus.PENDING_EMAIL:
        raise AuthError("email_verification_required", status=403)
    if user.status == UserStatus.DISABLED:
        raise AuthError("account_disabled", status=403)
    if user.status == UserStatus.PASSWORD_RESET_REQUIRED:
        raise AuthError("password_reset_required", status=403)

    # Successful authentication: clear failure counters, upgrade hash if needed.
    user.failed_login_count = 0
    user.locked_until = None
    if needs_rehash(stored_hash):
        user.password_hash = hash_password(password)
    await db.flush()
    await rate_limit.reset(
        db, bucket=_LOGIN_LOCK_BUCKET, identity=_lock_identity(lock_subject, ip_address),
    )

    pair = await sessions.create_session(
        db, user, settings=settings, user_agent=user_agent, ip_address=ip_address,
    )
    await audit.record(
        db, action="login", actor_id=str(user.id),
        detail={"session_id": pair.session_id}, ip_address=ip_address, user_agent=user_agent,
    )
    return pair


async def refresh(
    db: AsyncSession,
    *,
    refresh_token: str,
    settings: IAMSettings,
    ip_address: str | None = None,
    user_agent: str | None = None,
) -> TokenPairRecord:
    """Rotate a refresh token; raises :class:`AuthError` on any failure."""
    try:
        return await sessions.rotate(
            db, refresh_token, settings=settings, ip_address=ip_address, user_agent=user_agent,
        )
    except RefreshInProgress as exc:
        raise AuthError("refresh_in_progress", status=409) from exc
    except RefreshReuseDetected as exc:
        raise AuthError("refresh_token_reuse", status=401) from exc
    except InvalidRefreshToken as exc:
        raise AuthError(f"invalid_refresh_token:{exc}", status=401) from exc


async def logout(db: AsyncSession, *, session_id: str, user_id: str | None = None) -> bool:
    """Revoke a single session; returns whether a session was revoked."""
    revoked = await sessions.revoke_session(
        db, session_id, reason=sessions.SessionRevocationReason.LOGOUT, actor_id=user_id,
    )
    if revoked:
        await audit.record(db, action="logout", actor_id=user_id, detail={"session_id": session_id})
    return revoked


async def logout_all(db: AsyncSession, *, user_id: str, except_session_id: str | None = None) -> int:
    """Revoke all sessions for a user (optionally except one); returns count."""
    count = await sessions.revoke_all_user_sessions(
        db, user_id, reason=sessions.SessionRevocationReason.LOGOUT_ALL,
        except_session_id=except_session_id, actor_id=user_id,
    )
    await audit.record(
        db, action="logout_all", actor_id=user_id,
        detail={"count": count, "kept": except_session_id},
    )
    return count


async def change_password(
    db: AsyncSession,
    *,
    user_id: str,
    current_password: str,
    new_password: str,
    settings: IAMSettings,
) -> None:
    """Change the password after verifying the current one; revokes sessions."""
    user = await get_user_by_id(db, user_id)
    if user is None:
        raise AuthError("user_not_found", status=404)
    if not user.password_hash or not verify_password(current_password or "", user.password_hash):
        raise AuthError("invalid_credentials", status=401)
    try:
        validated = validate_password(new_password)
    except PasswordPolicyError as exc:
        raise AuthError(str(exc), status=422) from exc
    if verify_password(validated, user.password_hash):
        raise AuthError("password_reused", status=422)
    user.password_hash = hash_password(validated)
    await db.flush()
    await sessions.bump_authz_version(db, str(user.id), reason=sessions.SessionRevocationReason.PASSWORD_CHANGE)
    await audit.record(db, action="user.password_changed", actor_id=str(user.id))


async def forgot_password(
    db: AsyncSession, *, email: str, settings: IAMSettings, identity_for_rate_limit: str,
) -> None:
    """Issue a password-reset link if the user exists. Always returns (``202``)."""
    try:
        await rate_limit.check_and_consume(
            db, bucket="reset", identity=identity_for_rate_limit,
            limit=settings.reset_rate_limit, window_seconds=settings.rate_limit_window,
        )
    except RateLimited as exc:
        raise AuthError("rate_limited", status=429) from exc
    try:
        normalized = validate_and_normalize(email)
    except InvalidEmail:
        return
    user = await get_user_by_email_normalized(db, normalized)
    if user is None or user.status == UserStatus.PENDING_EMAIL:
        return
    raw_token = await email_tokens.issue(
        db, user_id=str(user.id), purpose=email_tokens.EmailTokenPurpose.PASSWORD_RESET, settings=settings,
    )
    from ..email import send_password_reset_email

    await send_password_reset_email(to=user.email, token=raw_token, settings=settings)


async def reset_password(
    db: AsyncSession, *, token: str, new_password: str, settings: IAMSettings,
) -> User:
    """Consume a reset token and set a new password; activates reset-required users."""
    try:
        record = await email_tokens.consume(db, raw_token=token, purpose=email_tokens.EmailTokenPurpose.PASSWORD_RESET)
    except email_tokens.TokenNotFound as exc:
        raise AuthError("invalid_or_expired_token", status=400) from exc
    try:
        validated = validate_password(new_password)
    except PasswordPolicyError as exc:
        raise AuthError(str(exc), status=422) from exc
    user = await db.get(User, record.user_id)
    if user is None:
        raise AuthError("invalid_or_expired_token", status=400)
    user.password_hash = hash_password(validated)
    if user.status == UserStatus.PASSWORD_RESET_REQUIRED:
        user.status = UserStatus.ACTIVE
    await db.flush()
    await sessions.bump_authz_version(db, str(user.id), reason=sessions.SessionRevocationReason.PASSWORD_CHANGE)
    await audit.record(db, action="user.password_reset", actor_id=str(user.id))
    return user
