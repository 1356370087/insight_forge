"""One-time email-verification & password-reset token service.

Tokens are 256-bit opaque strings; only an HMAC digest is stored. A new token
for a given ``(user, purpose)`` invalidates prior tokens of the same purpose
(single-use family). Consumption marks the token used and is atomic.
"""

from __future__ import annotations

from datetime import timedelta

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from ..models import EmailToken, EmailTokenPurpose, User, UserStatus, utcnow
from ..settings import IAMSettings
from ..tokens import digest_token, generate_token_with_digest


class EmailTokenError(Exception):
    """Base error for email-token failures."""


class TokenNotFound(EmailTokenError):
    """The supplied token is unknown, expired, or already used."""


def _ttl_for(purpose: str, settings: IAMSettings) -> int:
    """Return the TTL (seconds) for a token purpose."""
    if purpose == EmailTokenPurpose.EMAIL_VERIFICATION:
        return settings.email_verify_ttl
    if purpose == EmailTokenPurpose.PASSWORD_RESET:
        return settings.password_reset_ttl
    raise EmailTokenError(f"unknown_purpose:{purpose}")


async def issue(
    db: AsyncSession,
    *,
    user_id: str,
    purpose: str,
    settings: IAMSettings,
) -> str:
    """Issue a one-time token for ``(user_id, purpose)``; return the raw token.

    Invalidates any prior unused tokens of the same purpose for the user.
    The raw token is returned exactly once; only its digest is persisted.
    """
    ttl = _ttl_for(purpose, settings)
    now = utcnow()
    # Invalidate prior unused tokens of this purpose for this user.
    await db.execute(
        update(EmailToken)
        .where(
            EmailToken.user_id == str(user_id),
            EmailToken.purpose == purpose,
            EmailToken.used_at.is_(None),
        )
        .values(used_at=now)
    )
    raw_token, digest = generate_token_with_digest(purpose)
    db.add(
        EmailToken(
            user_id=str(user_id),
            purpose=purpose,
            digest=digest,
            expires_at=now + timedelta(seconds=ttl),
        )
    )
    await db.flush()
    return raw_token


async def consume(
    db: AsyncSession,
    *,
    raw_token: str,
    purpose: str,
) -> EmailToken:
    """Validate, consume and return the :class:`EmailToken` for ``raw_token``.

    Raises:
        TokenNotFound: if the token is unknown, expired, or already used.
    """
    digest = digest_token(raw_token, purpose=purpose)
    now = utcnow()
    # Lock the row so two concurrent presentations of the same one-time token
    # serialize: the loser sees used_at set and fails (same idiom as refresh
    # rotation).
    result = await db.execute(
        select(EmailToken)
        .where(
            EmailToken.digest == digest,
            EmailToken.purpose == purpose,
        )
        .with_for_update()
    )
    record = result.scalar_one_or_none()
    if record is None:
        raise TokenNotFound()
    if record.used_at is not None:
        raise TokenNotFound()
    if record.expires_at <= now:
        raise TokenNotFound()
    record.used_at = now
    await db.flush()
    return record


async def verify_email_with_token(
    db: AsyncSession,
    *,
    raw_token: str,
    settings: IAMSettings,
) -> User:
    """Consume an email-verification token and mark the user verified.

    Returns the verified user. Raises :class:`TokenNotFound` on any failure.
    """
    record = await consume(db, raw_token=raw_token, purpose=EmailTokenPurpose.EMAIL_VERIFICATION)
    user = await db.get(User, record.user_id)
    if user is None:
        raise TokenNotFound()
    user.email_verified_at = utcnow()
    if user.status == UserStatus.PENDING_EMAIL:
        user.status = UserStatus.PENDING_APPROVAL
    await db.flush()
    return user
