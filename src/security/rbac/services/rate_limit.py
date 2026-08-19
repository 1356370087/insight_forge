"""PostgreSQL-backed distributed rate limiting.

A fixed-window counter per ``(bucket, identity)`` stored in ``iam_rate_limits``.
When the window elapses the counter resets. Identities are opaque strings
chosen by the caller (email, normalized email, or IP). Designed to throttle
login, registration, resend and password-reset endpoints.

Concurrent consumers serialize on a row lock (``SELECT ... FOR UPDATE``, the
same idiom as refresh-token rotation) so parallel requests can never read the
same count and both pass; the insert path tolerates a race via
``ON CONFLICT DO NOTHING`` and re-locks the winning row.
"""

from __future__ import annotations

from datetime import timedelta

from sqlalchemy import delete, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from ..models import RateLimit, utcnow


class RateLimited(Exception):
    """Raised when an identity has exceeded a bucket's limit."""


async def _locked_row(db: AsyncSession, bucket: str, identity: str) -> RateLimit | None:
    """Return the ``(bucket, identity)`` row locked for the transaction."""
    result = await db.execute(
        select(RateLimit)
        .where(RateLimit.bucket == bucket, RateLimit.identity == identity)
        .with_for_update()
    )
    return result.scalar_one_or_none()


async def check_and_consume(
    db: AsyncSession,
    *,
    bucket: str,
    identity: str,
    limit: int,
    window_seconds: int,
) -> int:
    """Consume one slot from ``(bucket, identity)``; raise if over ``limit``.

    Returns the count *after* consuming. The existing-row path locks the row
    before reading, and the first-insert path relies on primary-key uniqueness
    with a follow-up locked read, so concurrent requests are always counted.
    """
    if limit < 1:
        raise RateLimited(bucket)
    now = utcnow()
    window_start = now - timedelta(seconds=window_seconds)

    row = await _locked_row(db, bucket, identity)
    if row is None:
        # No row yet: insert with count=1; a concurrent winner is tolerated
        # and accounted for by re-locking the surviving row below.
        stmt = (
            pg_insert(RateLimit)
            .values(bucket=bucket, identity=identity, count=1, window_started_at=now)
            .on_conflict_do_nothing(index_elements=["bucket", "identity"])
            .returning(RateLimit.count)
        )
        inserted = (await db.execute(stmt)).scalar_one_or_none()
        if inserted is not None:
            return int(inserted)
        row = await _locked_row(db, bucket, identity)
        if row is None:  # pragma: no cover - only if the row vanished mid-flight
            return 1

    if row.window_started_at <= window_start:
        # Window elapsed — reset (the current request counts as the first).
        row.count = 1
        row.window_started_at = now
    elif row.count < limit:
        row.count = int(row.count) + 1
    else:
        raise RateLimited(bucket)
    await db.flush()
    return int(row.count)


async def current_count(
    db: AsyncSession,
    *,
    bucket: str,
    identity: str,
    window_seconds: int | None = None,
) -> int:
    """Return the in-window count for ``(bucket, identity)``.

    With ``window_seconds`` the count is reported as zero once the window has
    elapsed (the stored row is left as-is; the next consume resets it).
    """
    row = await db.get(RateLimit, (bucket, identity))
    if row is None:
        return 0
    if window_seconds is not None:
        window_start = utcnow() - timedelta(seconds=window_seconds)
        if row.window_started_at <= window_start:
            return 0
    return int(row.count)


async def reset(db: AsyncSession, *, bucket: str, identity: str) -> None:
    """Clear a bucket row (used when a successful login clears its lockout)."""
    await db.execute(
        delete(RateLimit).where(RateLimit.bucket == bucket, RateLimit.identity == identity)
    )
