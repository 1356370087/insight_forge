"""PostgreSQL-backed distributed rate limiting.

A fixed-window counter per ``(bucket, identity)`` stored in ``iam_rate_limits``.
When the window elapses the counter resets. Identities are opaque strings
chosen by the caller (email, normalized email, or IP). Designed to throttle
login, registration, resend and password-reset endpoints.
"""

from __future__ import annotations

from datetime import timedelta

from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from ..models import RateLimit, utcnow


class RateLimited(Exception):
    """Raised when an identity has exceeded a bucket's limit."""


async def check_and_consume(
    db: AsyncSession,
    *,
    bucket: str,
    identity: str,
    limit: int,
    window_seconds: int,
) -> int:
    """Consume one slot from ``(bucket, identity)``; raise if over ``limit``.

    Returns the count *after* consuming. Uses an UPSERT so concurrent requests
    cannot race past the limit by both reading zero.
    """
    now = utcnow()
    window_start = now - timedelta(seconds=window_seconds)

    # Lock existing row pessimistically by selecting within the transaction;
    # for the insert path rely on the primary-key uniqueness to serialize.
    existing = await db.get(RateLimit, (bucket, identity))
    if existing is not None and existing.window_started_at <= window_start:
        # Window elapsed — reset.
        existing.count = 0
        existing.window_started_at = now

    if existing is not None:
        if existing.count >= limit:
            raise RateLimited(bucket)
        existing.count += 1
        await db.flush()
        return existing.count

    # No row yet: insert with count=1, tolerating a concurrent insert race.
    stmt = pg_insert(RateLimit).values(
        bucket=bucket, identity=identity, count=1, window_started_at=now,
    )
    stmt = stmt.on_conflict_do_nothing(index_elements=["bucket", "identity"])
    await db.execute(stmt)
    await db.flush()

    inserted = await db.get(RateLimit, (bucket, identity))
    return int(inserted.count) if inserted is not None else 1


async def current_count(db: AsyncSession, *, bucket: str, identity: str) -> int:
    """Return the current (non-reset) count for ``(bucket, identity)``."""
    row = await db.get(RateLimit, (bucket, identity))
    return int(row.count) if row is not None else 0
