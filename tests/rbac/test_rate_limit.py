"""DB integration: concurrent-correctness of the distributed rate limiter."""

from __future__ import annotations

import asyncio
from datetime import timedelta

import pytest
from sqlalchemy import update

from security.rbac import database as iam_db
from security.rbac.models import RateLimit, utcnow
from security.rbac.services import rate_limit
from security.rbac.services.rate_limit import RateLimited

pytestmark = [pytest.mark.asyncio, pytest.mark.db]


async def _consume_once(bucket: str, identity: str, limit: int) -> bool:
    """Consume one slot in its own transaction; return whether it was allowed."""
    async with iam_db.session_scope() as db:
        try:
            await rate_limit.check_and_consume(
                db, bucket=bucket, identity=identity, limit=limit, window_seconds=600,
            )
        except RateLimited:
            return False
        await db.commit()
        return True


class TestConcurrentConsumption:
    """Parallel requests must never exceed the configured limit."""

    async def test_concurrent_burst_admits_exactly_limit(self):
        """20 parallel consumers with limit 5 admit exactly 5."""
        results = await asyncio.gather(
            *(_consume_once("burst", "ip-a", 5) for _ in range(20))
        )
        assert sum(1 for allowed in results if allowed) == 5

    async def test_current_count_respects_window(self):
        """An expired window reports zero without touching the row."""
        async with iam_db.session_scope() as db:
            await rate_limit.check_and_consume(
                db, bucket="stale", identity="ip-b", limit=3, window_seconds=600,
            )
            await db.commit()
        async with iam_db.session_scope() as db:
            await db.execute(
                update(RateLimit)
                .where(RateLimit.bucket == "stale")
                .values(window_started_at=utcnow() - timedelta(seconds=700))
            )
            await db.commit()
        async with iam_db.session_scope() as db:
            assert await rate_limit.current_count(
                db, bucket="stale", identity="ip-b", window_seconds=600,
            ) == 0
            assert await rate_limit.current_count(db, bucket="stale", identity="ip-b") == 1

    async def test_reset_clears_the_bucket(self):
        """A successful login path can clear its lockout bucket."""
        async with iam_db.session_scope() as db:
            await rate_limit.check_and_consume(
                db, bucket="clear", identity="ip-c", limit=5, window_seconds=600,
            )
            await db.commit()
        async with iam_db.session_scope() as db:
            await rate_limit.reset(db, bucket="clear", identity="ip-c")
            await db.commit()
        async with iam_db.session_scope() as db:
            assert await rate_limit.current_count(db, bucket="clear", identity="ip-c") == 0
