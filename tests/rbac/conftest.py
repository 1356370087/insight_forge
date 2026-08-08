"""Shared fixtures for the IAM/RBAC database-backed integration tests.

These tests run against a real PostgreSQL when ``IAM_TEST_DATABASE_URL`` is
set; otherwise the DB tests are skipped. The schema is created once per
session (in a throwaway event loop) and seeded with the fixed permission
catalog and system roles; mutable tables are truncated before each test.

asyncpg connections are bound to the event loop that created them, so the
cached engine is reset per test and re-created lazily inside each test's own
function-scoped loop.
"""

from __future__ import annotations

import asyncio
import os
import re

import pytest
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import create_async_engine

from security.rbac import database as iam_db
from security.rbac import email as iam_email
from security.rbac import jwt_service
from security.rbac.models import Base, Permission, Role, RolePermission
from security.rbac.permissions import PERMISSIONS
from security.rbac.roles import SYSTEM_ROLES

TEST_DB_URL = os.environ.get("IAM_TEST_DATABASE_URL")
SKIP_REASON = "Set IAM_TEST_DATABASE_URL (e.g. postgresql+asyncpg://odr:odr@localhost:5432/iam) to run DB tests"

# Mutable tables cleared between tests (catalog/roles are seeded once, kept).
_MUTABLE_TABLES = (
    "iam_rate_limits",
    "iam_audit_events",
    "iam_email_tokens",
    "iam_refresh_tokens",
    "iam_sessions",
    "iam_user_roles",
    "iam_users",
)


def pytest_collection_modifyitems(config, items):
    """Auto-skip DB tests when no test database is configured."""
    if TEST_DB_URL:
        return
    skip_db = pytest.mark.skip(reason=SKIP_REASON)
    for item in items:
        if "db" in item.keywords:
            item.add_marker(skip_db)


@pytest.fixture(scope="session")
def test_db_url():
    """Return the configured test database URL (empty string when unset)."""
    return TEST_DB_URL


async def _create_schema_and_seed(url: str) -> None:
    """Create all tables and seed the catalog + system roles (throwaway engine)."""
    engine = create_async_engine(url)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    async with engine.begin() as conn:
        # Seed permissions.
        for perm in PERMISSIONS:
            await conn.execute(
                Permission.__table__.insert().values(
                    code=perm.code, name=perm.name, description=perm.description, domain=perm.domain,
                )
            )
        await conn.commit()
    # Seed roles + grants via ORM session for easier id resolution.
    factory = iam_db.get_session_factory()
    async with factory() as session:
        perm_rows = (await session.execute(select(Permission))).scalars().all()
        perm_by_code = {p.code: p for p in perm_rows}
        for role in SYSTEM_ROLES:
            role_obj = Role(code=role.code, name=role.name, description=role.description, is_system=True)
            session.add(role_obj)
            await session.flush()
            for code in role.permissions:
                session.add(RolePermission(role_id=role_obj.id, permission_id=perm_by_code[code].id))
        await session.commit()
    iam_db.reset_engine_for_tests()
    await engine.dispose()


@pytest.fixture(scope="session", autouse=True)
def _session_schema(test_db_url):
    """Create the schema once per session and seed it (no-op without a DB)."""
    if not test_db_url:
        yield
        return
    os.environ["IAM_DATABASE_URL"] = test_db_url
    os.environ.setdefault("APP_ENV", "development")
    os.environ.setdefault("IAM_MAIL_BACKEND", "console")
    iam_db.reset_engine_for_tests()
    jwt_service.reset_cache_for_tests()
    asyncio.run(_create_schema_and_seed(test_db_url))
    iam_db.reset_engine_for_tests()
    yield
    asyncio.run(_dispose_engine())
    iam_db.reset_engine_for_tests()


async def _dispose_engine() -> None:
    """Dispose the cached engine at session end."""
    try:
        await iam_db.dispose_engine()
    except Exception:  # noqa: BLE001 - best-effort teardown
        pass


@pytest.fixture(autouse=True)
def _per_test_isolation(test_db_url):
    """Reset the cached engine and truncate mutable tables before each test.

    No-op when no test database is configured. Truncation runs in a throwaway
    engine/loop so it never conflicts with an async test's own event loop; the
    cached IAM engine is reset afterward so async db tests lazily create a
    fresh engine bound to their own loop.
    """
    if not test_db_url:
        return
    iam_db.reset_engine_for_tests()
    jwt_service.reset_cache_for_tests()
    asyncio.run(_truncate_mutable(test_db_url))


async def _truncate_mutable(url: str) -> None:
    """Truncate all mutable tables in a short-lived throwaway engine."""
    engine = create_async_engine(url)
    async with engine.begin() as conn:
        for table in _MUTABLE_TABLES:
            await conn.execute(text(f"TRUNCATE TABLE {table} RESTART IDENTITY CASCADE"))
    await engine.dispose()


@pytest.fixture
def settings():
    """Return a fresh IAMSettings snapshot for the current environment."""
    from security.rbac.settings import get_settings

    return get_settings()


@pytest.fixture
def mail_recorder(monkeypatch):
    """Patch the mail sender to record messages instead of printing/sending.

    Returns the recorder list; tokens are extractable from recorded bodies.
    """
    recorded: list[dict] = []

    class _Recorder:
        async def send(self, *, to, subject, text_body, html_body=None):
            recorded.append({"to": to, "subject": subject, "text_body": text_body})

    monkeypatch.setattr(iam_email, "get_mail_sender", lambda settings=None: _Recorder())
    return recorded


def extract_token(mail_record_body: str) -> str:
    """Pull the one-time token out of a recorded email body."""
    match = re.search(r"token=([^\s]+)", mail_record_body)
    assert match, f"no token found in email body: {mail_record_body!r}"
    return match.group(1)
