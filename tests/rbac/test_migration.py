"""DB integration: Alembic upgrade on a real, empty PostgreSQL (SPEC §5).

Creates a throwaway database, runs ``alembic upgrade head``, and asserts the
schema + seed are present. Satisfies: "Alembic 在真实 PostgreSQL 上从空库升级到 head".
"""

from __future__ import annotations

import os

import pytest
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import create_async_engine

from security.rbac import database as iam_db
from security.rbac.import_users import import_users
from security.rbac.models import Role, User, UserRole, UserStatus

pytestmark = [pytest.mark.asyncio, pytest.mark.db]


def _admin_dsn(test_db_url: str) -> str:
    """Return a DSN pointing at the maintenance ``postgres`` database."""
    return test_db_url.rsplit("/", 1)[0] + "/postgres"


async def _create_database(admin_dsn: str, name: str) -> None:
    """Create a fresh database for the migration run."""
    engine = create_async_engine(admin_dsn, isolation_level="AUTOCOMMIT")
    async with engine.connect() as conn:
        await conn.execute(text(f'DROP DATABASE IF EXISTS "{name}"'))
        await conn.execute(text(f'CREATE DATABASE "{name}"'))
    await engine.dispose()


async def _drop_database(admin_dsn: str, name: str) -> None:
    """Drop the throwaway database."""
    engine = create_async_engine(admin_dsn, isolation_level="AUTOCOMMIT")
    async with engine.connect() as conn:
        await conn.execute(text(f'DROP DATABASE IF EXISTS "{name}"'))
    await engine.dispose()


async def test_alembic_upgrade_heads_from_empty(test_db_url):
    """The initial migration builds all tables and seeds the catalog on a fresh DB."""
    import time

    from alembic import command
    from alembic.config import Config

    db_name = f"iam_migrate_{int(time.time() * 1000)}"
    admin_dsn = _admin_dsn(test_db_url)
    target_dsn = test_db_url.rsplit("/", 1)[0] + f"/{db_name}"

    await _create_database(admin_dsn, db_name)
    try:
        # env.py reads IAM_DATABASE_URL from the environment.
        os.environ["IAM_DATABASE_URL"] = target_dsn
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        cfg = Config(os.path.join(repo_root, "alembic.ini"))
        cfg.set_main_option("script_location", os.path.join(repo_root, "src", "security", "rbac", "migrations"))
        # Alembic's env.py calls asyncio.run(); run it in a worker thread so it
        # has its own event loop instead of nesting in this test's loop.
        import asyncio

        await asyncio.to_thread(command.upgrade, cfg, "head")

        engine = create_async_engine(target_dsn)
        async with engine.connect() as conn:
            tables = {
                r[0] for r in (await conn.execute(
                    text("SELECT table_name FROM information_schema.tables WHERE table_schema='public'")
                )).all()
            }
            for required in (
                "iam_users", "iam_roles", "iam_permissions", "iam_user_roles",
                "iam_role_permissions", "iam_sessions", "iam_refresh_tokens",
                "iam_email_tokens", "iam_audit_events", "iam_rate_limits", "alembic_version",
            ):
                assert required in tables, f"missing table {required}"
            assert (await conn.execute(text("SELECT count(*) FROM iam_permissions"))).scalar() >= 25
            assert (await conn.execute(text("SELECT count(*) FROM iam_roles WHERE is_system"))).scalar() == 4
            assert (await conn.execute(text("SELECT count(*) FROM iam_role_permissions"))).scalar() == 36
            rev = (await conn.execute(text("SELECT version_num FROM alembic_version"))).scalar()
            assert rev == "0002_sandbox_permissions"
        await engine.dispose()

        await asyncio.to_thread(command.downgrade, cfg, "0001_iam_initial")
        engine = create_async_engine(target_dsn)
        async with engine.connect() as conn:
            remaining = (
                await conn.execute(
                    text(
                        "SELECT count(*) FROM iam_permissions "
                        "WHERE code LIKE 'research.security_approval.%' "
                        "OR code LIKE 'research.tool.file.%' "
                        "OR code = 'research.tool.shell.execute'"
                    )
                )
            ).scalar()
            revision = (
                await conn.execute(text("SELECT version_num FROM alembic_version"))
            ).scalar()
            assert remaining == 0
            assert revision == "0001_iam_initial"
        await engine.dispose()
        await asyncio.to_thread(command.upgrade, cfg, "head")
    finally:
        os.environ["IAM_DATABASE_URL"] = test_db_url  # restore for the rest of the session
        await _drop_database(admin_dsn, db_name)


async def test_identity_import_is_dry_run_by_default_and_preserves_uuid(settings):
    """Legacy identities are previewed first, then imported without credentials."""
    user_id = "8b187818-b93e-4fb5-b298-97d78203afaa"
    exported = [{
        "id": user_id,
        "email": " Legacy.User@Example.com ",
        "encrypted_password": "must-not-be-imported",
        "user_metadata": {"display_name": "Legacy User", "provider_token": "secret"},
        "email_confirmed_at": "2026-01-02T03:04:05Z",
    }]

    async with iam_db.session_scope() as db:
        preview = await import_users(db, exported, settings=settings)
        assert preview.created == 1
        assert (await db.get(User, user_id)) is None

    async with iam_db.session_scope() as db:
        result = await import_users(db, exported, settings=settings, dry_run=False)
        await db.commit()
        assert result.created == 1

    async with iam_db.session_scope() as db:
        user = await db.get(User, user_id)
        assert user is not None
        assert user.email == "legacy.user@example.com"
        assert user.password_hash is None
        assert user.display_name == "Legacy User"
        assert user.status == UserStatus.PASSWORD_RESET_REQUIRED
        assignment = (
            await db.execute(
                select(Role.code)
                .join(UserRole, UserRole.role_id == Role.id)
                .where(UserRole.user_id == user_id)
            )
        ).scalar_one()
        assert assignment == "viewer"
