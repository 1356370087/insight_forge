"""Alembic environment for the IAM/RBAC subsystem (async, asyncpg).

The database URL is read from ``IAM_DATABASE_URL``. The target metadata is the
IAM ORM ``Base.metadata``, so autogenerate is scoped to the ``iam_*`` tables
only (no other application tables are managed here).
"""

from __future__ import annotations

import asyncio
import os

# Make the IAM package importable when alembic is invoked from the repo root.
import sys  # noqa: E402
from logging.config import fileConfig

from alembic import context
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

from security.rbac.models import Base  # noqa: E402

config = context.config

# Resolve the database URL from the environment.
database_url = os.environ.get("IAM_DATABASE_URL")
if not database_url:
    raise RuntimeError("IAM_DATABASE_URL must be set to run IAM migrations")
config.set_main_option("sqlalchemy.url", database_url)

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode (emit SQL to stdout)."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    """Apply migrations within a live connection."""
    context.configure(connection=connection, target_metadata=target_metadata, compare_type=True)
    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online() -> None:
    """Run migrations in 'online' mode using an async engine."""
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())
