"""Async database infrastructure for the IAM subsystem.

Provides a process-wide async engine + session factory and a FastAPI
dependency that yields a request/task-scoped :class:`AsyncSession`. Sessions
are never shared across concurrent requests or background tasks, per the
SPEC and SQLAlchemy's session contract.
"""

from __future__ import annotations

import contextlib
import logging
from typing import AsyncIterator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from .settings import get_settings

logger = logging.getLogger(__name__)

_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def _build_engine(database_url: str) -> AsyncEngine:
    """Create an async engine for ``database_url``.

    Forces ``pool_pre_ping`` so dead connections (e.g. after a DB restart) are
    detected before use rather than surfacing as request errors.
    """
    return create_async_engine(database_url, pool_pre_ping=True, future=True)


def get_engine() -> AsyncEngine:
    """Return the lazily-created process-wide async engine."""
    global _engine
    settings = get_settings()
    if not settings.database_url:
        raise RuntimeError("IAM_DATABASE_URL is not configured")
    if _engine is None:
        _engine = _build_engine(settings.database_url)
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """Return the lazily-created process-wide session factory."""
    global _session_factory
    if _session_factory is None:
        _session_factory = async_sessionmaker(
            get_engine(),
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
        )
    return _session_factory


@contextlib.asynccontextmanager
async def session_scope() -> AsyncIterator[AsyncSession]:
    """Open a task-scoped session that rolls back on error.

    Usage::

        async with session_scope() as db:
            ...  # work with the session
    """
    factory = get_session_factory()
    async with factory() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise


async def get_db() -> AsyncIterator[AsyncSession]:
    """FastAPI dependency yielding a request-scoped :class:`AsyncSession`."""
    factory = get_session_factory()
    async with factory() as session:
        try:
            yield session
        finally:
            await session.close()


async def get_transactional_db() -> AsyncIterator[AsyncSession]:
    """FastAPI dependency yielding a session that commits on success.

    Use for mutating endpoints: services ``flush`` within the request and the
    dependency commits once (or rolls back on any exception).
    """
    factory = get_session_factory()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def dispose_engine() -> None:
    """Dispose the process-wide engine (used on application shutdown)."""
    global _engine, _session_factory
    if _engine is not None:
        await _engine.dispose()
    _engine = None
    _session_factory = None


def reset_engine_for_tests() -> None:
    """Reset cached engine/factory state so tests can swap the database URL."""
    global _engine, _session_factory
    _engine = None
    _session_factory = None
