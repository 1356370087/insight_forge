"""Application integration: router mounting and startup validation.

The host application (``server.py``) calls :func:`mount_rbac` at construction
time and :func:`startup_checks` during startup. When IAM is not configured
(no ``IAM_DATABASE_URL``) both are no-ops, so the legacy auth path and the
existing test suite remain unaffected.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI

from .database import dispose_engine, get_engine
from .keys import KeyError_, load_key_material
from .routers import admin_router, auth_router
from .settings import IAMSettings, get_settings

logger = logging.getLogger(__name__)


class StartupError(RuntimeError):
    """Raised when a required startup invariant is violated."""


def mount_rbac(app: FastAPI) -> None:
    """Mount the ``/auth/*`` and ``/admin/*`` routers when IAM is enabled."""
    settings = get_settings()
    if not settings.database_url:
        logger.info("IAM database not configured; auth/admin routers are unavailable in local bypass mode.")
        return
    app.include_router(auth_router)
    app.include_router(admin_router)
    logger.info("IAM enabled: mounted /auth and /admin routers.")


async def startup_checks(settings: IAMSettings | None = None) -> None:
    """Validate startup invariants; raise :class:`StartupError` on failure.

    Enforced only when IAM is enabled. Checks:
      * JWT keys (access + refresh) load successfully.
      * Production open-registration requires an SMTP backend.
      * Production refuses a console mail backend.
    The Alembic-version check is performed by :func:`assert_schema_current`,
    invoked separately so migrations can run as a dedicated deploy step.
    """
    settings = settings or get_settings()
    if not settings.iam_enabled:
        raise StartupError("IAM_DATABASE_URL is required unless LOCAL_DEV_AUTH_BYPASS is enabled in development")
    if not settings.database_url:
        logger.warning("IAM startup running in explicit local development bypass mode.")
        return
    for kind in ("access", "refresh"):
        try:
            load_key_material(kind, settings=settings)  # type: ignore[arg-type]
        except KeyError_ as exc:
            raise StartupError(f"jwt_key_invalid:{kind}:{exc}") from exc
    if settings.is_production:
        if not settings.token_digest_secret:
            raise StartupError("production_requires_token_digest_secret")
        if settings.open_registration and settings.mail_backend != "smtp":
            raise StartupError("open_registration_requires_smtp")
        if settings.mail_backend != "smtp":
            raise StartupError("production_requires_smtp")
    logger.info("IAM startup checks passed.")


async def assert_schema_current(expected_revision: str | None = None) -> None:
    """Assert the database schema is migrated (and optional revision pinned).

    The migration is run as a dedicated deploy step; the API must refuse to
    serve if the schema is stale. ``expected_revision`` lets operators pin a
    minimum revision.
    """
    import sqlalchemy as sa

    settings = get_settings()
    if not settings.database_url:
        return
    engine = get_engine()
    async with engine.connect() as conn:
        exists = await conn.execute(
            sa.text(
                "SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name='alembic_version')"
            )
        )
        if not exists.scalar():
            raise StartupError("schema_not_migrated")
        current = await conn.execute(sa.text("SELECT version_num FROM alembic_version"))
        revision = current.scalar()
    if expected_revision is not None and revision != expected_revision:
        raise StartupError(f"schema_revision_mismatch:got={revision}:expected={expected_revision}")


async def shutdown_rbac() -> None:
    """Dispose the IAM engine on application shutdown."""
    await dispose_engine()
