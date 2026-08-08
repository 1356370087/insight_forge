"""Identity & RBAC command-line interface.

Usage::

    python -m security.cli bootstrap-admin --email admin@example.com --password '...'

The ``import-supabase-users`` subcommand performs the one-time identity cutover
without importing provider password hashes or credentials.
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from .rbac.database import session_scope
from .rbac.import_users import import_users, load_export
from .rbac.services import bootstrap
from .rbac.settings import get_settings


def _cmd_bootstrap_admin(args: argparse.Namespace) -> int:
    """Create the first administrator (idempotent)."""

    async def _run() -> int:
        settings = get_settings()
        if not settings.database_url:
            print("error: IAM_DATABASE_URL is not configured", file=sys.stderr)  # noqa: T201
            return 2
        async with session_scope() as db:
            user = await bootstrap.bootstrap_admin(
                db, email=args.email, password=args.password, display_name=args.display_name,
            )
            await db.commit()
        if user is None:
            print("bootstrap-admin: an administrator already exists; nothing to do.")  # noqa: T201
            return 0
        print(f"bootstrap-admin: created administrator {user.email}")  # noqa: T201
        return 0

    return asyncio.run(_run())


def _cmd_import_users(args: argparse.Namespace) -> int:
    """Import a Supabase Auth JSON/JSONL export, preserving user UUIDs."""

    async def _run() -> int:
        settings = get_settings()
        if not settings.database_url:
            print("error: IAM_DATABASE_URL is not configured", file=sys.stderr)  # noqa: T201
            return 2
        items = load_export(args.input)
        async with session_scope() as db:
            result = await import_users(
                db,
                items,
                settings=settings,
                dry_run=args.dry_run,
                send_reset_emails=args.send_reset_emails,
                default_role=args.default_role,
            )
            if args.dry_run:
                await db.rollback()
            else:
                await db.commit()
        print(  # noqa: T201
            "import-supabase-users: "
            f"discovered={result.discovered} created={result.created} "
            f"skipped={result.skipped} invalid={result.invalid} "
            f"reset_emails_sent={result.reset_emails_sent} dry_run={args.dry_run}"
        )
        return 0

    return asyncio.run(_run())


def build_parser() -> argparse.ArgumentParser:
    """Return the CLI argument parser."""
    parser = argparse.ArgumentParser(prog="security.cli", description="Identity & RBAC tooling")
    sub = parser.add_subparsers(dest="command", required=True)

    bs = sub.add_parser("bootstrap-admin", help="Create the first administrator (idempotent)")
    bs.add_argument("--email", required=True)
    bs.add_argument("--password", required=True)
    bs.add_argument("--display-name", default=None)
    bs.set_defaults(func=_cmd_bootstrap_admin)

    migrate = sub.add_parser(
        "import-supabase-users",
        help="Import a Supabase Auth JSON/JSONL export (dry-run by default)",
    )
    migrate.add_argument("--input", required=True)
    migrate.add_argument("--default-role", default="viewer")
    migrate.add_argument("--apply", action="store_false", dest="dry_run")
    migrate.add_argument("--send-reset-emails", action="store_true")
    migrate.set_defaults(func=_cmd_import_users, dry_run=True)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the CLI; returns a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
