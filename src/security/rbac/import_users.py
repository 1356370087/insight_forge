"""One-time import of users exported from Supabase Auth.

Only public identity fields are accepted. Password hashes, provider tokens and
metadata credentials are deliberately ignored; imported users must establish a
new local password through the reset flow.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from .email import send_password_reset_email
from .emails import InvalidEmail, validate_and_normalize
from .models import User, UserRole, UserStatus
from .repositories import get_user_by_email_normalized, get_user_by_id
from .services import email_tokens
from .services.roles import get_role_by_code
from .settings import IAMSettings


@dataclass(frozen=True)
class ImportResult:
    """Counts produced by a user import."""

    discovered: int
    created: int
    skipped: int
    invalid: int
    reset_emails_sent: int


def load_export(path: str | Path) -> list[dict[str, Any]]:
    """Load a Supabase JSON object/list or JSONL export."""
    text = Path(path).read_text(encoding="utf-8")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    if isinstance(payload, dict):
        payload = payload.get("users", [])
    if not isinstance(payload, list):
        raise ValueError("import_payload_must_be_users_list")
    return [item for item in payload if isinstance(item, dict)]


def _display_name(item: dict[str, Any]) -> str | None:
    metadata = item.get("user_metadata")
    if not isinstance(metadata, dict):
        return None
    value = metadata.get("display_name") or metadata.get("full_name") or metadata.get("name")
    if not value:
        return None
    return str(value).strip()[:160] or None


def _confirmed_at(item: dict[str, Any]) -> datetime | None:
    raw = item.get("email_confirmed_at") or item.get("confirmed_at")
    if not raw:
        return None
    try:
        return datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except ValueError:
        return None


async def import_users(
    db: AsyncSession,
    items: list[dict[str, Any]],
    *,
    settings: IAMSettings,
    dry_run: bool = True,
    send_reset_emails: bool = False,
    default_role: str = "viewer",
) -> ImportResult:
    """Import identities while preserving their UUID and forcing password reset."""
    role = await get_role_by_code(db, default_role)
    if role is None:
        raise ValueError(f"default_role_not_found:{default_role}")
    created = skipped = invalid = sent = 0
    for item in items:
        try:
            user_id = str(UUID(str(item.get("id", ""))))
            normalized = validate_and_normalize(str(item.get("email", "")))
        except (ValueError, InvalidEmail):
            invalid += 1
            continue
        if await get_user_by_id(db, user_id) or await get_user_by_email_normalized(db, normalized):
            skipped += 1
            continue
        created += 1
        if dry_run:
            continue
        confirmed = _confirmed_at(item)
        user = User(
            id=user_id,
            email=normalized,
            email_normalized=normalized,
            password_hash=None,
            display_name=_display_name(item),
            status=UserStatus.PASSWORD_RESET_REQUIRED,
            email_verified_at=confirmed,
        )
        db.add(user)
        await db.flush()
        db.add(UserRole(user_id=user.id, role_id=role.id))
        if send_reset_emails:
            token = await email_tokens.issue(
                db,
                user_id=user_id,
                purpose=email_tokens.EmailTokenPurpose.PASSWORD_RESET,
                settings=settings,
            )
            await send_password_reset_email(to=normalized, token=token, settings=settings)
            sent += 1
    return ImportResult(len(items), created, skipped, invalid, sent)
