"""DB integration: bootstrap, admin user/role management, last-admin guard, audit."""

from __future__ import annotations

import pytest

from security.rbac import database as iam_db
from security.rbac.repositories import collect_user_permissions
from security.rbac.services import auth as auth_service
from security.rbac.services import bootstrap
from security.rbac.services import roles as roles_service
from security.rbac.services import users as users_service
from security.rbac.services.users import LastAdminError, UserError

pytestmark = [pytest.mark.asyncio, pytest.mark.db]


class TestBootstrap:
    """First-admin bootstrap is idempotent and grants admin+researcher."""

    async def test_bootstrap_creates_first_admin(self, settings):
        """Bootstrap creates exactly one admin with both roles."""
        async with iam_db.session_scope() as db:
            user = await bootstrap.bootstrap_admin(
                db, email="root@example.com", password="initial-admin-passphrase",
            )
            await db.commit()
        assert user is not None
        role_codes, perms = await _perms_for(user.id)
        assert {"admin", "researcher"} <= set(role_codes)
        # admin does not grant cross-user research-create directly, researcher does.
        assert "research.run.create" in perms

    async def test_bootstrap_idempotent(self, settings):
        """A second bootstrap call is a no-op once an admin exists."""
        async with iam_db.session_scope() as db:
            await bootstrap.bootstrap_admin(
                db, email="root@example.com", password="initial-admin-passphrase",
            )
            await db.commit()
        async with iam_db.session_scope() as db:
            again = await bootstrap.bootstrap_admin(
                db, email="root2@example.com", password="initial-admin-passphrase",
            )
            await db.commit()
        assert again is None


class TestApproval:
    """Admin approval assigns roles and activates the user."""

    async def test_approve_assigns_researcher_and_activates(self, mail_recorder, settings):
        """Approving a pending user with [researcher] makes them active."""
        admin = await _bootstrap_admin(settings)
        await _register_verified(mail_recorder, settings, "carol@example.com")
        async with iam_db.session_scope() as db:
            carol = await _user_by_email(db, "carol@example.com")
            await users_service.approve_user(
                db, str(carol.id), role_codes=["researcher"], actor_id=admin,
            )
            await db.commit()
        role_codes, perms = await _perms_for(carol.id)
        assert "researcher" in role_codes
        assert "research.run.create" in perms
        async with iam_db.session_scope() as db:
            refreshed = await users_service.get_user(db, str(carol.id))
        assert refreshed.status == "active"

    async def test_approve_requires_non_admin_role(self, mail_recorder, settings):
        """Approval with only the admin role is rejected."""
        admin = await _bootstrap_admin(settings)
        await _register_verified(mail_recorder, settings, "dave@example.com")
        async with iam_db.session_scope() as db:
            dave = await _user_by_email(db, "dave@example.com")
            with pytest.raises(UserError):
                await users_service.approve_user(
                    db, str(dave.id), role_codes=["admin"], actor_id=admin,
                )


class TestLastAdminGuard:
    """The last effective admin cannot be disabled or demoted."""

    async def test_cannot_disable_last_admin(self, settings):
        """Disabling the only active admin is blocked."""
        admin = await _bootstrap_admin(settings)
        async with iam_db.session_scope() as db:
            with pytest.raises(LastAdminError):
                await users_service.disable_user(db, admin, actor_id=admin)

    async def test_cannot_remove_admin_role_from_last_admin(self, settings):
        """Stripping the admin role from the only admin is blocked."""
        admin = await _bootstrap_admin(settings)
        async with iam_db.session_scope() as db:
            with pytest.raises(LastAdminError):
                await users_service.assign_roles(db, admin, ["researcher"], actor_id=admin)

    async def test_can_disable_second_admin_when_two_exist(self, mail_recorder, settings):
        """With two admins, one can be disabled/demoted (guard only protects the last)."""
        first = await _bootstrap_admin(settings)
        await _register_verified(mail_recorder, settings, "admin2@example.com")
        async with iam_db.session_scope() as db:
            admin2 = await _user_by_email(db, "admin2@example.com")
            await users_service.approve_user(
                db, str(admin2.id), role_codes=["admin", "researcher"], actor_id=first,
            )
            await db.commit()
            admin2_id = str(admin2.id)
        async with iam_db.session_scope() as db:
            # Now demoting admin2 is allowed (admin remains).
            await users_service.assign_roles(db, admin2_id, ["researcher"], actor_id=first)
            await db.commit()


class TestRoles:
    """Custom role CRUD, system-role protection, permission catalog."""

    async def test_create_custom_role_and_list(self, settings):
        """A custom role combining catalog permissions can be listed."""
        async with iam_db.session_scope() as db:
            created = await roles_service.create_role(
                db, code="analyst", name="Analyst", description="Read-only analyst",
                permission_codes=["research.run.read_own", "research.task_activity.read_own"],
            )
            await db.commit()
        assert created.code == "analyst"
        assert "research.run.read_own" in created.permission_codes
        async with iam_db.session_scope() as db:
            roles = await roles_service.list_roles(db)
        codes = {r.code for r in roles}
        assert {"viewer", "researcher", "developer", "admin", "analyst"} <= codes

    async def test_system_role_permissions_immutable(self, settings):
        """System roles reject permission changes."""
        async with iam_db.session_scope() as db:
            viewer = await roles_service.get_role_by_code(db, "viewer")
            with pytest.raises(roles_service.RoleError):
                await roles_service.set_role_permissions(
                    db, str(viewer.id), ["research.run.create"],
                )

    async def test_system_role_not_deletable(self, settings):
        """System roles cannot be deleted."""
        async with iam_db.session_scope() as db:
            viewer = await roles_service.get_role_by_code(db, "viewer")
            with pytest.raises(roles_service.RoleError):
                await roles_service.delete_role(db, str(viewer.id))

    async def test_permission_catalog_seeded(self, settings):
        """The fixed permission catalog is present after seeding."""
        async with iam_db.session_scope() as db:
            perms = await roles_service.list_permissions(db)
        codes = {p.code for p in perms}
        assert "research.run.read_own" in codes
        assert "iam.users.write" in codes
        assert len(perms) >= 17


class TestAudit:
    """Security-relevant operations write audit events."""

    async def test_login_records_audit(self, mail_recorder, settings):
        """A successful login writes an audit event."""
        await _register_verified(mail_recorder, settings, "eve@example.com")
        async with iam_db.session_scope() as db:
            await auth_service.login(
                db, email="eve@example.com", password="a-very-strong-passphrase",
                settings=settings, ip_address="ip-eve",
            )
            await db.commit()
        async with iam_db.session_scope() as db:
            from sqlalchemy import select

            from security.rbac.models import AuditEvent

            actions = [
                row[0]
                for row in (await db.execute(select(AuditEvent.action))).all()
            ]
        assert "login" in actions


# --------------------------------------------------------------------------- helpers

async def _perms_for(user_id):
    """Return (role_codes, permission_codes) for a user."""
    async with iam_db.session_scope() as db:
        return await collect_user_permissions(db, str(user_id))


async def _bootstrap_admin(settings) -> str:
    """Bootstrap an admin and return its user id."""
    async with iam_db.session_scope() as db:
        user = await bootstrap.bootstrap_admin(
            db, email="root@example.com", password="initial-admin-passphrase",
        )
        await db.commit()
    return str(user.id)


async def _register_verified(mail_recorder, settings, email):
    """Register and verify a user with the given email."""
    async with iam_db.session_scope() as db:
        await auth_service.register(
            db, email=email, password="a-very-strong-passphrase", display_name=email,
            settings=settings, identity_for_rate_limit=f"ip-{email}",
        )
        await db.commit()
    import re

    body = [m["text_body"] for m in mail_recorder if "verify" in m["subject"].lower()][-1]
    token = re.search(r"token=([^\s]+)", body).group(1)
    async with iam_db.session_scope() as db:
        await auth_service.verify_email(db, token=token, settings=settings)
        await db.commit()


async def _user_by_email(db, email):
    """Return the User row for a normalized email within an existing session."""
    from security.rbac.emails import normalize_email
    from security.rbac.repositories import get_user_by_email_normalized

    return await get_user_by_email_normalized(db, normalize_email(email))
