"""Tests for the permission catalog, built-in roles and Principal evaluation."""

from __future__ import annotations

import pytest

from security.rbac import permissions as P
from security.rbac import roles as R
from security.rbac.permissions import (
    PERMISSION_CODES,
    PERMISSIONS,
    is_known_permission,
    validate_permission_codes,
)
from security.rbac.principal import Principal, dev_permissions, synthetic_dev_principal


class TestPermissionCatalog:
    """The permission catalog is closed and stable."""

    def test_catalog_is_non_empty(self):
        """The catalog contains a meaningful set of permissions."""
        assert len(PERMISSIONS) >= 17

    def test_codes_are_unique(self):
        """Every permission code is unique."""
        codes = [p.code for p in PERMISSIONS]
        assert len(codes) == len(set(codes))

    def test_known_permission_lookup(self):
        """is_known_permission recognizes catalog entries only."""
        assert is_known_permission("research.run.read_own") is True
        assert is_known_permission("bogus.perm") is False

    def test_validate_permission_codes_dedups_and_preserves_order(self):
        """Validation deduplicates and preserves first-seen order."""
        out = validate_permission_codes(
            ["research.run.read_own", "iam.users.read", "research.run.read_own"],
        )
        assert out == ["research.run.read_own", "iam.users.read"]

    def test_validate_permission_codes_rejects_unknown(self):
        """Unknown codes raise ValueError."""
        with pytest.raises(ValueError):
            validate_permission_codes(["research.run.read_own", "nope.perm"])

    def test_research_and_iam_domains_present(self):
        """Both research and iam domains are represented."""
        domains = {p.domain for p in PERMISSIONS}
        assert {"research", "iam"} <= domains

    def test_admin_permissions_exclude_cross_user_research(self):
        """Admin catalog permissions are IAM-only (no cross-user run access)."""
        admin_codes = set(R.ADMIN.permissions)
        assert admin_codes
        assert not (admin_codes & {P.RESEARCH_RUN_READ_OWN.code, P.RESEARCH_RUN_CREATE.code})


class TestSystemRoles:
    """Built-in role composition matches the SPEC hierarchy."""

    def test_system_role_codes(self):
        """The four system role codes are recognized as system."""
        assert set(R.SYSTEM_ROLE_CODES) == {"viewer", "researcher", "developer", "admin"}
        for code in R.SYSTEM_ROLE_CODES:
            assert R.is_system_role_code(code) is True
        assert R.is_system_role_code("custom") is False

    def test_researcher_extends_viewer(self):
        """Researcher is a strict superset of viewer permissions."""
        assert set(R.VIEWER.permissions) <= set(R.RESEARCHER.permissions)

    def test_developer_extends_researcher(self):
        """Developer is a strict superset of researcher permissions."""
        assert set(R.RESEARCHER.permissions) <= set(R.DEVELOPER.permissions)

    def test_developer_has_advanced_tool_perms(self):
        """Developer grants MCP/browser/skill/observability/diagnostics."""
        dev = set(R.DEVELOPER.permissions)
        for code in (
            P.RESEARCH_TOOL_MCP.code, P.RESEARCH_TOOL_BROWSER.code, P.RESEARCH_TOOL_SKILL.code,
            P.RESEARCH_OBSERVABILITY_READ_OWN.code, P.RESEARCH_DIAGNOSTICS_PREVIEW.code,
        ):
            assert code in dev

    def test_all_role_perms_are_in_catalog(self):
        """Every system role grant references a real catalog permission."""
        for role in R.SYSTEM_ROLES:
            for code in role.permissions:
                assert code in PERMISSION_CODES


class TestPrincipal:
    """Principal permission evaluation and the dev-bypass principal."""

    def test_has_helpers(self):
        """has/has_all/has_any evaluate permission membership."""
        p = Principal(
            user_id="u", email="u@x", status="active", session_id="s",
            roles=frozenset({"researcher"}),
            permissions=frozenset({"a", "b", "c"}),
            authz_version=1,
        )
        assert p.has("a") is True
        assert p.has("z") is False
        assert p.has_all(["a", "b"]) is True
        assert p.has_all(["a", "z"]) is False
        assert p.has_any(["z", "c"]) is True
        assert p.has_any(["z"]) is False

    def test_active_property(self):
        """is_active reflects the active status."""
        active = Principal("u", "e", "active", None, frozenset(), frozenset(), 1)
        pending = Principal("u", "e", "pending_approval", None, frozenset(), frozenset(), 1)
        assert active.is_active is True
        assert pending.is_active is False
        assert pending.is_pending_approval is True

    def test_synthetic_dev_principal_has_no_iam_admin(self):
        """The dev-bypass principal grants researcher+developer, never IAM admin."""
        p = synthetic_dev_principal()
        assert p.is_active
        assert p.roles == {"researcher", "developer"}
        assert p.has("research.run.create") is True
        assert p.has("research.diagnostics.preview") is True
        assert p.has("iam.users.write") is False

    def test_dev_permissions_union(self):
        """dev_permissions is the researcher ∪ developer union."""
        union = dev_permissions()
        assert set(R.RESEARCHER.permissions) | set(R.DEVELOPER.permissions) == set(union)

    def test_runtime_dict_shape(self):
        """to_runtime_dict exposes identity/roles/permissions for legacy code."""
        p = Principal("uid", "e", "active", "sid", frozenset({"r"}), frozenset({"p"}), 9)
        d = p.to_runtime_dict()
        assert d["identity"] == "uid"
        assert d["roles"] == ["r"]
        assert "p" in d["effective_permissions"]
        assert d["authz_version"] == 9
