"""Self-hosted identity & RBAC subsystem for Open Deep Research.

Public API surface for application code and tests. Internal modules
(``models``, ``services.*``, ``routers.*``) are importable but not re-exported
here to keep the namespace focused.
"""

from __future__ import annotations

from .app_extension import (
    StartupError,
    check_database_connection,
    mount_rbac,
    shutdown_rbac,
    startup_checks,
)
from .dependencies import (
    apply_principal_to_config,
    build_principal,
    get_current_principal,
    reauthorize_session,
    register_ownership_checker,
    require_active_user,
    require_permissions,
    require_run_owner,
    require_run_owner_or_any,
    require_task_owner,
)
from .principal import Principal, synthetic_dev_principal
from .settings import IAMSettings, get_settings, local_dev_bypass_enabled

__all__ = [
    "IAMSettings",
    "Principal",
    "StartupError",
    "apply_principal_to_config",
    "build_principal",
    "check_database_connection",
    "get_current_principal",
    "get_settings",
    "local_dev_bypass_enabled",
    "mount_rbac",
    "reauthorize_session",
    "register_ownership_checker",
    "require_active_user",
    "require_permissions",
    "require_run_owner",
    "require_run_owner_or_any",
    "require_task_owner",
    "shutdown_rbac",
    "startup_checks",
    "synthetic_dev_principal",
]
