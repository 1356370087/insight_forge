"""Compatibility imports for the self-hosted IAM principal.

Supabase authentication was removed.  Older integrations may continue to
import ``get_current_user`` and ``apply_user_to_config`` while receiving the
new strongly typed :class:`security.rbac.Principal` implementation.
"""

from __future__ import annotations

from .rbac.dependencies import apply_principal_to_config, get_current_principal

get_current_user = get_current_principal
apply_user_to_config = apply_principal_to_config

__all__ = ["apply_user_to_config", "get_current_user"]
