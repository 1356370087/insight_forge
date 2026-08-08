"""FastAPI routers for the identity & RBAC subsystem."""

from __future__ import annotations

from .admin import router as admin_router
from .auth import router as auth_router

__all__ = ["admin_router", "auth_router"]
