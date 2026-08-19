"""Configuration for the self-hosted identity and RBAC subsystem.

All settings are environment driven. Production deployments must provide
explicit JWT keys, a database URL and (when registration is open) SMTP; the
:func:`startup_checks` helper enforces that and refuses to boot otherwise.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Literal, cast

TRUE_VALUES = {"1", "true", "yes", "on"}
MailBackend = Literal["console", "smtp"]


def _flag(name: str, default: str = "false") -> bool:
    """Return whether an env var is set to a truthy value."""
    return os.environ.get(name, default).strip().lower() in TRUE_VALUES


def _int(name: str, default: int) -> int:
    """Parse an env var as int, falling back to ``default`` on any error."""
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


# Token / session time-to-live defaults (seconds).
ACCESS_TOKEN_TTL = 15 * 60            # 15 minutes
REFRESH_IDLE_TTL = 30 * 24 * 60 * 60  # 30 days
SESSION_ABSOLUTE_TTL = 90 * 24 * 60 * 60  # 90 days
EMAIL_VERIFY_TTL = 24 * 60 * 60       # 24 hours
PASSWORD_RESET_TTL = 30 * 60          # 30 minutes
REFRESH_REUSE_GRACE = 5              # seconds of concurrent-refresh tolerance
SSE_REAUTH_INTERVAL = 60            # seconds between SSE authorization rechecks

# JWT identifiers.
ACCESS_AUDIENCE = "odr-api"
REFRESH_AUDIENCE = "odr-refresh"
ACCESS_TOKEN_TYPE = "odr-access+jwt"
REFRESH_TOKEN_TYPE = "odr-refresh+jwt"

# Algorithm is fixed and never read from the token header (RFC 8725).
JWT_ALGORITHM = "EdDSA"


@dataclass(frozen=True)
class IAMSettings:
    """Resolved identity/RBAC configuration sourced from the environment."""

    app_env: str = field(default_factory=lambda: os.environ.get("APP_ENV", "development"))
    issuer: str = field(default_factory=lambda: os.environ.get("IAM_JWT_ISSUER", "open-deep-research"))

    database_url: str = field(default_factory=lambda: os.environ.get("IAM_DATABASE_URL", ""))

    # TTL knobs.
    access_token_ttl: int = field(default_factory=lambda: _int("IAM_ACCESS_TOKEN_TTL", ACCESS_TOKEN_TTL))
    refresh_idle_ttl: int = field(default_factory=lambda: _int("IAM_REFRESH_IDLE_TTL", REFRESH_IDLE_TTL))
    session_absolute_ttl: int = field(default_factory=lambda: _int("IAM_SESSION_ABSOLUTE_TTL", SESSION_ABSOLUTE_TTL))
    email_verify_ttl: int = field(default_factory=lambda: _int("IAM_EMAIL_VERIFY_TTL", EMAIL_VERIFY_TTL))
    password_reset_ttl: int = field(default_factory=lambda: _int("IAM_PASSWORD_RESET_TTL", PASSWORD_RESET_TTL))
    refresh_reuse_grace: int = field(default_factory=lambda: _int("IAM_REFRESH_REUSE_GRACE", REFRESH_REUSE_GRACE))
    sse_reauth_interval: int = field(default_factory=lambda: _int("IAM_SSE_REAUTH_INTERVAL", SSE_REAUTH_INTERVAL))

    # JWT keys (PEM-encoded Ed25519). Current signing key + kid, plus an
    # optional kid -> public-key verification map for smooth rotation.
    access_signing_key_pem: str = field(default_factory=lambda: os.environ.get("IAM_JWT_ACCESS_SIGNING_KEY", ""))
    access_signing_kid: str = field(default_factory=lambda: os.environ.get("IAM_JWT_ACCESS_KID", ""))
    access_verify_keys_json: str = field(default_factory=lambda: os.environ.get("IAM_JWT_ACCESS_VERIFY_KEYS", ""))
    refresh_signing_key_pem: str = field(default_factory=lambda: os.environ.get("IAM_JWT_REFRESH_SIGNING_KEY", ""))
    refresh_signing_kid: str = field(default_factory=lambda: os.environ.get("IAM_JWT_REFRESH_KID", ""))
    refresh_verify_keys_json: str = field(default_factory=lambda: os.environ.get("IAM_JWT_REFRESH_VERIFY_KEYS", ""))

    # SMTP / mail.
    mail_backend: MailBackend = field(
        default_factory=lambda: cast(MailBackend, os.environ.get("IAM_MAIL_BACKEND", "console"))
    )
    smtp_host: str = field(default_factory=lambda: os.environ.get("IAM_SMTP_HOST", ""))
    smtp_port: int = field(default_factory=lambda: _int("IAM_SMTP_PORT", 587))
    smtp_user: str = field(default_factory=lambda: os.environ.get("IAM_SMTP_USER", ""))
    smtp_password: str = field(default_factory=lambda: os.environ.get("IAM_SMTP_PASSWORD", ""))
    mail_from: str = field(default_factory=lambda: os.environ.get("IAM_MAIL_FROM", "no-reply@localhost"))
    smtp_starttls: bool = field(default_factory=lambda: _flag("IAM_SMTP_STARTTLS", "true"))
    public_base_url: str = field(default_factory=lambda: os.environ.get("IAM_PUBLIC_BASE_URL", "http://localhost:3000"))
    token_digest_secret: str = field(default_factory=lambda: os.environ.get("IAM_TOKEN_DIGEST_SECRET", ""))

    # Behaviour toggles.
    local_dev_bypass: bool = field(default_factory=lambda: _flag("LOCAL_DEV_AUTH_BYPASS", "false"))
    open_registration: bool = field(default_factory=lambda: _flag("IAM_OPEN_REGISTRATION", "false"))

    # Distributed rate-limit bucket sizes (requests per window per identity).
    # Defaults to 0 (trust only the socket peer): a non-zero value must be set
    # deliberately to match the real proxy topology, otherwise clients could
    # spoof X-Forwarded-For and rotate their rate-limit identity at will.
    trusted_proxy_count: int = field(
        default_factory=lambda: max(0, _int("IAM_TRUSTED_PROXY_COUNT", 0))
    )
    login_rate_limit: int = field(default_factory=lambda: _int("IAM_LOGIN_RATE_LIMIT", 10))
    register_rate_limit: int = field(default_factory=lambda: _int("IAM_REGISTER_RATE_LIMIT", 5))
    resend_rate_limit: int = field(default_factory=lambda: _int("IAM_RESEND_RATE_LIMIT", 3))
    reset_rate_limit: int = field(default_factory=lambda: _int("IAM_RESET_RATE_LIMIT", 5))
    rate_limit_window: int = field(default_factory=lambda: _int("IAM_RATE_LIMIT_WINDOW", 600))

    # Bootstrap administrator (created on first migration run when set).
    bootstrap_admin_email: str = field(default_factory=lambda: os.environ.get("IAM_BOOTSTRAP_ADMIN_EMAIL", ""))
    bootstrap_admin_password: str = field(default_factory=lambda: os.environ.get("IAM_BOOTSTRAP_ADMIN_PASSWORD", ""))

    @property
    def is_production(self) -> bool:
        """Return whether the process runs in a production environment."""
        return self.app_env.strip().lower() == "production"

    @property
    def iam_enabled(self) -> bool:
        """Return whether IAM can authenticate requests.

        A database is mandatory outside the explicit development bypass. The
        legacy Supabase authentication path no longer exists.
        """
        return bool(self.database_url) or local_dev_bypass_enabled()


def get_settings() -> IAMSettings:
    """Return a freshly-resolved settings snapshot (env is read each call)."""
    return IAMSettings()


def local_dev_bypass_enabled() -> bool:
    """Return whether the explicit local-development auth bypass is enabled.

    Per the SPEC, the bypass is only honored when ``APP_ENV=development`` and
    returns a synthetic researcher/developer principal without IAM-admin
    permissions.
    """
    if not _flag("LOCAL_DEV_AUTH_BYPASS", "false"):
        return False
    return os.environ.get("APP_ENV", "development").strip().lower() == "development"
