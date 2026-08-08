"""Transactional email backend (SMTP or console).

Console backend logs the link to stdout for local development; SMTP backend
sends via ``aiosmtplib``. The backend is selected by ``IAM_MAIL_BACKEND``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Literal, Protocol

import aiosmtplib

from .settings import IAMSettings, get_settings

logger = logging.getLogger(__name__)

MailBackendKind = Literal["console", "smtp"]


class MailSender(Protocol):
    """Protocol implemented by console and SMTP mail backends."""

    async def send(self, *, to: str, subject: str, text_body: str, html_body: str | None = ...) -> None:
        """Send a transactional email."""
        ...


@dataclass
class ConsoleMailSender:
    """Dev mail backend that writes the rendered link to the log/stdout."""

    settings: IAMSettings

    async def send(self, *, to: str, subject: str, text_body: str, html_body: str | None = None) -> None:
        """Print the email to stdout (development only)."""
        print(  # noqa: T201 - intentional dev output
            f"\n[console-mail] to={to} subject={subject}\n{text_body}\n",
            flush=True,
        )


@dataclass
class SmtpMailSender:
    """SMTP mail backend using ``aiosmtplib`` with optional STARTTLS."""

    settings: IAMSettings

    async def send(self, *, to: str, subject: str, text_body: str, html_body: str | None = None) -> None:
        """Send an email over SMTP."""
        message = MIMEMultipart("alternative")
        message["From"] = self.settings.mail_from
        message["To"] = to
        message["Subject"] = subject
        message.attach(MIMEText(text_body, "plain", "utf-8"))
        if html_body:
            message.attach(MIMEText(html_body, "html", "utf-8"))
        await aiosmtplib.send(
            message,
            hostname=self.settings.smtp_host,
            port=self.settings.smtp_port,
            username=self.settings.smtp_user or None,
            password=self.settings.smtp_password or None,
            start_tls=self.settings.smtp_starttls,
        )


def get_mail_sender(settings: IAMSettings | None = None) -> MailSender:
    """Return the configured mail sender (``console`` or ``smtp``)."""
    settings = settings or get_settings()
    if settings.mail_backend == "smtp":
        return SmtpMailSender(settings=settings)
    return ConsoleMailSender(settings=settings)


def verification_url(base_url: str, token: str) -> str:
    """Return the email-verification deep link."""
    return f"{base_url.rstrip('/')}/verify-email?token={token}"


def reset_url(base_url: str, token: str) -> str:
    """Return the password-reset deep link."""
    return f"{base_url.rstrip('/')}/reset-password?token={token}"


async def send_verification_email(
    *,
    to: str,
    token: str,
    settings: IAMSettings | None = None,
) -> None:
    """Send the email-verification email with a one-time token link."""
    settings = settings or get_settings()
    base = settings.public_base_url
    text = (
        "Verify your Open Deep Research account.\n\n"
        f"Open this link to confirm your email address (valid 24 hours):\n{verification_url(base, token)}\n\n"
        "If you did not create an account, you can ignore this email."
    )
    await get_mail_sender(settings).send(to=to, subject="Verify your email", text_body=text)


async def send_password_reset_email(
    *,
    to: str,
    token: str,
    settings: IAMSettings | None = None,
) -> None:
    """Send the password-reset email with a one-time token link."""
    settings = settings or get_settings()
    base = settings.public_base_url
    text = (
        "Reset your Open Deep Research password.\n\n"
        f"Open this link to choose a new password (valid 30 minutes):\n{reset_url(base, token)}\n\n"
        "If you did not request a reset, your account is safe and you can ignore this email."
    )
    await get_mail_sender(settings).send(to=to, subject="Reset your password", text_body=text)
