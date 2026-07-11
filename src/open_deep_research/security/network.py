"""SSRF-resistant URL validation shared by HTTP fetch tools."""

from __future__ import annotations

import asyncio
import ipaddress
import socket
from urllib.parse import urlparse

_BLOCKED_HOSTS = {"localhost", "localhost.localdomain", "metadata.google.internal"}


def _is_forbidden_ip(value: str) -> bool:
    address = ipaddress.ip_address(value)
    return bool(
        address.is_private
        or address.is_loopback
        or address.is_link_local
        or address.is_multicast
        or address.is_reserved
        or address.is_unspecified
    )


async def validate_public_http_url(url: str) -> str:
    """Validate scheme, credentials, host, port, and every resolved address."""
    try:
        parsed = urlparse(url)
    except ValueError as exc:
        raise ValueError("Invalid URL") from exc
    if parsed.scheme.lower() not in {"http", "https"}:
        raise ValueError("Only http and https URLs are allowed")
    if not parsed.hostname:
        raise ValueError("URL must include a hostname")
    if parsed.username or parsed.password:
        raise ValueError("URL userinfo is not allowed")
    host = parsed.hostname.rstrip(".").lower()
    if host in _BLOCKED_HOSTS or host.endswith(".localhost"):
        raise ValueError("Local and metadata hosts are not allowed")
    try:
        if _is_forbidden_ip(host):
            raise ValueError("Private, local, and reserved addresses are not allowed")
        return url
    except ValueError as exc:
        # ip_address raises ValueError for ordinary DNS names. Re-raise only our
        # explicit policy error.
        if "not allowed" in str(exc):
            raise

    port = parsed.port or (443 if parsed.scheme.lower() == "https" else 80)
    try:
        addresses = await asyncio.to_thread(
            socket.getaddrinfo,
            host,
            port,
            type=socket.SOCK_STREAM,
        )
    except OSError as exc:
        raise ValueError("URL hostname could not be resolved") from exc
    resolved = {str(item[4][0]) for item in addresses if item and item[4]}
    if not resolved or any(_is_forbidden_ip(value) for value in resolved):
        raise ValueError("Hostname resolves to a private, local, or reserved address")
    return url
