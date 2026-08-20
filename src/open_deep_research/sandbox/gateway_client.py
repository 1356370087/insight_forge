"""API-side trusted control client for sandbox-gateway."""

from __future__ import annotations

import secrets
import time
from typing import Any

import httpx

from open_deep_research.configuration import Configuration
from open_deep_research.sandbox.crypto import SandboxDerivedKeys, sign_payload
from open_deep_research.sandbox.gateway import (
    GatewayRunRegistrationRequest,
    GatewayRunUnregisterRequest,
)


def split_gateway_registration(
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, str]]:
    """Separate frozen non-secrets from ephemeral per-run credentials."""
    configurable = dict(config.get("configurable", {}))
    configurable.pop("sandbox_root_signing_key", None)
    raw_api_keys = configurable.pop("apiKeys", {})
    credentials = (
        {
            str(key): str(value)
            for key, value in raw_api_keys.items()
            if isinstance(value, str)
        }
        if isinstance(raw_api_keys, dict)
        else {}
    )
    for key in list(configurable):
        normalized = key.lower()
        if normalized == "mcp_subject_token" or normalized.endswith(
            ("_api_key", "_secret_key", "_access_token", "_auth_token")
        ):
            value = configurable.pop(key)
            if isinstance(value, str) and value:
                credentials[key] = value
    return (
        {
            "configurable": configurable,
            "metadata": dict(config.get("metadata", {})),
        },
        credentials,
    )


class SandboxGatewayControlClient:
    """Register frozen run configuration and ephemeral credentials in Gateway."""

    def __init__(self, configurable: Configuration) -> None:
        """Initialize the trusted Gateway control client."""
        self.configurable = configurable
        self.keys = SandboxDerivedKeys.from_root(configurable.sandbox_root_signing_key or "")

    async def register_run(
        self,
        *,
        run_id: str,
        fence_token: int,
        frozen_config: dict[str, Any],
        api_keys: dict[str, str],
    ) -> None:
        """Register one frozen run and its non-persistent OAP credentials."""
        request = GatewayRunRegistrationRequest(
            run_id=run_id,
            fence_token=fence_token,
            frozen_config=frozen_config,
            api_keys=api_keys,
            service_timestamp=time.time(),
            service_nonce=secrets.token_urlsafe(24),
            service_signature="pending",
        )
        request.service_signature = sign_payload(
            request.signed_payload(), self.keys.service_auth
        )
        async with httpx.AsyncClient(
            base_url=self.configurable.sandbox_gateway_url,
            timeout=30,
        ) as client:
            response = await client.post(
                "/internal/v1/runs/register",
                content=request.model_dump_json(),
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()

    async def unregister_run(self, *, run_id: str, fence_token: int) -> None:
        """Erase one run's ephemeral Credential Vault entry idempotently."""
        request = GatewayRunUnregisterRequest(
            run_id=run_id,
            fence_token=fence_token,
            service_timestamp=time.time(),
            service_nonce=secrets.token_urlsafe(24),
            service_signature="pending",
        )
        request.service_signature = sign_payload(
            request.signed_payload(), self.keys.service_auth
        )
        async with httpx.AsyncClient(
            base_url=self.configurable.sandbox_gateway_url,
            timeout=30,
        ) as client:
            response = await client.post(
                "/internal/v1/runs/unregister",
                content=request.model_dump_json(),
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()
