"""API-side client for the Docker sandbox controller Unix socket."""

from __future__ import annotations

import secrets
import time

import httpx
from pydantic import BaseModel

from open_deep_research.configuration import Configuration
from open_deep_research.sandbox.controller import (
    ControllerActiveTask,
    ControllerCollectResponse,
    ControllerCreateResponse,
    ControllerReconcileRequest,
    ControllerRunRequest,
    ControllerStatusResponse,
    ControllerStopRequest,
    ControllerTaskRequest,
)
from open_deep_research.sandbox.crypto import SandboxDerivedKeys, sign_payload
from open_deep_research.sandbox.schema import SandboxPolicyBundle, policy_digest
from open_deep_research.sandbox.wire import SandboxTaskPayloadV1


class SandboxControllerClient:
    """Authenticated async client; it never opens Docker Socket itself."""

    def __init__(
        self,
        configurable: Configuration,
        bundle: SandboxPolicyBundle,
    ) -> None:
        """Initialize a UDS transport and derived service keys."""
        self.configurable = configurable
        self.bundle = bundle
        self.keys = SandboxDerivedKeys.from_root(configurable.sandbox_root_signing_key or "")
        self.transport = httpx.AsyncHTTPTransport(uds=configurable.sandbox_controller_socket)

    async def _post(
        self,
        path: str,
        request: BaseModel,
        *,
        timeout: float | None = 30,
    ) -> dict:
        async with httpx.AsyncClient(
            transport=self.transport,
            base_url="http://sandbox-controller",
            timeout=timeout,
        ) as client:
            response = await client.post(
                path,
                content=request.model_dump_json(),
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()
            return dict(response.json())

    def _task_request(self, container_id: str) -> ControllerTaskRequest:
        request = ControllerTaskRequest(
            container_id=container_id,
            deployment_id=self.bundle.deployment_id,
            service_timestamp=time.time(),
            service_nonce=secrets.token_urlsafe(24),
            service_signature="pending",
        )
        request.service_signature = sign_payload(
            request.signed_payload(), self.keys.service_auth
        )
        return request

    async def create_task(
        self,
        *,
        payload: SandboxTaskPayloadV1,
        task_token: str,
        runtime_digest_value: str,
    ) -> ControllerCreateResponse:
        """Create one authenticated task and return its container identity."""
        policy_signature = sign_payload(
            {
                "deployment_id": self.bundle.deployment_id,
                "policy_digest": policy_digest(self.bundle),
            },
            self.keys.policy_signature,
        )
        request = ControllerRunRequest(
            payload=payload,
            task_token=task_token,
            runtime_digest=runtime_digest_value,
            policy_signature=policy_signature,
            service_timestamp=time.time(),
            service_nonce=secrets.token_urlsafe(24),
            service_signature="pending",
        )
        request.service_signature = sign_payload(
            request.signed_payload(), self.keys.service_auth
        )
        response = await self._post("/v1/tasks/create", request)
        return ControllerCreateResponse.model_validate(response)

    async def start_task(self, container_id: str) -> ControllerStatusResponse:
        """Start one created task idempotently."""
        response = await self._post(
            "/v1/tasks/start", self._task_request(container_id)
        )
        return ControllerStatusResponse.model_validate(response)

    async def task_status(self, container_id: str) -> ControllerStatusResponse:
        """Read one task status without blocking for completion."""
        response = await self._post(
            "/v1/tasks/status", self._task_request(container_id)
        )
        return ControllerStatusResponse.model_validate(response)

    async def collect_task(self, container_id: str) -> ControllerCollectResponse:
        """Collect one terminal task result and apply Controller retention."""
        response = await self._post(
            "/v1/tasks/collect", self._task_request(container_id)
        )
        return ControllerCollectResponse.model_validate(response)

    async def collect_archive(self, container_id: str) -> bytes:
        """Download the Controller-validated canonical task archive."""
        request = self._task_request(container_id)
        async with httpx.AsyncClient(
            transport=self.transport,
            base_url="http://sandbox-controller",
            timeout=httpx.Timeout(120.0, connect=5.0),
        ) as client:
            response = await client.post(
                "/v1/tasks/archive",
                content=request.model_dump_json(),
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()
            return bytes(response.content)

    async def stop_task(
        self,
        container_id: str,
        *,
        timeout_seconds: int = 5,
    ) -> None:
        """Stop one deployment-owned container through Controller."""
        request = ControllerStopRequest(
            container_id=container_id,
            deployment_id=self.bundle.deployment_id,
            timeout_seconds=timeout_seconds,
            service_timestamp=time.time(),
            service_nonce=secrets.token_urlsafe(24),
            service_signature="pending",
        )
        request.service_signature = sign_payload(
            request.signed_payload(), self.keys.service_auth
        )
        await self._post("/v1/tasks/stop", request)

    async def reconcile_tasks(
        self,
        active_tasks: list[ControllerActiveTask],
    ) -> list[str]:
        """Stop deployment-owned Workers not present in the API ownership set."""
        request = ControllerReconcileRequest(
            deployment_id=self.bundle.deployment_id,
            active_tasks=active_tasks,
            service_timestamp=time.time(),
            service_nonce=secrets.token_urlsafe(24),
            service_signature="pending",
        )
        request.service_signature = sign_payload(
            request.signed_payload(), self.keys.service_auth
        )
        response = await self._post("/v1/tasks/reconcile", request)
        return [str(value) for value in response.get("stopped_container_ids", [])]
