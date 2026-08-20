"""Credential-owning sandbox Gateway for model and governed network operations."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import time
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any

import httpx
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, message_to_dict, messages_from_dict
from pydantic import ConfigDict, Field

from open_deep_research.configuration import Configuration
from open_deep_research.models.fallback import (
    ModelErrorKind,
    classify_model_error,
    invoke_with_model_fallback,
)
from open_deep_research.models.resolution import (
    build_model_config,
    get_configurable_model_template,
)
from open_deep_research.observability import (
    apply_helicone_config,
    invoke_model_with_retry_observability,
)
from open_deep_research.sandbox.approvals import SecurityApproval, SecurityApprovalStore
from open_deep_research.sandbox.crypto import (
    NonceReplayCache,
    SandboxDerivedKeys,
    decode_task_token,
    validate_timestamp,
    verify_payload,
)
from open_deep_research.sandbox.internal_api import (
    ApprovalConsumeRequest,
    ApprovalCreateRequest,
    ApprovalWaitRequest,
    BudgetFailRequest,
    BudgetReserveRequest,
    BudgetSettleRequest,
    OperationGetRequest,
    OperationTransitionRequest,
    SandboxInternalClient,
    ServiceRequest,
    ToolBudgetReserveRequest,
    ToolBudgetSettleRequest,
)
from open_deep_research.sandbox.policy import egress_target_from_url
from open_deep_research.sandbox.schema import (
    command_policy_decision,
    filesystem_path_allowed,
    network_target_decision,
    resolve_profile,
    tool_policy_decision,
)
from open_deep_research.sandbox.wire import (
    GatewayCatalogToolV1,
    GatewayModelOutcomeV1,
    GatewayModelRequestV1,
    GatewayOperationLookupOutcomeV1,
    GatewayOperationLookupRequestV1,
    GatewayToolCatalogOutcomeV1,
    GatewayToolCatalogRequestV1,
    GatewayToolOutcomeV1,
    GatewayToolRequestV1,
)

logger = logging.getLogger(__name__)


class GatewayRunRegistrationRequest(ServiceRequest):
    """Register one frozen run and its ephemeral OAP credentials."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    fence_token: int = Field(ge=1)
    frozen_config: dict[str, Any]
    api_keys: dict[str, str] = Field(default_factory=dict)


class GatewayRunUnregisterRequest(ServiceRequest):
    """Erase one run's in-memory credentials after its ownership epoch ends."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    fence_token: int = Field(ge=1)


@dataclass(slots=True)
class GatewayRunContext:
    """In-memory run configuration and credentials; never serialized by Gateway."""

    config: dict[str, Any]
    fence_token: int
    api_keys: dict[str, str] = field(default_factory=dict)
    registered_at: float = field(default_factory=time.time)


class RemoteBudgetGate:
    """Synchronous BudgetGate-compatible adapter backed by the API authority."""

    def __init__(
        self,
        *,
        internal: SandboxInternalClient,
        run_id: str,
        task_id: str,
        fence_token: int,
        stage: str,
        logical_operation_id: str,
        initial_attempt_count: int = 0,
    ) -> None:
        """Bind one logical operation to the API budget authority."""
        self.internal = internal
        self.run_id = run_id
        self.task_id = task_id
        self.fence_token = fence_token
        self.stage = stage
        self.logical_operation_id = logical_operation_id
        self._counter = max(0, initial_attempt_count)
        self._keys: dict[str, str] = {}
        self._pending_posts: list[tuple[str, ServiceRequest]] = []
        self.last_physical_attempt_id = ""

    def check_deadline(self, stage: str) -> None:
        """Delegate deadline enforcement to the API reserve endpoint."""
        del stage

    def _enqueue(self, path: str, request: ServiceRequest) -> None:
        self._pending_posts.append((path, request))

    async def flush_pending(self) -> None:
        """Await queued authority calls without blocking the Gateway event loop."""
        while self._pending_posts:
            path, request = self._pending_posts[0]
            await self.internal.post(path, request)
            del self._pending_posts[0]

    def reserve_model_call(
        self,
        op_key: str,
        *,
        estimated_input_tokens: int,
        estimated_output_tokens: int,
        model_name: str,
    ) -> None:
        """Reserve one deterministic physical model attempt before dispatch."""
        self._counter += 1
        physical_id = hashlib.sha256(
            f"{self.logical_operation_id}:{self._counter}".encode()
        ).hexdigest()
        self._keys[op_key] = physical_id
        self.last_physical_attempt_id = physical_id
        request = self.internal.signed(
            BudgetReserveRequest,
            run_id=self.run_id,
            task_id=self.task_id,
            fence_token=self.fence_token,
            stage=self.stage,
            logical_operation_id=self.logical_operation_id,
            physical_attempt_id=physical_id,
            model_name=model_name,
            estimated_input_tokens=max(1, estimated_input_tokens),
            estimated_output_tokens=max(1, estimated_output_tokens),
        )
        self._enqueue("/internal/sandbox/budgets/reserve", request)
        transition = self.internal.signed(
            OperationTransitionRequest,
            run_id=self.run_id,
            fence_token=self.fence_token,
            logical_operation_id=self.logical_operation_id,
            status="dispatched",
            outcome=None,
            error_type=None,
        )
        self._enqueue("/internal/sandbox/operations/transition", transition)

    def settle_model_call(
        self,
        op_key: str,
        *,
        input_tokens: int,
        output_tokens: int,
        model_name: str,
    ) -> None:
        """Settle one physical model attempt with provider usage."""
        physical_id = self._keys[op_key]
        request = self.internal.signed(
            BudgetSettleRequest,
            run_id=self.run_id,
            fence_token=self.fence_token,
            physical_attempt_id=physical_id,
            model_name=model_name,
            input_tokens=max(0, input_tokens),
            output_tokens=max(0, output_tokens),
        )
        self._enqueue("/internal/sandbox/budgets/settle", request)

    def fail_model_call(self, op_key: str, *, uncertain: bool) -> None:
        """Release or conservatively mark a failed physical attempt."""
        physical_id = self._keys.get(op_key)
        if not physical_id:
            return
        request = self.internal.signed(
            BudgetFailRequest,
            run_id=self.run_id,
            fence_token=self.fence_token,
            physical_attempt_id=physical_id,
            uncertain=uncertain,
        )
        self._enqueue("/internal/sandbox/budgets/fail", request)


class GatewayRuntime:
    """Run registry, token verifier and physical model invocation owner."""

    def __init__(self, configurable: Configuration) -> None:
        """Initialize credentials, run registry and API authority client."""
        self.configurable = configurable
        self.keys = SandboxDerivedKeys.from_root(configurable.sandbox_root_signing_key or "")
        self.runs: dict[str, GatewayRunContext] = {}
        self.operation_locks: dict[tuple[str, str], asyncio.Lock] = {}
        self.nonces = NonceReplayCache()
        self.internal = SandboxInternalClient(
            os.getenv("SANDBOX_API_INTERNAL_URL", "http://api:2024"),
            configurable.sandbox_root_signing_key or "",
        )

    def register(self, request: GatewayRunRegistrationRequest) -> None:
        """Register a signed frozen run and ephemeral credentials in memory."""
        validate_timestamp(request.service_timestamp)
        if not verify_payload(
            request.signed_payload(), request.service_signature, self.keys.service_auth
        ):
            raise ValueError("sandbox_service_auth_invalid")
        self.nonces.consume(
            f"gateway-register:{request.run_id}",
            request.service_nonce,
            expires_at=time.time() + 60,
        )
        existing = self.runs.get(request.run_id)
        if existing is not None and existing.fence_token > request.fence_token:
            raise ValueError("stale_fence")
        if existing is not None and existing.fence_token == request.fence_token:
            current_fingerprint = str(
                existing.config.get("metadata", {}).get("run_config_fingerprint")
                or ""
            )
            next_fingerprint = str(
                request.frozen_config.get("metadata", {}).get(
                    "run_config_fingerprint"
                )
                or ""
            )
            if (
                current_fingerprint
                and next_fingerprint
                and current_fingerprint != next_fingerprint
            ):
                raise ValueError("sandbox_gateway_frozen_config_mismatch")
        config = {
            "configurable": dict(request.frozen_config.get("configurable") or {}),
            "metadata": {
                **dict(request.frozen_config.get("metadata") or {}),
                "sandbox_gateway_physical": True,
                "run_fence_token": request.fence_token,
            },
        }
        # Physical observability is returned to the API; Gateway never opens .runs.
        config["configurable"].update(
            {
                "token_usage_accounting_enabled": False,
                "sqlite_observability_enabled": False,
                "event_log_enabled": False,
                "apiKeys": dict(request.api_keys),
                "_sandbox_credential_vault": {},
            }
        )
        if request.api_keys.get("mcp_subject_token"):
            config["configurable"]["mcp_subject_token"] = request.api_keys[
                "mcp_subject_token"
            ]
        if existing is not None and existing.fence_token == request.fence_token:
            prior_vault = existing.config.get("configurable", {}).get(
                "_sandbox_credential_vault"
            )
            if isinstance(prior_vault, dict):
                config["configurable"]["_sandbox_credential_vault"] = prior_vault
        self.runs[request.run_id] = GatewayRunContext(
            config=config,
            fence_token=request.fence_token,
            api_keys=dict(request.api_keys),
            registered_at=(
                existing.registered_at
                if existing is not None
                and existing.fence_token == request.fence_token
                else time.time()
            ),
        )

    def unregister(self, request: GatewayRunUnregisterRequest) -> None:
        """Replay-protected deletion of ephemeral run credentials and locks."""
        validate_timestamp(request.service_timestamp)
        if not verify_payload(
            request.signed_payload(), request.service_signature, self.keys.service_auth
        ):
            raise ValueError("sandbox_service_auth_invalid")
        self.nonces.consume(
            f"gateway-unregister:{request.run_id}",
            request.service_nonce,
            expires_at=time.time() + 60,
        )
        context = self.runs.get(request.run_id)
        if context is not None and context.fence_token != request.fence_token:
            raise ValueError("stale_fence")
        self.runs.pop(request.run_id, None)
        for key in [key for key in self.operation_locks if key[0] == request.run_id]:
            self.operation_locks.pop(key, None)

    def authorize_task(
        self,
        request: (
            GatewayModelRequestV1
            | GatewayToolRequestV1
            | GatewayOperationLookupRequestV1
            | GatewayToolCatalogRequestV1
        ),
        *,
        authorization: str,
        timestamp: float,
        nonce: str,
    ) -> tuple[Any, GatewayRunContext]:
        """Validate task claims, timestamp, nonce and live ownership epoch."""
        validate_timestamp(timestamp)
        if not authorization.startswith("Bearer "):
            raise ValueError("sandbox_task_token_missing")
        claims = decode_task_token(authorization[7:], self.keys.task_token)
        self.nonces.consume(claims.jti, nonce, expires_at=claims.expires_at)
        context = self.runs.get(request.run_id)
        if context is None:
            raise ValueError("sandbox_gateway_run_not_registered")
        if (
            claims.run_id != request.run_id
            or claims.task_id != request.task_id
            or claims.fence_token != context.fence_token
        ):
            raise ValueError("sandbox_task_token_claim_mismatch")
        return claims, context

    def authorize_api_model(
        self,
        request: GatewayModelRequestV1 | GatewayOperationLookupRequestV1,
        *,
        timestamp: float,
        nonce: str,
        fence_token: int,
        signature: str,
    ) -> GatewayRunContext:
        """Authenticate one trusted API model request with the service key."""
        validate_timestamp(timestamp)
        context = self.runs.get(request.run_id)
        if context is None:
            raise ValueError("sandbox_gateway_run_not_registered")
        if context.fence_token != fence_token:
            raise ValueError("stale_fence")
        signed = {
            "request": request.model_dump(mode="json"),
            "timestamp": timestamp,
            "nonce": nonce,
            "fence_token": fence_token,
        }
        if not verify_payload(signed, signature, self.keys.service_auth):
            raise ValueError("sandbox_service_auth_invalid")
        self.nonces.consume(
            f"api-model:{request.run_id}:{fence_token}",
            nonce,
            expires_at=time.time() + 60,
        )
        return context

    async def lookup_model_operation(
        self,
        request: GatewayOperationLookupRequestV1,
        context: GatewayRunContext,
    ) -> GatewayOperationLookupOutcomeV1:
        """Read a journaled model outcome without dispatching a Provider call."""
        lookup = self.internal.signed(
            OperationGetRequest,
            run_id=request.run_id,
            fence_token=context.fence_token,
            logical_operation_id=request.logical_operation_id,
        )
        existing = await self.internal.post(
            "/internal/sandbox/operations/get",
            lookup,
        )
        operation = existing.get("operation") if existing.get("found") else None
        raw_outcome = operation.get("outcome") if isinstance(operation, dict) else None
        return GatewayOperationLookupOutcomeV1(
            found=raw_outcome is not None,
            outcome=(
                GatewayModelOutcomeV1.model_validate(raw_outcome)
                if raw_outcome is not None
                else None
            ),
        )

    async def tool_catalog(
        self,
        request: GatewayToolCatalogRequestV1,
        context: GatewayRunContext,
    ) -> GatewayToolCatalogOutcomeV1:
        """Return permission-filtered Gateway tool schemas, never implementations."""
        from open_deep_research.tools.base import (
            ToolExecutionZone,
            tool_to_model_definition,
        )
        from open_deep_research.tools.governance import (
            AgentRole,
            filter_tools_by_permission,
        )
        from open_deep_research.tools.registry import assemble_toolset

        role = AgentRole(request.role)
        assembled = await assemble_toolset(role, context.config)
        permitted = filter_tools_by_permission(assembled, role, context.config)
        configuration = Configuration.from_runnable_config(context.config)
        catalog: list[GatewayCatalogToolV1] = []
        for tool in permitted:
            if tool.execution_zone is not ToolExecutionZone.GATEWAY:
                continue
            definition = await tool_to_model_definition(
                tool,
                max_description_chars=configuration.max_tool_description_chars,
            )
            prompt = tool.prompt(context.config)
            catalog.append(
                GatewayCatalogToolV1(
                    name=tool.name,
                    definition=definition,
                    prompt=(
                        str(prompt)[: configuration.max_tool_description_chars]
                        if prompt
                        else None
                    ),
                    origin=getattr(tool.origin, "value", str(tool.origin)),
                    effect=getattr(tool.effect, "value", str(tool.effect)),
                    retryable=tool.retryable,
                    concurrency_safe=tool.concurrency_safe,
                    max_output_chars=tool.max_output_chars,
                )
            )
        return GatewayToolCatalogOutcomeV1(tools=catalog)

    async def _wait_for_approval(
        self,
        request: GatewayToolRequestV1,
        context: GatewayRunContext,
        *,
        kind: str,
        capability: str,
        target: dict[str, Any],
        expires_at: float,
    ) -> SecurityApproval:
        """Reuse or durably wait for one run-scoped security decision."""
        fingerprint = SecurityApprovalStore.fingerprint(kind, capability, target)
        current_request = self.internal.signed(
            ApprovalWaitRequest,
            run_id=request.run_id,
            fence_token=context.fence_token,
            after_version=0,
            timeout_seconds=0.1,
        )
        current = await self.internal.post(
            "/internal/sandbox/approvals/wait", current_request
        )
        approval = next(
            (
                SecurityApproval.model_validate(item)
                for item in current.get("approvals", [])
                if item.get("target_fingerprint") == fingerprint
                and item.get("fence_token") == context.fence_token
                and (
                    item.get("decision") == "allow_run"
                    or (
                        item.get("decision") == "allow_once"
                        and item.get("status") == "resolved"
                        and item.get("operation_id")
                        == request.logical_operation_id
                    )
                )
            ),
            None,
        )
        if approval is None:
            create = self.internal.signed(
                ApprovalCreateRequest,
                run_id=request.run_id,
                task_id=request.task_id,
                fence_token=context.fence_token,
                kind=kind,
                capability=capability,
                target=target,
                operation_id=request.logical_operation_id,
                expires_at=expires_at,
                stage=request.stage,
            )
            approval = SecurityApproval.model_validate(
                await self.internal.post(
                    "/internal/sandbox/approvals/request", create
                )
            )
            version = int(current.get("version", 0))
            while approval.status == "pending" and time.time() < expires_at:
                wait = self.internal.signed(
                    ApprovalWaitRequest,
                    run_id=request.run_id,
                    fence_token=context.fence_token,
                    after_version=version,
                    timeout_seconds=25,
                )
                update = await self.internal.post(
                    "/internal/sandbox/approvals/wait", wait
                )
                version = int(update.get("version", version))
                match = next(
                    (
                        item
                        for item in update.get("approvals", [])
                        if item.get("approval_id") == approval.approval_id
                    ),
                    None,
                )
                if match is not None:
                    approval = SecurityApproval.model_validate(match)
        if approval.decision in {"allow_once", "allow_run"}:
            consume = self.internal.signed(
                ApprovalConsumeRequest,
                run_id=request.run_id,
                fence_token=context.fence_token,
                approval_id=approval.approval_id,
                operation_id=request.logical_operation_id,
            )
            await self.internal.post(
                "/internal/sandbox/approvals/consume", consume
            )
        return approval

    async def invoke_tool(
        self,
        request: GatewayToolRequestV1,
        context: GatewayRunContext,
    ) -> GatewayToolOutcomeV1:
        """Execute one authoritative Gateway-zone tool operation."""
        from open_deep_research.tools.governance import (
            AgentRole,
            execute_governed_tool_call,
        )
        from open_deep_research.tools.registry import assemble_toolset

        if request.execution_zone != "gateway":
            return GatewayToolOutcomeV1(
                logical_operation_id=request.logical_operation_id,
                tool_call_id=request.tool_call_id,
                status="failed",
                error={
                    "error_type": "tool_execution_zone_mismatch",
                    "message": "Gateway execution RPC requires zone=gateway.",
                },
            )
        role = AgentRole(request.role)
        tools = await assemble_toolset(role, context.config)
        tools_by_name = {tool.name: tool for tool in tools}
        tool = tools_by_name.get(request.tool_name)
        if tool is None:
            return GatewayToolOutcomeV1(
                logical_operation_id=request.logical_operation_id,
                tool_call_id=request.tool_call_id,
                status="failed",
                error={"error_type": "tool_not_found", "message": "Tool is not registered in Gateway."},
            )
        from open_deep_research.tools.base import ToolExecutionZone

        if tool.execution_zone is not ToolExecutionZone.GATEWAY:
            return GatewayToolOutcomeV1(
                logical_operation_id=request.logical_operation_id,
                tool_call_id=request.tool_call_id,
                status="failed",
                error={
                    "error_type": "tool_execution_zone_mismatch",
                    "message": "Sandbox-local tools require authorization-only RPC.",
                },
            )
        configuration = Configuration.from_runnable_config(context.config)
        _bundle, _profile_id, profile = resolve_profile(configuration)
        effect = getattr(tool.effect, "value", str(tool.effect))
        profile_tool_decision = tool_policy_decision(
            profile,
            tool_name=tool.name,
            effect=effect,
        )
        approval_deadline = min(
            time.time() + profile.resources.approval_timeout_seconds,
            context.registered_at
            + float(
                configuration.run_deadline_seconds
                or profile.resources.approval_timeout_seconds
            ),
        )
        if profile_tool_decision == "deny" or (
            profile_tool_decision == "ask"
            and profile.approval_policy == "never"
        ):
            return GatewayToolOutcomeV1(
                logical_operation_id=request.logical_operation_id,
                tool_call_id=request.tool_call_id,
                status="failed",
                error={
                    "error_type": "sandbox_tool_policy_denied",
                    "message": f"Tool '{tool.name}' is denied by the sandbox profile.",
                },
            )
        if profile_tool_decision == "ask":
            tool_approval = await self._wait_for_approval(
                request,
                context,
                kind="tool_effect",
                capability=f"tool.execute:{tool.name}",
                target={
                    "tool": tool.name,
                    "effect": effect,
                    "execution_zone": getattr(
                        tool.execution_zone,
                        "value",
                        str(tool.execution_zone),
                    ),
                },
                expires_at=approval_deadline,
            )
            if tool_approval.decision not in {"allow_once", "allow_run"}:
                return GatewayToolOutcomeV1(
                    logical_operation_id=request.logical_operation_id,
                    tool_call_id=request.tool_call_id,
                    status="failed",
                    error={
                        "error_type": "sandbox_tool_policy_denied",
                        "message": f"Tool '{tool.name}' was denied or approval timed out.",
                    },
                )

        authorized_hosts: list[str] = []
        raw_urls = tool.egress_urls(request.arguments)
        parsed_targets = [egress_target_from_url(url) for url in raw_urls]
        targets = sorted({target for target in parsed_targets if target is not None})
        if any(target is None for target in parsed_targets):
            return GatewayToolOutcomeV1(
                logical_operation_id=request.logical_operation_id,
                tool_call_id=request.tool_call_id,
                status="failed",
                error={
                    "error_type": "egress_target_invalid",
                    "message": "Tool declared an invalid outbound target.",
                },
            )
        if targets:
            for host, port in targets:
                target_decision = network_target_decision(
                    profile.network,
                    host,
                    port,
                )
                if target_decision == "allow":
                    authorized_hosts.append(host)
                    continue
                if target_decision == "deny" or profile.approval_policy == "never":
                    return GatewayToolOutcomeV1(
                        logical_operation_id=request.logical_operation_id,
                        tool_call_id=request.tool_call_id,
                        status="failed",
                        error={
                            "error_type": "egress_domain_denied",
                            "message": f"Target '{host}:{port}' is not allowed by the sandbox profile.",
                            "domain": host,
                            "port": port,
                        },
                    )
                target = {"domain": host, "port": port}
                approval = await self._wait_for_approval(
                    request,
                    context,
                    kind="network",
                    capability="tool.egress",
                    target=target,
                    expires_at=approval_deadline,
                )
                if approval.decision not in {"allow_once", "allow_run"}:
                    return GatewayToolOutcomeV1(
                        logical_operation_id=request.logical_operation_id,
                        tool_call_id=request.tool_call_id,
                        status="failed",
                        error={
                            "error_type": "egress_domain_denied",
                            "message": f"Domain '{host}' was denied or approval timed out.",
                            "domain": host,
                            "port": port,
                        },
                    )
                authorized_hosts.append(host)
        from open_deep_research.security.network import validate_public_http_url

        try:
            for raw_url in raw_urls:
                await validate_public_http_url(raw_url)
        except ValueError as exc:
            return GatewayToolOutcomeV1(
                logical_operation_id=request.logical_operation_id,
                tool_call_id=request.tool_call_id,
                status="failed",
                error={
                    "error_type": "sandbox_private_destination_denied",
                    "message": str(exc)[:500],
                },
            )
        budget_request = self.internal.signed(
            ToolBudgetReserveRequest,
            run_id=request.run_id,
            task_id=request.task_id,
            fence_token=context.fence_token,
            stage=request.stage,
            logical_operation_id=request.logical_operation_id,
        )
        await self.internal.post(
            "/internal/sandbox/budgets/tool-reserve", budget_request
        )
        call = {
            "name": request.tool_name,
            "args": request.arguments,
            "id": request.tool_call_id,
        }
        execution_config = {
            **context.config,
            "metadata": {
                **context.config.get("metadata", {}),
                "sandbox_gateway_authorized_hosts": authorized_hosts,
            },
        }
        governed = await execute_governed_tool_call(
            call,
            tools_by_name,
            role,
            execution_config,
            operation_id=request.logical_operation_id,
        )
        if (
            governed.error is not None
            and governed.error.error_type.value == "interaction_required"
            and governed.error.detail.get("interaction_url")
        ):
            interaction_url = str(governed.error.detail["interaction_url"])
            approval = await self._wait_for_approval(
                request,
                context,
                kind="mcp_oauth",
                capability=f"mcp.oauth:{request.tool_name}",
                target={"url": interaction_url, "tool": request.tool_name},
                expires_at=approval_deadline,
            )
            if approval.decision in {"allow_once", "allow_run"}:
                governed = await execute_governed_tool_call(
                    call,
                    tools_by_name,
                    role,
                    execution_config,
                    operation_id=request.logical_operation_id,
                )
        settle_request = self.internal.signed(
            ToolBudgetSettleRequest,
            run_id=request.run_id,
            fence_token=context.fence_token,
            logical_operation_id=request.logical_operation_id,
        )
        await self.internal.post(
            "/internal/sandbox/budgets/tool-settle",
            settle_request,
        )
        if governed.error is not None:
            return GatewayToolOutcomeV1(
                logical_operation_id=request.logical_operation_id,
                tool_call_id=request.tool_call_id,
                status="failed",
                error=governed.error.model_dump(mode="json"),
            )
        return GatewayToolOutcomeV1(
            logical_operation_id=request.logical_operation_id,
            tool_call_id=request.tool_call_id,
            status="completed",
            output=governed.result.output if governed.result is not None else governed.message.content,
        )

    async def authorize_local_tool(
        self,
        request: GatewayToolRequestV1,
        context: GatewayRunContext,
    ) -> GatewayToolOutcomeV1:
        """Authorize, but never physically execute, a sandbox-local tool call."""
        from open_deep_research.security.redaction import redact_text
        from open_deep_research.tools.base import ToolExecutionZone
        from open_deep_research.tools.governance import (
            AgentRole,
            filter_tools_by_permission,
        )
        from open_deep_research.tools.registry import assemble_toolset

        if request.execution_zone != "sandbox_local":
            return GatewayToolOutcomeV1(
                logical_operation_id=request.logical_operation_id,
                tool_call_id=request.tool_call_id,
                status="failed",
                error={
                    "error_type": "tool_execution_zone_mismatch",
                    "message": "Local authorization RPC requires zone=sandbox_local.",
                },
            )
        role = AgentRole(request.role)
        tools = await assemble_toolset(role, context.config)
        tool = next((item for item in tools if item.name == request.tool_name), None)
        if tool is None or tool.execution_zone is not ToolExecutionZone.SANDBOX_LOCAL:
            return GatewayToolOutcomeV1(
                logical_operation_id=request.logical_operation_id,
                tool_call_id=request.tool_call_id,
                status="failed",
                error={
                    "error_type": "tool_execution_zone_mismatch",
                    "message": "Tool is not registered as sandbox-local.",
                },
            )
        permitted = filter_tools_by_permission([tool], role, context.config)
        if not permitted:
            return GatewayToolOutcomeV1(
                logical_operation_id=request.logical_operation_id,
                tool_call_id=request.tool_call_id,
                status="failed",
                error={
                    "error_type": "permission_denied",
                    "message": "Caller is not permitted to use this local tool.",
                },
            )
        try:
            tool.input_schema.model_validate(request.arguments)
        except Exception as exc:  # noqa: BLE001 - return a bounded schema denial
            return GatewayToolOutcomeV1(
                logical_operation_id=request.logical_operation_id,
                tool_call_id=request.tool_call_id,
                status="failed",
                error={
                    "error_type": "tool_input_invalid",
                    "message": str(exc)[:500],
                },
            )

        configuration = Configuration.from_runnable_config(context.config)
        _bundle, _profile_id, profile = resolve_profile(configuration)
        effect = getattr(tool.effect, "value", str(tool.effect))
        decisions = [
            tool_policy_decision(
                profile,
                tool_name=tool.name,
                effect=effect,
            )
        ]
        approval_kind = "tool_effect"
        target: dict[str, Any] = {
            "tool": tool.name,
            "effect": effect,
            "arguments_digest": hashlib.sha256(
                json.dumps(
                    request.arguments,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest(),
        }
        if tool.name == "ShellExec":
            command = str(request.arguments.get("command") or "")
            decisions.append(command_policy_decision(profile, command))
            approval_kind = "command"
            target.update(
                {
                    "command_preview": redact_text(command)[:1000],
                    "cwd": str(request.arguments.get("cwd") or ".")[:1024],
                }
            )
        elif tool.name in {"ReadFile", "WriteFile"}:
            path = str(request.arguments.get("path") or "")
            write = tool.name == "WriteFile"
            if not filesystem_path_allowed(profile, path, write=write):
                decisions.append("deny")
            approval_kind = "filesystem"
            target.update(
                {
                    "path": path[:1024],
                    "operation": "write" if write else "read",
                }
            )

        decision = (
            "deny"
            if "deny" in decisions
            else "ask"
            if "ask" in decisions
            else "allow"
        )
        if decision == "ask" and profile.approval_policy == "never":
            decision = "deny"
        if decision == "deny":
            return GatewayToolOutcomeV1(
                logical_operation_id=request.logical_operation_id,
                tool_call_id=request.tool_call_id,
                status="failed",
                error={
                    "error_type": "sandbox_local_tool_denied",
                    "message": f"Local tool '{tool.name}' is denied by the sandbox profile.",
                },
            )
        if decision == "ask":
            deadline = min(
                time.time() + profile.resources.approval_timeout_seconds,
                context.registered_at
                + float(
                    configuration.run_deadline_seconds
                    or profile.resources.approval_timeout_seconds
                ),
            )
            approval = await self._wait_for_approval(
                request,
                context,
                kind=approval_kind,
                capability=f"tool.execute:{tool.name}",
                target=target,
                expires_at=deadline,
            )
            if approval.decision not in {"allow_once", "allow_run"}:
                return GatewayToolOutcomeV1(
                    logical_operation_id=request.logical_operation_id,
                    tool_call_id=request.tool_call_id,
                    status="failed",
                    error={
                        "error_type": "sandbox_local_tool_denied",
                        "message": f"Local tool '{tool.name}' was denied or approval timed out.",
                    },
                )
        budget_request = self.internal.signed(
            ToolBudgetReserveRequest,
            run_id=request.run_id,
            task_id=request.task_id,
            fence_token=context.fence_token,
            stage=request.stage,
            logical_operation_id=request.logical_operation_id,
        )
        await self.internal.post(
            "/internal/sandbox/budgets/tool-reserve", budget_request
        )
        settle_request = self.internal.signed(
            ToolBudgetSettleRequest,
            run_id=request.run_id,
            fence_token=context.fence_token,
            logical_operation_id=request.logical_operation_id,
        )
        await self.internal.post(
            "/internal/sandbox/budgets/tool-settle",
            settle_request,
        )
        return GatewayToolOutcomeV1(
            logical_operation_id=request.logical_operation_id,
            tool_call_id=request.tool_call_id,
            status="completed",
            output={
                "authorized": True,
                "execution_zone": ToolExecutionZone.SANDBOX_LOCAL.value,
            },
        )

    @staticmethod
    def _role_settings(configuration: Configuration, role: str) -> tuple[str, int]:
        mapping = {
            "supervisor": (configuration.research_model, configuration.research_model_max_tokens),
            "researcher": (configuration.research_model, configuration.research_model_max_tokens),
            "summarization": (configuration.summarization_model, configuration.summarization_model_max_tokens),
            "message_summary": (configuration.message_summary_model, configuration.message_summary_model_max_tokens),
            "compression": (configuration.compression_model, configuration.compression_model_max_tokens),
            "final_report": (configuration.final_report_model, configuration.final_report_model_max_tokens),
            "quality_evaluation": (
                configuration.quality_evaluation_model,
                configuration.quality_evaluation_model_max_tokens,
            ),
            "quality_evaluator": (
                configuration.quality_evaluation_model,
                configuration.quality_evaluation_model_max_tokens,
            ),
        }
        if role not in mapping or not mapping[role][0]:
            raise ValueError(f"sandbox_gateway_unknown_model_role:{role}")
        return str(mapping[role][0]), int(mapping[role][1])

    @staticmethod
    def _usage(message: AIMessage) -> dict[str, int]:
        usage = getattr(message, "usage_metadata", None) or {}
        response_usage = message.response_metadata.get("token_usage", {})
        return {
            "input_tokens": int(usage.get("input_tokens", response_usage.get("prompt_tokens", 0)) or 0),
            "output_tokens": int(usage.get("output_tokens", response_usage.get("completion_tokens", 0)) or 0),
        }

    async def invoke_model_operation(
        self,
        request: GatewayModelRequestV1,
        context: GatewayRunContext,
    ) -> GatewayModelOutcomeV1:
        """Serialize duplicate logical operations before consulting the journal."""
        key = (request.run_id, request.logical_operation_id)
        lock = self.operation_locks.setdefault(key, asyncio.Lock())
        async with lock:
            return await self._invoke_model_operation_locked(request, context)

    async def _invoke_model_operation_locked(
        self,
        request: GatewayModelRequestV1,
        context: GatewayRunContext,
    ) -> GatewayModelOutcomeV1:
        """Execute or recover one idempotent logical model operation."""
        lookup = self.internal.signed(
            OperationGetRequest,
            run_id=request.run_id,
            fence_token=context.fence_token,
            logical_operation_id=request.logical_operation_id,
        )
        existing = await self.internal.post("/internal/sandbox/operations/get", lookup)
        prior_attempt_count = 0
        if existing.get("found"):
            operation = existing["operation"]
            prior_attempt_count = len(operation.get("physical_attempts") or [])
            if operation.get("status") == "completed" and operation.get("outcome"):
                return GatewayModelOutcomeV1.model_validate(operation["outcome"])
            if operation.get("status") == "uncertain" and operation.get("outcome"):
                return GatewayModelOutcomeV1.model_validate(operation["outcome"])
            if operation.get("status") == "failed" and operation.get("outcome"):
                failed = GatewayModelOutcomeV1.model_validate(operation["outcome"])
                if failed.error_type not in {
                    ModelErrorKind.RATE_LIMITED.value,
                    ModelErrorKind.TRANSIENT.value,
                    ModelErrorKind.MODEL_UNAVAILABLE.value,
                }:
                    return failed
            if operation.get("status") == "dispatched":
                return GatewayModelOutcomeV1(
                    logical_operation_id=request.logical_operation_id,
                    physical_attempt_id=str(operation.get("physical_attempt_id") or "unknown"),
                    status="uncertain",
                    error_type="model_operation_uncertain",
                    error_message="A prior physical dispatch has no terminal outcome.",
                )
        configuration = Configuration.from_runnable_config(context.config)
        primary_model, max_tokens = self._role_settings(configuration, request.role)
        messages = messages_from_dict(request.messages)
        fallback_events: list[dict[str, Any]] = []
        budget = RemoteBudgetGate(
            internal=self.internal,
            run_id=request.run_id,
            task_id=request.task_id,
            fence_token=context.fence_token,
            stage=request.stage,
            logical_operation_id=request.logical_operation_id,
            initial_attempt_count=prior_attempt_count,
        )

        async def call(model_id: str, call_messages: list[Any]) -> AIMessage:
            fake_provider = os.getenv(
                "SANDBOX_GATEWAY_FAKE_PROVIDER", "false"
            ).lower() in {"1", "true", "yes", "on"}
            if fake_provider:
                from open_deep_research.sandbox.fake_provider import (
                    DeterministicGatewayModel,
                )

                model = DeterministicGatewayModel(role=request.role)
            else:
                model = get_configurable_model_template()
            if request.tools and hasattr(model, "bind_tools"):
                model = model.bind_tools(request.tools)
            if request.model_kwargs and hasattr(model, "bind"):
                model = model.bind(**request.model_kwargs)
            model_config = (
                {}
                if fake_provider
                else apply_helicone_config(
                    build_model_config(
                        model_id,
                        max_tokens,
                        context.config,
                        role=request.role,
                    ),
                    context.config,
                    span_name=f"gateway.{request.role}.model",
                    agent_role=request.role,
                )
            )
            if hasattr(model, "with_config"):
                model = model.with_config(model_config)
            response = await invoke_model_with_retry_observability(
                model,
                call_messages,
                context.config,
                span_name=f"gateway.{request.role}.model",
                agent_role=request.role,
                model_name=model_id,
                stage=request.stage,
                attributes={"gateway": True, "logical_operation_id": request.logical_operation_id},
                budget_gate=budget,
            )
            if not isinstance(response, AIMessage):
                raise RuntimeError("sandbox_gateway_provider_returned_non_ai_message")
            return response

        try:
            response = await invoke_with_model_fallback(
                call,
                messages,
                primary_model=primary_model,
                model_fallbacks=configuration.model_fallbacks,
                role=request.role,
                config=context.config,
                on_fallback=lambda event: fallback_events.append(dict(event)),
            )
            usage = self._usage(response)
            outcome = GatewayModelOutcomeV1(
                logical_operation_id=request.logical_operation_id,
                physical_attempt_id=budget.last_physical_attempt_id,
                status="completed",
                message=message_to_dict(response),
                usage=usage,
                fallback_events=fallback_events,
                provider_ttft_ms=(
                    float(response.response_metadata["provider_ttft_ms"])
                    if response.response_metadata.get("provider_ttft_ms") is not None
                    else None
                ),
            )
            transition = self.internal.signed(
                OperationTransitionRequest,
                run_id=request.run_id,
                fence_token=context.fence_token,
                logical_operation_id=request.logical_operation_id,
                status="completed",
                outcome=outcome.model_dump(mode="json"),
                error_type=None,
            )
            await self.internal.post("/internal/sandbox/operations/transition", transition)
            return outcome
        except Exception as exc:
            kind = classify_model_error(exc, primary_model)
            uncertain = kind in {ModelErrorKind.CANCELLED, ModelErrorKind.UNKNOWN}
            logger.warning(
                "Gateway provider operation failed run_id=%s operation_id=%s "
                "error_type=%s exception_type=%s",
                request.run_id,
                request.logical_operation_id,
                kind.value,
                type(exc).__name__,
            )
            outcome = GatewayModelOutcomeV1(
                logical_operation_id=request.logical_operation_id,
                physical_attempt_id=budget.last_physical_attempt_id or "unreserved",
                status="uncertain" if uncertain else "failed",
                fallback_events=fallback_events,
                error_type=kind.value,
                error_message="Provider operation failed.",
            )
            transition = self.internal.signed(
                OperationTransitionRequest,
                run_id=request.run_id,
                fence_token=context.fence_token,
                logical_operation_id=request.logical_operation_id,
                status="uncertain" if uncertain else "failed",
                outcome=outcome.model_dump(mode="json"),
                error_type=kind.value,
            )
            with suppress(httpx.HTTPError, ValueError, KeyError):
                await self.internal.post("/internal/sandbox/operations/transition", transition)
            return outcome


def create_gateway_app(runtime: GatewayRuntime) -> FastAPI:
    """Create the task-data and trusted-control Gateway application."""
    app = FastAPI(title="InsightForge Sandbox Gateway", docs_url=None, redoc_url=None)

    @app.get("/healthz")
    async def healthz() -> dict[str, Any]:
        return {"status": "ok", "registered_runs": len(runtime.runs)}

    @app.post("/internal/v1/runs/register")
    async def register_run(request: GatewayRunRegistrationRequest) -> dict[str, str]:
        try:
            runtime.register(request)
        except ValueError as exc:
            raise HTTPException(status_code=401, detail=str(exc)) from exc
        return {"status": "registered"}

    @app.post("/internal/v1/runs/unregister")
    async def unregister_run(
        request: GatewayRunUnregisterRequest,
    ) -> dict[str, str]:
        try:
            runtime.unregister(request)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {"status": "unregistered"}

    @app.post("/v1/models/invoke", response_model=GatewayModelOutcomeV1)
    async def invoke_model(
        request: GatewayModelRequestV1,
        authorization: str = Header(default="", alias="Authorization"),
        timestamp: float = Header(alias="X-Sandbox-Timestamp"),
        nonce: str = Header(alias="X-Sandbox-Nonce"),
        service_signature: str = Header(
            default="",
            alias="X-Sandbox-Service-Signature",
        ),
        fence_token: int | None = Header(
            default=None,
            alias="X-Sandbox-Fence-Token",
        ),
    ) -> GatewayModelOutcomeV1:
        try:
            if authorization.startswith("Bearer "):
                _claims, context = runtime.authorize_task(
                    request,
                    authorization=authorization,
                    timestamp=timestamp,
                    nonce=nonce,
                )
            else:
                if fence_token is None or not service_signature:
                    raise ValueError("sandbox_model_auth_missing")
                context = runtime.authorize_api_model(
                    request,
                    timestamp=timestamp,
                    nonce=nonce,
                    fence_token=fence_token,
                    signature=service_signature,
                )
        except ValueError as exc:
            raise HTTPException(status_code=401, detail=str(exc)) from exc
        return await runtime.invoke_model_operation(request, context)

    @app.post("/v1/models/stream")
    async def stream_model(
        request: GatewayModelRequestV1,
        authorization: str = Header(default="", alias="Authorization"),
        timestamp: float = Header(alias="X-Sandbox-Timestamp"),
        nonce: str = Header(alias="X-Sandbox-Nonce"),
        service_signature: str = Header(
            default="",
            alias="X-Sandbox-Service-Signature",
        ),
        fence_token: int | None = Header(
            default=None,
            alias="X-Sandbox-Fence-Token",
        ),
    ) -> StreamingResponse:
        try:
            if authorization.startswith("Bearer "):
                _claims, context = runtime.authorize_task(
                    request,
                    authorization=authorization,
                    timestamp=timestamp,
                    nonce=nonce,
                )
            else:
                if fence_token is None or not service_signature:
                    raise ValueError("sandbox_model_auth_missing")
                context = runtime.authorize_api_model(
                    request,
                    timestamp=timestamp,
                    nonce=nonce,
                    fence_token=fence_token,
                    signature=service_signature,
                )
        except ValueError as exc:
            raise HTTPException(status_code=401, detail=str(exc)) from exc

        async def events():
            yield json.dumps({"type": "started"}, separators=(",", ":")) + "\n"
            outcome = await runtime.invoke_model_operation(request, context)
            yield json.dumps(
                {
                    "type": "result",
                    "outcome": outcome.model_dump(mode="json"),
                },
                separators=(",", ":"),
            ) + "\n"

        return StreamingResponse(events(), media_type="application/x-ndjson")

    @app.post(
        "/v1/models/lookup",
        response_model=GatewayOperationLookupOutcomeV1,
    )
    async def lookup_model(
        request: GatewayOperationLookupRequestV1,
        authorization: str = Header(default="", alias="Authorization"),
        timestamp: float = Header(alias="X-Sandbox-Timestamp"),
        nonce: str = Header(alias="X-Sandbox-Nonce"),
        service_signature: str = Header(
            default="",
            alias="X-Sandbox-Service-Signature",
        ),
        fence_token: int | None = Header(
            default=None,
            alias="X-Sandbox-Fence-Token",
        ),
    ) -> GatewayOperationLookupOutcomeV1:
        try:
            if authorization.startswith("Bearer "):
                _claims, context = runtime.authorize_task(
                    request,
                    authorization=authorization,
                    timestamp=timestamp,
                    nonce=nonce,
                )
            else:
                if fence_token is None or not service_signature:
                    raise ValueError("sandbox_model_auth_missing")
                context = runtime.authorize_api_model(
                    request,
                    timestamp=timestamp,
                    nonce=nonce,
                    fence_token=fence_token,
                    signature=service_signature,
                )
        except ValueError as exc:
            raise HTTPException(status_code=401, detail=str(exc)) from exc
        return await runtime.lookup_model_operation(request, context)

    @app.post(
        "/v1/tools/catalog",
        response_model=GatewayToolCatalogOutcomeV1,
    )
    async def tool_catalog(
        request: GatewayToolCatalogRequestV1,
        authorization: str = Header(default="", alias="Authorization"),
        timestamp: float = Header(alias="X-Sandbox-Timestamp"),
        nonce: str = Header(alias="X-Sandbox-Nonce"),
    ) -> GatewayToolCatalogOutcomeV1:
        try:
            _claims, context = runtime.authorize_task(
                request,
                authorization=authorization,
                timestamp=timestamp,
                nonce=nonce,
            )
        except ValueError as exc:
            raise HTTPException(status_code=401, detail=str(exc)) from exc
        return await runtime.tool_catalog(request, context)

    @app.post("/v1/tools/call", response_model=GatewayToolOutcomeV1)
    async def invoke_tool(
        request: GatewayToolRequestV1,
        authorization: str = Header(default="", alias="Authorization"),
        timestamp: float = Header(alias="X-Sandbox-Timestamp"),
        nonce: str = Header(alias="X-Sandbox-Nonce"),
    ) -> GatewayToolOutcomeV1:
        try:
            _claims, context = runtime.authorize_task(
                request,
                authorization=authorization,
                timestamp=timestamp,
                nonce=nonce,
            )
        except ValueError as exc:
            raise HTTPException(status_code=401, detail=str(exc)) from exc
        return await runtime.invoke_tool(request, context)

    @app.post("/v1/tools/authorize-local", response_model=GatewayToolOutcomeV1)
    async def authorize_local_tool(
        request: GatewayToolRequestV1,
        authorization: str = Header(default="", alias="Authorization"),
        timestamp: float = Header(alias="X-Sandbox-Timestamp"),
        nonce: str = Header(alias="X-Sandbox-Nonce"),
    ) -> GatewayToolOutcomeV1:
        try:
            _claims, context = runtime.authorize_task(
                request,
                authorization=authorization,
                timestamp=timestamp,
                nonce=nonce,
            )
        except ValueError as exc:
            raise HTTPException(status_code=401, detail=str(exc)) from exc
        return await runtime.authorize_local_tool(request, context)

    return app


def main() -> None:
    """Run the Gateway service on the internal task network."""
    import asyncio

    import uvicorn

    configurable = Configuration.from_runnable_config(None)
    if not configurable.sandbox_enabled:
        raise SystemExit("SANDBOX_ENABLED must be true for sandbox-gateway")
    if os.getenv("SANDBOX_GATEWAY_FAKE_PROVIDER", "false").lower() in {
        "1",
        "true",
        "yes",
        "on",
    } and os.getenv("APP_ENV", "production").lower() not in {
        "development",
        "test",
    }:
        raise SystemExit(
            "SANDBOX_GATEWAY_FAKE_PROVIDER is restricted to development/test"
        )
    runtime = GatewayRuntime(configurable)

    async def serve() -> None:
        from open_deep_research.sandbox.egress_proxy import GatewayEgressProxy

        proxy = GatewayEgressProxy(runtime)
        proxy_server = await asyncio.start_server(
            proxy.handle,
            "0.0.0.0",
            int(os.getenv("SANDBOX_GATEWAY_PROXY_PORT", "8080")),
        )
        server = uvicorn.Server(
            uvicorn.Config(
                create_gateway_app(runtime),
                host="0.0.0.0",
                port=int(os.getenv("SANDBOX_GATEWAY_PORT", "8081")),
                log_level="info",
            )
        )
        async with proxy_server:
            await asyncio.gather(proxy_server.serve_forever(), server.serve())

    asyncio.run(serve())


if __name__ == "__main__":
    main()
