"""Authenticated API-owned budget, operation and approval control plane."""

from __future__ import annotations

import asyncio
import secrets
import time
from dataclasses import dataclass
from typing import Any, Callable, Literal

import httpx
from fastapi import APIRouter, HTTPException
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, ConfigDict, Field

from open_deep_research.budgets import BudgetGate
from open_deep_research.configuration import Configuration
from open_deep_research.events.public import event_publisher_from_config
from open_deep_research.sandbox.approvals import ApprovalKind, SecurityApprovalStore
from open_deep_research.sandbox.crypto import (
    NonceReplayCache,
    SandboxDerivedKeys,
    sign_payload,
    validate_timestamp,
    verify_payload,
)
from open_deep_research.sandbox.operations import ModelOperationStore
from open_deep_research.tasks.registry import TaskStatus, get_task_registry


class ServiceRequest(BaseModel):
    """Common replay-protected service authentication envelope."""

    model_config = ConfigDict(extra="forbid")

    service_timestamp: float
    service_nonce: str = Field(min_length=16, max_length=256)
    service_signature: str

    def signed_payload(self) -> dict[str, Any]:
        """Return the canonical service-auth payload."""
        return self.model_dump(mode="json", exclude={"service_signature"})


class BudgetReserveRequest(ServiceRequest):
    """Reserve one physical model attempt and its estimated usage."""
    run_id: str
    task_id: str
    fence_token: int
    stage: str
    logical_operation_id: str
    physical_attempt_id: str
    model_name: str
    estimated_input_tokens: int = Field(ge=1)
    estimated_output_tokens: int = Field(ge=1)


class BudgetSettleRequest(ServiceRequest):
    """Settle one physical model attempt with actual usage."""
    run_id: str
    fence_token: int
    physical_attempt_id: str
    model_name: str
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)


class BudgetFailRequest(ServiceRequest):
    """Finalize failed model reservations as released or uncertain."""
    run_id: str
    fence_token: int
    physical_attempt_id: str
    uncertain: bool


class ToolBudgetReserveRequest(ServiceRequest):
    """Reserve one logical Gateway tool call."""
    run_id: str
    task_id: str
    fence_token: int
    stage: str
    logical_operation_id: str


class ToolBudgetSettleRequest(ServiceRequest):
    """Settle one previously reserved logical Gateway tool call."""

    run_id: str
    fence_token: int
    logical_operation_id: str


class OperationTransitionRequest(ServiceRequest):
    """Advance one durable logical model operation."""
    run_id: str
    fence_token: int
    logical_operation_id: str
    status: Literal["dispatched", "completed", "failed", "uncertain"]
    outcome: dict[str, Any] | None = None
    error_type: str | None = None


class OperationGetRequest(ServiceRequest):
    """Read one logical model operation through authenticated POST."""
    run_id: str
    fence_token: int
    logical_operation_id: str


class ApprovalCreateRequest(ServiceRequest):
    """Create an idempotent durable security approval."""
    run_id: str
    task_id: str
    fence_token: int
    kind: ApprovalKind
    capability: str
    target: dict[str, Any]
    operation_id: str
    expires_at: float
    stage: str = "researching"


class ApprovalWaitRequest(ServiceRequest):
    """Long-poll the run approval index after a version cursor."""
    run_id: str
    fence_token: int
    after_version: int = Field(ge=0)
    timeout_seconds: float = Field(default=25, gt=0, le=25)


class ApprovalConsumeRequest(ServiceRequest):
    """Consume an allow-once approval or validate allow-run reuse."""
    run_id: str
    fence_token: int
    approval_id: str
    operation_id: str


@dataclass(frozen=True, slots=True)
class InternalRunContext:
    """Facts the API must authoritatively resolve for one live run."""

    config: RunnableConfig
    configurable: Configuration
    fence_token: int
    started_at: float


def build_internal_sandbox_router(
    resolve_run: Callable[[str], InternalRunContext | None],
) -> APIRouter:
    """Build the internal-only, HMAC-authenticated sandbox control router."""
    router = APIRouter(prefix="/internal/sandbox", tags=["sandbox-internal"])
    replay = NonceReplayCache()

    def authorize(request: ServiceRequest, context: InternalRunContext) -> None:
        validate_timestamp(request.service_timestamp)
        root = context.configurable.sandbox_root_signing_key
        if not root:
            raise ValueError("sandbox_unavailable:root_signing_key")
        keys = SandboxDerivedKeys.from_root(root)
        if not verify_payload(request.signed_payload(), request.service_signature, keys.service_auth):
            raise ValueError("sandbox_service_auth_invalid")
        replay.consume(
            f"service:{context.fence_token}",
            request.service_nonce,
            expires_at=time.time() + 60,
        )

    def authority(request: ServiceRequest, run_id: str, fence_token: int) -> InternalRunContext:
        context = resolve_run(run_id)
        if context is None:
            raise HTTPException(status_code=404, detail="run_not_active")
        try:
            authorize(request, context)
        except ValueError as exc:
            raise HTTPException(status_code=401, detail=str(exc)) from exc
        if fence_token != context.fence_token:
            raise HTTPException(status_code=409, detail="stale_fence")
        return context

    def sync_waiting_tasks(run_id: str, approvals: list[Any]) -> None:
        """Project the durable concurrent queue onto legacy scalar task status."""
        pending_by_task: dict[str, list[Any]] = {}
        for approval in approvals:
            if getattr(approval, "status", None) == "pending":
                pending_by_task.setdefault(str(approval.task_id), []).append(approval)
        registry = get_task_registry()
        for task in registry.list(run_id=run_id):
            pending = pending_by_task.get(task.task_id, [])
            if pending:
                first = pending[0]
                task.pending_domain = str(first.target.get("domain") or "") or None
                task.pending_domain_tool = first.capability
                if task.status == TaskStatus.RUNNING:
                    registry.update_status(
                        task.task_id,
                        TaskStatus.WAITING_FOR_CONFIRMATION,
                    )
            elif task.status == TaskStatus.WAITING_FOR_CONFIRMATION:
                task.pending_domain = None
                task.pending_domain_tool = None
                registry.update_status(task.task_id, TaskStatus.RUNNING)

    @router.post("/budgets/reserve")
    async def reserve_budget(request: BudgetReserveRequest) -> dict[str, Any]:
        context = authority(request, request.run_id, request.fence_token)
        gate = BudgetGate.from_config(
            context.configurable,
            request.run_id,
            started_at=context.started_at,
        )
        await asyncio.to_thread(
            gate.reserve_model_call,
            request.physical_attempt_id,
            estimated_input_tokens=request.estimated_input_tokens,
            estimated_output_tokens=request.estimated_output_tokens,
            model_name=request.model_name,
        )
        store = ModelOperationStore(request.run_id, runs_dir=context.configurable.runs_dir)
        record = await asyncio.to_thread(
            store.reserve,
            task_id=request.task_id,
            stage=request.stage,
            logical_operation_id=request.logical_operation_id,
            physical_attempt_id=request.physical_attempt_id,
        )
        return record.model_dump(mode="json")

    @router.post("/budgets/settle")
    async def settle_budget(request: BudgetSettleRequest) -> dict[str, str]:
        context = authority(request, request.run_id, request.fence_token)
        gate = BudgetGate.from_config(
            context.configurable,
            request.run_id,
            started_at=context.started_at,
        )
        await asyncio.to_thread(
            gate.settle_model_call,
            request.physical_attempt_id,
            input_tokens=request.input_tokens,
            output_tokens=request.output_tokens,
            model_name=request.model_name,
        )
        return {"status": "settled"}

    @router.post("/budgets/fail")
    async def fail_budget(request: BudgetFailRequest) -> dict[str, str]:
        context = authority(request, request.run_id, request.fence_token)
        gate = BudgetGate.from_config(
            context.configurable,
            request.run_id,
            started_at=context.started_at,
        )
        await asyncio.to_thread(
            gate.fail_model_call,
            request.physical_attempt_id,
            uncertain=request.uncertain,
        )
        return {"status": "uncertain" if request.uncertain else "released"}

    @router.post("/budgets/tool-reserve")
    async def reserve_tool_budget(request: ToolBudgetReserveRequest) -> dict[str, str]:
        context = authority(request, request.run_id, request.fence_token)
        gate = BudgetGate.from_config(
            context.configurable,
            request.run_id,
            started_at=context.started_at,
        )
        await asyncio.to_thread(
            gate.reserve_tool_call,
            request.logical_operation_id,
        )
        return {"status": "reserved"}

    @router.post("/budgets/tool-settle")
    async def settle_tool_budget(
        request: ToolBudgetSettleRequest,
    ) -> dict[str, str]:
        context = authority(request, request.run_id, request.fence_token)
        gate = BudgetGate.from_config(
            context.configurable,
            request.run_id,
            started_at=context.started_at,
        )
        await asyncio.to_thread(
            gate.settle_tool_call,
            request.logical_operation_id,
        )
        return {"status": "settled"}

    @router.post("/operations/get")
    async def get_operation(request: OperationGetRequest) -> dict[str, Any]:
        context = authority(request, request.run_id, request.fence_token)
        record = await asyncio.to_thread(
            ModelOperationStore(
                request.run_id, runs_dir=context.configurable.runs_dir
            ).get,
            request.logical_operation_id,
        )
        if record is None:
            return {"found": False}
        return {"found": True, "operation": record.model_dump(mode="json")}

    @router.post("/operations/transition")
    async def transition_operation(request: OperationTransitionRequest) -> dict[str, Any]:
        context = authority(request, request.run_id, request.fence_token)
        expected = {
            "dispatched": {"reserved"},
            "completed": {"dispatched"},
            "failed": {"reserved", "dispatched"},
            "uncertain": {"reserved", "dispatched"},
        }[request.status]
        record = await asyncio.to_thread(
            ModelOperationStore(
                request.run_id, runs_dir=context.configurable.runs_dir
            ).transition,
            request.logical_operation_id,
            expected=expected,
            status=request.status,
            outcome=request.outcome,
            error_type=request.error_type,
        )
        return record.model_dump(mode="json")

    @router.post("/approvals/request")
    async def request_approval(request: ApprovalCreateRequest) -> dict[str, Any]:
        context = authority(request, request.run_id, request.fence_token)
        approval = await asyncio.to_thread(
            SecurityApprovalStore(
                request.run_id, runs_dir=context.configurable.runs_dir
            ).request,
            task_id=request.task_id,
            fence_token=request.fence_token,
            kind=request.kind,
            capability=request.capability,
            target=request.target,
            operation_id=request.operation_id,
            expires_at=request.expires_at,
        )
        task = get_task_registry().get(request.task_id)
        if task is not None and task.run_id == request.run_id:
            task.pending_domain = str(request.target.get("domain") or "") or None
            task.pending_domain_tool = request.capability
            if task.status == TaskStatus.RUNNING:
                get_task_registry().update_status(
                    request.task_id, TaskStatus.WAITING_FOR_CONFIRMATION
                )
        await event_publisher_from_config(context.config).publish(
            "security.approval.required",
            stage=request.stage,
            payload={
                "approval_id": approval.approval_id,
                "task_id": approval.task_id,
                "kind": approval.kind,
                "capability": approval.capability,
                "target": approval.target,
                "status": approval.status,
                "expires_at": approval.expires_at,
            },
            dedupe_key=f"security-approval:{approval.approval_id}:required",
        )
        return approval.model_dump(mode="json")

    @router.post("/approvals/wait")
    async def wait_approvals(request: ApprovalWaitRequest) -> dict[str, Any]:
        context = authority(request, request.run_id, request.fence_token)
        version, approvals = await SecurityApprovalStore(
            request.run_id, runs_dir=context.configurable.runs_dir
        ).wait_for_change(
            request.after_version,
            timeout_seconds=request.timeout_seconds,
        )
        sync_waiting_tasks(request.run_id, approvals)
        for approval in approvals:
            if approval.status == "expired":
                await event_publisher_from_config(context.config).publish(
                    "security.approval.resolved",
                    stage="researching",
                    payload={
                        "approval_id": approval.approval_id,
                        "task_id": approval.task_id,
                        "kind": approval.kind,
                        "capability": approval.capability,
                        "decision": "deny",
                        "status": "expired",
                    },
                    dedupe_key=(
                        f"security-approval:{approval.approval_id}:expired:"
                        f"{approval.version}"
                    ),
                )
        return {
            "version": version,
            "approvals": [item.model_dump(mode="json") for item in approvals],
        }

    @router.post("/approvals/consume")
    async def consume_approval(request: ApprovalConsumeRequest) -> dict[str, Any]:
        context = authority(request, request.run_id, request.fence_token)
        try:
            approval = await asyncio.to_thread(
                SecurityApprovalStore(
                    request.run_id, runs_dir=context.configurable.runs_dir
                ).consume,
                request.approval_id,
                operation_id=request.operation_id,
                expected_fence_token=request.fence_token,
            )
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        _version, approvals = await asyncio.to_thread(
            SecurityApprovalStore(
                request.run_id,
                runs_dir=context.configurable.runs_dir,
            ).list
        )
        sync_waiting_tasks(request.run_id, approvals)
        return approval.model_dump(mode="json")

    return router


class SandboxInternalClient:
    """Gateway-side authenticated client for the API authority."""

    def __init__(self, base_url: str, root_key: str) -> None:
        """Initialize an authenticated client for one API internal origin."""
        self.base_url = base_url.rstrip("/")
        self.keys = SandboxDerivedKeys.from_root(root_key)

    def signed(self, model_type, **values):
        """Construct and sign one service request with a fresh nonce."""
        request = model_type(
            **values,
            service_timestamp=time.time(),
            service_nonce=secrets.token_urlsafe(24),
            service_signature="pending",
        )
        request.service_signature = sign_payload(
            request.signed_payload(), self.keys.service_auth
        )
        return request

    async def post(self, path: str, request: ServiceRequest) -> dict[str, Any]:
        """POST one signed request and return its JSON object."""
        async with httpx.AsyncClient(base_url=self.base_url, timeout=60) as client:
            response = await client.post(
                path,
                content=request.model_dump_json(),
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()
            return response.json()

    async def get(self, path: str) -> dict[str, Any] | None:
        """GET a non-mutating internal resource, returning None for 404."""
        async with httpx.AsyncClient(base_url=self.base_url, timeout=30) as client:
            response = await client.get(path)
            if response.status_code == 404:
                return None
            response.raise_for_status()
            return response.json()
