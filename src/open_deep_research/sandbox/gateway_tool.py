"""Worker-side Tool proxy for Gateway-owned implementations."""

# Delegating properties intentionally mirror the structural Tool protocol.
# ruff: noqa: D102

from __future__ import annotations

import hashlib
import json
import os
import secrets
import time
from dataclasses import dataclass
from typing import Any, Optional

import httpx
from pydantic import BaseModel

from open_deep_research.sandbox.wire import GatewayToolOutcomeV1, GatewayToolRequestV1
from open_deep_research.tools.base import (
    ProgressCallback,
    Tool,
    ToolContext,
    ToolExecutionZone,
    ToolResult,
)


def _request_for(
    tool: Tool,
    input: BaseModel,
    context: ToolContext,
) -> GatewayToolRequestV1:
    metadata = context.config.get("metadata", {})
    arguments = input.model_dump(mode="json")
    stable = {
        "run_id": metadata.get("run_id"),
        "task_id": metadata.get("task_id"),
        "tool_call_id": context.tool_call_id,
        "tool": tool.name,
        "arguments": arguments,
    }
    logical_id = context.operation_id or (
        "gateway-tool:"
        + hashlib.sha256(
            json.dumps(stable, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    )
    return GatewayToolRequestV1(
        run_id=str(metadata.get("run_id") or "default"),
        task_id=str(metadata.get("task_id") or "researcher"),
        role=context.role,
        stage=str(metadata.get("sandbox_tool_stage") or "researching"),
        execution_zone=getattr(
            tool.execution_zone,
            "value",
            str(tool.execution_zone),
        ),
        logical_operation_id=logical_id,
        tool_call_id=context.tool_call_id,
        tool_name=tool.name,
        arguments=arguments,
    )


async def _call_gateway(
    path: str,
    tool: Tool,
    input: BaseModel,
    context: ToolContext,
) -> GatewayToolOutcomeV1:
    gateway_url = os.environ.get("SANDBOX_GATEWAY_URL", "")
    task_token = os.environ.get("SANDBOX_TASK_TOKEN", "")
    if not gateway_url or not task_token:
        raise RuntimeError("sandbox_gateway_not_configured")
    headers = {
        "Authorization": f"Bearer {task_token}",
        "Content-Type": "application/json",
        "X-Sandbox-Timestamp": str(time.time()),
        "X-Sandbox-Nonce": secrets.token_urlsafe(24),
    }
    async with httpx.AsyncClient(base_url=gateway_url, timeout=None) as client:
        response = await client.post(
            path,
            content=_request_for(tool, input, context).model_dump_json(),
            headers=headers,
        )
        response.raise_for_status()
        return GatewayToolOutcomeV1.model_validate(response.json())


@dataclass(frozen=True, slots=True)
class GatewayToolProxy:
    """Preserve a tool's model contract while moving physical execution to Gateway."""

    delegate: Tool

    @property
    def name(self):
        return self.delegate.name

    @property
    def input_schema(self):
        return self.delegate.input_schema

    @property
    def origin(self):
        return self.delegate.origin

    @property
    def effect(self):
        return self.delegate.effect

    @property
    def retryable(self):
        return self.delegate.retryable

    @property
    def concurrency_safe(self):
        return self.delegate.concurrency_safe

    @property
    def supports_idempotency(self):
        return getattr(self.delegate, "supports_idempotency", False)

    @property
    def max_output_chars(self):
        return self.delegate.max_output_chars

    @property
    def execution_zone(self) -> ToolExecutionZone:
        return ToolExecutionZone.GATEWAY

    async def description(self, input: Optional[BaseModel] = None) -> str:
        return await self.delegate.description(input)

    def prompt(self, config):
        return self.delegate.prompt(config)

    def is_enabled(self, config):
        return self.delegate.is_enabled(config)

    def egress_urls(self, input: dict[str, Any]) -> list[str]:
        return self.delegate.egress_urls(input)

    async def call(
        self,
        input: BaseModel,
        context: ToolContext,
        on_progress: Optional[ProgressCallback[Any]] = None,
    ) -> ToolResult[Any]:
        del on_progress
        outcome = await _call_gateway(
            "/v1/tools/call",
            self.delegate,
            input,
            context,
        )
        if outcome.status != "completed":
            message = (outcome.error or {}).get("message") or "sandbox_gateway_tool_failed"
            raise RuntimeError(str(message))
        return ToolResult(output=outcome.output)


@dataclass(frozen=True, slots=True)
class AuthorizedLocalToolProxy(GatewayToolProxy):
    """Require Gateway authorization before invoking a Worker-local tool."""

    @property
    def execution_zone(self) -> ToolExecutionZone:
        return ToolExecutionZone.SANDBOX_LOCAL

    async def call(
        self,
        input: BaseModel,
        context: ToolContext,
        on_progress: Optional[ProgressCallback[Any]] = None,
    ) -> ToolResult[Any]:
        outcome = await _call_gateway(
            "/v1/tools/authorize-local",
            self.delegate,
            input,
            context,
        )
        if outcome.status != "completed":
            message = (outcome.error or {}).get("message") or "sandbox_local_tool_denied"
            raise RuntimeError(str(message))
        return await self.delegate.call(input, context, on_progress)


def proxy_gateway_tools(tools: list[Tool]) -> list[Tool]:
    """Proxy Gateway tools and authorize every sandbox-local execution."""
    return [
        GatewayToolProxy(tool)
        if getattr(tool, "execution_zone", ToolExecutionZone.GATEWAY)
        is ToolExecutionZone.GATEWAY
        else AuthorizedLocalToolProxy(tool)
        for tool in tools
    ]
