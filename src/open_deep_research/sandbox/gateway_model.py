"""Secret-free LangChain chat model used inside sandbox Workers."""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import time
from typing import Any, AsyncIterator, Sequence

import httpx
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    message_to_dict,
    messages_from_dict,
)
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import Field

from open_deep_research.configuration import Configuration
from open_deep_research.sandbox.crypto import SandboxDerivedKeys, sign_payload
from open_deep_research.sandbox.wire import (
    GatewayModelOutcomeV1,
    GatewayModelRequestV1,
)

_ROLE_STAGE = {
    "supervisor": "planning",
    "researcher": "researching",
    "summarization": "researching",
    "message_summary": "researching",
    "compression": "synthesizing",
    "final_report": "writing",
    "quality_evaluation": "finalizing",
    "quality_evaluator": "finalizing",
}


class GatewayChatModel(BaseChatModel):
    """Proxy model that never resolves a Provider credential in the Worker."""

    gateway_url: str = Field(default_factory=lambda: os.environ.get("SANDBOX_GATEWAY_URL", ""))
    task_token: str = Field(default_factory=lambda: os.environ.get("SANDBOX_TASK_TOKEN", ""))
    role: str = "researcher"
    bound_tools: list[dict[str, Any]] = Field(default_factory=list)
    bound_tool_choice: str | dict[str, Any] | bool | None = None
    is_sandbox_gateway_model: bool = True

    @property
    def _llm_type(self) -> str:
        return "insightforge-sandbox-gateway"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"gateway_url": self.gateway_url, "role": self.role}

    def for_role(self, role: str) -> GatewayChatModel:
        """Return an immutable role-bound copy used by the Query engine."""
        return self.model_copy(update={"role": role})

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | Any],
        *,
        tool_choice: str | dict[str, Any] | bool | None = None,
        **kwargs: Any,
    ) -> GatewayChatModel:
        """Bind JSON tool definitions without importing their implementations."""
        del kwargs
        definitions = []
        for tool in tools:
            try:
                definitions.append(convert_to_openai_tool(tool))
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    "GatewayChatModel accepts JSON tool definitions only"
                ) from exc
        return self.model_copy(
            update={
                "bound_tools": definitions,
                "bound_tool_choice": tool_choice,
            }
        )

    def _operation_id(
        self,
        messages: list[BaseMessage],
        config: dict[str, Any] | None,
        model_kwargs: dict[str, Any],
    ) -> str:
        metadata = (config or {}).get("metadata") or {}
        role = str(metadata.get("sandbox_model_role") or self.role)
        stable = {
            "run_id": metadata.get("run_id"),
            "task_id": metadata.get("task_id"),
            "role": role,
            "turn": metadata.get("query_turn", metadata.get("turn", 0)),
            "messages": [message_to_dict(message) for message in messages],
            "tools": self.bound_tools,
            "tool_choice": self.bound_tool_choice,
            "model_kwargs": model_kwargs,
        }
        digest = hashlib.sha256(
            json.dumps(stable, sort_keys=True, default=str, separators=(",", ":")).encode()
        ).hexdigest()
        return f"gateway-model:{digest}"

    def _request(
        self,
        messages: list[BaseMessage],
        config: dict[str, Any] | None,
        model_kwargs: dict[str, Any] | None = None,
    ) -> GatewayModelRequestV1:
        metadata = (config or {}).get("metadata") or {}
        role = str(metadata.get("sandbox_model_role") or self.role)
        safe_model_kwargs = {
            key: value
            for key, value in (model_kwargs or {}).items()
            if key == "response_format"
        }
        return GatewayModelRequestV1(
            run_id=str(metadata.get("run_id") or "default"),
            task_id=str(metadata.get("task_id") or "researcher"),
            role=role,
            stage=str(
                metadata.get("sandbox_model_stage")
                or _ROLE_STAGE.get(role, "researching")
            ),
            logical_operation_id=self._operation_id(
                messages,
                config,
                safe_model_kwargs,
            ),
            messages=[message_to_dict(message) for message in messages],
            tools=self.bound_tools,
            tool_choice=self.bound_tool_choice,
            model_kwargs=safe_model_kwargs,
        )

    def _headers(
        self,
        request: GatewayModelRequestV1,
        config: dict[str, Any] | None,
    ) -> dict[str, str]:
        if not self.gateway_url:
            raise RuntimeError("sandbox_gateway_not_configured")
        timestamp = time.time()
        nonce = secrets.token_urlsafe(24)
        headers = {
            "Content-Type": "application/json",
            "X-Sandbox-Timestamp": str(timestamp),
            "X-Sandbox-Nonce": nonce,
        }
        if self.task_token:
            headers["Authorization"] = f"Bearer {self.task_token}"
            return headers
        configurable = Configuration.from_runnable_config(config)
        fence_token = int((config or {}).get("metadata", {}).get("run_fence_token") or 0)
        if (
            not configurable.sandbox_enabled
            or not configurable.sandbox_root_signing_key
            or fence_token < 1
        ):
            raise RuntimeError("sandbox_gateway_service_auth_unavailable")
        signed = {
            "request": request.model_dump(mode="json"),
            "timestamp": timestamp,
            "nonce": nonce,
            "fence_token": fence_token,
        }
        keys = SandboxDerivedKeys.from_root(configurable.sandbox_root_signing_key)
        headers["X-Sandbox-Timestamp"] = str(timestamp)
        headers["X-Sandbox-Fence-Token"] = str(fence_token)
        headers["X-Sandbox-Service-Signature"] = sign_payload(
            signed,
            keys.service_auth,
        )
        return headers

    @staticmethod
    def _message(outcome: GatewayModelOutcomeV1) -> AIMessage:
        if outcome.status != "completed" or outcome.message is None:
            raise RuntimeError(
                outcome.error_type or outcome.error_message or "sandbox_gateway_model_failed"
            )
        values = messages_from_dict([outcome.message])
        if len(values) != 1 or not isinstance(values[0], AIMessage):
            raise RuntimeError("sandbox_gateway_returned_invalid_message")
        return values[0]

    async def ainvoke(
        self,
        input: list[BaseMessage],
        config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> AIMessage:
        """Invoke while preserving the run/task metadata carried by RunnableConfig."""
        result = await self._agenerate(input, config=config, **kwargs)
        message = result.generations[0].message
        if not isinstance(message, AIMessage):
            raise RuntimeError("sandbox_gateway_returned_non_ai_message")
        return message

    def invoke(
        self,
        input: list[BaseMessage],
        config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> AIMessage:
        """Invoke synchronously for LangChain structured wrappers."""
        result = self._generate(input, config=config, **kwargs)
        message = result.generations[0].message
        if not isinstance(message, AIMessage):
            raise RuntimeError("sandbox_gateway_returned_non_ai_message")
        return message

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        del stop, run_manager
        config = kwargs.pop("config", None)
        request = self._request(messages, config, kwargs)
        started = time.monotonic()
        async with httpx.AsyncClient(base_url=self.gateway_url, timeout=None) as client:
            response = await client.post(
                "/v1/models/invoke",
                content=request.model_dump_json(),
                headers=self._headers(request, config),
            )
            response.raise_for_status()
            outcome = GatewayModelOutcomeV1.model_validate(response.json())
        message = self._message(outcome)
        message.response_metadata = {
            **message.response_metadata,
            "provider_ttft_ms": outcome.provider_ttft_ms,
            "rpc_ttft_ms": (time.monotonic() - started) * 1000,
        }
        return ChatResult(generations=[ChatGeneration(message=message)])

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        del stop, run_manager
        config = kwargs.pop("config", None)
        request = self._request(messages, config, kwargs)
        started = time.monotonic()
        with httpx.Client(base_url=self.gateway_url, timeout=None) as client:
            response = client.post(
                "/v1/models/invoke",
                content=request.model_dump_json(),
                headers=self._headers(request, config),
            )
            response.raise_for_status()
            outcome = GatewayModelOutcomeV1.model_validate(response.json())
        message = self._message(outcome)
        message.response_metadata = {
            **message.response_metadata,
            "provider_ttft_ms": outcome.provider_ttft_ms,
            "rpc_ttft_ms": (time.monotonic() - started) * 1000,
        }
        return ChatResult(generations=[ChatGeneration(message=message)])

    async def astream(
        self,
        input: list[BaseMessage],
        config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[AIMessageChunk]:
        """Yield one merged chunk while preserving Provider and RPC TTFT fields."""
        request = self._request(input, config, kwargs)
        started = time.monotonic()
        rpc_ttft_ms: float | None = None
        outcome: GatewayModelOutcomeV1 | None = None
        async with httpx.AsyncClient(base_url=self.gateway_url, timeout=None) as client:
            async with client.stream(
                "POST",
                "/v1/models/stream",
                content=request.model_dump_json(),
                headers=self._headers(request, config),
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    if rpc_ttft_ms is None:
                        rpc_ttft_ms = (time.monotonic() - started) * 1000
                    event = json.loads(line)
                    if event.get("type") == "result":
                        outcome = GatewayModelOutcomeV1.model_validate(
                            event.get("outcome")
                        )
        if outcome is None:
            raise RuntimeError("sandbox_gateway_stream_missing_result")
        message = self._message(outcome)
        message.response_metadata = {
            **message.response_metadata,
            "provider_ttft_ms": outcome.provider_ttft_ms,
            "rpc_ttft_ms": rpc_ttft_ms,
        }
        yield AIMessageChunk(
            content=message.content,
            additional_kwargs=message.additional_kwargs,
            response_metadata=message.response_metadata,
            tool_calls=list(getattr(message, "tool_calls", []) or []),
            usage_metadata=getattr(message, "usage_metadata", None),
        )
