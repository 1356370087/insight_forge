"""Deterministic no-network chat model for sandbox CI fault/throughput tests."""

from __future__ import annotations

import re
import uuid
from typing import Any, Sequence

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field


def _function(tool: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    function = tool.get("function") if isinstance(tool.get("function"), dict) else tool
    return str(function.get("name") or tool.get("name") or ""), dict(
        function.get("parameters")
        or function.get("input_schema")
        or tool.get("parameters")
        or {}
    )


def _sample_schema(schema: dict[str, Any], root: dict[str, Any], key: str = "") -> Any:
    if "$ref" in schema:
        target: Any = root
        for part in str(schema["$ref"]).removeprefix("#/").split("/"):
            if part:
                target = target[part]
        return _sample_schema(dict(target), root, key)
    variants = schema.get("anyOf") or schema.get("oneOf")
    if isinstance(variants, list):
        selected = next(
            (item for item in variants if item.get("type") != "null"),
            variants[0],
        )
        return _sample_schema(dict(selected), root, key)
    if schema.get("enum"):
        return schema["enum"][0]
    if "const" in schema:
        return schema["const"]
    value_type = schema.get("type")
    if value_type == "object" or schema.get("properties"):
        properties = dict(schema.get("properties") or {})
        required = set(schema.get("required") or properties)
        return {
            name: _sample_schema(dict(value), root, name)
            for name, value in properties.items()
            if name in required or "default" in value
        }
    if value_type == "array":
        minimum = int(schema.get("minItems") or 0)
        return [
            _sample_schema(dict(schema.get("items") or {}), root, key)
            for _ in range(minimum)
        ]
    if value_type == "boolean":
        return bool(schema.get("default", False))
    if value_type == "integer":
        return int(schema.get("default", schema.get("minimum", 1)))
    if value_type == "number":
        return float(schema.get("default", schema.get("minimum", 1.0)))
    if "default" in schema:
        return schema["default"]
    text_by_key = {
        "research_brief": "Deterministic sandbox integration research brief.",
        "objective": "Verify the deterministic sandbox execution path.",
        "question": "Verify the deterministic sandbox execution path.",
        "title": "Sandbox integration",
        "summary": "Deterministic sandbox result.",
        "decision": "complete",
        "action": "complete",
    }
    return text_by_key.get(key, "deterministic")


class DeterministicGatewayModel(BaseChatModel):
    """Produce stable tool calls/content without any Provider network access."""

    role: str
    tools: list[dict[str, Any]] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "insightforge-deterministic-gateway"

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | Any],
        **_kwargs: Any,
    ) -> DeterministicGatewayModel:
        """Bind JSON tool definitions for deterministic structured responses."""
        return self.model_copy(
            update={
                "tools": [dict(tool) for tool in tools if isinstance(tool, dict)]
            }
        )

    @staticmethod
    def _call(name: str, args: dict[str, Any]) -> AIMessage:
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": name,
                    "args": args,
                    "id": f"fake-{uuid.uuid4().hex}",
                }
            ],
            usage_metadata={
                "input_tokens": 10,
                "output_tokens": 5,
                "total_tokens": 15,
            },
        )

    def _response(self, messages: list[BaseMessage]) -> AIMessage:
        functions = dict(_function(tool) for tool in self.tools)
        if self.role == "compression":
            requirement_ids = sorted(
                set(
                    re.findall(
                        r"COV-[0-9]{2}-[0-9a-f]{12}",
                        "\n".join(str(message.content) for message in messages),
                        flags=re.IGNORECASE,
                    )
                )
            )
            coverage = "\n".join(
                f"- {requirement_id}: deterministic sandbox coverage verified."
                for requirement_id in requirement_ids
            )
            return AIMessage(
                content=(
                    "Deterministic sandbox researcher result.\n\n"
                    "## Requirement Coverage\n"
                    f"{coverage or '- No delegated requirements.'}"
                ),
                usage_metadata={
                    "input_tokens": 10,
                    "output_tokens": 10,
                    "total_tokens": 20,
                },
            )
        if self.role == "researcher" and "ResearchComplete" in functions:
            return self._call("ResearchComplete", {})
        if self.role == "supervisor" and functions:
            tool_messages = [message for message in messages if isinstance(message, ToolMessage)]
            combined = "\n".join(str(message.content) for message in tool_messages)
            task_ids = re.findall(
                r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
                combined,
                flags=re.IGNORECASE,
            )
            if not task_ids and "StartResearchTask" in functions:
                schema = functions["StartResearchTask"]
                args = {
                    "research_topic": "Deterministic sandbox integration task",
                    "display_title": "Sandbox integration",
                    "requirement_ids": [],
                }
                requirement_schema = (
                    schema.get("properties", {}).get("requirement_ids", {})
                )
                enum = requirement_schema.get("items", {}).get("enum")
                if isinstance(enum, list):
                    args["requirement_ids"] = list(enum)
                return self._call("StartResearchTask", args)
            if "completed" in combined.lower() and "ResearchComplete" in functions:
                return self._call("ResearchComplete", {})
            last_name = tool_messages[-1].name if tool_messages else ""
            if last_name != "WaitForResearchUpdates" and "WaitForResearchUpdates" in functions:
                # A real Docker Worker has cold-import and network setup cost;
                # keep the deterministic supervisor parked long enough for the
                # task result instead of exhausting its own turn budget first.
                return self._call("WaitForResearchUpdates", {"timeout_seconds": 15})
            if task_ids and "CheckResearchTask" in functions:
                return self._call("CheckResearchTask", {"task_ids": [task_ids[-1]]})
            if "ResearchComplete" in functions:
                return self._call("ResearchComplete", {})
        if len(functions) == 1:
            name, schema = next(iter(functions.items()))
            return self._call(name, _sample_schema(schema, schema))
        return AIMessage(
            content=(
                "Deterministic sandbox synthesis completed. "
                "No external provider or network was used."
            ),
            usage_metadata={
                "input_tokens": 10,
                "output_tokens": 10,
                "total_tokens": 20,
            },
        )

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        del stop, run_manager, kwargs
        return ChatResult(generations=[ChatGeneration(message=self._response(messages))])

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        return self._generate(messages, stop, run_manager, **kwargs)
