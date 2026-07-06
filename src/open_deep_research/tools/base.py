"""Project-owned tool contract and default implementation."""

from __future__ import annotations

import inspect
import json
from dataclasses import dataclass
from enum import Enum
from typing import (
    Any,
    Awaitable,
    Callable,
    Generic,
    Optional,
    Protocol,
    TypeVar,
    Union,
    runtime_checkable,
)

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel

InputT = TypeVar("InputT", bound=BaseModel)
OutputT = TypeVar("OutputT")
ProgressT = TypeVar("ProgressT")
ProgressT_co = TypeVar("ProgressT_co", covariant=True)


class ToolOrigin(str, Enum):
    """Where a tool implementation originates."""

    SYSTEM = "system"
    SEARCH = "search"
    MCP = "mcp"
    PROVIDER_NATIVE = "provider_native"
    SKILL = "skill"


@dataclass(frozen=True, slots=True)
class ToolContext:
    """Runtime facts shared with every tool invocation."""

    config: RunnableConfig
    role: str
    tool_call_id: str


@dataclass(frozen=True, slots=True)
class ToolResult(Generic[OutputT]):
    """A successful tool result before transport-specific rendering."""

    output: OutputT


ProgressCallback = Callable[[ProgressT], Union[Awaitable[None], None]]
DescriptionFn = Callable[[Optional[InputT]], Union[str, Awaitable[str]]]
@runtime_checkable
class Tool(Protocol, Generic[InputT, OutputT, ProgressT_co]):
    """Structural interface implemented by every executable project tool."""

    @property
    def name(self) -> str:
        """Return the unique tool name."""
        ...

    @property
    def input_schema(self) -> type[InputT]:
        """Return the Pydantic input schema."""
        ...

    @property
    def origin(self) -> ToolOrigin:
        """Return the declared tool origin."""
        ...

    @property
    def retryable(self) -> bool:
        """Return whether transient failures may be retried."""
        ...

    async def description(self, input: Optional[InputT] = None) -> str:
        """Return the model-facing or invocation-specific description."""
        ...

    async def call(
        self,
        input: InputT,
        context: ToolContext,
        on_progress: Optional[ProgressCallback[ProgressT_co]] = None,
    ) -> ToolResult[OutputT]:
        """Execute the tool using validated input."""
        ...


@dataclass(frozen=True, slots=True)
class BuiltTool(Generic[InputT, OutputT, ProgressT]):
    """Default immutable implementation produced by :func:`build_tool`."""

    name: str
    input_schema: type[InputT]
    origin: ToolOrigin
    retryable: bool
    _description: Callable[[Optional[InputT]], Union[str, Awaitable[str]]]
    _call: Callable[
        [InputT, ToolContext, Optional[ProgressCallback[ProgressT]]],
        Union[ToolResult[OutputT], Awaitable[ToolResult[OutputT]]],
    ]

    async def description(self, input: Optional[InputT] = None) -> str:
        """Resolve a static or dynamic description."""
        value = self._description(input)
        if inspect.isawaitable(value):
            value = await value
        return str(value)

    async def call(
        self,
        input: InputT,
        context: ToolContext,
        on_progress: Optional[ProgressCallback[ProgressT]] = None,
    ) -> ToolResult[OutputT]:
        """Delegate execution to the configured callable."""
        result = self._call(input, context, on_progress)
        if inspect.isawaitable(result):
            result = await result
        if not isinstance(result, ToolResult):
            raise TypeError(
                f"Tool '{self.name}' returned {type(result).__name__}; "
                "call() must return ToolResult"
            )
        return result


def build_tool(
    *,
    name: str,
    description: Union[str, DescriptionFn[InputT]],
    input_schema: type[InputT],
    call: Callable[
        [InputT, ToolContext, Optional[ProgressCallback[ProgressT]]],
        Union[ToolResult[OutputT], Awaitable[ToolResult[OutputT]]],
    ],
    origin: ToolOrigin,
    retryable: bool = False,
) -> Tool[InputT, OutputT, ProgressT]:
    """Build a tool with conservative defaults and eager contract checks."""
    if not name or not name.strip():
        raise ValueError("Tool name must be a non-empty string")
    if not isinstance(input_schema, type) or not issubclass(input_schema, BaseModel):
        raise TypeError("input_schema must be a Pydantic BaseModel subclass")
    if not callable(call):
        raise TypeError("call must be callable")
    if isinstance(description, str):
        static_description = description

        def describe(_: Optional[InputT] = None) -> str:
            return static_description

        description_fn: DescriptionFn[InputT] = describe
    elif callable(description):
        description_fn = description
    else:
        raise TypeError("description must be a string or callable")
    return BuiltTool(
        name=name.strip(),
        input_schema=input_schema,
        origin=origin,
        retryable=bool(retryable),
        _description=description_fn,
        _call=call,
    )


def serialize_tool_output(output: Any) -> str:
    """Render arbitrary tool output into stable ToolMessage content."""
    if isinstance(output, str):
        return output
    if isinstance(output, BaseModel):
        return output.model_dump_json()
    if output is None:
        return ""
    if isinstance(output, dict | list | tuple | bool | int | float):
        return json.dumps(
            output,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
    return str(output)


async def tool_to_model_definition(
    tool: Tool[Any, Any, Any],
) -> dict[str, Any]:
    """Project a Tool into the schema-only shape accepted by bind_tools()."""
    return {
        "name": tool.name,
        "description": await tool.description(None),
        "parameters": tool.input_schema.model_json_schema(),
    }


async def tools_to_model_definitions(
    tools: list[Tool[Any, Any, Any]],
) -> list[dict[str, Any]]:
    """Project tools for model binding without exposing an execution bypass."""
    return [await tool_to_model_definition(tool) for tool in tools]


def build_tool_registry(
    tools: list[Tool[Any, Any, Any]],
) -> dict[str, Tool[Any, Any, Any]]:
    """Index tools by name and reject malformed or duplicate registrations."""
    registry: dict[str, Tool[Any, Any, Any]] = {}
    for candidate in tools:
        if not isinstance(candidate, Tool):
            raise TypeError(f"Registered object does not satisfy Tool: {candidate!r}")
        if candidate.name in registry:
            raise ValueError(f"Duplicate tool name: {candidate.name}")
        registry[candidate.name] = candidate
    return registry
