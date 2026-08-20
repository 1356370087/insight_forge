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
    BROWSER = "browser"


class ToolEffect(str, Enum):
    """Externally visible effect a tool invocation may have."""

    READ_ONLY = "read_only"
    SENSITIVE_READ = "sensitive_read"
    EXTERNAL_WRITE = "external_write"
    LOCAL_WRITE = "local_write"
    DESTRUCTIVE = "destructive"


class ToolExecutionZone(str, Enum):
    """Physical trust boundary in which a tool implementation executes."""

    SANDBOX_LOCAL = "sandbox_local"
    GATEWAY = "gateway"
    HOST_CONTROL = "host_control"


@dataclass(frozen=True, slots=True)
class ToolContext:
    """Runtime facts shared with every tool invocation."""

    config: RunnableConfig
    role: str
    tool_call_id: str
    operation_id: str = ""
    attempt: int = 1


@dataclass(frozen=True, slots=True)
class ToolResult(Generic[OutputT]):
    """A successful tool result before transport-specific rendering."""

    output: OutputT


ProgressCallback = Callable[[ProgressT], Union[Awaitable[None], None]]
DescriptionFn = Callable[[Optional[InputT]], Union[str, Awaitable[str]]]
PromptFn = Callable[[RunnableConfig], Optional[str]]
EnabledFn = Callable[[RunnableConfig], bool]
EgressUrlsFn = Callable[[dict[str, Any]], list[str]]


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

    @property
    def effect(self) -> ToolEffect:
        """Return the externally visible effect of an invocation."""
        ...

    @property
    def execution_zone(self) -> ToolExecutionZone:
        """Return the authoritative physical execution zone."""
        ...

    @property
    def concurrency_safe(self) -> bool:
        """Return whether independent calls may execute concurrently."""
        ...

    @property
    def max_output_chars(self) -> Optional[int]:
        """Return the per-tool serialized output budget, if configured."""
        ...

    def prompt(self, config: RunnableConfig) -> Optional[str]:
        """Return detailed model guidance for this tool, if any."""
        ...

    def is_enabled(self, config: RunnableConfig) -> bool:
        """Return whether this tool is enabled for the supplied run config."""
        ...

    def egress_urls(self, input: dict[str, Any]) -> list[str]:
        """Extract outbound URLs whose hosts must pass egress governance."""
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
    effect: ToolEffect
    execution_zone: ToolExecutionZone
    retryable: bool
    concurrency_safe: bool
    supports_idempotency: bool
    max_output_chars: Optional[int]
    _description: Callable[[Optional[InputT]], Union[str, Awaitable[str]]]
    _prompt: PromptFn
    _is_enabled: EnabledFn
    _egress_urls: EgressUrlsFn
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

    def prompt(self, config: RunnableConfig) -> Optional[str]:
        """Resolve detailed model guidance for the supplied run config."""
        value = self._prompt(config)
        return None if value is None else str(value)

    def is_enabled(self, config: RunnableConfig) -> bool:
        """Resolve declarative availability for the supplied run config."""
        return bool(self._is_enabled(config))

    def egress_urls(self, input: dict[str, Any]) -> list[str]:
        """Return declared egress URLs for validated raw invocation arguments."""
        urls = self._egress_urls(input)
        if not isinstance(urls, list) or any(not isinstance(url, str) for url in urls):
            raise TypeError(f"Tool '{self.name}' egress_urls() must return list[str]")
        return urls

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
    effect: ToolEffect = ToolEffect.READ_ONLY,
    execution_zone: Optional[ToolExecutionZone] = None,
    retryable: bool = False,
    concurrency_safe: bool = False,
    supports_idempotency: bool = False,
    prompt: Union[str, PromptFn, None] = None,
    is_enabled: Optional[EnabledFn] = None,
    egress_urls: Optional[EgressUrlsFn] = None,
    max_output_chars: Optional[int] = None,
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
    if isinstance(prompt, str):
        static_prompt = prompt

        def render_prompt(_: RunnableConfig) -> str:
            return static_prompt

        prompt_fn: PromptFn = render_prompt
    elif prompt is None:
        def no_prompt(_: RunnableConfig) -> None:
            return None

        prompt_fn = no_prompt
    elif callable(prompt):
        prompt_fn = prompt
    else:
        raise TypeError("prompt must be a string, callable, or None")
    if is_enabled is not None and not callable(is_enabled):
        raise TypeError("is_enabled must be callable or None")
    if egress_urls is not None and not callable(egress_urls):
        raise TypeError("egress_urls must be callable or None")
    if max_output_chars is not None and (
        isinstance(max_output_chars, bool)
        or not isinstance(max_output_chars, int)
        or max_output_chars <= 0
    ):
        raise ValueError("max_output_chars must be a positive integer or None")
    return BuiltTool(
        name=name.strip(),
        input_schema=input_schema,
        origin=origin,
        effect=effect,
        execution_zone=(
            execution_zone
            or (
                ToolExecutionZone.SANDBOX_LOCAL
                if origin is ToolOrigin.SYSTEM
                else ToolExecutionZone.GATEWAY
            )
        ),
        retryable=bool(retryable),
        concurrency_safe=bool(concurrency_safe),
        supports_idempotency=bool(supports_idempotency),
        max_output_chars=max_output_chars,
        _description=description_fn,
        _prompt=prompt_fn,
        _is_enabled=is_enabled or (lambda _: True),
        _egress_urls=egress_urls or (lambda _: []),
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
    *,
    max_description_chars: int = 2_000,
) -> dict[str, Any]:
    """Project a Tool into the schema-only shape accepted by bind_tools()."""
    remote_definition = getattr(tool, "model_definition", None)
    if isinstance(remote_definition, dict):
        projected = dict(remote_definition)
        projected["description"] = str(
            projected.get("description") or ""
        )[:max_description_chars]
        return projected
    description = (await tool.description(None)).strip()
    description = description[:max_description_chars]
    return {
        "name": tool.name,
        "description": description,
        "parameters": tool.input_schema.model_json_schema(),
    }


async def tools_to_model_definitions(
    tools: list[Tool[Any, Any, Any]],
    *,
    max_description_chars: int = 2_000,
) -> list[dict[str, Any]]:
    """Project tools for model binding without exposing an execution bypass."""
    return [
        await tool_to_model_definition(tool, max_description_chars=max_description_chars)
        for tool in tools
    ]


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
