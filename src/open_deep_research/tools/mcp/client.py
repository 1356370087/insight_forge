"""MCP client built directly on the mcp 2.x SDK.

Replaces the ``langchain-mcp-adapters`` ``MultiServerMCPClient`` seam with an
in-process equivalent: the constructor still accepts the same connection-dict
shape and :meth:`MultiServerMCPClient.get_tools` still returns LangChain
``StructuredTool`` objects whose invocations open a fresh connection per call
(the session-per-call semantics the adapters shipped in 0.3.x).
"""

from __future__ import annotations

import json
import re
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Any, Literal, Optional

from langchain_core.tools import StructuredTool, ToolException
from mcp import Client
from mcp.client.sse import sse_client
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.client.streamable_http import (
    create_mcp_http_client,
    streamable_http_client,
)
from mcp.types import CallToolResult, Tool
from pydantic import Field, create_model

_HTTP_TRANSPORTS = frozenset({"http", "streamable_http"})
_SUPPORTED_TRANSPORTS = frozenset({"stdio", "sse"}) | _HTTP_TRANSPORTS

_JSON_SCHEMA_TYPE_MAP: dict[str, type] = {
    "string": str,
    "number": float,
    "integer": int,
    "boolean": bool,
    "array": list,
    "object": dict,
    "null": type(None),
}


@asynccontextmanager
async def open_mcp_connection(connection: dict[str, Any]):
    """Open one MCP transport described by a connection dict.

    Yields the read/write stream pair the ``Client`` handshake drives. The
    caller owns the context; exiting closes the transport and any HTTP client
    created for header injection.
    """
    async with AsyncExitStack() as stack:
        yield await _enter_transport(stack, connection)


async def _enter_transport(stack: AsyncExitStack, connection: dict[str, Any]):
    transport = str(connection.get("transport") or "streamable_http")
    if transport not in _SUPPORTED_TRANSPORTS:
        raise ValueError(
            f"Unsupported MCP transport {transport!r}; expected one of "
            f"{sorted(_SUPPORTED_TRANSPORTS)}"
        )
    if transport == "stdio":
        command = connection.get("command")
        if not command:
            raise ValueError("stdio MCP connection requires a 'command' entry")
        params = StdioServerParameters(
            command=str(command),
            args=[str(arg) for arg in connection.get("args") or []],
            env={
                str(key): str(value)
                for key, value in (connection.get("env") or {}).items()
            }
            or None,
        )
        return await stack.enter_async_context(stdio_client(params))
    url = connection.get("url")
    if not url:
        raise ValueError(f"{transport} MCP connection requires a 'url' entry")
    headers = connection.get("headers")
    if transport == "sse":
        return await stack.enter_async_context(
            sse_client(str(url), headers=dict(headers) if headers else None)
        )
    http_client = None
    if headers:
        http_client = await stack.enter_async_context(
            create_mcp_http_client(headers=dict(headers))
        )
    return await stack.enter_async_context(
        streamable_http_client(str(url), http_client=http_client)
    )


def _annotation_from_schema(schema: dict[str, Any]) -> Any:
    """Map a JSON Schema property descriptor onto a Python annotation."""
    if "enum" in schema and schema["enum"]:
        values = tuple(schema["enum"])
        if all(isinstance(value, str) for value in values):
            return Literal[values]
    schema_type = schema.get("type")
    if isinstance(schema_type, list):
        non_null = [item for item in schema_type if item != "null"]
        nullable = len(non_null) < len(schema_type)
        branches = [
            _JSON_SCHEMA_TYPE_MAP.get(item, Any)
            for item in non_null
            if isinstance(item, str)
        ] or [Any]
        annotation: Any = branches[0]
        for branch in branches[1:]:
            annotation = annotation | branch
        return annotation | None if nullable else annotation
    if schema_type == "array":
        items = schema.get("items")
        if isinstance(items, dict):
            # Item annotation is a runtime value derived from the remote schema.
            return list[_annotation_from_schema(items)]  # type: ignore[misc]
        return list
    if "anyOf" in schema and isinstance(schema["anyOf"], list):
        branches = [
            _annotation_from_schema(branch)
            for branch in schema["anyOf"]
            if isinstance(branch, dict)
        ] or [Any]
        merged: Any = branches[0]
        for branch in branches[1:]:
            merged = merged | branch
        return merged
    if not isinstance(schema_type, str):
        return Any
    return _JSON_SCHEMA_TYPE_MAP.get(schema_type, Any)


def build_args_schema(tool_name: str, input_schema: dict[str, Any]) -> type:
    """Compile an MCP ``inputSchema`` into a Pydantic model for LangChain."""
    properties = input_schema.get("properties") if isinstance(input_schema, dict) else None
    properties = properties if isinstance(properties, dict) else {}
    required = set(
        input_schema.get("required") or [] if isinstance(input_schema, dict) else []
    )
    fields: dict[str, tuple[Any, Any]] = {}
    for name, raw_schema in properties.items():
        schema = raw_schema if isinstance(raw_schema, dict) else {}
        annotation = _annotation_from_schema(schema)
        if name in required and "default" not in schema:
            fields[str(name)] = (annotation, Field(...))
        else:
            fields[str(name)] = (annotation, schema.get("default"))
    model_name = "MCPArgs_" + re.sub(r"\W", "_", tool_name or "tool")
    # Field definitions are (annotation, default) tuples built at runtime.
    return create_model(model_name, **fields)  # type: ignore[call-overload]


def _text_of(block: Any) -> Optional[str]:
    if getattr(block, "type", None) != "text":
        return None
    text = getattr(block, "text", None)
    return text if isinstance(text, str) else None


def convert_call_tool_result(result: CallToolResult) -> Any:
    """Render a ``CallToolResult`` into the tool-output shape the seam expects.

    Text-only results collapse to a newline-joined string; mixed content
    (images, files, resources) serializes to stable JSON under the same
    output budget governance applies to every tool. ``isError`` results raise
    ``ToolException`` so the governance layer can classify and surface them.
    """
    blocks = list(result.content or [])
    texts = [text for text in (_text_of(block) for block in blocks) if text is not None]
    if getattr(result, "is_error", False):
        raise ToolException("\n".join(texts) or "MCP tool returned an error with no text content")
    if not blocks:
        return ""
    if len(texts) == len(blocks):
        return "\n".join(texts)
    return json.dumps(
        [block.model_dump(exclude_none=True) for block in blocks],
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )


def _tool_to_structured_tool(mcp_tool: Tool, connection: dict[str, Any]) -> StructuredTool:
    async def _invoke(**kwargs: Any) -> Any:
        async with Client(open_mcp_connection(connection)) as client:
            result = await client.call_tool(mcp_tool.name, kwargs or None)
        return convert_call_tool_result(result)

    return StructuredTool(
        name=mcp_tool.name,
        description=mcp_tool.description or "",
        args_schema=build_args_schema(mcp_tool.name, dict(mcp_tool.input_schema)),
        coroutine=_invoke,
    )


class MultiServerMCPClient:
    """Discover LangChain tools from MCP servers on the mcp 2.x SDK."""

    def __init__(self, connections: dict[str, dict[str, Any]] | None) -> None:
        """Store the per-server connection dictionaries for later discovery."""
        self.connections: dict[str, dict[str, Any]] = dict(connections or {})

    async def _list_tools(self, connection: dict[str, Any]) -> list[Tool]:
        async with Client(open_mcp_connection(connection)) as client:
            tools: list[Tool] = []
            cursor: str | None = None
            while True:
                result = await client.list_tools(cursor=cursor)
                tools.extend(result.tools)
                cursor = getattr(result, "next_cursor", None)
                if not cursor:
                    return tools

    async def get_tools(self) -> list[StructuredTool]:
        """List tools from every configured server as LangChain tools."""
        loaded: list[StructuredTool] = []
        for connection in self.connections.values():
            loaded.extend(
                _tool_to_structured_tool(mcp_tool, connection)
                for mcp_tool in await self._list_tools(connection)
            )
        return loaded


__all__ = [
    "MultiServerMCPClient",
    "build_args_schema",
    "convert_call_tool_result",
    "open_mcp_connection",
]
