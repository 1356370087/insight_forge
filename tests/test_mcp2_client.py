"""Tests for the in-process MCP client built on the mcp 2.x SDK."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest
from langchain_core.tools import StructuredTool, ToolException
from mcp import MCPError
from mcp.types import (
    URL_ELICITATION_REQUIRED,
    CallToolResult,
    ImageContent,
    TextContent,
)
from pydantic import BaseModel

from open_deep_research.tools.mcp.client import (
    MultiServerMCPClient,
    build_args_schema,
    convert_call_tool_result,
)
from open_deep_research.tools.mcp.oauth import wrap_mcp_authenticate_tool

_FIXTURES_DIR = Path(__file__).resolve().parent


def _stdio_connection(fixture: str) -> dict[str, dict[str, Any]]:
    return {
        "server_1": {
            "transport": "stdio",
            "command": sys.executable,
            "args": [str(_FIXTURES_DIR / fixture)],
        }
    }


class _NoArgs(BaseModel):
    pass


def _failing_tool(exc: BaseException) -> StructuredTool:
    async def failing(**kwargs: Any) -> str:
        raise exc

    return StructuredTool(
        name="failing",
        description="d",
        args_schema=_NoArgs,
        coroutine=failing,
    )


@pytest.mark.asyncio
async def test_stdio_discovery_and_per_call_invocation() -> None:
    tools = await MultiServerMCPClient(_stdio_connection("mcp_stdio_server_fixture.py")).get_tools()

    assert [tool.name for tool in tools] == ["echo"]
    assert tools[0].description == "Echo the provided text back to the caller."
    first = await tools[0].ainvoke({"text": "one"})
    second = await tools[0].ainvoke({"text": "two"})
    assert first == "echo: one"
    assert second == "echo: two"


@pytest.mark.asyncio
async def test_http_transport_propagates_headers_and_invokes() -> None:
    import socket
    import subprocess

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    server = subprocess.Popen(
        [
            sys.executable,
            str(_FIXTURES_DIR / "mcp_http_server_fixture.py"),
            str(port),
        ],
        cwd=_FIXTURES_DIR.parent,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        connection = {
            "server_1": {
                "transport": "streamable_http",
                "url": f"http://127.0.0.1:{port}/mcp",
                "headers": {"Authorization": "Bearer test-token"},
            }
        }
        import anyio

        for _ in range(50):
            try:
                with socket.socket() as client:
                    client.settimeout(0.2)
                    client.connect(("127.0.0.1", port))
                break
            except OSError:
                await anyio.sleep(0.2)
        else:
            pytest.fail("fixture HTTP server never started listening")

        tools = await MultiServerMCPClient(connection).get_tools()
        assert [tool.name for tool in tools] == ["greet"]
        assert await tools[0].ainvoke({"name": "http"}) == "hello http"
    finally:
        server.terminate()
        server.wait(timeout=10)


@pytest.mark.asyncio
async def test_unsupported_transport_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported MCP transport"):
        await MultiServerMCPClient(
            {"server_1": {"transport": "websocket", "url": "ws://example.invalid"}}
        ).get_tools()


def test_build_args_schema_maps_common_json_schema_types() -> None:
    schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "limit": {"type": "integer", "default": 3},
            "ratio": {"type": "number"},
            "verbose": {"type": "boolean"},
            "tags": {"type": "array", "items": {"type": "string"}},
            "payload": {"type": "object"},
            "mode": {"enum": ["fast", "deep"]},
            "optional_mode": {"type": ["string", "null"]},
        },
        "required": ["query"],
    }

    model = build_args_schema("schema_tool", schema)

    validated = model.model_validate(
        {"query": "q", "tags": ["a"], "mode": "fast", "optional_mode": None}
    )
    assert validated.query == "q"
    assert validated.limit == 3
    assert validated.tags == ["a"]
    with pytest.raises(ValueError):
        model.model_validate({})


def test_build_args_schema_handles_empty_and_untyped_properties() -> None:
    empty = build_args_schema("empty_tool", {"type": "object", "properties": {}})
    assert empty.model_validate({}) == empty()

    untyped = build_args_schema(
        "untyped_tool",
        {"type": "object", "properties": {"anything": {"description": "no type"}}},
    )
    assert untyped.model_validate({"anything": [1, 2, 3]}).anything == [1, 2, 3]


def test_convert_call_tool_result_joins_text_blocks() -> None:
    result = CallToolResult(
        content=[
            TextContent(type="text", text="alpha"),
            TextContent(type="text", text="beta"),
        ]
    )

    assert convert_call_tool_result(result) == "alpha\nbeta"


def test_convert_call_tool_result_serializes_mixed_content() -> None:
    result = CallToolResult(
        content=[
            TextContent(type="text", text="caption"),
            ImageContent(type="image", data="abc", mimeType="image/png"),
        ]
    )

    payload = json.loads(convert_call_tool_result(result))
    assert payload[0] == {"text": "caption", "type": "text"}
    assert payload[1]["data"] == "abc"


def test_convert_call_tool_result_empty_content_returns_empty_string() -> None:
    assert convert_call_tool_result(CallToolResult(content=[])) == ""


def test_convert_call_tool_result_error_raises_tool_exception() -> None:
    result = CallToolResult(
        content=[TextContent(type="text", text="invalid arguments")], isError=True
    )

    with pytest.raises(ToolException, match="invalid arguments"):
        convert_call_tool_result(result)


class _GroupLike(RuntimeError):
    """Mimic the ``exceptions`` attribute contract of ExceptionGroup (3.11+)."""

    def __init__(self, *nested: BaseException) -> None:
        super().__init__("task group failed")
        self.exceptions = tuple(nested)


@pytest.mark.asyncio
async def test_auth_wrapper_translates_url_elicitation_required() -> None:
    inner = MCPError(
        code=URL_ELICITATION_REQUIRED,
        message="auth needed",
        data={
            "elicitations": [
                {"message": "Authorize at", "url": "https://auth.example/consent"}
            ]
        },
    )
    tool = wrap_mcp_authenticate_tool(_failing_tool(_GroupLike(inner)))

    with pytest.raises(ToolException) as excinfo:
        await tool.ainvoke({})

    assert str(excinfo.value) == "Authorize at https://auth.example/consent"


@pytest.mark.asyncio
async def test_auth_wrapper_keeps_legacy_interaction_code() -> None:
    legacy = MCPError(
        code=-32003,
        message="Required interaction",
        data={"message": {"text": "Please sign in"}, "url": "https://login.example"},
    )
    tool = wrap_mcp_authenticate_tool(_failing_tool(legacy))

    with pytest.raises(ToolException) as excinfo:
        await tool.ainvoke({})

    assert str(excinfo.value) == "Please sign in https://login.example"


@pytest.mark.asyncio
async def test_auth_wrapper_propagates_unrelated_mcp_errors() -> None:
    tool = wrap_mcp_authenticate_tool(_failing_tool(MCPError(code=-32602, message="bad params")))

    with pytest.raises(MCPError, match="bad params"):
        await tool.ainvoke({})
