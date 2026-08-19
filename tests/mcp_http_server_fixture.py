"""Minimal streamable-http MCP server used by the mcp 2.x client integration tests."""

import sys

from mcp.server.mcpserver import MCPServer

server = MCPServer("test-http-server")


@server.tool()
def greet(name: str) -> str:
    """Greet the caller by name."""
    return f"hello {name}"


if __name__ == "__main__":
    server.run("streamable-http", host="127.0.0.1", port=int(sys.argv[1]))
