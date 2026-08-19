"""Minimal stdio MCP server used by the mcp 2.x client integration tests."""

from mcp.server.mcpserver import MCPServer

server = MCPServer("test-stdio-server")


@server.tool()
def echo(text: str) -> str:
    """Echo the provided text back to the caller."""
    return f"echo: {text}"


if __name__ == "__main__":
    server.run("stdio")
