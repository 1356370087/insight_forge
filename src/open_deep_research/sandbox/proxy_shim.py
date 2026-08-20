"""Local proxy shim that adds task capability and per-connection nonce headers."""

from __future__ import annotations

import argparse
import asyncio
import os
import secrets
import time


class ProxyShim:
    """Inject task authentication into standard HTTP proxy requests."""

    def __init__(self, host: str, port: int, task_token: str) -> None:
        """Bind the shim to one Gateway proxy and task token."""
        self.host = host
        self.port = port
        self.task_token = task_token

    async def handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        """Forward one client connection with fresh timestamp and nonce headers."""
        upstream_writer = None
        try:
            raw = await reader.readuntil(b"\r\n\r\n")
            head = raw[:-4]
            injected_headers = (
                f"\r\nProxy-Authorization: Bearer {self.task_token}"
                f"\r\nX-Sandbox-Timestamp: {time.time()}"
                f"\r\nX-Sandbox-Nonce: {secrets.token_urlsafe(24)}\r\n\r\n"
            ).encode()
            injected = head + injected_headers
            upstream_reader, upstream_writer = await asyncio.open_connection(self.host, self.port)
            upstream_writer.write(injected)
            await upstream_writer.drain()

            async def copy(source, destination):
                while data := await source.read(64 * 1024):
                    destination.write(data)
                    await destination.drain()

            await asyncio.gather(copy(reader, upstream_writer), copy(upstream_reader, writer))
        finally:
            if upstream_writer is not None:
                upstream_writer.close()
            writer.close()


async def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--listen-host", default="127.0.0.1")
    parser.add_argument("--listen-port", type=int, default=3128)
    parser.add_argument("--gateway-host", default="sandbox-gateway")
    parser.add_argument("--gateway-port", type=int, default=8080)
    args = parser.parse_args()
    token = os.environ.get("SANDBOX_TASK_TOKEN", "")
    if not token:
        raise SystemExit("SANDBOX_TASK_TOKEN is required")
    shim = ProxyShim(args.gateway_host, args.gateway_port, token)
    server = await asyncio.start_server(shim.handle, args.listen_host, args.listen_port)
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    asyncio.run(_main())
