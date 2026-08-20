"""Network-boundary tests for the task-authenticated Gateway proxy."""

from __future__ import annotations

import asyncio
import socket
from types import SimpleNamespace

import pytest

from open_deep_research.sandbox import egress_proxy
from open_deep_research.sandbox.egress_proxy import GatewayEgressProxy, _public_address


class MemoryWriter:
    def __init__(self) -> None:
        self.data = bytearray()
        self.closed = False

    def write(self, data: bytes) -> None:
        self.data.extend(data)

    async def drain(self) -> None:
        return None

    def close(self) -> None:
        self.closed = True

    async def wait_closed(self) -> None:
        return None


def test_public_address_rejects_cgnat_and_benchmark_networks() -> None:
    assert _public_address("100.64.0.1") is False
    assert _public_address("198.18.0.1") is False
    assert _public_address("8.8.8.8") is True


def test_proxy_normalizes_idn_hosts_before_policy_and_dns() -> None:
    assert GatewayEgressProxy._normalize_host("BÜCHER.Example.") == (
        "xn--bcher-kva.example"
    )


@pytest.mark.asyncio
async def test_dns_resolution_rejects_any_private_answer(monkeypatch) -> None:
    monkeypatch.setattr(
        egress_proxy.socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("8.8.8.8", 443)),
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 443)),
        ],
    )
    proxy = GatewayEgressProxy(SimpleNamespace())

    with pytest.raises(ValueError, match="sandbox_private_destination_denied"):
        await proxy._resolve("example.com", 443, False)


@pytest.mark.asyncio
async def test_error_reason_cannot_inject_response_headers() -> None:
    writer = MemoryWriter()

    await GatewayEgressProxy._reply(
        writer,
        403,
        "denied\r\nX-Injected: yes",
    )

    head = bytes(writer.data).split(b"\r\n\r\n", 1)[0]
    assert b"X-Injected" not in head
    assert head.startswith(b"HTTP/1.1 403 Forbidden")


@pytest.mark.asyncio
async def test_connect_uses_the_once_resolved_ip(monkeypatch) -> None:
    proxy = GatewayEgressProxy(SimpleNamespace(keys=SimpleNamespace(task_token=b"k")))
    proxy._authorize = lambda **_kwargs: asyncio.sleep(  # type: ignore[method-assign]
        0,
        result=(True, None, False),
    )
    proxy._resolve = lambda *_args: asyncio.sleep(  # type: ignore[method-assign]
        0,
        result="8.8.8.8",
    )
    monkeypatch.setattr(
        egress_proxy,
        "decode_task_token",
        lambda *_args: SimpleNamespace(task_id="task-1"),
    )
    connected: list[tuple[str, int]] = []

    async def open_connection(host: str, port: int):
        connected.append((host, port))
        upstream_reader = asyncio.StreamReader()
        upstream_reader.feed_eof()
        return upstream_reader, MemoryWriter()

    monkeypatch.setattr(egress_proxy.asyncio, "open_connection", open_connection)
    reader = asyncio.StreamReader()
    reader.feed_data(
        b"CONNECT example.com:443 HTTP/1.1\r\n"
        b"Proxy-Authorization: Bearer token\r\n"
        b"X-Sandbox-Timestamp: 1\r\n"
        b"X-Sandbox-Nonce: nonce\r\n\r\n"
    )
    reader.feed_eof()
    writer = MemoryWriter()

    await proxy.handle(reader, writer)

    assert connected == [("8.8.8.8", 443)]
    assert bytes(writer.data).startswith(
        b"HTTP/1.1 200 Connection Established\r\n\r\n"
    )
