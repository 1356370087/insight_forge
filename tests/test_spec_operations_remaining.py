"""Acceptance coverage for the remaining operations-control SPEC workstreams."""

from __future__ import annotations

import argparse
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from open_deep_research import server
from open_deep_research.api_governance import ConnectionLimiter, FixedWindowRateLimiter
from open_deep_research.configuration import Configuration
from open_deep_research.logging_config import (
    JSONFormatter,
    RequestContextFilter,
    bind_request_id,
)
from open_deep_research.memory import maintenance
from open_deep_research.observability.core import SQLiteTraceStore, TokenUsage
from open_deep_research.run_context import RunContextStore
from open_deep_research.sandbox import manager as sandbox_manager
from security.rbac.principal import Principal


def _principal(user_id: str, *, admin: bool = False) -> Principal:
    return Principal(
        user_id=user_id,
        email=f"{user_id}@example.test",
        status="active",
        session_id="session",
        roles=frozenset({"admin"} if admin else {"researcher"}),
        permissions=frozenset({"research.run.control_own"}),
        authz_version=1,
    )


def _terminal_run(
    runs_dir: Path,
    run_id: str,
    *,
    owner: str = "owner",
    ended_at: float | None = None,
    payload_bytes: int = 0,
) -> RunContextStore:
    store = RunContextStore(run_id, runs_dir=str(runs_dir))
    store.initialize(owner, {"configurable": {"runs_dir": str(runs_dir)}})
    store._update_manifest(  # noqa: SLF001
        status="completed",
        ended_at=ended_at or time.time(),
    )
    if payload_bytes:
        (store.context_dir / "payload.bin").write_bytes(b"x" * payload_bytes)
    return store


@pytest.mark.asyncio
async def test_manual_purge_cascades_directory_trace_and_memory(tmp_path) -> None:
    runs_dir = tmp_path / "runs"
    trace_path = tmp_path / "traces.sqlite3"
    run_id = "purge-run"
    store = _terminal_run(runs_dir, run_id)
    traces = SQLiteTraceStore(str(trace_path))
    traces.start_run(run_id, "owner", {})
    traces.start_span(
        span_id="span-1",
        run_id=run_id,
        parent_span_id=None,
        name="test",
        kind="llm",
        agent_role="researcher",
        attributes={},
        input_preview=None,
        provider="test",
        model="test",
    )
    traces.add_usage(run_id, "span-1", "test", "test", TokenUsage(input_tokens=1))
    traces.record_retry_event(
        run_id=run_id,
        span_id="span-1",
        attempt=1,
        error_type="transient",
    )
    config = Configuration(
        runs_dir=str(runs_dir),
        trace_store_path=str(trace_path),
        prometheus_enabled=False,
    )

    result = await server._purge_run_artifacts(
        run_id,
        config,
        reason="manual",
        actor="owner",
    )

    assert result["status"] == "deleted"
    assert not store.context_dir.parent.exists()
    assert traces.get_run(run_id) is None
    assert traces.list_spans(run_id) == []
    assert traces.get_usage(run_id)["total_tokens"] == 0


@pytest.mark.asyncio
async def test_delete_run_enforces_owner_isolation(tmp_path, monkeypatch) -> None:
    runs_dir = tmp_path / "runs"
    trace_path = tmp_path / "traces.sqlite3"
    _terminal_run(runs_dir, "owned", owner="owner")
    config = Configuration(runs_dir=str(runs_dir), trace_store_path=str(trace_path))
    monkeypatch.setattr(
        server.Configuration,
        "from_runnable_config",
        lambda _config: config,
    )

    with pytest.raises(HTTPException) as denied:
        await server.delete_run("owned", user=_principal("other"))
    assert denied.value.status_code == 404

    result = await server.delete_run("owned", user=_principal("owner"))
    assert result["status"] == "deleted"

    active = RunContextStore("active-owned", runs_dir=str(runs_dir))
    active.initialize("owner", {"configurable": {"runs_dir": str(runs_dir)}})
    with pytest.raises(HTTPException) as conflict:
        await server.delete_run("active-owned", user=_principal("owner"))
    assert conflict.value.status_code == 409

    _terminal_run(runs_dir, "admin-target", owner="someone-else")
    admin_result = await server.delete_run(
        "admin-target",
        user=_principal("administrator", admin=True),
    )
    assert admin_result["status"] == "deleted"


@pytest.mark.asyncio
async def test_retention_sweep_deletes_expired_terminal_only(tmp_path) -> None:
    runs_dir = tmp_path / "runs"
    old = time.time() - 10 * 86400
    _terminal_run(runs_dir, "expired", ended_at=old)
    retained = _terminal_run(runs_dir, "recent")
    active = RunContextStore("active", runs_dir=str(runs_dir))
    active.initialize("owner", {"configurable": {"runs_dir": str(runs_dir)}})
    config = Configuration(
        runs_dir=str(runs_dir),
        trace_store_path=str(tmp_path / "traces.sqlite3"),
        run_retention_days=1,
        trace_retention_days=0,
        prometheus_enabled=False,
    )

    result = await server._run_retention_sweep(config)

    assert result["deleted_by_age"] == 1
    assert not (runs_dir / "expired").exists()
    assert retained.context_dir.parent.exists()
    assert active.context_dir.parent.exists()


@pytest.mark.asyncio
async def test_retention_zero_keeps_runs_and_quota_reclaims_to_watermark(tmp_path) -> None:
    keep_dir = tmp_path / "keep"
    _terminal_run(keep_dir, "forever", ended_at=time.time() - 100 * 86400)
    keep_config = Configuration(
        runs_dir=str(keep_dir),
        trace_store_path=str(tmp_path / "keep-traces.sqlite3"),
        run_retention_days=0,
        trace_retention_days=0,
    )
    result = await server._run_retention_sweep(keep_config)
    assert result["deleted_by_age"] == 0
    assert (keep_dir / "forever").exists()

    quota_dir = tmp_path / "quota"
    _terminal_run(quota_dir, "old-a", ended_at=time.time() - 20, payload_bytes=100_000)
    _terminal_run(quota_dir, "old-b", ended_at=time.time() - 10, payload_bytes=100_000)
    quota_config = Configuration(
        runs_dir=str(quota_dir),
        trace_store_path=str(tmp_path / "quota-traces.sqlite3"),
        run_retention_days=0,
        trace_retention_days=0,
        runs_dir_max_bytes=150_000,
    )
    quota_result = await server._run_retention_sweep(quota_config)
    assert quota_result["deleted_by_quota"] >= 1
    assert server._runs_dir_size_bytes(quota_dir) <= 135_000


@pytest.mark.asyncio
async def test_trace_retention_is_independent_from_run_directory_retention(tmp_path) -> None:
    runs_dir = tmp_path / "runs"
    run_id = "trace-expired"
    _terminal_run(runs_dir, run_id)
    trace_path = tmp_path / "traces.sqlite3"
    traces = SQLiteTraceStore(str(trace_path))
    traces.start_run(run_id, "owner", {})
    with traces._lock, traces._connect() as conn:  # noqa: SLF001
        conn.execute(
            "UPDATE runs SET status='completed', ended_at=? WHERE run_id=?",
            (time.time() - 10 * 86400, run_id),
        )
    config = Configuration(
        runs_dir=str(runs_dir),
        trace_store_path=str(trace_path),
        run_retention_days=0,
        trace_retention_days=1,
    )

    result = await server._run_retention_sweep(config)

    assert result["trace_runs_deleted"] == 1
    assert traces.get_run(run_id) is None
    assert (runs_dir / run_id).exists()


def test_json_logging_and_request_id_round_trip(monkeypatch) -> None:
    bind_request_id("test-123")
    record = logging.LogRecord("spec", logging.INFO, __file__, 1, "hello", (), None)
    RequestContextFilter().filter(record)
    payload = json.loads(JSONFormatter().format(record))
    assert payload["request_id"] == "test-123"

    monkeypatch.setenv("MAX_REQUEST_BODY_BYTES", "1024")
    client = TestClient(server.app)
    response = client.get("/healthz", headers={"X-Request-ID": "test-123"})
    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == "test-123"
    too_large = client.post(
        "/healthz",
        content=b"x" * 2048,
        headers={"Content-Type": "application/octet-stream"},
    )
    assert too_large.status_code == 413

    def chunks():
        yield b"x" * 700
        yield b"y" * 700

    streamed = client.post(
        "/healthz",
        content=chunks(),
        headers={"Content-Type": "application/octet-stream"},
    )
    assert streamed.status_code == 413


def test_request_id_is_written_to_run_metadata() -> None:
    bind_request_id("run-request-123")
    config = server._config_from_request(
        server.RunRequest(messages=[{"role": "user", "content": "test"}]),
        _principal("owner"),
    )
    assert config["metadata"]["request_id"] == "run-request-123"


def test_fixed_window_rate_limiter_rejects_eleventh_request() -> None:
    limiter = FixedWindowRateLimiter()
    for _ in range(10):
        assert limiter.allow("principal", 10, now=0) == (True, 0)
    allowed, retry_after = limiter.allow("principal", 10, now=0)
    assert not allowed
    assert retry_after == 60
    assert limiter.allow("principal", 10, now=61) == (True, 0)


def test_run_create_guard_rejects_eleventh_and_terminal_releases_slot() -> None:
    user = _principal("rate-owner")
    config = Configuration(
        api_run_create_per_minute=10,
        max_concurrent_runs_per_user=0,
        prometheus_enabled=False,
    )
    server._api_rate_limiter.clear()
    for _ in range(10):
        server._enforce_run_create_limits(user, config)
    with pytest.raises(HTTPException) as limited:
        server._enforce_run_create_limits(user, config)
    assert limited.value.status_code == 429
    assert limited.value.headers["Retry-After"]

    engine = SimpleNamespace(config={"metadata": {"owner": user.user_id}})
    active = server.RunRecord(run_id="active-limit", engine=engine, status="running")
    terminal = server.RunRecord(
        run_id="terminal-limit",
        engine=engine,
        status="completed",
    )
    server._runs[active.run_id] = active
    server._runs[terminal.run_id] = terminal
    try:
        assert server._active_runs_for_user(user.user_id) == 1
        active.status = "completed"
        assert server._active_runs_for_user(user.user_id) == 0
    finally:
        server._runs.pop(active.run_id, None)
        server._runs.pop(terminal.run_id, None)


@pytest.mark.asyncio
async def test_connection_limiter_releases_terminal_slot() -> None:
    limiter = ConnectionLimiter()
    assert await limiter.acquire(1)
    assert not await limiter.acquire(1)
    await limiter.release(1)
    assert await limiter.acquire(1)


def test_run_limiter_failure_is_fail_open(monkeypatch) -> None:
    config = Configuration(
        api_run_create_per_minute=1,
        max_concurrent_runs_per_user=0,
        prometheus_enabled=False,
    )
    monkeypatch.setattr(
        server._api_rate_limiter,
        "allow",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("broken")),
    )
    server._enforce_run_create_limits(_principal("owner"), config)


def test_sqlite_busy_timeout_supports_concurrent_usage_writes(tmp_path) -> None:
    store = SQLiteTraceStore(str(tmp_path / "concurrent.sqlite3"))
    store.start_run("run", "owner", {})

    def write_usage(index: int) -> None:
        store.add_usage(
            "run",
            f"span-{index}",
            "test",
            "test",
            TokenUsage(input_tokens=1),
        )

    with ThreadPoolExecutor(max_workers=10) as pool:
        list(pool.map(write_usage, range(100)))

    assert store.get_usage("run")["input_tokens"] == 100
    with store._connect() as conn:  # noqa: SLF001
        assert conn.execute("PRAGMA busy_timeout").fetchone()[0] == 5000
        assert conn.execute("PRAGMA synchronous").fetchone()[0] == 1


@pytest.mark.asyncio
async def test_memory_maintenance_loop_uses_configured_interval(monkeypatch) -> None:
    calls: list[int] = []
    delays: list[float] = []

    async def fake_daily(_args, _config):
        calls.append(1)
        return {"iteration": len(calls)}

    async def fake_sleep(delay: float) -> None:
        delays.append(delay)

    monkeypatch.setattr(maintenance, "_run_daily", fake_daily)
    args = argparse.Namespace(interval_hours=2, dry_run=True, user_id=None, loop=True)
    results = await maintenance._run_daily_loop(
        args,
        Configuration(),
        sleep=fake_sleep,
        max_iterations=2,
    )
    assert results == [{"iteration": 1}, {"iteration": 2}]
    assert delays == [7200]


def test_sandbox_secret_injection_can_be_disabled(monkeypatch) -> None:
    monkeypatch.setenv("SANDBOX_SECRET_ENV_KEYS", "")
    assert sandbox_manager._sandbox_secret_env_keys() == ()
