"""Health, graceful shutdown, and startup recovery tests."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from open_deep_research import server
from open_deep_research.configuration import Configuration
from open_deep_research.public_events import RunEventStore
from open_deep_research.run_context import RunContextStore
from open_deep_research.tasks.lease import LeaderLeaseManager


@pytest.fixture(autouse=True)
def _reset_server_lifecycle_state():
    server._shutting_down.clear()
    server._sse_shutdown.clear()
    server._runs.clear()
    yield
    server._shutting_down.clear()
    server._sse_shutdown.clear()
    server._runs.clear()


def test_healthz_is_public_and_dependency_free() -> None:
    client = TestClient(server.app)

    response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_readyz_reports_degraded_search_without_failing(monkeypatch) -> None:
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.setenv("OBSERVABILITY_ENABLED", "false")
    monkeypatch.setattr(server, "_probe_runs_directory", lambda _config: None)
    monkeypatch.setattr(
        server,
        "get_iam_settings",
        lambda: SimpleNamespace(database_url=""),
    )

    report, ready = await server._readiness_report()

    assert ready is True
    assert report["status"] == "degraded"
    assert report["components"]["search"]["reason"] == "api_key_missing"


@pytest.mark.asyncio
async def test_readyz_fails_when_runs_directory_is_not_writable(monkeypatch) -> None:
    monkeypatch.setenv("OBSERVABILITY_ENABLED", "false")
    monkeypatch.setattr(
        server,
        "_probe_runs_directory",
        lambda _config: (_ for _ in ()).throw(PermissionError("denied")),
    )
    monkeypatch.setattr(
        server,
        "get_iam_settings",
        lambda: SimpleNamespace(database_url=""),
    )

    report, ready = await server._readiness_report()

    assert ready is False
    assert report["status"] == "failed"
    assert report["components"]["runs_dir"] == {
        "status": "failed",
        "error_type": "PermissionError",
    }


@pytest.mark.asyncio
async def test_readyz_immediately_fails_while_shutting_down(monkeypatch) -> None:
    monkeypatch.setenv("OBSERVABILITY_ENABLED", "false")
    monkeypatch.setattr(server, "_probe_runs_directory", lambda _config: None)
    monkeypatch.setattr(
        server,
        "get_iam_settings",
        lambda: SimpleNamespace(database_url=""),
    )
    server._shutting_down.set()

    response = await server.readyz()

    assert response.status_code == 503
    assert json.loads(response.body)["components"]["server"]["reason"] == "shutting_down"


@pytest.mark.asyncio
async def test_shutdown_drain_persists_trace_and_cancels_run(tmp_path, monkeypatch) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{}", encoding="utf-8")
    manifest_updates = []
    trace_finishes = []
    public_events = []

    class FakeStore:
        def __init__(self):
            self.manifest_path = manifest_path

        def _update_manifest(self, **updates):
            manifest_updates.append(updates)

    class FakeRecorder:
        def finish_run(self, run_id, status):
            trace_finishes.append((run_id, status))

    class FakePublisher:
        async def publish(self, event_type, **kwargs):
            public_events.append((event_type, kwargs["payload"]["status"]))

    cancelled = asyncio.Event()

    async def live_task():
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    config = {
        "configurable": {"observability_enabled": False},
        "metadata": {"run_id": "drain-run"},
    }
    engine = SimpleNamespace(config=config, context_store=FakeStore())
    record = server._new_run_record(
        run_id="drain-run",
        engine=engine,
        status="running",
        config=config,
    )
    record.task = asyncio.create_task(live_task())
    server._remember_run(record, config)
    monkeypatch.setattr(server, "get_trace_recorder", lambda _config: FakeRecorder())
    monkeypatch.setattr(
        server,
        "event_publisher_from_config",
        lambda _config: FakePublisher(),
    )

    await server._drain_inflight_runs(1)

    assert record.status == "interrupted"
    assert manifest_updates == [{"status": "interrupted"}]
    assert trace_finishes == [("drain-run", "interrupted")]
    assert public_events == [("run.interrupted", "interrupted")]
    assert record.task.cancelled()
    assert cancelled.is_set()


@pytest.mark.asyncio
async def test_startup_sweep_interrupts_expired_but_not_live_lease(tmp_path) -> None:
    config = {
        "configurable": {
            "runs_dir": str(tmp_path),
            "observability_enabled": False,
        },
        "metadata": {},
    }
    for run_id in ("orphan-run", "active-run"):
        store = RunContextStore(run_id, runs_dir=str(tmp_path))
        store.initialize("user-1", {**config, "metadata": {"run_id": run_id}})
        store._update_manifest(status="running")

    expired = LeaderLeaseManager(
        runs_dir=str(tmp_path),
        run_id="orphan-run",
        owner_id="expired-owner",
    )
    expired_lease = await expired.acquire()
    await expired.release(expected_fence_token=expired_lease.fence_token)
    active = LeaderLeaseManager(
        runs_dir=str(tmp_path),
        run_id="active-run",
        owner_id="active-owner",
    )
    active_lease = await active.acquire()

    configurable = Configuration(
        runs_dir=str(tmp_path),
        observability_enabled=False,
    )
    try:
        interrupted = await server._run_recovery_sweep(configurable)
    finally:
        await active.release(expected_fence_token=active_lease.fence_token)

    assert interrupted == 1
    assert RunContextStore(
        "orphan-run", runs_dir=str(tmp_path)
    ).load_manifest().status == "interrupted"
    assert RunContextStore(
        "active-run", runs_dir=str(tmp_path)
    ).load_manifest().status == "running"
    assert [
        event.type
        for event in RunEventStore("orphan-run", runs_dir=str(tmp_path)).read()
    ] == ["run.interrupted"]
