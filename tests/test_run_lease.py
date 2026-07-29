from __future__ import annotations

import asyncio

import pytest
from langchain_core.messages import HumanMessage

from open_deep_research.agents.query_engine import QueryEngine
from open_deep_research.configuration import freeze_run_config
from open_deep_research.run_context import RunContextStore
from open_deep_research.tasks.lease import FenceLostError, LeaderLeaseManager


@pytest.mark.asyncio
async def test_lease_takeover_increments_fence_token(tmp_path):
    first = LeaderLeaseManager(
        runs_dir=str(tmp_path),
        run_id="run-fence",
        lease_seconds=0.01,
    )
    first_lease = await first.acquire()
    assert first_lease.fence_token == 1

    await asyncio.sleep(0.02)
    second = LeaderLeaseManager(
        runs_dir=str(tmp_path),
        run_id="run-fence",
        lease_seconds=10,
    )
    second.owner_id = "second-owner"
    second_lease = await second.acquire()

    assert second_lease.fence_token == 2
    with pytest.raises(RuntimeError, match="Lost Lead lease"):
        await first.renew()


@pytest.mark.asyncio
async def test_same_lease_owner_preserves_fence_token(tmp_path):
    manager = LeaderLeaseManager(
        runs_dir=str(tmp_path),
        run_id="run-same-owner",
    )

    acquired = await manager.acquire()
    renewed = await manager.renew()

    assert renewed.fence_token == acquired.fence_token


@pytest.mark.asyncio
async def test_expired_same_owner_starts_new_fence_epoch(tmp_path):
    manager = LeaderLeaseManager(
        runs_dir=str(tmp_path),
        run_id="run-expired-owner",
        lease_seconds=0.01,
    )
    first = await manager.acquire()

    await asyncio.sleep(0.02)

    with pytest.raises(FenceLostError, match="expired"):
        await manager.renew(expected_fence_token=first.fence_token)

    second = await manager.acquire()
    assert second.fence_token == first.fence_token + 1
    assert second.acquired_at > first.acquired_at


@pytest.mark.asyncio
async def test_stale_fence_cannot_release_successor_lease(tmp_path):
    first = LeaderLeaseManager(
        runs_dir=str(tmp_path),
        run_id="run-stale-release",
        lease_seconds=0.01,
        owner_id="first-owner",
    )
    first_lease = await first.acquire()
    await asyncio.sleep(0.02)

    second = LeaderLeaseManager(
        runs_dir=str(tmp_path),
        run_id="run-stale-release",
        lease_seconds=10,
        owner_id="second-owner",
    )
    second_lease = await second.acquire()

    with pytest.raises(FenceLostError):
        await first.release(expected_fence_token=first_lease.fence_token)
    assert await second.is_owner(expected_fence_token=second_lease.fence_token)


@pytest.mark.asyncio
async def test_fenced_operation_rejects_stale_epoch(tmp_path):
    first = LeaderLeaseManager(
        runs_dir=str(tmp_path),
        run_id="run-fenced-operation",
        lease_seconds=0.01,
        owner_id="first-owner",
    )
    first_lease = await first.acquire()
    await asyncio.sleep(0.02)

    second = LeaderLeaseManager(
        runs_dir=str(tmp_path),
        run_id="run-fenced-operation",
        lease_seconds=10,
        owner_id="second-owner",
    )
    second_lease = await second.acquire()

    with pytest.raises(FenceLostError):
        await first.run_fenced(first_lease.fence_token, lambda: "stale")
    assert await second.run_fenced(second_lease.fence_token, lambda: "current") == "current"


@pytest.mark.asyncio
async def test_run_context_rejects_writes_from_superseded_lease(tmp_path):
    first = LeaderLeaseManager(
        runs_dir=str(tmp_path),
        run_id="context-fence",
        owner_id="first",
    )
    first_lease = await first.acquire()
    stale_store = RunContextStore("context-fence", runs_dir=str(tmp_path))
    stale_store.bind_fence_token(first_lease.fence_token, first.owner_id)
    stale_store.initialize("user", {"metadata": {"run_id": "context-fence"}})

    await first.release(expected_fence_token=first_lease.fence_token)
    second = LeaderLeaseManager(
        runs_dir=str(tmp_path),
        run_id="context-fence",
        owner_id="second",
    )
    second_lease = await second.acquire()
    current_store = RunContextStore("context-fence", runs_dir=str(tmp_path))
    current_store.bind_fence_token(second_lease.fence_token, second.owner_id)

    with pytest.raises(FenceLostError):
        stale_store.write_text_atomic("stale.txt", "must not commit")
    current_store.write_text_atomic("current.txt", "committed")

    assert not (tmp_path / "context-fence" / "context" / "stale.txt").exists()
    assert (tmp_path / "context-fence" / "context" / "current.txt").exists()


def _engine_config(tmp_path, run_id: str, *, owner: str = "user"):
    return {
        "configurable": {
            "runs_dir": str(tmp_path),
            "event_log_enabled": False,
            "search_api": "none",
        },
        "metadata": {"run_id": run_id, "owner": owner},
    }


@pytest.mark.asyncio
async def test_stream_message_releases_lease_when_startup_publication_fails(
    monkeypatch,
    tmp_path,
):
    engine = QueryEngine(_engine_config(tmp_path, "startup-failure"))
    acquired_token = None

    async def fail_publication(*_args, **_kwargs):
        nonlocal acquired_token
        acquired_token = engine.run_fence_token
        raise RuntimeError("public event unavailable")

    monkeypatch.setattr(engine, "_publish_public", fail_publication)
    try:
        with pytest.raises(RuntimeError, match="public event unavailable"):
            async for _ in engine.stream_message([HumanMessage(content="research")]):
                pass

        assert acquired_token is not None
        assert engine.run_fence_token is None
        assert not await engine.run_lease.is_owner(
            expected_fence_token=acquired_token
        )
    finally:
        await engine.release_run_lease()


@pytest.mark.asyncio
async def test_stream_message_releases_lease_when_fence_binding_fails(
    monkeypatch,
    tmp_path,
):
    engine = QueryEngine(_engine_config(tmp_path, "fence-binding-failure"))
    assert engine.context_store is not None

    def fail_binding(_self, *_args, **_kwargs):
        raise RuntimeError("fence binding failed")

    monkeypatch.setattr(RunContextStore, "bind_fence_token", fail_binding)
    try:
        with pytest.raises(RuntimeError, match="fence binding failed"):
            async for _ in engine.stream_message([HumanMessage(content="research")]):
                pass

        assert engine.run_fence_token is None
        assert engine.run_lease.fence_token is None
    finally:
        await engine.release_run_lease()


@pytest.mark.asyncio
async def test_stream_message_rebinds_run_scoped_resources_when_run_id_changes(
    monkeypatch,
    tmp_path,
):
    engine = QueryEngine(_engine_config(tmp_path, "original-run"))

    async def fail_after_acquire(*_args, **_kwargs):
        raise RuntimeError("stop after resource binding")

    monkeypatch.setattr(engine, "_publish_public", fail_after_acquire)
    try:
        with pytest.raises(RuntimeError, match="stop after resource binding"):
            async for _ in engine.stream_message(
                [HumanMessage(content="research")],
                _engine_config(tmp_path, "replacement-run"),
            ):
                pass

        assert engine.run_lease.run_id == "replacement-run"
        assert engine.context_store is not None
        assert engine.context_store.run_id == "replacement-run"
    finally:
        await engine.release_run_lease()


@pytest.mark.asyncio
async def test_stream_resume_does_not_acquire_lease_when_owner_validation_fails(
    tmp_path,
):
    run_id = "resume-owner-mismatch"
    store = RunContextStore(run_id, runs_dir=str(tmp_path))
    store.initialize(
        "expected-owner",
        freeze_run_config(_engine_config(tmp_path, run_id)),
    )
    engine = QueryEngine.load(
        run_id,
        runs_dir=str(tmp_path),
        config=_engine_config(tmp_path, run_id, owner="intruder"),
    )

    try:
        with pytest.raises(PermissionError, match="run_owner_mismatch"):
            async for _ in engine.stream_resume():
                pass

        assert engine.run_fence_token is None
        assert engine.run_lease.fence_token is None
        assert engine.context_store is not None
        assert engine.context_store.load_manifest().fence_token == 0
    finally:
        await engine.release_run_lease()


@pytest.mark.asyncio
async def test_heartbeat_io_failure_requests_run_cancellation(monkeypatch, tmp_path):
    config = _engine_config(tmp_path, "heartbeat-io-failure")
    config["configurable"].update({
        "leader_heartbeat_seconds": 0.01,
        "leader_lease_seconds": 2,
    })
    engine = QueryEngine(config)
    engine.run_fence_token = 7
    engine.status = "running"

    async def fail_renewal(*, expected_fence_token):
        assert expected_fence_token == 7
        raise OSError("lease storage unavailable")

    monkeypatch.setattr(engine.run_lease, "renew", fail_renewal)
    await asyncio.wait_for(engine._lease_heartbeat(), 1)

    assert engine.cancellation_scope.is_cancelled is True
    assert engine.cancellation_scope.reason == "lease_lost"
    assert engine.cancelled is True
    assert engine.status == "cancelling"
