from __future__ import annotations

import asyncio

import pytest

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
