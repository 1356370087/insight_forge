from __future__ import annotations

import asyncio

import pytest

from open_deep_research.runtime_control import CancellationScope, RunCancelled


@pytest.mark.asyncio
async def test_cancellation_scope_cancels_and_drains_active_work():
    scope = CancellationScope()
    started = asyncio.Event()
    finalized = asyncio.Event()

    async def work() -> None:
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            finalized.set()

    task = asyncio.create_task(scope.run(work(), stage="tool"))
    await started.wait()
    assert scope.request("user_cancelled") is True

    with pytest.raises(RunCancelled) as raised:
        await task

    assert raised.value.reason == "user_cancelled"
    assert raised.value.stage == "tool"
    assert finalized.is_set()
    assert scope.request("later_reason") is False
    assert scope.reason == "user_cancelled"


@pytest.mark.asyncio
async def test_cancellation_scope_returns_completed_work():
    scope = CancellationScope()

    result = await scope.run(asyncio.sleep(0, result="done"), stage="model")

    assert result == "done"


@pytest.mark.asyncio
async def test_cancellation_scope_times_out_and_drains_work():
    scope = CancellationScope()
    finalized = asyncio.Event()

    async def work() -> None:
        try:
            await asyncio.Event().wait()
        finally:
            finalized.set()

    with pytest.raises(TimeoutError, match="hook exceeded"):
        await scope.run(work(), stage="hook", timeout_seconds=0.01)

    assert finalized.is_set()
