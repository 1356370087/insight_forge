"""Process-local run retention and event buffer tests."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from open_deep_research import server
from open_deep_research.configuration import Configuration


@pytest.fixture(autouse=True)
def _clear_run_memory():
    server._runs.clear()
    for task in server._run_eviction_tasks.values():
        task.cancel()
    server._run_eviction_tasks.clear()
    yield
    server._runs.clear()
    for task in server._run_eviction_tasks.values():
        task.cancel()
    server._run_eviction_tasks.clear()


def _config(**overrides):
    return {"configurable": overrides}


def _record(run_id: str, config, *, status: str = "running"):
    engine = SimpleNamespace(config=config)
    return server._new_run_record(
        run_id=run_id,
        engine=engine,
        status=status,
        config=config,
    )


def test_event_buffer_keeps_only_configured_recent_events() -> None:
    config = _config(inflight_event_buffer_size=3)
    record = _record("bounded-events", config)

    for sequence in range(5):
        record.events.append({"sequence": sequence})

    assert record.events.maxlen == 3
    assert [event["sequence"] for event in record.events] == [2, 3, 4]


@pytest.mark.asyncio
async def test_terminal_run_is_evicted_after_retention() -> None:
    config = _config(
        inflight_run_memory_retention_seconds=0,
        max_inflight_runs_in_memory=10,
    )
    record = _record("finished", config, status="completed")
    server._remember_run(record, config)

    server._schedule_run_eviction(record, config)
    task = server._run_eviction_tasks[record.run_id]
    await task

    assert record.finished_at is not None
    assert record.run_id not in server._runs


def test_capacity_evicts_oldest_terminal_record() -> None:
    config = _config(max_inflight_runs_in_memory=2)
    oldest = _record("oldest", config, status="completed")
    oldest.finished_at = 1.0
    newer = _record("newer", config, status="failed")
    newer.finished_at = 2.0
    active = _record("active", config)

    server._remember_run(oldest, config)
    server._remember_run(newer, config)
    server._remember_run(active, config)

    assert set(server._runs) == {"newer", "active"}


@pytest.mark.asyncio
async def test_stale_eviction_does_not_remove_resumed_replacement() -> None:
    config = _config(
        inflight_run_memory_retention_seconds=0,
        max_inflight_runs_in_memory=10,
    )
    old = _record("same-run", config, status="completed")
    server._remember_run(old, config)
    server._schedule_run_eviction(old, config)
    stale_task = server._run_eviction_tasks[old.run_id]

    replacement = _record("same-run", config)
    server._remember_run(replacement, config)
    await asyncio.gather(stale_task, return_exceptions=True)

    assert server._runs[old.run_id] is replacement


def test_memory_configuration_uses_environment_precedence(monkeypatch) -> None:
    monkeypatch.setenv("INFLIGHT_EVENT_BUFFER_SIZE", "7")

    configured = Configuration.from_runnable_config(
        _config(inflight_event_buffer_size=3)
    )

    assert configured.inflight_event_buffer_size == 7
