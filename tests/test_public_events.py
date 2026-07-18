import asyncio
import json

import pytest
from fastapi.testclient import TestClient

from open_deep_research.public_events import (
    PublicEventLogCorrupted,
    RunEventStore,
    canonical_public_source,
    extract_public_sources,
    project_public_events,
    sanitize_public_payload,
)
from open_deep_research.run_context import RunContextStore
from open_deep_research.run_control import RunControlStore


@pytest.mark.asyncio
async def test_public_event_store_is_ordered_and_idempotent(tmp_path):
    store = RunEventStore("run-1", runs_dir=str(tmp_path))

    first = await store.append(
        "run.created",
        payload={"status": "pending"},
        dedupe_key="run:created",
    )
    duplicate = await store.append(
        "run.created",
        payload={"status": "different"},
        dedupe_key="run:created",
    )
    second = await store.append(
        "run.started",
        payload={"status": "running"},
        dedupe_key="run:started",
    )

    assert first.sequence == duplicate.sequence == 1
    assert first.event_id == duplicate.event_id
    assert second.sequence == 2
    assert [event.sequence for event in store.read()] == [1, 2]


@pytest.mark.asyncio
async def test_public_event_store_serializes_concurrent_publishers(tmp_path):
    stores = [RunEventStore("run-concurrent", runs_dir=str(tmp_path)) for _ in range(4)]

    events = await asyncio.gather(
        *[
            stores[index % len(stores)].append(
                "research.task.progress",
                stage="researching",
                payload={"task_id": f"task-{index}", "iteration": index},
                dedupe_key=f"task:{index}:progress",
            )
            for index in range(40)
        ]
    )

    assert sorted(event.sequence for event in events) == list(range(1, 41))
    assert [event.sequence for event in stores[0].read()] == list(range(1, 41))
    assert len({event.event_id for event in events}) == 40


@pytest.mark.asyncio
async def test_event_store_repairs_only_a_truncated_tail(tmp_path):
    store = RunEventStore("run-tail", runs_dir=str(tmp_path))
    await store.append(
        "run.created",
        payload={"status": "pending"},
        dedupe_key="run:created",
    )
    with store.path.open("ab") as handle:
        handle.write(b'{"partial":')

    records = store.read()

    assert len(records) == 1
    assert store.path.read_bytes().endswith(b"\n")


@pytest.mark.asyncio
async def test_event_store_rejects_middle_corruption_and_marks_manifest_failed(tmp_path):
    context = RunContextStore("run-corrupt", runs_dir=str(tmp_path))
    context.initialize("user-1", {"metadata": {"run_id": "run-corrupt"}})
    store = RunEventStore("run-corrupt", runs_dir=str(tmp_path))
    await store.append("run.created", payload={"status": "pending"}, dedupe_key="run:created")
    await store.append("run.started", payload={"status": "running"}, dedupe_key="run:started")
    lines = store.path.read_bytes().splitlines()
    store.path.write_bytes(lines[0] + b"\n{not-json}\n" + lines[1] + b"\n")

    with pytest.raises(PublicEventLogCorrupted):
        store.read()

    manifest = context.load_manifest()
    assert manifest.status == "failed"
    assert manifest.persistence_degraded is True
    assert manifest.persistence_error.startswith("event_persistence_failed")


@pytest.mark.asyncio
async def test_run_control_commands_are_idempotent_and_path_safe(tmp_path):
    store = RunControlStore("run-control", runs_dir=str(tmp_path))

    first = await store.enqueue("feedback", {"message": "one"}, command_id="feedback-1")
    duplicate = await store.enqueue("feedback", {"message": "two"}, command_id="feedback-1")

    assert first == duplicate
    assert len(await store.pending()) == 1
    await store.ack(first)
    assert await store.pending() == []
    assert (await store.enqueue("feedback", {}, command_id="feedback-1")) == first
    with pytest.raises(ValueError, match="Invalid command_id"):
        await store.enqueue("cancel", {}, command_id="../escape")


def test_public_sanitizer_and_source_canonicalization():
    payload = sanitize_public_payload(
        "research.task.progress",
        {
            "task_id": "task-1",
            "phase": "researching",
            "api_key": "secret",
            "tool_result": "raw body",
            "unknown": "not allowed",
        },
    )
    source = canonical_public_source(
        "https://user:password@example.com/path?q=secret#fragment",
        "Example",
    )

    assert payload == {"task_id": "task-1", "phase": "researching"}
    assert source is not None
    assert source["url"] == "https://example.com/path"
    assert "password" not in json.dumps(source)


def test_public_sources_are_extracted_from_compressed_text_without_leaking_body():
    sources = extract_public_sources({
        "compressed_research": (
            "Finding with [official source](https://user:secret@example.com/docs?q=private#part) "
            "and duplicate https://example.com/docs?other=value."
        ),
        "raw_notes": ["Also see https://numpy.org/devdocs/reference/thread_safety.html)."],
    })

    assert [source["url"] for source in sources] == [
        "https://example.com/docs",
        "https://numpy.org/devdocs/reference/thread_safety.html",
    ]
    assert all("private" not in json.dumps(source) for source in sources)


@pytest.mark.asyncio
async def test_projection_reduces_plan_tasks_waves_and_findings(tmp_path):
    store = RunEventStore("run-projection", runs_dir=str(tmp_path))
    await store.append(
        "plan.created",
        stage="planning",
        payload={"plan_id": "plan-1", "revision": 1, "objective": "Test", "stages": []},
        dedupe_key="plan:created",
    )
    await store.append(
        "plan.task.added",
        stage="researching",
        payload={"task_id": "task-1", "wave_id": "wave-1", "title": "Task", "mode": "sync", "status": "pending"},
        dedupe_key="plan:task:1",
    )
    await store.append(
        "research.wave.started",
        stage="researching",
        payload={"wave_id": "wave-1", "mode": "sync", "task_ids": ["task-1"], "task_count": 1},
        dedupe_key="wave:1:started",
    )
    await store.append(
        "research.task.completed",
        stage="researching",
        payload={"task_id": "task-1", "status": "completed", "phase": "completed", "mode": "sync"},
        dedupe_key="task:1:completed",
    )
    await store.append(
        "findings.updated",
        stage="researching",
        payload={"task_id": "task-1", "summary": "Finding", "sources": [], "source_count": 0},
        dedupe_key="task:1:findings",
    )
    await store.append(
        "research.wave.completed",
        stage="researching",
        payload={"wave_id": "wave-1", "mode": "sync", "task_ids": ["task-1"], "task_count": 1, "completed": 1, "failed": 0, "rejected": 0},
        dedupe_key="wave:1:completed",
    )

    projection = project_public_events(store.read())

    assert projection.tasks == {
        "total": 1,
        "pending": 0,
        "running": 0,
        "completed": 1,
        "failed": 0,
        "cancelled": 0,
        "timed_out": 0,
    }
    assert projection.waves == {"total": 1, "completed": 1}
    assert projection.latest_findings[0]["summary"] == "Finding"


def test_sse_replays_after_last_event_id(monkeypatch, tmp_path):
    from open_deep_research import server
    from security.auth import get_current_user

    monkeypatch.setenv("RUNS_DIR", str(tmp_path))
    run_id = "run-sse"
    context = RunContextStore(run_id, runs_dir=str(tmp_path))
    context.initialize("user-1", {"configurable": {"runs_dir": str(tmp_path)}, "metadata": {"run_id": run_id}})
    store = RunEventStore(run_id, runs_dir=str(tmp_path))
    asyncio.run(store.append("run.created", payload={"status": "pending"}, dedupe_key="run:created"))
    asyncio.run(store.append("run.completed", payload={"status": "completed", "result_ref": f"/runs/{run_id}"}, dedupe_key="run:completed"))
    server.app.dependency_overrides[get_current_user] = lambda: {"identity": "user-1"}
    try:
        client = TestClient(server.app)
        response = client.get(
            f"/runs/{run_id}/events?after=0",
            headers={"Last-Event-ID": "1"},
        )
    finally:
        server.app.dependency_overrides.pop(get_current_user, None)

    assert response.status_code == 200
    assert "id: 1" not in response.text
    assert "id: 2" in response.text
    assert "event: run.completed" in response.text
    data_line = next(line for line in response.text.splitlines() if line.startswith("data: "))
    assert "dedupe_key" not in data_line
