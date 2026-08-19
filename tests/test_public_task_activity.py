import asyncio
import json

from open_deep_research.events.public import RunEventStore
from open_deep_research.events.task_activity import (
    TaskActivityPublisher,
    TaskActivityStore,
    activity_summary,
    derive_trace_activity,
    publish_task_activity,
)
from open_deep_research.server import _task_activity_iterator


def _append(store: TaskActivityStore, event_type: str, dedupe_key: str, **overrides):
    values = {
        "kind": "model",
        "phase": "reasoning",
        "status": "success",
        "title": "模型规划",
        "summary": "完成了一轮可公开的模型规划。",
        "iteration": 1,
        "duration_ms": 25,
        "payload": {"model": "qwen-max", "input_tokens": 12},
        "dedupe_key": dedupe_key,
    }
    values.update(overrides)
    return asyncio.run(store.append(event_type, **values))


def test_store_sequences_deduplicates_and_pages(tmp_path):
    store = TaskActivityStore("run-1", "task-1", runs_dir=str(tmp_path))
    first = _append(store, "model.completed", "model:1")
    duplicate = _append(store, "model.completed", "model:1")
    second = _append(
        store,
        "tool.completed",
        "tool:1",
        kind="tool",
        phase="tool_execution",
        payload={"tool_name": "tavily_search", "result_chars": 120},
    )

    assert first.sequence == duplicate.sequence == 1
    assert second.sequence == 2
    items, has_more, last_sequence = store.page(limit=1)
    assert [item.sequence for item in items] == [2]
    assert has_more is True
    assert last_sequence == 2


def test_store_repairs_partial_tail(tmp_path):
    store = TaskActivityStore("run-1", "task-1", runs_dir=str(tmp_path))
    _append(store, "model.completed", "model:1")
    with store.path.open("ab") as handle:
        handle.write(b'{"partial":')

    assert [event.sequence for event in store.read()] == [1]
    assert store.path.read_bytes().endswith(b"\n")


def test_payload_projection_removes_secrets_and_unsafe_urls(tmp_path):
    store = TaskActivityStore("run-1", "task-1", runs_dir=str(tmp_path))
    event = _append(
        store,
        "tool.completed",
        "tool:1",
        kind="tool",
        phase="tool_execution",
        payload={
            "tool_name": "fetch",
            "result_chars": 55,
            "authorization": "Bearer secret",
            "tool_result": "raw body",
            "urls": ["https://example.com/a?token=secret", "javascript:alert(1)"],
        },
    )

    assert "authorization" not in event.payload
    assert "tool_result" not in event.payload
    assert event.payload["urls"] == ["https://example.com/a"]


def test_publisher_is_fail_open():
    class BrokenStore:
        async def append(self, *_args, **_kwargs):
            raise OSError("disk full")

    publisher = TaskActivityPublisher(BrokenStore())  # type: ignore[arg-type]
    assert asyncio.run(publisher.publish("model.started")) is None


def test_summary_counts_completed_calls_and_warnings(tmp_path):
    store = TaskActivityStore("run-1", "task-1", runs_dir=str(tmp_path))
    _append(store, "model.completed", "model:1")
    _append(store, "model.retrying", "model:retry", status="warning")
    summary = activity_summary(store.read())
    assert summary["model_call_count"] == 1
    assert summary["retry_count"] == 1
    assert summary["warning_count"] == 1
    assert summary["activity_available"] is True


def test_legacy_trace_is_scoped_to_conduct_research_task():
    spans = [
        {"span_id": "root-a", "parent_span_id": None, "name": "tool.ConductResearch", "kind": "tool", "started_at": 1, "attributes_json": json.dumps({"tool_call_id": "task-a"})},
        {"span_id": "llm-a", "parent_span_id": "root-a", "name": "researcher.model", "kind": "llm", "started_at": 2, "duration_ms": 10, "status": "ok", "attributes_json": "{}"},
        {"span_id": "root-b", "parent_span_id": None, "name": "tool.ConductResearch", "kind": "tool", "started_at": 3, "attributes_json": json.dumps({"tool_call_id": "task-b"})},
        {"span_id": "tool-b", "parent_span_id": "root-b", "name": "tool.search", "kind": "tool", "started_at": 4, "status": "ok", "attributes_json": "{}"},
    ]
    events = derive_trace_activity("run-1", "task-a", spans)
    assert len(events) == 1
    assert events[0].task_id == "task-a"
    assert events[0].kind == "model"


def test_warning_completion_projects_as_completed_terminal(tmp_path):
    config = {
        "configurable": {"runs_dir": str(tmp_path)},
        "metadata": {
            "run_id": "run-1",
            "task_id": "task-1",
            "research_mode": "sync",
            "research_wave_id": "wave-1",
        },
    }
    asyncio.run(publish_task_activity(
        config,
        "task.completed",
        kind="lifecycle",
        phase="terminal",
        status="warning",
        title="Subagent 已完成，交接需补证",
        summary="研究执行已经结束，但交接未通过质量门禁。",
        iteration=None,
        duration_ms=None,
        payload={"mode": "sync", "wave_id": "wave-1", "admission_status": "rejected"},
        dedupe_key="task:task-1:completed:rejected",
        update_run_summary=True,
    ))

    run_events = RunEventStore("run-1", runs_dir=str(tmp_path)).read()
    progress = run_events[-1]
    assert progress.type == "research.task.progress"
    assert progress.payload["status"] == "completed"
    assert progress.payload["phase"] == "completed"


def test_stream_replays_legacy_tail_after_early_terminal_then_closes(tmp_path):
    store = TaskActivityStore("run-1", "task-1", runs_dir=str(tmp_path))
    _append(
        store,
        "task.completed",
        "task:completed",
        kind="lifecycle",
        phase="terminal",
        status="success",
        payload={"mode": "sync", "admission_status": "pending"},
    )
    _append(
        store,
        "quality.completed",
        "quality:handoff",
        kind="quality",
        phase="handoff",
        payload={"evaluation_type": "subagent_handoff", "decision": "accepted"},
    )

    async def collect(after: int):
        return [item async for item in _task_activity_iterator(store, after=after)]

    replay = asyncio.run(asyncio.wait_for(collect(0), timeout=1))
    closed_tail = asyncio.run(asyncio.wait_for(collect(2), timeout=1))
    assert len(replay) == 2
    assert closed_tail == []
