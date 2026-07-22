"""Tests for async SubAgent research modules.

Covers: events.py, task_registry.py, and (later) async_tools.py,
background_executor.py, recovery.py.
"""

import asyncio
import json
import os
import tempfile
import time
from pathlib import Path

import pytest

from open_deep_research.tasks.events import (
    EventType,
    JSONLEventWriter,
    ResearchEvent,
)
from open_deep_research.tasks.registry import (
    TaskPhase,
    TaskRecord,
    TaskRegistry,
    TaskStatus,
    get_task_registry,
)
from open_deep_research.tasks.state import (
    MemoryTaskStateStore,
    TaskSnapshot,
    reset_memory_task_state_store,
)


@pytest.fixture(autouse=True)
def clear_task_state_store():
    reset_memory_task_state_store()
    yield
    reset_memory_task_state_store()


# ---------------------------------------------------------------------------
# Registry / snapshot — waiting-for-confirmation state
# ---------------------------------------------------------------------------


class TestWaitingForConfirmation:
    def test_count_active_includes_waiting(self):
        registry = TaskRegistry()
        rec = registry.create("topic", run_id="run-1")
        registry.update_status(rec.task_id, TaskStatus.WAITING_FOR_CONFIRMATION)
        assert registry.count_running(run_id="run-1") == 0
        assert registry.count_active(run_id="run-1") == 1

    def test_count_active_includes_running(self):
        registry = TaskRegistry()
        rec = registry.create("topic", run_id="run-1")
        registry.update_status(rec.task_id, TaskStatus.RUNNING)
        assert registry.count_active(run_id="run-1") == 1

    def test_waiting_is_not_terminal(self):
        registry = TaskRegistry()
        rec = registry.create("topic", run_id="run-1")
        registry.update_status(rec.task_id, TaskStatus.WAITING_FOR_CONFIRMATION)
        assert registry.all_completed(run_id="run-1") is False

    def test_pending_domain_fields_serialize_in_snapshot(self):
        record = TaskRecord(task_id="t1", run_id="run-1", research_topic="topic")
        record.status = TaskStatus.WAITING_FOR_CONFIRMATION
        record.pending_domain = "untrusted.example.com"
        record.pending_domain_tool = "fetch_webpage"
        snapshot = TaskSnapshot.from_record(record)
        assert snapshot.status == TaskStatus.WAITING_FOR_CONFIRMATION
        assert snapshot.pending_domain == "untrusted.example.com"
        assert snapshot.pending_domain_tool == "fetch_webpage"
        # Round-trips through JSON for the file-backed task snapshot.
        dumped = snapshot.model_dump_json()
        assert "untrusted.example.com" in dumped


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------

class TestResearchEvent:
    """Schema and construction tests for ResearchEvent."""

    def test_default_factory_fields_are_populated(self):
        event = ResearchEvent(
            event_type=EventType.TASK_STARTED,
            task_id="task-1",
            run_id="run-1",
        )
        assert event.event_id  # non-empty uuid4 string
        assert len(event.event_id) == 36
        assert event.timestamp > 0

    def test_all_event_type_values_are_constructable(self):
        for et in EventType:
            event = ResearchEvent(
                event_type=et,
                task_id="t",
                run_id="r",
                phase="researching",
                data={"key": "val"},
            )
            assert event.event_type == et
            assert event.phase == "researching"
            assert event.data == {"key": "val"}

    def test_model_dump_json_produces_valid_json(self):
        event = ResearchEvent(
            event_type=EventType.TASK_COMPLETED,
            task_id="abc",
            run_id="xyz",
        )
        raw = event.model_dump_json()
        parsed = json.loads(raw)
        assert parsed["event_type"] == "task.completed"
        assert parsed["task_id"] == "abc"


class TestJSONLEventWriter:
    """JSONL write / read-back tests."""

    def test_write_single_event(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = JSONLEventWriter(run_id="run-1", runs_dir=tmpdir)
            event = ResearchEvent(
                event_type=EventType.TASK_STARTED,
                task_id="t1",
                run_id="run-1",
            )
            writer.write(event)
            writer.close()

            # Read back
            events_path = os.path.join(tmpdir, "run-1", "events.jsonl")
            assert os.path.isfile(events_path)
            lines = Path(events_path).read_text().strip().split("\n")
            assert len(lines) == 1
            parsed = json.loads(lines[0])
            assert parsed["event_type"] == "task.started"
            assert parsed["task_id"] == "t1"

    def test_write_multiple_events_appends(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = JSONLEventWriter(run_id="run-2", runs_dir=tmpdir)
            for i in range(3):
                writer.write(ResearchEvent(
                    event_type=EventType.TASK_TOOL_CALL,
                    task_id=f"t{i}",
                    run_id="run-2",
                    data={"index": i},
                ))
            writer.close()

            events_path = os.path.join(tmpdir, "run-2", "events.jsonl")
            lines = Path(events_path).read_text().strip().split("\n")
            assert len(lines) == 3
            for i, line in enumerate(lines):
                assert json.loads(line)["data"]["index"] == i

    def test_context_manager_closes_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with JSONLEventWriter(run_id="run-ctx", runs_dir=tmpdir) as writer:
                writer.write(ResearchEvent(
                    event_type=EventType.TASK_CREATED,
                    task_id="t",
                    run_id="run-ctx",
                ))
            # After __exit__ the file should be closed
            assert writer._file.closed

    def test_run_directory_is_created_if_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runs_dir = os.path.join(tmpdir, "nested", "runs")
            writer = JSONLEventWriter(run_id="deep-run", runs_dir=runs_dir)
            assert os.path.isdir(writer.run_dir)
            writer.close()

    def test_file_path_property(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = JSONLEventWriter(run_id="fp-test", runs_dir=tmpdir)
            expected = os.path.join(tmpdir, "fp-test", "events.jsonl")
            assert writer.file_path == expected
            writer.close()


# ---------------------------------------------------------------------------
# Task Registry
# ---------------------------------------------------------------------------

class TestTaskRecord:
    """Unit tests for TaskRecord dataclass."""

    def test_defaults_are_sensible(self):
        record = TaskRecord(research_topic="test topic")
        assert record.task_id
        assert record.status == TaskStatus.PENDING
        assert record.phase == TaskPhase.RESEARCHING
        assert record.created_at > 0
        assert record.result is None
        assert record.error_message is None
        assert record.query_count == 0

    def test_elapsed_seconds_running(self):
        record = TaskRecord(research_topic="t")
        record.created_at = time.time() - 5.0
        assert 4.5 <= record.elapsed_seconds <= 5.5

    def test_elapsed_seconds_completed(self):
        record = TaskRecord(research_topic="t")
        record.created_at = 100.0
        record.completed_at = 110.0
        assert record.elapsed_seconds == 10.0

    def test_custom_task_id(self):
        record = TaskRecord(task_id="custom-123", research_topic="t")
        assert record.task_id == "custom-123"


class TestTaskRegistry:
    """Unit tests for TaskRegistry CRUD and queries."""

    def test_create_and_get(self):
        registry = TaskRegistry()
        record = registry.create("research A")
        assert record.status == TaskStatus.PENDING

        fetched = registry.get(record.task_id)
        assert fetched is record
        assert fetched.research_topic == "research A"

    def test_get_missing_returns_none(self):
        registry = TaskRegistry()
        assert registry.get("no-such-id") is None

    def test_list_all(self):
        registry = TaskRegistry()
        registry.create("a")
        registry.create("b")
        assert len(registry.list()) == 2

    def test_list_filter_by_status(self):
        registry = TaskRegistry()
        r1 = registry.create("a")
        r2 = registry.create("b")
        registry.update_status(r1.task_id, TaskStatus.RUNNING)

        running = registry.list(status_filter=TaskStatus.RUNNING)
        pending = registry.list(status_filter=TaskStatus.PENDING)
        assert len(running) == 1
        assert running[0].task_id == r1.task_id
        assert len(pending) == 1
        assert pending[0].task_id == r2.task_id

    def test_update_status_modifies_record(self):
        registry = TaskRegistry()
        record = registry.create("t")
        registry.update_status(
            record.task_id,
            TaskStatus.COMPLETED,
            completed_at=time.time(),
            result={"compressed_research": "done"},
        )
        assert record.status == TaskStatus.COMPLETED
        assert record.completed_at is not None
        assert record.result == {"compressed_research": "done"}

    def test_update_status_unknown_id_is_noop(self):
        registry = TaskRegistry()
        # Should not raise
        registry.update_status("no-such-id", TaskStatus.COMPLETED)

    def test_all_completed_true_when_all_terminal(self):
        registry = TaskRegistry()
        r1 = registry.create("a")
        r2 = registry.create("b")
        registry.update_status(r1.task_id, TaskStatus.COMPLETED)
        registry.update_status(r2.task_id, TaskStatus.FAILED)
        assert registry.all_completed() is True

    def test_all_completed_false_when_any_running(self):
        registry = TaskRegistry()
        registry.create("a")
        r2 = registry.create("b")
        registry.update_status(r2.task_id, TaskStatus.RUNNING)
        assert registry.all_completed() is False

    def test_count_running(self):
        registry = TaskRegistry()
        r1 = registry.create("a")
        r2 = registry.create("b")
        registry.update_status(r1.task_id, TaskStatus.RUNNING)
        registry.update_status(r2.task_id, TaskStatus.RUNNING)
        assert registry.count_running() == 2
        registry.update_status(r1.task_id, TaskStatus.COMPLETED)
        assert registry.count_running() == 1

    def test_len(self):
        registry = TaskRegistry()
        registry.create("a")
        registry.create("b")
        assert len(registry) == 2


class TestGetTaskRegistrySingleton:
    """Tests for the module-level singleton accessor."""

    def test_returns_same_instance(self):
        r1 = get_task_registry()
        r2 = get_task_registry()
        assert r1 is r2

    def test_registry_is_functional(self):
        registry = get_task_registry()
        record = registry.create("singleton test")
        assert registry.get(record.task_id) is not None


# ---------------------------------------------------------------------------
# Shared Task State
# ---------------------------------------------------------------------------

class TestTaskSnapshot:
    """Tests for the serializable latest-state snapshot."""

    def test_from_record_round_trip(self):
        record = TaskRecord(task_id="task-1", research_topic="topic", run_id="run-1")
        record.status = TaskStatus.COMPLETED
        record.phase = TaskPhase.COMPLETED
        record.result = {"compressed_research": "done", "raw_notes": ["n"]}
        record.source_count = 3

        snapshot = TaskSnapshot.from_record(record, version=2)
        restored = TaskSnapshot.model_validate_json(snapshot.model_dump_json())

        assert restored.task_id == "task-1"
        assert restored.run_id == "run-1"
        assert restored.status == TaskStatus.COMPLETED
        assert restored.result["compressed_research"] == "done"
        assert restored.metrics["source_count"] == 3
        assert restored.version == 2


class TestMemoryTaskStateStore:
    """Store contract tests for the in-process shared-state backend."""

    @pytest.mark.asyncio
    async def test_create_update_get_and_list(self):
        store = MemoryTaskStateStore()
        record = TaskRecord(task_id="task-a", research_topic="A", run_id="run-1")

        first = await store.update_from_record(record)
        assert first.version == 1

        record.status = TaskStatus.RUNNING
        second = await store.update_from_record(record)
        assert second.version == 2

        fetched = await store.get("task-a")
        assert fetched.status == TaskStatus.RUNNING
        assert await store.count_running(run_id="run-1") == 1
        assert len(await store.list(run_id="run-1")) == 1

    @pytest.mark.asyncio
    async def test_collect_completed(self):
        store = MemoryTaskStateStore()
        r1 = TaskRecord(task_id="a", research_topic="A", run_id="run-1")
        r2 = TaskRecord(task_id="b", research_topic="B", run_id="run-1")
        r1.status = TaskStatus.COMPLETED
        r1.result = {"compressed_research": "done"}
        r2.status = TaskStatus.RUNNING

        await store.update_from_record(r1)
        await store.update_from_record(r2)

        completed = await store.collect_completed(run_id="run-1")
        assert [s.task_id for s in completed] == ["a"]

    @pytest.mark.asyncio
    async def test_rejects_stale_fence_epoch(self):
        store = MemoryTaskStateStore()
        record = TaskRecord(task_id="task-fence", research_topic="A", run_id="run-1")

        await store.update_from_record(record, fence_token=2)

        with pytest.raises(RuntimeError, match="stale_fence_token"):
            await store.update_from_record(record, fence_token=1)


# ---------------------------------------------------------------------------
# Async Tools — tool model validation
# ---------------------------------------------------------------------------

class TestAsyncToolModels:
    """Verify Pydantic models construct and validate correctly."""

    def test_start_research_task_minimal(self):
        from open_deep_research.tasks.async_tools import StartResearchTask
        tool = StartResearchTask(research_topic="test topic")
        assert tool.research_topic == "test topic"

    def test_check_research_task(self):
        from open_deep_research.tasks.async_tools import CheckResearchTask
        tool = CheckResearchTask(task_ids=["a", "b"])
        assert tool.task_ids == ["a", "b"]

    def test_list_research_tasks_no_filter(self):
        from open_deep_research.tasks.async_tools import ListResearchTasks
        tool = ListResearchTasks()
        assert tool.status_filter is None

    def test_list_research_tasks_with_filter(self):
        from open_deep_research.tasks.async_tools import ListResearchTasks
        tool = ListResearchTasks(status_filter="running")
        assert tool.status_filter == "running"

    def test_update_research_task(self):
        from open_deep_research.tasks.async_tools import UpdateResearchTask
        tool = UpdateResearchTask(task_id="x", instruction="add more sources")
        assert tool.task_id == "x"
        assert tool.instruction == "add more sources"

    def test_cancel_research_task(self):
        from open_deep_research.tasks.async_tools import CancelResearchTask
        tool = CancelResearchTask(task_ids=["a"], reason="not needed")
        assert tool.task_ids == ["a"]
        assert tool.reason == "not needed"

    def test_approve_research_domain(self):
        from open_deep_research.tasks.async_tools import ApproveResearchDomain
        tool = ApproveResearchDomain(task_id="x", domain="example.com", allow=True)
        assert tool.task_id == "x"
        assert tool.domain == "example.com"
        assert tool.allow is True


# ---------------------------------------------------------------------------
# Async Tools — handler functions (unit-level)
# ---------------------------------------------------------------------------

class TestHandleStartResearchTask:
    """Tests for handle_start_research_task."""

    @pytest.mark.asyncio
    async def test_creates_task_and_returns_tool_message(self):
        from open_deep_research.tasks.async_tools import handle_start_research_task
        from open_deep_research.tasks.registry import TaskRegistry, TaskStatus

        registry = TaskRegistry()
        tool_call = {
            "id": "call-1",
            "name": "StartResearchTask",
            "args": {"research_topic": "test"},
        }

        async def fake_launch(record, cfg):
            registry.update_status(record.task_id, TaskStatus.COMPLETED)

        result = await handle_start_research_task(
            tool_call,
            {"configurable": {"task_state_backend": "memory"}},
            registry,
            fake_launch,
        )
        assert "task_id" in result.content
        assert len(registry) == 1

    @pytest.mark.asyncio
    async def test_respects_max_in_flight(self):
        from open_deep_research.tasks.async_tools import handle_start_research_task
        from open_deep_research.tasks.registry import TaskRegistry, TaskStatus

        registry = TaskRegistry()
        # Fill to the limit (max_in_flight_tasks=1 via configurable), same run_id
        r = registry.create("running", run_id="default")
        registry.update_status(r.task_id, TaskStatus.RUNNING)

        tool_call = {
            "id": "call-2",
            "name": "StartResearchTask",
            "args": {"research_topic": "test"},
        }

        async def fake_launch(record, cfg):
            pass

        result = await handle_start_research_task(
            tool_call,
            {"configurable": {"max_in_flight_tasks": 1}},
            registry,
            fake_launch,
        )
        assert "Cannot start" in result.content

    @pytest.mark.asyncio
    async def test_pending_task_holds_an_admission_slot(self):
        from open_deep_research.tasks.async_tools import handle_start_research_task

        registry = TaskRegistry()

        async def fake_launch(record, cfg):
            del record, cfg

        config = {
            "configurable": {
                "task_state_backend": "memory",
                "max_in_flight_tasks": 1,
            },
            "metadata": {"run_id": "admission-run"},
        }
        first = await handle_start_research_task(
            {"id": "first", "args": {"research_topic": "first"}},
            config,
            registry,
            fake_launch,
        )
        second = await handle_start_research_task(
            {"id": "second", "args": {"research_topic": "second"}},
            config,
            registry,
            fake_launch,
        )

        assert "task_id" in first.content
        assert "Cannot start" in second.content
        assert len(registry.list(run_id="admission-run")) == 1

    @pytest.mark.asyncio
    async def test_launch_failure_persists_terminal_state(self):
        from open_deep_research.tasks.async_tools import handle_start_research_task

        registry = TaskRegistry()

        async def fail_launch(record, cfg):
            del record, cfg
            raise RuntimeError("pool unavailable")

        with pytest.raises(RuntimeError, match="pool unavailable"):
            await handle_start_research_task(
                {"id": "failed", "args": {"research_topic": "topic"}},
                {"configurable": {"task_state_backend": "memory"}},
                registry,
                fail_launch,
            )

        [record] = registry.list(run_id="default")
        assert record.status == TaskStatus.FAILED
        assert record.error_message == "Unable to launch task: pool unavailable"


class TestHandleCheckResearchTask:
    """Tests for handle_check_research_task."""

    @pytest.mark.asyncio
    async def test_unknown_task(self):
        from open_deep_research.tasks.async_tools import handle_check_research_task
        from open_deep_research.tasks.registry import TaskRegistry

        registry = TaskRegistry()
        tool_call = {"id": "c", "name": "CheckResearchTask", "args": {"task_ids": ["no-such"]}}
        result = await handle_check_research_task(tool_call, registry)
        assert "UNKNOWN" in result.content

    @pytest.mark.asyncio
    async def test_rejects_task_from_another_run(self):
        from open_deep_research.tasks.async_tools import handle_check_research_task

        registry = TaskRegistry()
        record = registry.create("secret topic", run_id="other-run")
        registry.update_status(record.task_id, TaskStatus.COMPLETED)
        result = await handle_check_research_task(
            {"id": "c", "name": "CheckResearchTask", "args": {"task_ids": [record.task_id]}},
            registry,
            run_id="current-run",
        )
        assert "UNKNOWN" in result.content
        assert "secret topic" not in result.content

    @pytest.mark.asyncio
    async def test_completed_task_returns_results(self):
        from open_deep_research.tasks.async_tools import handle_check_research_task
        from open_deep_research.tasks.registry import TaskRegistry, TaskStatus

        registry = TaskRegistry()
        r = registry.create("topic")
        registry.update_status(
            r.task_id, TaskStatus.COMPLETED,
            result={"compressed_research": "findings here", "raw_notes": []},
        )
        tool_call = {"id": "c", "name": "CheckResearchTask", "args": {"task_ids": [r.task_id]}}
        result = await handle_check_research_task(tool_call, registry)
        assert "COMPLETED" in result.content
        assert "findings here" in result.content

    @pytest.mark.asyncio
    async def test_running_task_shows_status(self):
        from open_deep_research.tasks.async_tools import handle_check_research_task
        from open_deep_research.tasks.registry import TaskRegistry, TaskStatus

        registry = TaskRegistry()
        r = registry.create("topic")
        registry.update_status(r.task_id, TaskStatus.RUNNING)
        tool_call = {"id": "c", "name": "CheckResearchTask", "args": {"task_ids": [r.task_id]}}
        result = await handle_check_research_task(tool_call, registry)
        assert "RUNNING" in result.content


class TestHandleListResearchTasks:
    """Tests for handle_list_research_tasks."""

    @pytest.mark.asyncio
    async def test_empty_registry(self):
        from open_deep_research.tasks.async_tools import handle_list_research_tasks
        from open_deep_research.tasks.registry import TaskRegistry

        registry = TaskRegistry()
        tool_call = {"id": "l", "name": "ListResearchTasks", "args": {}}
        result = await handle_list_research_tasks(tool_call, registry)
        assert "No research tasks" in result.content

    @pytest.mark.asyncio
    async def test_with_tasks(self):
        from open_deep_research.tasks.async_tools import handle_list_research_tasks
        from open_deep_research.tasks.registry import TaskRegistry, TaskStatus

        registry = TaskRegistry()
        r1 = registry.create("topic A")
        r2 = registry.create("topic B")
        registry.update_status(r1.task_id, TaskStatus.RUNNING)
        registry.update_status(r2.task_id, TaskStatus.COMPLETED)

        tool_call = {"id": "l", "name": "ListResearchTasks", "args": {}}
        result = await handle_list_research_tasks(tool_call, registry)
        assert "running" in result.content
        assert "completed" in result.content


class TestHandleUpdateResearchTask:
    """Tests for handle_update_research_task."""

    @pytest.mark.asyncio
    async def test_queues_instruction_for_running_task(self):
        from open_deep_research.tasks.async_tools import handle_update_research_task
        from open_deep_research.tasks.registry import TaskRegistry, TaskStatus

        registry = TaskRegistry()
        r = registry.create("topic")
        registry.update_status(r.task_id, TaskStatus.RUNNING)
        tool_call = {
            "id": "u",
            "name": "UpdateResearchTask",
            "args": {"task_id": r.task_id, "instruction": "use official sources"},
        }
        result = await handle_update_research_task(tool_call, registry)
        assert "queued" in result.content
        # Check the instruction landed on the control queue
        msg = r.control_queue.get_nowait()
        assert msg["instruction"] == "use official sources"

    @pytest.mark.asyncio
    async def test_rejects_non_running_task(self):
        from open_deep_research.tasks.async_tools import handle_update_research_task
        from open_deep_research.tasks.registry import TaskRegistry, TaskStatus

        registry = TaskRegistry()
        r = registry.create("topic")
        registry.update_status(r.task_id, TaskStatus.COMPLETED)
        tool_call = {
            "id": "u",
            "name": "UpdateResearchTask",
            "args": {"task_id": r.task_id, "instruction": "x"},
        }
        result = await handle_update_research_task(tool_call, registry)
        assert "not RUNNING" in result.content


class TestHandleCancelResearchTask:
    """Tests for handle_cancel_research_task."""

    @pytest.mark.asyncio
    async def test_cancels_running_task(self):
        from open_deep_research.tasks.async_tools import handle_cancel_research_task
        from open_deep_research.tasks.registry import TaskRegistry, TaskStatus

        registry = TaskRegistry()
        r = registry.create("topic")
        registry.update_status(r.task_id, TaskStatus.RUNNING)
        tool_call = {
            "id": "x",
            "name": "CancelResearchTask",
            "args": {"task_ids": [r.task_id], "reason": "done"},
        }
        result = await handle_cancel_research_task(tool_call, registry)
        assert "cancelled" in result.content
        assert r.cancelled.is_set()
        assert r.status == TaskStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_skips_already_cancelled(self):
        from open_deep_research.tasks.async_tools import handle_cancel_research_task
        from open_deep_research.tasks.registry import TaskRegistry, TaskStatus

        registry = TaskRegistry()
        r = registry.create("topic")
        registry.update_status(r.task_id, TaskStatus.CANCELLED)
        tool_call = {
            "id": "x",
            "name": "CancelResearchTask",
            "args": {"task_ids": [r.task_id]},
        }
        result = await handle_cancel_research_task(tool_call, registry)
        assert "already cancelled" in result.content

    @pytest.mark.asyncio
    async def test_cancel_waiting_task_preserves_sibling_approvals(self):
        from open_deep_research.tasks.async_tools import handle_cancel_research_task
        from open_deep_research.tasks.registry import TaskRegistry, TaskStatus
        from open_deep_research.tasks.domain_approvals import (
            get_domain_approval_registry,
            reset_domain_approval_registry,
        )

        reset_domain_approval_registry()
        registry = TaskRegistry()
        r = registry.create("topic", run_id="run-x")
        registry.update_status(r.task_id, TaskStatus.WAITING_FOR_CONFIRMATION)
        sibling = registry.create("sibling", run_id="run-x")
        registry.update_status(sibling.task_id, TaskStatus.RUNNING)
        approvals = get_domain_approval_registry()
        req = approvals.request_decision("run-x", "pending.example", "fetch_webpage")
        req.future = asyncio.get_running_loop().create_future()
        tool_call = {
            "id": "x",
            "name": "CancelResearchTask",
            "args": {"task_ids": [r.task_id]},
        }
        await handle_cancel_research_task(tool_call, registry)
        assert r.status == TaskStatus.CANCELLED
        # A single task cancellation must not cancel other tasks in the run.
        assert not req.future.cancelled()
        reset_domain_approval_registry()


class TestHandleApproveResearchDomain:
    """Tests for handle_approve_research_domain + the waiting snapshot branch."""

    @pytest.fixture(autouse=True)
    def _reset_approvals(self):
        from open_deep_research.tasks.domain_approvals import (
            reset_domain_approval_registry,
        )

        reset_domain_approval_registry()
        yield
        reset_domain_approval_registry()

    @pytest.mark.asyncio
    async def test_approve_waiting_task_resolves_and_resumes(self):
        from open_deep_research.tasks.async_tools import handle_approve_research_domain
        from open_deep_research.tasks.registry import TaskRegistry, TaskStatus
        from open_deep_research.tasks.domain_approvals import get_domain_approval_registry

        registry = TaskRegistry()
        r = registry.create("topic", run_id="run-x")
        registry.update_status(r.task_id, TaskStatus.WAITING_FOR_CONFIRMATION)
        r.pending_domain = "untrusted.example"
        r.pending_domain_tool = "fetch_webpage"
        approvals = get_domain_approval_registry()
        req = approvals.request_decision("run-x", "untrusted.example", "fetch_webpage")
        req.future = asyncio.get_running_loop().create_future()

        tool_call = {
            "id": "ap1",
            "name": "ApproveResearchDomain",
            "args": {"task_id": r.task_id, "domain": "untrusted.example", "allow": True},
        }
        config = {"configurable": {"search_api": "none"}, "metadata": {"run_id": "run-x"}}
        result = await handle_approve_research_domain(tool_call, config, registry)

        assert "approved" in result.content
        assert r.status == TaskStatus.RUNNING
        assert r.pending_domain is None
        # future resolved True
        assert req.future.result() is True
        # decision cached for the run
        assert approvals.is_allowed("run-x", "untrusted.example") is True
        # The approval Future is the single wake-up path; no duplicate control
        # marker should remain to race with later supervisor updates.
        assert r.control_queue.empty()

    @pytest.mark.asyncio
    async def test_deny_waiting_task(self):
        from open_deep_research.tasks.async_tools import handle_approve_research_domain
        from open_deep_research.tasks.registry import TaskRegistry, TaskStatus
        from open_deep_research.tasks.domain_approvals import get_domain_approval_registry

        registry = TaskRegistry()
        r = registry.create("topic", run_id="run-x")
        registry.update_status(r.task_id, TaskStatus.WAITING_FOR_CONFIRMATION)
        approvals = get_domain_approval_registry()
        req = approvals.request_decision("run-x", "bad.example", "fetch_webpage")
        req.future = asyncio.get_running_loop().create_future()

        tool_call = {
            "id": "ap1",
            "name": "ApproveResearchDomain",
            "args": {"task_id": r.task_id, "domain": "bad.example", "allow": False},
        }
        config = {"configurable": {"search_api": "none"}, "metadata": {"run_id": "run-x"}}
        result = await handle_approve_research_domain(tool_call, config, registry)

        assert "denied" in result.content
        assert approvals.is_allowed("run-x", "bad.example") is False
        assert req.future.result() is False

    @pytest.mark.asyncio
    async def test_approve_requires_waiting_or_running(self):
        from open_deep_research.tasks.async_tools import handle_approve_research_domain
        from open_deep_research.tasks.registry import TaskRegistry, TaskStatus

        registry = TaskRegistry()
        r = registry.create("topic", run_id="run-x")
        registry.update_status(r.task_id, TaskStatus.COMPLETED)
        tool_call = {
            "id": "ap1",
            "name": "ApproveResearchDomain",
            "args": {"task_id": r.task_id, "domain": "x.example", "allow": True},
        }
        config = {"configurable": {"search_api": "none"}, "metadata": {"run_id": "run-x"}}
        result = await handle_approve_research_domain(tool_call, config, registry)
        assert "completed" in result.content

    @pytest.mark.asyncio
    async def test_approve_unknown_task(self):
        from open_deep_research.tasks.async_tools import handle_approve_research_domain
        from open_deep_research.tasks.registry import TaskRegistry

        registry = TaskRegistry()
        tool_call = {
            "id": "ap1",
            "name": "ApproveResearchDomain",
            "args": {"task_id": "nope", "domain": "x.example", "allow": True},
        }
        config = {"configurable": {"search_api": "none"}, "metadata": {"run_id": "run-x"}}
        result = await handle_approve_research_domain(tool_call, config, registry)
        assert "not found" in result.content

    def test_format_snapshot_waiting_branch(self):
        from open_deep_research.tasks.async_tools import format_task_snapshot_for_context

        record = TaskRecord(task_id="t1", run_id="run-x", research_topic="topic")
        record.status = TaskStatus.WAITING_FOR_CONFIRMATION
        record.pending_domain = "untrusted.example"
        record.pending_domain_tool = "fetch_webpage"
        snapshot = TaskSnapshot.from_record(record)
        text = format_task_snapshot_for_context(snapshot)
        assert "WAITING FOR DOMAIN APPROVAL" in text
        assert "untrusted.example" in text
        assert "ApproveResearchDomain" in text
        assert "t1" in text


class TestCollectCompletedTaskOutputs:
    """Tests for collect_completed_task_outputs."""

    @pytest.mark.asyncio
    async def test_collects_only_completed(self):
        from open_deep_research.tasks.async_tools import collect_completed_task_outputs
        from open_deep_research.tasks.registry import TaskRegistry, TaskStatus

        registry = TaskRegistry()
        r1 = registry.create("topic A")
        r2 = registry.create("topic B")
        r3 = registry.create("topic C")
        registry.update_status(
            r1.task_id, TaskStatus.COMPLETED,
            result={"compressed_research": "findings A", "raw_notes": ["n1"]},
        )
        registry.update_status(r2.task_id, TaskStatus.RUNNING)
        registry.update_status(
            r3.task_id, TaskStatus.COMPLETED,
            result={"compressed_research": "findings C", "raw_notes": ["n2"]},
        )

        outputs = await collect_completed_task_outputs(registry)
        assert len(outputs) == 2
        topics = {o["research_topic"] for o in outputs}
        assert topics == {"topic A", "topic C"}

    @pytest.mark.asyncio
    async def test_preserves_public_source_registries(self):
        from open_deep_research.tasks.async_tools import collect_completed_task_outputs
        from open_deep_research.tasks.registry import TaskRegistry, TaskStatus

        registry = TaskRegistry()
        record = registry.create("topic")
        registry.update_status(
            record.task_id,
            TaskStatus.COMPLETED,
            result={
                "compressed_research": "finding https://example.com/doc",
                "raw_notes": [],
                "candidate_registry": [{"url": "https://example.com/doc"}],
                "document_registry": [{"url": "https://example.com/doc"}],
                "evidence_registry": [{"source_url": "https://example.com/doc"}],
                "web_research_iterations": [{"iteration": 1}],
            },
        )

        output = (await collect_completed_task_outputs(registry))[0]

        assert output["candidate_registry"]
        assert output["document_registry"]
        assert output["evidence_registry"]
        assert output["web_research_iterations"]


# ---------------------------------------------------------------------------
# Recovery — CheckpointManager
# ---------------------------------------------------------------------------

class TestResearcherCheckpoint:
    """Serialisation round-trip tests."""

    def test_to_dict_and_from_dict_round_trip(self):
        from open_deep_research.tasks.recovery import ResearcherCheckpoint

        cp = ResearcherCheckpoint(
            task_id="task-1",
            phase="researching",
            next_step="compress",
            fence_token=7,
            committed_tool_call_ids=["call-1"],
            artifact_refs=[{"path": "artifact.json", "sha256": "a" * 64}],
            completion_decision={"action": "complete_partial"},
            messages_snapshot=[{"type": "human", "content": "hello"}],
            tool_call_iterations=3,
            completed_queries=["q1", "q2"],
            fetched_sources=["https://a.com"],
        )
        data = cp.to_dict()
        restored = ResearcherCheckpoint.from_dict(data)
        assert restored.task_id == cp.task_id
        assert restored.phase == cp.phase
        assert restored.tool_call_iterations == cp.tool_call_iterations
        assert restored.completed_queries == cp.completed_queries
        assert restored.fetched_sources == cp.fetched_sources
        assert restored.schema_version == 2
        assert restored.next_step == "compress"
        assert restored.fence_token == 7
        assert restored.committed_tool_call_ids == ["call-1"]
        assert restored.artifact_refs == [{"path": "artifact.json", "sha256": "a" * 64}]
        assert restored.completion_decision == {"action": "complete_partial"}

    def test_defaults_are_sensible(self):
        from open_deep_research.tasks.recovery import ResearcherCheckpoint

        cp = ResearcherCheckpoint(task_id="t", phase="compressing")
        assert cp.messages_snapshot == []
        assert cp.completed_queries == []
        assert cp.compressed_research is None
        assert cp.timestamp > 0


class TestCheckpointManager:
    """Persistence tests for CheckpointManager."""

    def test_save_and_load(self):
        import tempfile

        from open_deep_research.tasks.recovery import CheckpointManager, ResearcherCheckpoint

        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(runs_dir=tmpdir, run_id="run-1")
            cp = ResearcherCheckpoint(
                task_id="task-a",
                phase="researching",
                completed_queries=["q1"],
            )
            mgr.save(cp)

            loaded = mgr.load("task-a")
            assert loaded is not None
            assert loaded.task_id == "task-a"
            assert loaded.completed_queries == ["q1"]

    def test_load_missing_returns_none(self):
        import tempfile

        from open_deep_research.tasks.recovery import CheckpointManager

        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(runs_dir=tmpdir, run_id="run-2")
            assert mgr.load("no-such-task") is None

    def test_delete_removes_file(self):
        import os
        import tempfile

        from open_deep_research.tasks.recovery import CheckpointManager, ResearcherCheckpoint

        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(runs_dir=tmpdir, run_id="run-3")
            cp = ResearcherCheckpoint(task_id="todel", phase="researching")
            mgr.save(cp)
            assert os.path.isfile(mgr._path("todel"))

            mgr.delete("todel")
            assert not os.path.isfile(mgr._path("todel"))
            assert mgr.load("todel") is None

    def test_delete_missing_is_noop(self):
        import tempfile

        from open_deep_research.tasks.recovery import CheckpointManager

        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(runs_dir=tmpdir, run_id="run-4")
            # Should not raise
            mgr.delete("no-such-file")

    def test_rejects_path_traversal_identifiers(self, tmp_path: Path):
        from open_deep_research.tasks.recovery import CheckpointManager

        with pytest.raises(ValueError):
            CheckpointManager(runs_dir=str(tmp_path), run_id="../outside")


class TestControlledExecutor:
    """Regression tests for live update/cancel handling."""

    @pytest.mark.asyncio
    async def test_controlled_executor_persists_current_fence_token(
        self,
        monkeypatch,
        tmp_path,
    ):
        from open_deep_research.tasks import executor
        from open_deep_research.tasks.state import FileTaskStateStore

        async def no_publish(*args, **kwargs):
            del args, kwargs

        monkeypatch.setattr(executor, "publish_task_update", no_publish)
        registry = TaskRegistry()
        record = registry.restore(
            TaskRecord(
                task_id="fenced-task",
                run_id="fenced-run",
                research_topic="topic",
            )
        )

        async def research(_state, _config):
            return {"compressed_research": "done", "raw_notes": [], "metrics": {}}

        await executor.run_task_with_control(
            record,
            {
                "configurable": {
                    "runs_dir": str(tmp_path),
                    "task_state_backend": "file",
                    "event_log_enabled": False,
                },
                "metadata": {"run_id": "fenced-run", "run_fence_token": 7},
            },
            registry,
            research,
            runs_dir=str(tmp_path),
            run_id="fenced-run",
            event_log_enabled=False,
            fence_token=7,
        )

        snapshot = await FileTaskStateStore(str(tmp_path)).get(
            "fenced-task",
            run_id="fenced-run",
        )
        assert snapshot is not None
        assert snapshot.fence_token == 7

    @pytest.mark.asyncio
    async def test_update_interrupts_and_restarts_research(self, monkeypatch):
        from open_deep_research.tasks import executor

        async def no_publish(*args, **kwargs):
            del args, kwargs

        class FakeContextStore:
            def __init__(self, *args, **kwargs):
                del args, kwargs

            def persist_task_result(self, *args, **kwargs):
                del args, kwargs
                return "0" * 64

        monkeypatch.setattr(executor, "publish_task_update", no_publish)
        monkeypatch.setattr(executor, "RunContextStore", FakeContextStore)
        registry = TaskRegistry()
        record = registry.restore(TaskRecord(task_id="update-task", run_id="update-run", research_topic="topic"))
        started = asyncio.Event()
        calls: list[list[str]] = []

        async def research(state, _config):
            calls.append([str(message.content) for message in state["researcher_messages"]])
            if len(calls) == 1:
                started.set()
                await asyncio.Event().wait()
            return {"compressed_research": "done", "raw_notes": [], "metrics": {}}

        config = {
            "configurable": {
                "task_state_backend": "memory",
                "task_checkpoint_enabled": False,
                "event_log_enabled": False,
                "task_timeout_seconds": 60,
            },
            "metadata": {"run_id": "update-run"},
        }
        task = asyncio.create_task(
            executor.run_task_with_control(
                record, config, registry, research, run_id="update-run", event_log_enabled=False
            )
        )
        await started.wait()
        await record.control_queue.put({"type": "update", "instruction": "use official sources"})
        await asyncio.wait_for(task, timeout=2)

        assert len(calls) == 2
        assert "[Supervisor Instruction] use official sources" in calls[1]
        assert record.status == TaskStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_mailbox_publish_failure_does_not_fail_research(self, monkeypatch):
        from open_deep_research.tasks import executor

        async def fail_publish(*args, **kwargs):
            del args, kwargs
            raise OSError("mailbox unavailable")

        class FakeContextStore:
            def __init__(self, *args, **kwargs):
                del args, kwargs

            def persist_task_result(self, *args, **kwargs):
                del args, kwargs
                return "0" * 64

        monkeypatch.setattr(executor, "publish_task_update", fail_publish)
        monkeypatch.setattr(executor, "RunContextStore", FakeContextStore)
        registry = TaskRegistry()
        record = registry.restore(TaskRecord(task_id="failed-start", run_id="run", research_topic="topic"))

        async def research(state, _config):
            del state, _config
            return {"compressed_research": "done", "raw_notes": [], "metrics": {}}

        await executor.run_task_with_control(
            record,
            {"configurable": {"task_state_backend": "memory", "event_log_enabled": False}},
            registry,
            research,
            run_id="run",
            event_log_enabled=False,
        )
        assert record.status == TaskStatus.COMPLETED


class TestAsyncSupervisorTermination:
    """Async supervisors must not silently finish with active tasks."""

    @pytest.mark.asyncio
    async def test_rejects_empty_tool_turn_while_task_is_active(self):
        from langchain_core.messages import AIMessage

        from open_deep_research.agents.deep_researcher import _execute_supervisor_tools
        from open_deep_research.configuration import Configuration
        from open_deep_research.tasks.state import get_task_state_store

        config = {
            "configurable": {
                "enable_async_research": True,
                "task_state_backend": "memory",
            },
            "metadata": {"run_id": "supervisor-run"},
        }
        record = TaskRecord(task_id="active-task", run_id="supervisor-run", research_topic="topic")
        configurable = Configuration.from_runnable_config(config)
        await get_task_state_store(configurable).update_from_record(record)

        with pytest.raises(RuntimeError, match="remain active"):
            await _execute_supervisor_tools(
                {
                    "enable_async_research": True,
                    "research_iterations": 0,
                    "supervisor_messages": [AIMessage(content="", tool_calls=[])],
                },
                config,
            )

    @pytest.mark.asyncio
    async def test_empty_tool_turn_finalizes_terminal_tasks_and_wave(self, monkeypatch, tmp_path):
        from langchain_core.messages import AIMessage

        from open_deep_research.agents import deep_researcher
        from open_deep_research.configuration import Configuration
        from open_deep_research.public_events import RunEventStore
        from open_deep_research.tasks.registry import get_task_registry
        from open_deep_research.tasks.state import get_task_state_store

        async def fake_summary(_output, _config):
            return "- public finding"

        monkeypatch.setattr(deep_researcher, "summarize_public_findings", fake_summary)
        run_id = "supervisor-terminal-run"
        config = {
            "configurable": {
                "enable_async_research": True,
                "quality_evaluation_enabled": False,
                "task_state_backend": "memory",
                "runs_dir": str(tmp_path),
            },
            "metadata": {"run_id": run_id},
        }
        record = TaskRecord(
            task_id="terminal-task",
            run_id=run_id,
            research_topic="topic",
            wave_id="wave-2",
            status=TaskStatus.COMPLETED,
            result={
                "compressed_research": "finding https://example.com/doc?secret=value",
                "raw_notes": [],
                "metrics": {},
            },
        )
        get_task_registry().restore(record)
        configurable = Configuration.from_runnable_config(config)
        await get_task_state_store(configurable).update_from_record(record)

        command = await deep_researcher._execute_supervisor_tools(
            {
                "enable_async_research": True,
                "research_iterations": 1,
                "research_brief": "brief",
                "supervisor_messages": [AIMessage(content="", tool_calls=[])],
            },
            config,
        )

        events = RunEventStore(run_id, runs_dir=str(tmp_path)).read()
        assert command.goto == "__end__"
        assert command.update["completed_task_outputs"][0]["task_id"] == "terminal-task"
        assert sum(event.type == "research.wave.completed" for event in events) == 1
        source_event = next(event for event in events if event.type == "research.source.discovered")
        assert source_event.payload["url"] == "https://example.com/doc"
