"""Reliability tests for file Mailbox coordination and persistent teammates."""

from __future__ import annotations

import asyncio
import hashlib
import json
import multiprocessing
import time
from pathlib import Path

import pytest

from open_deep_research.configuration import Configuration
from open_deep_research.tasks.coordination import (
    claim_lead_updates,
    get_mailbox,
    publish_task_update,
)
from open_deep_research.tasks.events import EventType
from open_deep_research.tasks.lease import LeaderLeaseManager
from open_deep_research.tasks.mailbox import (
    CoordinationCorruptedError,
    CoordinationError,
    FileMailbox,
    MailboxFile,
)
from open_deep_research.tasks.registry import TaskRecord, TaskRegistry, TaskStatus
from open_deep_research.tasks.state import FileTaskStateStore, TaskSnapshot
from open_deep_research.tasks.teammate_pool import TeammatePool


def _send_messages(runs_dir: str, worker: int, count: int) -> None:
    async def send_all() -> None:
        mailbox = FileMailbox(runs_dir=runs_dir, run_id="concurrent")
        for index in range(count):
            await mailbox.send(
                recipient="lead",
                sender=f"writer-{worker}",
                message_type="task_progress",
                dedupe_key=f"{worker}:{index}",
                payload={"worker": worker, "index": index},
            )

    asyncio.run(send_all())


def test_multiple_processes_append_without_lost_updates(tmp_path: Path) -> None:
    """Independent writers serialize through the stable inbox lock."""
    processes = [
        multiprocessing.Process(target=_send_messages, args=(str(tmp_path), worker, 15))
        for worker in range(4)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=20)
        assert process.exitcode == 0

    path = tmp_path / "concurrent" / "coordination" / "inboxes" / "lead.json"
    mailbox = MailboxFile.model_validate_json(path.read_text(encoding="utf-8"))
    assert len(mailbox.messages) == 60
    assert [message.sequence for message in mailbox.messages] == list(range(1, 61))


@pytest.mark.asyncio
async def test_claim_ack_dedupe_priority_and_expired_lease(tmp_path: Path) -> None:
    mailbox = FileMailbox(
        runs_dir=str(tmp_path),
        run_id="lease-run",
        claim_lease_seconds=0.05,
    )
    first = await mailbox.send(
        recipient="worker-1",
        sender="lead",
        message_type="task_assignment",
        priority=20,
        dedupe_key="same",
    )
    duplicate = await mailbox.send(
        recipient="worker-1",
        sender="lead",
        message_type="task_assignment",
        priority=20,
        dedupe_key="same",
    )
    urgent = await mailbox.send(
        recipient="worker-1",
        sender="lead",
        message_type="cancel_request",
        priority=0,
    )

    claimed = await mailbox.claim(agent_id="worker-1", consumer_id="consumer-a", limit=1)
    assert claimed[0].message_id == urgent.message_id
    await asyncio.sleep(0.07)
    redelivered = await mailbox.claim(agent_id="worker-1", consumer_id="consumer-b", limit=1)
    assert redelivered[0].message_id == urgent.message_id
    assert redelivered[0].attempt_count == 2
    await mailbox.ack(
        agent_id="worker-1",
        consumer_id="consumer-b",
        message_ids=[urgent.message_id],
    )
    assert first.message_id == duplicate.message_id


@pytest.mark.asyncio
async def test_corrupt_mailbox_marks_run_and_raises(tmp_path: Path) -> None:
    mailbox = FileMailbox(runs_dir=str(tmp_path), run_id="corrupt-run")
    inbox = tmp_path / "corrupt-run" / "coordination" / "inboxes" / "lead.json"
    inbox.parent.mkdir(parents=True, exist_ok=True)
    inbox.write_text("{broken", encoding="utf-8")

    with pytest.raises(CoordinationCorruptedError):
        await mailbox.claim(agent_id="lead", consumer_id="consumer")
    assert (tmp_path / "corrupt-run" / "coordination" / "coordination_corrupted.json").exists()


@pytest.mark.asyncio
async def test_file_task_store_versions_snapshots(tmp_path: Path) -> None:
    store = FileTaskStateStore(str(tmp_path))
    snapshot = TaskSnapshot(task_id="task-1", run_id="run-1", research_topic="topic")
    await store.upsert(snapshot)
    snapshot.status = TaskStatus.RUNNING
    await store.upsert(snapshot)

    restored = await store.get("task-1")
    assert restored is not None
    assert restored.version == 2
    assert restored.status == TaskStatus.RUNNING


@pytest.mark.asyncio
async def test_lead_verifies_completed_artifact_before_ack(tmp_path: Path) -> None:
    configurable = Configuration(runs_dir=str(tmp_path), task_state_backend="file")
    store = FileTaskStateStore(str(tmp_path))
    artifact = tmp_path / "artifact-run" / "context" / "artifacts" / "research_tasks" / "task-1.json"
    artifact.parent.mkdir(parents=True)
    content = b'{"result":"ok"}'
    artifact.write_bytes(content)
    snapshot = TaskSnapshot(
        task_id="task-1",
        run_id="artifact-run",
        research_topic="topic",
        status=TaskStatus.COMPLETED,
        result_artifact_path="context/artifacts/research_tasks/task-1.json",
        result_artifact_sha256=hashlib.sha256(content).hexdigest(),
    )
    await store.upsert(snapshot)
    await publish_task_update(configurable, snapshot, EventType.TASK_COMPLETED)

    messages, _ = await claim_lead_updates(
        configurable,
        run_id="artifact-run",
        consumer_id="lead-test",
    )
    assert len(messages) == 1

    artifact.write_bytes(b"tampered")
    await get_mailbox(configurable, "artifact-run").nack(
        agent_id="lead",
        consumer_id="lead-test",
        message_ids=[messages[0].message_id],
    )
    with pytest.raises(CoordinationError, match="hash mismatch"):
        await claim_lead_updates(
            configurable,
            run_id="artifact-run",
            consumer_id="lead-test-2",
        )


@pytest.mark.asyncio
async def test_leader_lease_rejects_live_owner_and_allows_takeover(tmp_path: Path) -> None:
    first = LeaderLeaseManager(runs_dir=str(tmp_path), run_id="lease-owner", lease_seconds=0.05)
    second = LeaderLeaseManager(runs_dir=str(tmp_path), run_id="lease-owner", lease_seconds=0.05)
    first.owner_id = "owner-a"
    second.owner_id = "owner-b"
    await first.acquire()
    with pytest.raises(RuntimeError):
        await second.acquire()
    await asyncio.sleep(0.07)
    lease = await second.acquire()
    assert lease.owner_instance_id == "owner-b"


@pytest.mark.asyncio
async def test_persistent_teammate_reuses_identity_with_clean_task_state(tmp_path: Path) -> None:
    seen_topics: list[str] = []
    seen_message_counts: list[int] = []

    async def fake_research(state, _config):
        seen_topics.append(str(state["research_topic"]))
        seen_message_counts.append(len(state["researcher_messages"]))
        await asyncio.sleep(0.02)
        return {
            **state,
            "compressed_research": f"result:{state['research_topic']}",
            "raw_notes": [],
        }

    registry = TaskRegistry()
    config = {
        "configurable": {
            "runs_dir": str(tmp_path),
            "task_state_backend": "file",
            "max_persistent_teammates": 1,
            "mailbox_poll_interval_ms": 50,
            "task_checkpoint_enabled": False,
            "event_log_enabled": False,
            "query_session_persistence_enabled": False,
            "leader_heartbeat_seconds": 60,
        },
        "metadata": {"run_id": "pool-run"},
    }
    pool = TeammatePool(config=config, registry=registry, execute_research=fake_research)
    first = registry.restore(TaskRecord(task_id="task-a", run_id="pool-run", research_topic="A"))
    second = registry.restore(TaskRecord(task_id="task-b", run_id="pool-run", research_topic="B"))
    await pool.submit(first)
    await pool.submit(second)

    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if first.status == TaskStatus.COMPLETED and second.status == TaskStatus.COMPLETED:
            break
        await asyncio.sleep(0.05)
    team_path = tmp_path / "pool-run" / "coordination" / "team.json"
    team = json.loads(team_path.read_text())
    while time.monotonic() < deadline and team["members"][0]["tasks_completed"] < 2:
        await asyncio.sleep(0.05)
        team = json.loads(team_path.read_text())
    await pool.shutdown()

    assert [member["teammate_id"] for member in team["members"]] == ["teammate-1"]
    assert team["members"][0]["tasks_completed"] == 2
    assert seen_topics == ["A", "B"]
    assert seen_message_counts == [1, 1]
