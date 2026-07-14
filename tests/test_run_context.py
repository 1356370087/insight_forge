"""Tests for file-backed Query session persistence."""

import asyncio
import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from open_deep_research.run_context import (
    JournalCorruptedError,
    ResearchBriefPersistenceError,
    RunContextStore,
)


def _store(tmp_path, run_id: str = "run-1", *, inline: int = 32768) -> RunContextStore:
    store = RunContextStore(run_id, runs_dir=str(tmp_path), inline_content_max_chars=inline)
    store.initialize("user-1", {"configurable": {"research_model": "openai:gpt-4.1"}})
    return store


def test_research_brief_is_exact_and_hash_verified(tmp_path) -> None:
    store = _store(tmp_path)
    brief = "# 研究目标\n\n完整保留 Unicode 与结尾。\n"

    digest = store.persist_research_brief(brief)

    assert store.load_research_brief() == brief
    assert store.load_manifest().research_brief_sha256 == digest
    assert not list(store.context_dir.glob("*.tmp"))


def test_research_brief_hash_mismatch_is_rejected(tmp_path) -> None:
    store = _store(tmp_path)
    store.persist_research_brief("original")
    store.brief_path.write_text("tampered", encoding="utf-8")

    with pytest.raises(ResearchBriefPersistenceError, match="research_brief_hash_mismatch"):
        store.load_research_brief()


@pytest.mark.asyncio
async def test_journal_replays_state_and_externalized_messages(tmp_path) -> None:
    store = _store(tmp_path, inline=1024)
    store.persist_research_brief("authoritative brief")
    large = "x" * 2048

    await store.append(
        channel="lead",
        record_type="state_delta",
        stage="received",
        payload={
            "scope": "main",
            "update": {
                "messages": {
                    "type": "override",
                    "value": [HumanMessage(content="hello"), AIMessage(content=large)],
                }
            },
        },
    )
    await store.checkpoint("research_brief_written", "plan_approval")

    replay = store.replay()

    assert [message.content for message in replay.state["messages"]] == ["hello", large]
    assert replay.state["research_brief"] == "authoritative brief"
    assert replay.manifest.next_stage == "plan_approval"
    assert list((store.context_dir / "artifacts" / "messages").glob("*.json"))


@pytest.mark.asyncio
async def test_journal_concurrent_appends_have_contiguous_sequence(tmp_path) -> None:
    store = _store(tmp_path)

    await asyncio.gather(*(
        store.append(
            channel="lead",
            record_type="state_delta",
            stage="received",
            payload={"scope": "main", "update": {"value": index}},
        )
        for index in range(20)
    ))

    records = store.replay().records
    assert [record.seq for record in records] == list(range(1, 21))


@pytest.mark.asyncio
async def test_replay_ignores_only_a_partial_final_line(tmp_path) -> None:
    store = _store(tmp_path)
    await store.append(
        channel="lead",
        record_type="state_delta",
        stage="received",
        payload={"scope": "main", "update": {"value": 1}},
    )
    with store.journal_path.open("ab") as handle:
        handle.write(b'{"schema_version":1')

    assert len(store.replay().records) == 1
    appended = await store.append(
        channel="lead",
        record_type="state_delta",
        stage="received",
        payload={"scope": "main", "update": {"value": 2}},
    )
    assert appended.seq == 2
    assert len(store.replay().records) == 2


@pytest.mark.asyncio
async def test_replay_rejects_corruption_before_final_record(tmp_path) -> None:
    store = _store(tmp_path)
    for index in range(2):
        await store.append(
            channel="lead",
            record_type="state_delta",
            stage="received",
            payload={"scope": "main", "update": {"value": index}},
        )
    lines = store.journal_path.read_text(encoding="utf-8").splitlines()
    store.journal_path.write_text("not-json\n" + lines[1] + "\n", encoding="utf-8")

    with pytest.raises(JournalCorruptedError, match="journal_corrupted"):
        store.replay()


@pytest.mark.asyncio
async def test_persisted_payload_redacts_secrets(tmp_path) -> None:
    store = _store(tmp_path)
    await store.append(
        channel="lead",
        record_type="state_delta",
        stage="received",
        payload={
            "scope": "main",
            "update": {
                "api_key": "sk-secret",
                "message": "Authorization: bearer-secret",
                "safe": "visible",
            },
        },
    )

    raw = store.journal_path.read_text(encoding="utf-8")
    assert "sk-secret" not in raw
    assert "bearer-secret" not in raw
    assert "visible" in raw
    assert "[REDACTED]" in raw


def test_artifact_path_cannot_escape_context(tmp_path) -> None:
    store = _store(tmp_path)

    with pytest.raises(ValueError, match="inside the run context"):
        store.write_json_atomic("../escape.json", {"x": 1})


def test_task_result_is_hash_verified_and_task_scoped(tmp_path) -> None:
    store = _store(tmp_path)
    digest = store.persist_task_result("task-1", {"raw_notes": ["evidence"]})

    assert store.load_task_result(
        "task-1",
        expected_sha256=digest,
    ) == {"raw_notes": ["evidence"]}

    artifact = store.context_dir / "artifacts" / "research_tasks" / "task-1.json"
    artifact.write_text('{"raw_notes":["tampered"]}', encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        store.load_task_result("task-1", expected_sha256=digest)

    with pytest.raises(ValueError, match="Invalid task_id"):
        store.persist_task_result("../escape", {"raw_notes": []})


def test_manifest_never_contains_plain_credentials(tmp_path) -> None:
    store = RunContextStore("run-secure", runs_dir=str(tmp_path))
    store.initialize(
        "user-1",
        {
            "configurable": {"api_key": "top-secret", "research_model": "model"},
            "metadata": {"authorization": "Bearer token"},
        },
    )

    payload = json.loads(store.manifest_path.read_text(encoding="utf-8"))
    serialized = json.dumps(payload)
    assert "top-secret" not in serialized
    assert "Bearer token" not in serialized


@pytest.mark.asyncio
async def test_manifest_can_be_rebuilt_from_journal(tmp_path) -> None:
    store = _store(tmp_path)
    await store.append(
        channel="lead",
        record_type="state_delta",
        stage="received",
        payload={
            "scope": "main",
            "update": {"messages": [HumanMessage(content="hello")]},
            "owner_id": "user-1",
            "config": {"configurable": {"research_model": "model"}},
        },
    )
    await store.checkpoint("research_brief_written", "plan_approval")
    store.manifest_path.write_text("broken", encoding="utf-8")

    manifest = store.load_manifest()

    assert manifest.owner_id == "user-1"
    assert manifest.last_stable_stage == "research_brief_written"
    assert manifest.next_stage == "plan_approval"
    assert manifest.last_journal_seq == 2
