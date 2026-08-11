"""Product-facing API contract tests for the research frontend."""

import asyncio

from fastapi.testclient import TestClient

from open_deep_research import server
from open_deep_research.public_events import RunEventStore
from open_deep_research.public_task_activity import TaskActivityStore
from open_deep_research.run_context import RunContextStore
from security.auth import get_current_user
from tests.auth_helpers import research_principal


def _client(identity: str) -> TestClient:
    server.app.dependency_overrides[get_current_user] = lambda: research_principal(identity)
    return TestClient(server.app)


def test_capabilities_exposes_only_explicit_frontend_fields():
    client = _client("user-1")
    try:
        payload = client.get("/capabilities").json()
    finally:
        server.app.dependency_overrides.clear()

    keys = set(payload["editable_config_keys"])
    assert payload["public_event_schema_version"] == 2
    assert payload["public_task_activity_schema_version"] == 1
    assert payload["features"]["subagent_activity"] is True
    assert "research_model" in keys
    assert "mcp_config" not in keys
    assert "sandbox_allowed_domains" not in keys
    assert "langfuse_secret_key" not in keys
    assert payload["config_schema"]["additionalProperties"] is False


def test_capabilities_defaults_reflect_effective_environment(monkeypatch):
    monkeypatch.setenv("RESEARCH_MODEL", "openai:deepseek-v4-flash")
    monkeypatch.setenv(
        "QUALITY_EVALUATION_MODEL",
        "openai:deepseek-v4-flash",
    )
    client = _client("user-1")
    try:
        payload = client.get("/capabilities").json()
    finally:
        server.app.dependency_overrides.clear()

    assert payload["defaults"]["research_model"] == "openai:deepseek-v4-flash"
    assert (
        payload["defaults"]["quality_evaluation_model"]
        == "openai:deepseek-v4-flash"
    )


def test_run_history_is_owner_scoped_sorted_and_legacy_title_falls_back(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("RUNS_DIR", str(tmp_path))
    for run_id, owner, created_at, title in (
        ("old", "user-1", 1.0, None),
        ("new", "user-1", 2.0, "New research"),
        ("private", "user-2", 3.0, "Other user"),
    ):
        store = RunContextStore(run_id, runs_dir=str(tmp_path))
        store.initialize(owner, {"configurable": {"runs_dir": str(tmp_path)}})
        store._update_manifest(created_at=created_at, title=title)  # noqa: SLF001

    client = _client("user-1")
    try:
        response = client.get("/runs?limit=10")
    finally:
        server.app.dependency_overrides.clear()

    assert response.status_code == 200
    assert [item["run_id"] for item in response.json()["items"]] == ["new", "old"]
    assert response.json()["items"][1]["title"] == "old"


def test_task_activity_is_owner_scoped_and_replays_terminal_stream(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("RUNS_DIR", str(tmp_path))
    run_id = "activity-run"
    task_id = "task-1"
    context = RunContextStore(run_id, runs_dir=str(tmp_path))
    context.initialize(
        "user-1",
        {
            "configurable": {"runs_dir": str(tmp_path)},
            "metadata": {"run_id": run_id},
        },
    )
    run_events = RunEventStore(run_id, runs_dir=str(tmp_path))
    asyncio.run(run_events.append(
        "research.task.started",
        stage="researching",
        payload={
            "task_id": task_id,
            "wave_id": "wave-1",
            "title": "验证活动接口",
            "status": "running",
            "phase": "researching",
            "mode": "sync",
        },
        dedupe_key="task:started",
    ))
    activity = TaskActivityStore(run_id, task_id, runs_dir=str(tmp_path))
    asyncio.run(activity.append(
        "task.completed",
        kind="lifecycle",
        phase="terminal",
        status="success",
        title="Subagent 已完成",
        summary="任务安全完成。",
        iteration=1,
        duration_ms=12,
        payload={"mode": "sync", "wave_id": "wave-1"},
        dedupe_key="terminal",
    ))

    owner = _client("user-1")
    try:
        page = owner.get(f"/runs/{run_id}/tasks/{task_id}/activity")
        stream = owner.get(
            f"/runs/{run_id}/tasks/{task_id}/activity/stream?after=0"
        )
    finally:
        server.app.dependency_overrides.clear()
    assert page.status_code == 200
    assert page.json()["source"] == "native"
    assert page.json()["items"][0]["type"] == "task.completed"
    assert "dedupe_key" not in page.text
    assert stream.status_code == 200
    assert "event: task.completed" in stream.text

    other = _client("user-2")
    try:
        denied = other.get(f"/runs/{run_id}/tasks/{task_id}/activity")
    finally:
        server.app.dependency_overrides.clear()
    assert denied.status_code == 404
