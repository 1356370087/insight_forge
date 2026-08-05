"""Product-facing API contract tests for the research frontend."""

from fastapi.testclient import TestClient

from open_deep_research import server
from open_deep_research.run_context import RunContextStore
from security.auth import get_current_user


def _client(identity: str) -> TestClient:
    server.app.dependency_overrides[get_current_user] = lambda: {
        "identity": identity,
        "permissions": [],
    }
    return TestClient(server.app)


def test_capabilities_exposes_only_explicit_frontend_fields():
    client = _client("user-1")
    try:
        payload = client.get("/capabilities").json()
    finally:
        server.app.dependency_overrides.clear()

    keys = set(payload["editable_config_keys"])
    assert payload["public_event_schema_version"] == 2
    assert "research_model" in keys
    assert "mcp_config" not in keys
    assert "sandbox_allowed_domains" not in keys
    assert "langfuse_secret_key" not in keys
    assert payload["config_schema"]["additionalProperties"] is False


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
