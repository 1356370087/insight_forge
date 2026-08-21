"""Static guards for the V7 sandbox trust boundaries."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from open_deep_research.sandbox.manager import DockerSandboxManager

ROOT = Path(__file__).parents[1] / "src" / "open_deep_research"
WORKER_BOUNDARY = (
    ROOT / "sandbox" / "worker.py",
    ROOT / "sandbox" / "gateway_model.py",
    ROOT / "sandbox" / "gateway_tool.py",
    ROOT / "sandbox" / "local_provider.py",
)
FORBIDDEN_WORKER_IMPORTS = (
    "openai",
    "anthropic",
    "google.generativeai",
    "tavily",
    "open_deep_research.models.resolution",
    "open_deep_research.models.fallback",
    "open_deep_research.models.circuit",
)


def _imports(path: Path) -> list[tuple[str, int]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.extend((item.name, node.lineno) for item in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.append((node.module, node.lineno))
    return found


def test_worker_boundary_does_not_import_provider_or_physical_model_stacks() -> None:
    violations = []
    for path in WORKER_BOUNDARY:
        for module, line in _imports(path):
            if module.startswith(FORBIDDEN_WORKER_IMPORTS):
                violations.append(f"{path.name}:{line}:{module}")
    assert not violations, violations


def test_only_controller_opens_the_docker_sdk() -> None:
    violations = []
    for path in ROOT.rglob("*.py"):
        relative = path.relative_to(ROOT).as_posix()
        for module, line in _imports(path):
            if (module == "docker" or module.startswith("docker.")) and relative != "sandbox/controller.py":
                violations.append(f"{relative}:{line}:{module}")
    assert not violations, violations


def test_manager_rejects_legacy_docker_client_injection() -> None:
    """A dynamically supplied SDK client must never revive the host path."""
    with pytest.raises(RuntimeError, match="sandbox_controller_required"):
        DockerSandboxManager(docker_client=object())

    manager = DockerSandboxManager()
    with pytest.raises(RuntimeError, match="sandbox_controller_required"):
        manager._get_client()


def test_public_nginx_denies_internal_sandbox_control_plane() -> None:
    nginx = (
        ROOT.parents[1] / "deploy" / "research-console.nginx.conf"
    ).read_text(encoding="utf-8")
    assert "location = /api/research/internal/sandbox" in nginx
    assert "location ^~ /api/research/internal/sandbox/" in nginx
    assert nginx.count("return 404;") >= 2


def test_gateway_rpc_construction_declares_stage_operation_and_zone() -> None:
    required_by_type = {
        "GatewayModelRequestV1": {"stage", "logical_operation_id"},
        "GatewayToolRequestV1": {
            "stage",
            "logical_operation_id",
            "execution_zone",
        },
    }
    violations = []
    for path in ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                continue
            required = required_by_type.get(node.func.id)
            if required is None:
                continue
            supplied = {keyword.arg for keyword in node.keywords if keyword.arg}
            missing = sorted(required - supplied)
            if missing:
                relative = path.relative_to(ROOT).as_posix()
                violations.append(f"{relative}:{node.lineno}:{','.join(missing)}")
    assert not violations, violations


def test_run_orchestration_never_constructs_provider_models_directly() -> None:
    """Provider constructors live only in resolution or Gateway-owned tools."""
    allowed = {
        "models/resolution.py",
        "tools/tavily_search/definition.py",
        "tools/tavily_search/summarization.py",
        "tools/web_research/pipeline.py",
    }
    violations = []
    for path in ROOT.rglob("*.py"):
        relative = path.relative_to(ROOT).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "init_chat_model"
                and relative not in allowed
            ):
                violations.append(f"{relative}:{node.lineno}")
    assert not violations, violations
