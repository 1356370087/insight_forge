"""Tests for Docker sandbox configuration and lifecycle helpers."""

import json
from pathlib import Path

import pytest
from langchain_core.messages import HumanMessage

from open_deep_research.configuration import Configuration, MCPConfig
from open_deep_research.sandbox.manager import DockerSandboxManager, SandboxWorkspace
from open_deep_research.sandbox.policy import (
    allowed_domains,
    egress_host_from_url,
    is_enforced_mode,
)
from open_deep_research.tasks.events import EventType
from open_deep_research.tasks.registry import TaskRecord


class FakeContainer:
    """Minimal Docker container fake used by DockerSandboxManager tests."""

    def __init__(self, output_dir: Path) -> None:
        self.id = "abcdef1234567890"
        self.output_dir = output_dir
        self.started = False
        self.removed = False

    def start(self) -> None:
        self.started = True

    def wait(self, timeout=None):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "result.json").write_text(
            json.dumps(
                {
                    "task_id": "task-1",
                    "status": "completed",
                    "compressed_research": "sandbox findings",
                    "raw_notes": ["note"],
                    "metrics": {"sources_read": 3, "tool_calls": 2, "failures": 0},
                }
            ),
            encoding="utf-8",
        )
        return {"StatusCode": 0}

    def logs(self, stdout=True, stderr=True):
        return b"worker log"

    def remove(self, force=False) -> None:
        self.removed = True


class FakeContainers:
    def __init__(self) -> None:
        self.create_kwargs = None
        self.created = None

    def create(self, **kwargs):
        self.create_kwargs = kwargs
        output_mount = next(
            mount for mount in kwargs["mounts"]
            if _mount_target(mount) == "/workspace/output"
        )
        self.created = FakeContainer(Path(_mount_source(output_mount)))
        return self.created

    def get(self, container_id):
        return self.created


class FakeDockerClient:
    def __init__(self) -> None:
        self.containers = FakeContainers()


def _mount_target(mount):
    return getattr(mount, "target", None) or mount.get("target") or mount["Target"]


def _mount_source(mount):
    return getattr(mount, "source", None) or mount.get("source") or mount["Source"]


def _mount_read_only(mount):
    if not isinstance(mount, dict):
        return getattr(mount, "read_only", None)
    return mount.get("read_only", mount["ReadOnly"])


class TestSandboxConfiguration:
    def test_defaults_are_disabled_and_conservative(self):
        cfg = Configuration()
        assert cfg.enable_docker_sandbox is False
        assert cfg.sandbox_provider == "docker"
        assert cfg.sandbox_network_mode == "allow-search-only"
        assert cfg.sandbox_read_only_rootfs is True

    def test_allowed_domains_accept_comma_separated_env_shape(self):
        cfg = Configuration(sandbox_allowed_domains="a.example,b.example")
        assert cfg.sandbox_allowed_domains == ["a.example", "b.example"]


class TestEgressPolicy:
    def test_allowed_domains_extracts_hosts(self):
        cfg = Configuration(
            research_model="openai:gpt-4.1",
            search_api="tavily",
            mcp_config=MCPConfig(url="https://mcp.example.com/mcp"),
        )
        domains = allowed_domains(cfg)
        assert "api.openai.com" in domains
        assert "api.tavily.com" in domains
        assert "mcp.example.com" in domains

    def test_allowed_domains_includes_user_allowlist(self):
        cfg = Configuration(
            sandbox_network_mode="allowlist-domain",
            sandbox_allowed_domains=["trusted.example.com"],
        )
        assert "trusted.example.com" in allowed_domains(cfg)

    def test_allowed_domains_manager_delegates(self):
        cfg = Configuration(research_model="anthropic:claude", search_api="anthropic")
        manager = DockerSandboxManager()
        assert manager._allowed_domains(cfg) == allowed_domains(cfg)
        assert "api.anthropic.com" in allowed_domains(cfg)

    def test_egress_host_from_url_normalizes(self):
        assert egress_host_from_url("https://Example.COM/path?x=1") == "example.com"
        assert egress_host_from_url("not a url") is None
        assert egress_host_from_url("") is None

    def test_is_enforced_mode(self):
        assert is_enforced_mode(Configuration(sandbox_network_mode="allow-search-only"))
        assert is_enforced_mode(Configuration(sandbox_network_mode="allowlist-domain"))
        assert is_enforced_mode(Configuration(sandbox_network_mode="no-network"))
        assert not is_enforced_mode(Configuration(sandbox_network_mode="open-network"))


class TestSandboxEvents:
    def test_sandbox_event_types_are_constructable(self):
        assert EventType.SANDBOX_WORKSPACE_CREATED == "sandbox.workspace_created"
        assert EventType.SANDBOX_FAILED == "sandbox.failed"


class TestSandboxWorkspace:
    def test_workspace_dirs_and_mount_policies(self, tmp_path):
        workspace = SandboxWorkspace.create(tmp_path / "run" / "task-1")
        for subdir in ("input", "output", "tmp", "logs", "artifacts"):
            assert (workspace.root / subdir).is_dir()

        mounts = workspace.mount_specs()
        input_mount = next(m for m in mounts if m["target"] == "/workspace/input")
        output_mount = next(m for m in mounts if m["target"] == "/workspace/output")
        assert input_mount["read_only"] is True
        assert output_mount["read_only"] is False


class TestDockerSandboxManager:
    def test_no_network_requires_search_none(self, tmp_path):
        cfg = Configuration(
            runs_dir=str(tmp_path),
            sandbox_network_mode="no-network",
            search_api="tavily",
        )
        manager = DockerSandboxManager(docker_client=FakeDockerClient())
        with pytest.raises(ValueError, match="search_api='none'"):
            manager._validate_network_policy(cfg)

    def test_build_container_kwargs_security_and_mounts(self, tmp_path):
        cfg = Configuration(
            runs_dir=str(tmp_path),
            sandbox_network_mode="no-network",
            search_api="none",
            sandbox_image="sandbox:test",
        )
        manager = DockerSandboxManager(docker_client=FakeDockerClient())
        workspace = manager.prepare_workspace(
            configurable=cfg,
            run_id="run-1",
            task_id="task-1",
        )
        spec = manager.build_spec(cfg, {"configurable": {}})
        kwargs = manager.build_container_kwargs(workspace, spec)

        assert kwargs["image"] == "sandbox:test"
        assert kwargs["network_mode"] == "none"
        assert kwargs["read_only"] is True
        assert kwargs["cap_drop"] == ["ALL"]
        assert kwargs["security_opt"] == ["no-new-privileges:true"]
        assert kwargs["mem_limit"] == "1g"
        assert kwargs["pids_limit"] == 256

        mounts = kwargs["mounts"]
        input_mount = next(m for m in mounts if _mount_target(m) == "/workspace/input")
        output_mount = next(m for m in mounts if _mount_target(m) == "/workspace/output")
        assert _mount_read_only(input_mount) is True
        assert _mount_read_only(output_mount) is False

    @pytest.mark.asyncio
    async def test_run_researcher_task_collects_result_and_events(self, tmp_path):
        fake_client = FakeDockerClient()
        manager = DockerSandboxManager(docker_client=fake_client)
        record = TaskRecord(
            task_id="task-1",
            research_topic="topic",
            run_id="run-1",
        )
        config = {
            "configurable": {
                "enable_docker_sandbox": True,
                "runs_dir": str(tmp_path),
                "sandbox_network_mode": "no-network",
                "search_api": "none",
                "sandbox_image": "sandbox:test",
                "sandbox_cleanup_policy": "always",
                "task_timeout_seconds": 5,
            },
            "metadata": {"run_id": "run-1"},
        }
        state = {
            "researcher_messages": [HumanMessage(content="topic")],
            "research_topic": "topic",
        }

        result = await manager.run_researcher_task(
            record,
            config,
            state,
            runs_dir=str(tmp_path),
            run_id="run-1",
            event_log_enabled=True,
        )

        assert result.result["compressed_research"] == "sandbox findings"
        assert result.result["metrics"]["sources_read"] == 3
        assert record.container_id == "abcdef1234567890"
        assert record.workspace_path
        assert result.output_archive_path
        assert record.output_archive_path == result.output_archive_path
        assert Path(result.output_archive_path).is_file()

        events_path = tmp_path / "run-1" / "events.jsonl"
        events = [
            json.loads(line)["event_type"]
            for line in events_path.read_text(encoding="utf-8").splitlines()
        ]
        assert "sandbox.workspace_created" in events
        assert "sandbox.container_started" in events
        assert "sandbox.output_collected" in events
        assert "sandbox.tmp_cleaned" in events
