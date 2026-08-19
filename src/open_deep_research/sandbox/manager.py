"""Docker sandbox lifecycle for isolated async Researcher tasks."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from langchain_core.messages import message_to_dict
from langchain_core.runnables import RunnableConfig

from open_deep_research.configuration import Configuration, SearchAPI
from open_deep_research.sandbox.policy import allowed_domains
from open_deep_research.tasks.events import EventType, JSONLEventWriter, ResearchEvent
from open_deep_research.tasks.registry import TaskRecord

CONTAINER_WORKSPACE = "/workspace"
TASK_PAYLOAD_NAME = "task_payload.json"
TASK_RESULT_NAME = "result.json"


SANDBOX_SECRET_ENV_KEYS = (
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GOOGLE_API_KEY",
    "TAVILY_API_KEY",
    "GROQ_API_KEY",
    "DEEPSEEK_API_KEY",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "AWS_REGION",
    "AWS_DEFAULT_REGION",
    "BEDROCK_AWS_REGION",
)


def _sandbox_secret_env_keys() -> tuple[str, ...]:
    """Return configured sandbox secret names; an explicit empty value disables injection."""
    configured = os.environ.get("SANDBOX_SECRET_ENV_KEYS")
    if configured is None:
        return SANDBOX_SECRET_ENV_KEYS
    return tuple(key.strip() for key in configured.split(",") if key.strip())

PROXY_ENV_KEYS = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "no_proxy",
)


def _emit_event(
    event: ResearchEvent,
    runs_dir: str,
    run_id: str,
    enabled: bool = True,
) -> None:
    """Write one sandbox event through a short-lived JSONL writer."""
    if not enabled:
        return
    writer = JSONLEventWriter(run_id=run_id, runs_dir=runs_dir)
    try:
        writer.write(event)
    finally:
        writer.close()


def _safe_error(exc: BaseException) -> str:
    """Return a bounded error string without attempting to include context."""
    return str(exc)[:1000]


def _short_container_id(container_id: Optional[str]) -> Optional[str]:
    if not container_id:
        return None
    return container_id[:12]


@dataclass(frozen=True)
class SandboxWorkspace:
    """Host-side workspace paths for a single sandboxed research task."""

    root: Path
    input_dir: Path
    output_dir: Path
    tmp_dir: Path
    logs_dir: Path
    artifacts_dir: Path

    @classmethod
    def create(cls, root: Path) -> SandboxWorkspace:
        """Create a workspace and all expected task subdirectories."""
        resolved_root = root.resolve()
        workspace = cls(
            root=resolved_root,
            input_dir=resolved_root / "input",
            output_dir=resolved_root / "output",
            tmp_dir=resolved_root / "tmp",
            logs_dir=resolved_root / "logs",
            artifacts_dir=resolved_root / "artifacts",
        )
        for path in (
            workspace.input_dir,
            workspace.output_dir,
            workspace.tmp_dir,
            workspace.logs_dir,
            workspace.artifacts_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)
        return workspace

    def mount_specs(self) -> list[dict[str, Any]]:
        """Return bind mount specs. Input is read-only; outputs are writable."""
        return [
            {
                "type": "bind",
                "source": str(self.input_dir),
                "target": f"{CONTAINER_WORKSPACE}/input",
                "read_only": True,
            },
            {
                "type": "bind",
                "source": str(self.output_dir),
                "target": f"{CONTAINER_WORKSPACE}/output",
                "read_only": False,
            },
            {
                "type": "bind",
                "source": str(self.tmp_dir),
                "target": f"{CONTAINER_WORKSPACE}/tmp",
                "read_only": False,
            },
            {
                "type": "bind",
                "source": str(self.logs_dir),
                "target": f"{CONTAINER_WORKSPACE}/logs",
                "read_only": False,
            },
            {
                "type": "bind",
                "source": str(self.artifacts_dir),
                "target": f"{CONTAINER_WORKSPACE}/artifacts",
                "read_only": False,
            },
        ]


@dataclass(frozen=True)
class SandboxSpec:
    """Container runtime specification for one sandboxed task."""

    image: str
    command: list[str]
    environment: dict[str, str]
    network_mode: str
    allowed_domains: list[str]
    memory: str
    cpus: float
    pids_limit: int
    read_only_rootfs: bool
    user: str
    timeout_seconds: int
    cleanup_policy: str


@dataclass(frozen=True)
class SandboxResult:
    """Result collected from a finished sandbox container."""

    container_id: Optional[str]
    exit_code: int
    result: dict[str, Any]
    output_archive_path: Optional[str]


class DockerSandboxManager:
    """Prepare workspaces and run Researcher workers inside Docker containers."""

    def __init__(self, docker_client: Any = None) -> None:
        """Create a manager, optionally with an injected Docker client."""
        self._client = docker_client

    async def run_researcher_task(
        self,
        task_record: TaskRecord,
        config: RunnableConfig,
        researcher_state: dict[str, Any],
        *,
        runs_dir: str = ".runs",
        run_id: str = "default",
        event_log_enabled: bool = True,
    ) -> SandboxResult:
        """Run one researcher task in a Docker sandbox and collect its output."""
        configurable = Configuration.from_runnable_config(config)
        workspace = self.prepare_workspace(
            configurable=configurable,
            run_id=run_id,
            task_id=task_record.task_id,
        )

        task_record.sandbox_enabled = True
        task_record.workspace_path = str(workspace.root)
        task_record.sandbox_network_mode = configurable.sandbox_network_mode

        self.write_payload(
            workspace=workspace,
            task_record=task_record,
            config=config,
            researcher_state=researcher_state,
        )

        self._record_event(
            task_record,
            EventType.SANDBOX_WORKSPACE_CREATED,
            runs_dir,
            run_id,
            event_log_enabled,
            data={
                "workspace_path": str(workspace.root),
                "network_mode": configurable.sandbox_network_mode,
            },
        )

        self._validate_network_policy(configurable)
        spec = self.build_spec(configurable, config)
        self._record_event(
            task_record,
            EventType.SANDBOX_NETWORK_POLICY_APPLIED,
            runs_dir,
            run_id,
            event_log_enabled,
            data={
                "network_mode": configurable.sandbox_network_mode,
                "docker_network_mode": spec.network_mode,
                "allowed_domains": spec.allowed_domains,
                "proxy_enforced": self._proxy_configured(spec.environment),
            },
        )

        result = await asyncio.to_thread(
            self._run_container_blocking,
            workspace,
            spec,
            task_record,
            runs_dir,
            run_id,
            event_log_enabled,
        )
        task_record.output_archive_path = result.output_archive_path
        return result

    def prepare_workspace(
        self,
        *,
        configurable: Configuration,
        run_id: str,
        task_id: str,
    ) -> SandboxWorkspace:
        """Create a task workspace under the configured sandbox root."""
        root_base = Path(
            configurable.sandbox_workspace_root
            or Path(configurable.runs_dir) / run_id / "workspaces"
        ).resolve()
        workspace_root = (root_base / task_id).resolve()
        if root_base not in workspace_root.parents and workspace_root != root_base:
            raise ValueError("Sandbox workspace escaped configured workspace root.")
        return SandboxWorkspace.create(workspace_root)

    def write_payload(
        self,
        *,
        workspace: SandboxWorkspace,
        task_record: TaskRecord,
        config: RunnableConfig,
        researcher_state: dict[str, Any],
    ) -> Path:
        """Write the JSON payload consumed by the container worker."""
        safe_configurable = Configuration.from_runnable_config(config).model_dump(mode="json")
        safe_configurable.pop("enable_memory", None)
        safe_configurable.pop("memory_auto_write", None)
        safe_configurable["enable_memory"] = False
        safe_configurable["memory_auto_write"] = False

        payload = {
            "task_id": task_record.task_id,
            "research_topic": task_record.research_topic,
            "researcher_state": {
                **researcher_state,
                "researcher_messages": [
                    message_to_dict(message)
                    for message in researcher_state.get("researcher_messages", [])
                ],
            },
            "configurable": safe_configurable,
            "metadata": {
                "run_id": task_record.run_id,
                "user_id": task_record.user_id,
                "task_id": task_record.task_id,
                "trace_parent_span_id": task_record.trace_parent_span_id,
                "langfuse_parent_span_id": task_record.langfuse_parent_span_id,
            },
        }

        payload_path = workspace.input_dir / TASK_PAYLOAD_NAME
        payload_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return payload_path

    def build_spec(
        self,
        configurable: Configuration,
        config: RunnableConfig,
    ) -> SandboxSpec:
        """Build the container spec from runtime configuration."""
        env = self._build_environment(configurable, config)
        network_mode = "none" if configurable.sandbox_network_mode == "no-network" else "bridge"
        timeout = configurable.sandbox_timeout_seconds or configurable.task_timeout_seconds
        return SandboxSpec(
            image=configurable.sandbox_image,
            command=["python", "-m", "open_deep_research.sandbox.worker"],
            environment=env,
            network_mode=network_mode,
            allowed_domains=self._allowed_domains(configurable),
            memory=configurable.sandbox_memory,
            cpus=configurable.sandbox_cpus,
            pids_limit=configurable.sandbox_pids_limit,
            read_only_rootfs=configurable.sandbox_read_only_rootfs,
            user=configurable.sandbox_user,
            timeout_seconds=timeout,
            cleanup_policy=configurable.sandbox_cleanup_policy,
        )

    def build_container_kwargs(
        self,
        workspace: SandboxWorkspace,
        spec: SandboxSpec,
    ) -> dict[str, Any]:
        """Build Docker SDK container creation kwargs."""
        return {
            "image": spec.image,
            "command": spec.command,
            "detach": True,
            "environment": spec.environment,
            "network_mode": spec.network_mode,
            "mounts": self._coerce_mounts_for_docker(workspace.mount_specs()),
            "working_dir": CONTAINER_WORKSPACE,
            "read_only": spec.read_only_rootfs,
            "cap_drop": ["ALL"],
            "security_opt": ["no-new-privileges:true"],
            "mem_limit": spec.memory,
            "nano_cpus": int(spec.cpus * 1_000_000_000),
            "pids_limit": spec.pids_limit,
            "user": spec.user,
        }

    def stop_container(self, container_id: str, *, timeout: int = 5) -> None:
        """Best-effort stop and remove for a running sandbox container."""
        client = self._get_client()
        container = client.containers.get(container_id)
        try:
            container.stop(timeout=timeout)
        finally:
            try:
                container.remove(force=True)
            except Exception:
                pass

    def _run_container_blocking(
        self,
        workspace: SandboxWorkspace,
        spec: SandboxSpec,
        task_record: TaskRecord,
        runs_dir: str,
        run_id: str,
        event_log_enabled: bool,
    ) -> SandboxResult:
        client = self._get_client()
        container = None
        exit_code = 1
        try:
            container = client.containers.create(
                **self.build_container_kwargs(workspace, spec)
            )
            task_record.container_id = getattr(container, "id", None)
            self._record_event(
                task_record,
                EventType.SANDBOX_CONTAINER_CREATED,
                runs_dir,
                run_id,
                event_log_enabled,
                data={"container_id": _short_container_id(task_record.container_id)},
            )

            container.start()
            self._record_event(
                task_record,
                EventType.SANDBOX_CONTAINER_STARTED,
                runs_dir,
                run_id,
                event_log_enabled,
                data={"container_id": _short_container_id(task_record.container_id)},
            )

            wait_result = container.wait(timeout=spec.timeout_seconds)
            if isinstance(wait_result, dict):
                exit_code = int(wait_result.get("StatusCode", 1))
            else:
                exit_code = int(wait_result)

            logs = container.logs(stdout=True, stderr=True)
            if isinstance(logs, bytes):
                logs = logs.decode("utf-8", errors="replace")
            (workspace.logs_dir / "container.log").write_text(str(logs), encoding="utf-8")

            if exit_code != 0:
                raise RuntimeError(f"Sandbox container exited with code {exit_code}.")

            result = self._read_worker_result(workspace)
            archive_path = self._archive_output(workspace, runs_dir, run_id, task_record.task_id)
            self._record_event(
                task_record,
                EventType.SANDBOX_OUTPUT_COLLECTED,
                runs_dir,
                run_id,
                event_log_enabled,
                data={
                    "container_id": _short_container_id(task_record.container_id),
                    "exit_code": exit_code,
                    "output_archive_path": archive_path,
                },
            )
            return SandboxResult(
                container_id=task_record.container_id,
                exit_code=exit_code,
                result=result,
                output_archive_path=archive_path,
            )

        except Exception as exc:
            self._record_event(
                task_record,
                EventType.SANDBOX_FAILED,
                runs_dir,
                run_id,
                event_log_enabled,
                data={
                    "container_id": _short_container_id(task_record.container_id),
                    "exit_code": exit_code,
                    "error": _safe_error(exc),
                },
            )
            raise
        finally:
            cleanup = spec.cleanup_policy
            should_cleanup = cleanup == "always" or (cleanup == "on_success" and exit_code == 0)
            if should_cleanup:
                self._cleanup_tmp(workspace, task_record, runs_dir, run_id, event_log_enabled)
                if container is not None:
                    try:
                        container.remove(force=True)
                        self._record_event(
                            task_record,
                            EventType.SANDBOX_CONTAINER_REMOVED,
                            runs_dir,
                            run_id,
                            event_log_enabled,
                            data={"container_id": _short_container_id(task_record.container_id)},
                        )
                    except Exception:
                        pass

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            import docker
        except ImportError as exc:
            raise RuntimeError(
                "Docker sandbox is enabled but the Docker SDK is not installed. "
                "Install dependencies with `uv sync` or add `docker>=7.1.0`."
            ) from exc
        self._client = docker.from_env()
        return self._client

    def _coerce_mounts_for_docker(self, mount_specs: list[dict[str, Any]]) -> list[Any]:
        try:
            from docker.types import Mount
        except ImportError:
            return mount_specs
        return [
            Mount(
                target=spec["target"],
                source=spec["source"],
                type=spec["type"],
                read_only=spec["read_only"],
            )
            for spec in mount_specs
        ]

    def _read_worker_result(self, workspace: SandboxWorkspace) -> dict[str, Any]:
        result_path = workspace.output_dir / TASK_RESULT_NAME
        if not result_path.exists():
            raise RuntimeError("Sandbox worker did not produce output/result.json.")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("status") == "failed":
            raise RuntimeError(result.get("error", "Sandbox worker failed."))
        return {
            "compressed_research": result.get("compressed_research", ""),
            "raw_notes": result.get("raw_notes", []),
            "metrics": result.get("metrics", {}),
        }

    def _archive_output(
        self,
        workspace: SandboxWorkspace,
        runs_dir: str,
        run_id: str,
        task_id: str,
    ) -> str:
        artifacts_root = Path(runs_dir) / run_id / "artifacts"
        artifacts_root.mkdir(parents=True, exist_ok=True)
        archive_path = artifacts_root / f"{task_id}.zip"
        with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for base_dir in (workspace.output_dir, workspace.logs_dir, workspace.artifacts_dir):
                if not base_dir.exists():
                    continue
                for path in base_dir.rglob("*"):
                    if path.is_file():
                        zf.write(path, path.relative_to(workspace.root))
        return str(archive_path)

    def _cleanup_tmp(
        self,
        workspace: SandboxWorkspace,
        task_record: TaskRecord,
        runs_dir: str,
        run_id: str,
        event_log_enabled: bool,
    ) -> None:
        if workspace.tmp_dir.exists():
            shutil.rmtree(workspace.tmp_dir)
        workspace.tmp_dir.mkdir(parents=True, exist_ok=True)
        self._record_event(
            task_record,
            EventType.SANDBOX_TMP_CLEANED,
            runs_dir,
            run_id,
            event_log_enabled,
            data={"workspace_path": str(workspace.root)},
        )

    def _record_event(
        self,
        task_record: TaskRecord,
        event_type: EventType,
        runs_dir: str,
        run_id: str,
        event_log_enabled: bool,
        *,
        data: Optional[dict[str, Any]] = None,
    ) -> None:
        task_record.last_sandbox_event = event_type.value
        payload = {
            "network_mode": task_record.sandbox_network_mode,
            "workspace_path": task_record.workspace_path,
            **(data or {}),
        }
        _emit_event(
            ResearchEvent(
                event_type=event_type,
                task_id=task_record.task_id,
                run_id=run_id,
                data=payload,
            ),
            runs_dir=runs_dir,
            run_id=run_id,
            enabled=event_log_enabled,
        )

    def _validate_network_policy(self, configurable: Configuration) -> None:
        if configurable.sandbox_network_mode != "no-network":
            return
        search_api = configurable.search_api
        if isinstance(search_api, SearchAPI):
            search_value = search_api.value
        else:
            search_value = str(search_api)
        if search_value != SearchAPI.NONE.value:
            raise ValueError(
                "sandbox_network_mode='no-network' requires search_api='none'."
            )
        if configurable.mcp_config and configurable.mcp_config.url:
            raise ValueError(
                "sandbox_network_mode='no-network' cannot be used with networked MCP servers."
            )

    def _build_environment(
        self,
        configurable: Configuration,
        config: RunnableConfig,
    ) -> dict[str, str]:
        env: dict[str, str] = {
            "PYTHONUNBUFFERED": "1",
            "GET_API_KEYS_FROM_CONFIG": "false",
            "SANDBOX_TASK_PAYLOAD_PATH": f"{CONTAINER_WORKSPACE}/input/{TASK_PAYLOAD_NAME}",
            "SANDBOX_RESULT_PATH": f"{CONTAINER_WORKSPACE}/output/{TASK_RESULT_NAME}",
            "SANDBOX_LOG_PATH": f"{CONTAINER_WORKSPACE}/logs/worker.log",
            "SANDBOX_NETWORK_MODE": configurable.sandbox_network_mode,
            "SANDBOX_EGRESS_ALLOWED_DOMAINS": ",".join(
                self._allowed_domains(configurable)
            ),
        }

        api_keys = config.get("configurable", {}).get("apiKeys", {}) if config else {}
        for key in _sandbox_secret_env_keys():
            value = api_keys.get(key) or os.getenv(key)
            if value:
                env[key] = value

        if configurable.sandbox_network_mode in {"allow-search-only", "allowlist-domain"}:
            for key in PROXY_ENV_KEYS:
                value = os.getenv(key)
                if value:
                    env[key] = value

        return env

    def _allowed_domains(self, configurable: Configuration) -> list[str]:
        return allowed_domains(configurable)

    def _proxy_configured(self, env: dict[str, str]) -> bool:
        return bool(env.get("HTTPS_PROXY") or env.get("https_proxy"))


def stop_sandbox_container(container_id: str) -> None:
    """Stop a sandbox container by ID using a fresh Docker client."""
    DockerSandboxManager().stop_container(container_id)
