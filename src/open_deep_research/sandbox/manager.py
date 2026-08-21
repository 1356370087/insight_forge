"""Docker sandbox lifecycle for isolated async Researcher tasks."""

from __future__ import annotations

import asyncio
import os
import shutil
import stat
import time
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from langchain_core.messages import message_to_dict
from langchain_core.runnables import RunnableConfig

from open_deep_research.configuration import Configuration
from open_deep_research.sandbox.crypto import SandboxDerivedKeys, encode_task_token
from open_deep_research.sandbox.safe_io import ArchiveLimits, extract_safe_tar
from open_deep_research.sandbox.schema import (
    SandboxProfile,
    policy_digest,
    resolve_profile,
    runtime_digest,
)
from open_deep_research.sandbox.wire import (
    SandboxTaskPayloadV1,
    SandboxTaskResultV1,
    TaskTokenClaimsV1,
)
from open_deep_research.tasks.events import EventType, JSONLEventWriter, ResearchEvent
from open_deep_research.tasks.registry import TaskRecord

CONTAINER_WORKSPACE = "/workspace"
TASK_PAYLOAD_NAME = "task_payload.json"
TASK_RESULT_NAME = "result.json"
# Security allowlist: every future Configuration field is excluded until it is
# explicitly reviewed for Worker semantics and added here.
_SANDBOX_RUNTIME_CONFIG_KEYS = frozenset(
    {
        "max_structured_output_retries",
        "model_transport_max_attempts",
        "model_circuit_breaker_enabled",
        "model_first_packet_probe",
        "context_recovery_max_attempts",
        "output_token_escalation_enabled",
        "output_continuation_max_attempts",
        "model_fallbacks",
        "model_context_window_overrides",
        "model_max_output_tokens_overrides",
        "unknown_model_context_window_tokens",
        "max_concurrent_tool_calls",
        "max_tool_batch_size",
        "model_call_timeout_seconds",
        "tool_call_timeout_seconds",
        "research_tool_call_timeout_seconds",
        "hook_timeout_seconds",
        "max_concurrent_research_units",
        "search_api",
        "max_researcher_iterations",
        "max_react_tool_calls",
        "summarization_model",
        "summarization_model_max_tokens",
        "max_content_length",
        "research_model",
        "research_model_max_tokens",
        "compression_model",
        "compression_model_max_tokens",
        "researcher_tool_whitelist",
        "researcher_blocked_origins",
        "max_tool_retries",
        "tool_retry_base_delay",
        "tool_retry_max_delay",
        "tool_param_constraints",
        "role_tool_blacklist",
        "role_blocked_origins",
        "task_timeout_seconds",
        "web_pipeline_mode",
        "web_pipeline_shadow_sample_rate",
        "fetch_backend_order",
        "external_extract_backends",
        "fetch_top_k",
        "web_min_source_authority",
        "search_candidate_limit",
        "max_fetches_per_researcher",
        "max_fetches_per_run",
        "fetch_global_concurrency",
        "fetch_per_host_concurrency",
        "web_rerank_model",
        "web_evidence_model",
        "html_max_bytes",
        "pdf_max_bytes",
        "pdf_max_pages",
        "respect_robots_txt",
        "browser_render_fallback_enabled",
        "enable_message_summarization",
        "message_summary_trigger_tokens",
        "message_summary_keep_last",
        "message_summary_model",
        "message_summary_model_max_tokens",
        "quality_evaluation_enabled",
        "quality_evaluation_model",
        "quality_evaluation_model_max_tokens",
        "quality_evaluation_fail_open",
        "quality_evaluation_rigor",
        "quality_evaluation_min_sources",
        "quality_evaluation_max_input_chars",
        "quality_risk_mode",
        "quality_caveat_admission_enabled",
        "quality_gap_recovery_max_attempts",
        "query_context_compaction_enabled",
        "query_context_trigger_ratio",
        "query_context_recent_window_ratio",
        "query_context_summary_max_tokens",
        "query_journal_inline_content_max_chars",
        "prompt_injection_protection_enabled",
        "external_content_fail_closed",
        "max_external_content_bytes",
        "max_mcp_description_chars",
        "max_tool_description_chars",
        "max_mcp_output_chars",
        "sandbox_profile_id",
        "gateway_protocol_version",
        "sandbox_enabled",
        "enable_async_research",
        "enable_memory",
        "memory_auto_write",
        "observability_enabled",
        "sqlite_observability_enabled",
        "token_usage_accounting_enabled",
        "token_usage_estimation_enabled",
        "event_log_enabled",
        "task_checkpoint_enabled",
        "query_session_persistence_enabled",
        "task_state_backend",
        "runs_dir",
        "trace_store_path",
        "langfuse_enabled",
        "prometheus_enabled",
        "helicone_enabled",
        "browser_mcp_enabled",
        "mcp_config",
        "run_deadline_seconds",
        "max_run_model_calls",
        "max_run_tool_calls",
        "max_run_input_tokens",
        "max_run_output_tokens",
        "max_run_cost_micro_usd",
    }
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


def _write_regular_file(path: Path, data: bytes, *, max_bytes: int) -> None:
    """Write a bounded host file without following a container-created link."""
    bounded = data[:max_bytes]
    if path.is_symlink():
        raise RuntimeError(f"Refusing to follow sandbox symlink: {path}")
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        os.write(descriptor, bounded)
    finally:
        os.close(descriptor)


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
    memory: int
    cpus: float
    pids_limit: int
    read_only_rootfs: bool
    user: str
    timeout_seconds: int
    retention: str
    output_bytes: int
    log_bytes: int
    artifact_bytes: int
    max_files: int
    stop_grace_seconds: int
    labels: dict[str, str]


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
        """Create a Controller-only manager and reject the removed SDK seam."""
        if docker_client is not None:
            raise RuntimeError("sandbox_controller_required")
        self._client = None

    async def _create_controller_task(
        self,
        controller: Any,
        *,
        payload: Any,
        task_token: str,
        runtime_digest_value: str,
        stop_grace_seconds: int,
    ) -> Any:
        """Finish an in-flight create and stop its container if caller cancels."""
        create = asyncio.create_task(
            controller.create_task(
                payload=payload,
                task_token=task_token,
                runtime_digest_value=runtime_digest_value,
            )
        )
        try:
            return await asyncio.shield(create)
        except asyncio.CancelledError:
            created = None
            try:
                created = await asyncio.shield(create)
            except Exception:
                pass
            if created is not None:
                stop = asyncio.create_task(
                    controller.stop_task(
                        created.container_id,
                        timeout_seconds=stop_grace_seconds,
                    )
                )
                try:
                    await asyncio.shield(stop)
                except Exception:
                    pass
            raise

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
        bundle, profile_id, profile = resolve_profile(configurable)
        if self._client is not None:
            raise RuntimeError("sandbox_controller_required")
        if self._client is None:
            from open_deep_research.sandbox.controller_client import (
                SandboxControllerClient,
            )
            from open_deep_research.sandbox.gateway_client import (
                SandboxGatewayControlClient,
                split_gateway_registration,
            )

            metadata = config.get("metadata", {}) if config else {}
            fence_token = max(1, int(metadata.get("run_fence_token") or 1))
            now = time.time()
            task_token_expires_at = (
                now + profile.resources.timeout_seconds + 60
            )
            frozen_for_gateway, credentials = split_gateway_registration(
                dict(config or {})
            )
            await SandboxGatewayControlClient(configurable).register_run(
                run_id=task_record.run_id or str(metadata.get("run_id") or "default"),
                fence_token=fence_token,
                frozen_config=frozen_for_gateway,
                api_keys=credentials,
                expires_at=task_token_expires_at,
            )

            payload = self.build_payload(
                task_record=task_record,
                config=config,
                researcher_state=researcher_state,
                profile_id=profile_id,
                policy_digest_value=policy_digest(bundle),
            )
            claims = TaskTokenClaimsV1(
                run_id=payload.run_id,
                task_id=payload.task_id,
                fence_token=payload.fence_token,
                profile_id=profile_id,
                policy_digest=payload.policy_digest,
                issued_at=now,
                expires_at=task_token_expires_at,
                jti=str(uuid.uuid4()),
            )
            keys = SandboxDerivedKeys.from_root(configurable.sandbox_root_signing_key or "")
            controller = SandboxControllerClient(configurable, bundle)
            created = await self._create_controller_task(
                controller,
                payload=payload,
                task_token=encode_task_token(claims, keys.task_token),
                runtime_digest_value=runtime_digest(profile),
                stop_grace_seconds=profile.resources.stop_grace_seconds,
            )
            task_record.sandbox_enabled = True
            task_record.workspace_path = f"controller://{payload.task_id}"
            task_record.sandbox_network_mode = profile.network.mode
            task_record.container_id = created.container_id
            self._record_event(
                task_record,
                EventType.SANDBOX_CONTAINER_CREATED,
                runs_dir,
                run_id,
                event_log_enabled,
                data={"container_id": _short_container_id(created.container_id)},
            )
            # Persist the container identity before start so crash recovery and
            # cancellation never depend on a terminal Controller response.
            from open_deep_research.tasks.state import get_task_state_store

            await get_task_state_store(configurable).update_from_record(
                task_record,
                fence_token=payload.fence_token,
            )
            try:
                if created.status == "created":
                    status = await controller.start_task(created.container_id)
                else:
                    status = await controller.task_status(created.container_id)
                self._record_event(
                    task_record,
                    EventType.SANDBOX_CONTAINER_STARTED,
                    runs_dir,
                    run_id,
                    event_log_enabled,
                    data={"container_id": _short_container_id(created.container_id)},
                )
                deadline = time.monotonic() + profile.resources.timeout_seconds
                while status.status not in {"result_ready", "exited", "dead"}:
                    if task_record.cancelled.is_set():
                        raise asyncio.CancelledError
                    if time.monotonic() >= deadline:
                        raise TimeoutError("sandbox task timed out")
                    await asyncio.sleep(0.2)
                    status = await controller.task_status(created.container_id)
                response = await controller.collect_task(created.container_id)
                canonical_archive = await controller.collect_archive(
                    created.container_id
                )
                archive_path = self._archive_controller_output(
                    canonical_archive,
                    response.logs,
                    runs_dir=runs_dir,
                    run_id=run_id,
                    task_id=task_record.task_id,
                    profile=profile,
                )
                task_record.output_archive_path = archive_path
                self._record_event(
                    task_record,
                    EventType.SANDBOX_OUTPUT_COLLECTED,
                    runs_dir,
                    run_id,
                    event_log_enabled,
                    data={
                        "container_id": _short_container_id(created.container_id),
                        "exit_code": response.exit_code,
                        "output_archive_path": archive_path,
                    },
                )
                if (
                    status.timed_out
                    or response.exit_code != 0
                    or response.result.status == "failed"
                ):
                    raise RuntimeError(response.result.error or "Sandbox worker failed.")
                return SandboxResult(
                    container_id=response.container_id,
                    exit_code=response.exit_code,
                    result=response.result.model_dump(
                        mode="json",
                        exclude={"schema_version", "status", "task_id", "error"},
                    ),
                    output_archive_path=archive_path,
                )
            finally:
                try:
                    await controller.stop_task(
                        created.container_id,
                        timeout_seconds=profile.resources.stop_grace_seconds,
                    )
                    if profile.runtime.retention == "remove":
                        self._record_event(
                            task_record,
                            EventType.SANDBOX_CONTAINER_REMOVED,
                            runs_dir,
                            run_id,
                            event_log_enabled,
                            data={
                                "container_id": _short_container_id(
                                    created.container_id
                                )
                            },
                        )
                except Exception as stop_exc:
                    self._record_event(
                        task_record,
                        EventType.SANDBOX_FAILED,
                        runs_dir,
                        run_id,
                        event_log_enabled,
                        data={
                            "container_id": _short_container_id(
                                created.container_id
                            ),
                            "error": f"sandbox_termination_failed:{_safe_error(stop_exc)}",
                        },
                    )
                    raise RuntimeError("sandbox_termination_failed") from stop_exc
        workspace = self.prepare_workspace(
            configurable=configurable,
            run_id=run_id,
            task_id=task_record.task_id,
        )

        task_record.sandbox_enabled = True
        task_record.workspace_path = str(workspace.root)
        task_record.sandbox_network_mode = profile.network.mode

        self.write_payload(
            workspace=workspace,
            task_record=task_record,
            config=config,
            researcher_state=researcher_state,
            profile_id=profile_id,
        )

        self._record_event(
            task_record,
            EventType.SANDBOX_WORKSPACE_CREATED,
            runs_dir,
            run_id,
            event_log_enabled,
            data={
                "workspace_path": str(workspace.root),
                "network_mode": profile.network.mode,
                "profile_id": profile_id,
            },
        )

        spec = self.build_spec(configurable, config)
        self._record_event(
            task_record,
            EventType.SANDBOX_NETWORK_POLICY_APPLIED,
            runs_dir,
            run_id,
            event_log_enabled,
            data={
                "network_mode": profile.network.mode,
                "docker_network_mode": spec.network_mode,
                "allowed_domains": spec.allowed_domains,
                "proxy_enforced": bool(configurable.sandbox_gateway_url),
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
        root_base = (Path(configurable.runs_dir) / run_id / "workspaces").resolve()
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
        profile_id: str | None = None,
    ) -> Path:
        """Write the JSON payload consumed by the container worker."""
        payload = self.build_payload(
            task_record=task_record,
            config=config,
            researcher_state=researcher_state,
            profile_id=profile_id,
        )
        payload_path = workspace.input_dir / TASK_PAYLOAD_NAME
        payload_path.write_text(payload.model_dump_json(), encoding="utf-8")
        return payload_path

    def build_payload(
        self,
        *,
        task_record: TaskRecord,
        config: RunnableConfig,
        researcher_state: dict[str, Any],
        profile_id: str | None = None,
        policy_digest_value: str | None = None,
    ) -> SandboxTaskPayloadV1:
        """Build a strict, secret-free payload without touching host storage."""
        configurable = Configuration.from_runnable_config(config)
        _bundle, resolved_profile_id, _profile = resolve_profile(configurable)
        safe_configurable = configurable.model_dump(
            mode="json",
            include=_SANDBOX_RUNTIME_CONFIG_KEYS,
        )
        safe_configurable.update(
            {
                "sandbox_enabled": False,
                "enable_async_research": False,
                "model_circuit_breaker_enabled": False,
                "model_first_packet_probe": "off",
                "model_fallbacks": {},
                "enable_memory": False,
                "memory_auto_write": False,
                "observability_enabled": False,
                "sqlite_observability_enabled": False,
                "token_usage_accounting_enabled": False,
                "event_log_enabled": False,
                "task_checkpoint_enabled": False,
                "query_session_persistence_enabled": False,
                "task_state_backend": "memory",
                "runs_dir": "/workspace/tmp/runs",
                "trace_store_path": "/workspace/tmp/traces.sqlite3",
                "langfuse_enabled": False,
                "prometheus_enabled": False,
                "helicone_enabled": False,
                "browser_mcp_enabled": False,
                "mcp_config": None,
                "run_deadline_seconds": None,
                "max_run_model_calls": None,
                "max_run_tool_calls": None,
                "max_run_input_tokens": None,
                "max_run_output_tokens": None,
                "max_run_cost_micro_usd": None,
            }
        )
        auth_user = config.get("configurable", {}).get("langgraph_auth_user") if config else None
        if isinstance(auth_user, dict):
            safe_configurable["langgraph_auth_user"] = {
                "identity": str(auth_user.get("identity") or ""),
                "roles": [str(value) for value in auth_user.get("roles", [])],
                "permissions": [
                    str(value) for value in auth_user.get("permissions", [])
                ],
                "is_authenticated": bool(auth_user.get("is_authenticated", True)),
            }
        safe_state = {
            key: value
            for key, value in researcher_state.items()
            if key != "_query_checkpoint_callback"
        }
        safe_state["researcher_messages"] = [
            message_to_dict(message)
            for message in researcher_state.get("researcher_messages", [])
        ]
        metadata = config.get("metadata", {}) if config else {}
        return SandboxTaskPayloadV1(
            task_id=task_record.task_id,
            run_id=task_record.run_id or str(metadata.get("run_id") or "default"),
            research_topic=task_record.research_topic,
            researcher_state=safe_state,
            runtime_config=safe_configurable,
            profile_id=profile_id or resolved_profile_id,
            policy_digest=policy_digest_value or configurable.sandbox_policy_digest,
            fence_token=max(1, int(metadata.get("run_fence_token") or 1)),
        )

    def build_spec(
        self,
        configurable: Configuration,
        config: RunnableConfig,
    ) -> SandboxSpec:
        """Build the container spec from runtime configuration."""
        bundle, _profile_id, profile = resolve_profile(configurable)
        env = self._build_environment(configurable, config, profile)
        metadata = config.get("metadata", {}) if config else {}
        return SandboxSpec(
            image=profile.runtime.worker_image_digest,
            command=["python", "-m", "open_deep_research.sandbox.worker"],
            environment=env,
            network_mode="none",
            allowed_domains=list(profile.network.allow_domains),
            memory=profile.resources.memory_bytes,
            cpus=profile.resources.cpu_cores,
            pids_limit=profile.resources.pids,
            read_only_rootfs=profile.runtime.read_only_rootfs,
            user=f"{profile.runtime.uid}:{profile.runtime.gid}",
            timeout_seconds=profile.resources.timeout_seconds,
            retention=profile.runtime.retention,
            output_bytes=profile.resources.output_bytes,
            log_bytes=profile.resources.log_bytes,
            artifact_bytes=profile.resources.artifact_bytes,
            max_files=profile.resources.max_files,
            stop_grace_seconds=profile.resources.stop_grace_seconds,
            labels={
                "com.insightforge.sandbox.deployment_id": bundle.deployment_id,
                "com.insightforge.sandbox.run_id": str(metadata.get("run_id") or "default"),
                "com.insightforge.sandbox.task_id": str(metadata.get("task_id") or "pending"),
                "com.insightforge.sandbox.fence_token": str(metadata.get("run_fence_token") or 0),
                "com.insightforge.sandbox.profile_id": configurable.sandbox_profile_id,
                "com.insightforge.sandbox.policy_digest": configurable.sandbox_policy_digest,
            },
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
            "labels": spec.labels,
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
                log_bytes = logs
            else:
                log_bytes = str(logs).encode("utf-8", errors="replace")
            _write_regular_file(
                workspace.logs_dir / "container.log",
                log_bytes,
                max_bytes=spec.log_bytes,
            )

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
            if container is not None:
                stop = getattr(container, "stop", None)
                if callable(stop):
                    try:
                        stop(timeout=spec.stop_grace_seconds)
                    except Exception:
                        pass
            self._cleanup_tmp(workspace, task_record, runs_dir, run_id, event_log_enabled)
            if container is not None and spec.retention == "remove":
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
        """Reject the removed direct Docker execution path unconditionally."""
        raise RuntimeError("sandbox_controller_required")

    def _coerce_mounts_for_docker(self, mount_specs: list[dict[str, Any]]) -> list[Any]:
        """Return SDK-neutral mount data for the explicit test seam."""
        return mount_specs

    def _read_worker_result(self, workspace: SandboxWorkspace) -> dict[str, Any]:
        result_path = workspace.output_dir / TASK_RESULT_NAME
        if not result_path.exists():
            raise RuntimeError("Sandbox worker did not produce output/result.json.")
        info = result_path.lstat()
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
            raise RuntimeError("Sandbox result must be a regular file.")
        if info.st_size > 64 * 1024 * 1024:
            raise RuntimeError("Sandbox result exceeds the 64 MiB contract.")
        result = SandboxTaskResultV1.model_validate_json(
            result_path.read_text(encoding="utf-8")
        )
        if result.status == "failed":
            raise RuntimeError(result.error or "Sandbox worker failed.")
        return result.model_dump(mode="json", exclude={"schema_version", "status", "task_id", "error"})

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
        if archive_path.is_symlink():
            raise RuntimeError("Refusing to overwrite sandbox archive symlink.")
        total_bytes = 0
        total_files = 0
        with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for base_dir in (workspace.output_dir, workspace.logs_dir, workspace.artifacts_dir):
                if not base_dir.exists():
                    continue
                for path in base_dir.rglob("*"):
                    info = path.lstat()
                    if stat.S_ISLNK(info.st_mode):
                        raise RuntimeError(f"Sandbox output contains symlink: {path}")
                    if stat.S_ISDIR(info.st_mode):
                        continue
                    if not stat.S_ISREG(info.st_mode):
                        raise RuntimeError(f"Sandbox output contains special file: {path}")
                    total_files += 1
                    total_bytes += info.st_size
                    if total_files > 10_000 or total_bytes > 512 * 1024 * 1024:
                        raise RuntimeError("Sandbox output archive exceeds collection limits.")
                    zf.write(path, path.relative_to(workspace.root))
        return str(archive_path)

    def _archive_controller_output(
        self,
        archive: bytes,
        controller_logs: str,
        *,
        runs_dir: str,
        run_id: str,
        task_id: str,
        profile: SandboxProfile,
    ) -> str:
        """Validate Controller output again and create the durable host ZIP."""
        runs_root = Path(runs_dir).resolve()
        collection_root = (
            runs_root
            / run_id
            / "sandbox"
            / "collections"
            / f".{uuid.uuid4().hex}"
        ).resolve()
        if runs_root not in collection_root.parents:
            raise RuntimeError("sandbox collection path escaped runs directory")
        collection_root.mkdir(parents=True, exist_ok=False)
        workspace = SandboxWorkspace(
            root=collection_root,
            input_dir=collection_root / "input",
            output_dir=collection_root / "output",
            tmp_dir=collection_root / "tmp",
            logs_dir=collection_root / "logs",
            artifacts_dir=collection_root / "artifacts",
        )
        try:
            extract_safe_tar(
                archive,
                collection_root,
                limits=ArchiveLimits(
                    max_bytes=(
                        profile.resources.output_bytes
                        + profile.resources.log_bytes
                        + profile.resources.artifact_bytes
                    ),
                    max_files=profile.resources.max_files,
                ),
            )
            for path in (
                workspace.input_dir,
                workspace.output_dir,
                workspace.tmp_dir,
                workspace.logs_dir,
                workspace.artifacts_dir,
            ):
                path.mkdir(parents=True, exist_ok=True)
            if controller_logs:
                _write_regular_file(
                    workspace.logs_dir / "controller.log",
                    controller_logs.encode("utf-8", errors="replace"),
                    max_bytes=profile.resources.log_bytes,
                )
            return self._archive_output(
                workspace,
                runs_dir,
                run_id,
                task_id,
            )
        finally:
            shutil.rmtree(collection_root, ignore_errors=True)

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

    def _build_environment(
        self,
        configurable: Configuration,
        config: RunnableConfig,
        profile: SandboxProfile,
    ) -> dict[str, str]:
        del config
        return {
            "PYTHONUNBUFFERED": "1",
            "GET_API_KEYS_FROM_CONFIG": "false",
            "SANDBOX_TASK_PAYLOAD_PATH": f"{CONTAINER_WORKSPACE}/input/{TASK_PAYLOAD_NAME}",
            "SANDBOX_RESULT_PATH": f"{CONTAINER_WORKSPACE}/output/{TASK_RESULT_NAME}",
            "SANDBOX_LOG_PATH": f"{CONTAINER_WORKSPACE}/logs/worker.log",
            "SANDBOX_NETWORK_POLICY_MODE": profile.network.mode,
            "SANDBOX_GATEWAY_URL": configurable.sandbox_gateway_url,
            "SANDBOX_PROFILE_ID": configurable.sandbox_profile_id,
            "TMPDIR": f"{CONTAINER_WORKSPACE}/tmp",
            "TMP": f"{CONTAINER_WORKSPACE}/tmp",
            "TEMP": f"{CONTAINER_WORKSPACE}/tmp",
        }
