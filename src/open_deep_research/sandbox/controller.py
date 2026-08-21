"""Trusted Docker controller; the only InsightForge process owning Docker Socket."""

from __future__ import annotations

import asyncio
import hashlib
import io
import os
import queue
import stat
import tarfile
import threading
import time
from contextlib import suppress
from datetime import datetime
from typing import Any

from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, ConfigDict, Field

from open_deep_research.configuration import Configuration
from open_deep_research.sandbox.crypto import (
    NonceReplayCache,
    SandboxDerivedKeys,
    decode_task_token,
    validate_timestamp,
    verify_payload,
)
from open_deep_research.sandbox.safe_io import (
    ArchiveLimits,
    repack_docker_archives,
)
from open_deep_research.sandbox.schema import (
    SandboxProfile,
    load_policy_bundle,
    policy_digest,
    runtime_digest,
)
from open_deep_research.sandbox.wire import (
    WORKER_EXIT_CODE_PATH,
    SandboxTaskPayloadV1,
    SandboxTaskResultV1,
)

CONTROLLER_EXEC_TIMEOUT_SECONDS = 2.0
WATCHDOG_BATCH_TIMEOUT_SECONDS = 10.0


class SandboxWorkerProtocolError(RuntimeError):
    """Raised when a Worker replaces a fixed control file with an unsafe type."""


class ControllerRunRequest(BaseModel):
    """Authenticated request to create and run exactly one Worker container."""

    model_config = ConfigDict(extra="forbid")

    payload: SandboxTaskPayloadV1
    task_token: str
    runtime_digest: str
    policy_signature: str
    service_timestamp: float
    service_nonce: str = Field(min_length=16, max_length=256)
    service_signature: str

    def signed_payload(self) -> dict[str, Any]:
        """Return the canonical service-auth payload."""
        return self.model_dump(mode="json", exclude={"service_signature"})


class ControllerCreateResponse(BaseModel):
    """Identity returned as soon as the isolated Worker has been created."""

    model_config = ConfigDict(extra="forbid")

    container_id: str
    status: str


class ControllerTaskRequest(BaseModel):
    """Authenticated operation on one deployment-owned sandbox task."""

    model_config = ConfigDict(extra="forbid")

    container_id: str
    deployment_id: str
    service_timestamp: float
    service_nonce: str = Field(min_length=16, max_length=256)
    service_signature: str

    def signed_payload(self) -> dict[str, Any]:
        """Return the canonical service-auth payload."""
        return self.model_dump(mode="json", exclude={"service_signature"})


class ControllerStatusResponse(BaseModel):
    """Non-blocking status projection for one Worker container."""

    model_config = ConfigDict(extra="forbid")

    container_id: str
    status: str
    exit_code: int | None = None
    timed_out: bool = False


class ControllerCollectResponse(BaseModel):
    """Bounded terminal result returned to the API process."""

    model_config = ConfigDict(extra="forbid")

    container_id: str
    exit_code: int
    result: SandboxTaskResultV1
    logs: str = ""


class ControllerStopRequest(BaseModel):
    """Authenticated stop request for a deployment-owned container."""

    model_config = ConfigDict(extra="forbid")

    container_id: str
    deployment_id: str
    timeout_seconds: int = Field(default=5, ge=1, le=60)
    service_timestamp: float
    service_nonce: str = Field(min_length=16, max_length=256)
    service_signature: str

    def signed_payload(self) -> dict[str, Any]:
        """Return the canonical service-auth payload."""
        return self.model_dump(mode="json", exclude={"service_signature"})


class ControllerActiveTask(BaseModel):
    """One run/task/fence identity still owned by the API."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    task_id: str
    fence_token: int = Field(ge=1)


class ControllerReconcileRequest(BaseModel):
    """Authenticated deployment-scoped orphan reconciliation request."""

    model_config = ConfigDict(extra="forbid")

    deployment_id: str
    active_tasks: list[ControllerActiveTask] = Field(default_factory=list)
    service_timestamp: float
    service_nonce: str = Field(min_length=16, max_length=256)
    service_signature: str

    def signed_payload(self) -> dict[str, Any]:
        """Return the canonical service-auth payload."""
        return self.model_dump(mode="json", exclude={"service_signature"})


def _tar_payload(
    name: str,
    data: bytes,
    *,
    uid: int,
    gid: int,
) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as bundle:
        info = tarfile.TarInfo(name=name)
        info.size = len(data)
        # The volume is mounted read-only into exactly one task container.  Use
        # a universally readable mode so Docker volume drivers that do not
        # preserve archive uid/gid ownership cannot make the payload unreadable
        # to the configured unprivileged Worker uid.
        info.mode = 0o444
        info.uid = uid
        info.gid = gid
        bundle.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


def _tmpfs_options(size: int, *, uid: int, gid: int) -> str:
    return (
        f"rw,noexec,nosuid,nodev,size={int(size)},"
        f"uid={uid},gid={gid},mode=0700"
    )


class DockerControllerRuntime:
    """Apply a validated Profile using a Docker SDK client."""

    def __init__(
        self,
        *,
        docker_client: Any,
        configurable: Configuration,
        gateway_container: str | None = None,
    ) -> None:
        """Initialize the Docker-owning runtime from administrator state."""
        self.client = docker_client
        self.configurable = configurable
        self.bundle = load_policy_bundle(configurable.sandbox_policy_path)
        self.keys = SandboxDerivedKeys.from_root(configurable.sandbox_root_signing_key or "")
        self.gateway_container = gateway_container or os.getenv(
            "SANDBOX_GATEWAY_CONTAINER", "sandbox-gateway"
        )
        self.nonces = NonceReplayCache()

    def _authorize(self, request: ControllerRunRequest) -> tuple[SandboxProfile, Any]:
        validate_timestamp(request.service_timestamp)
        if not verify_payload(
            request.signed_payload(), request.service_signature, self.keys.service_auth
        ):
            raise ValueError("sandbox_service_auth_invalid")
        claims = decode_task_token(request.task_token, self.keys.task_token)
        self.nonces.consume(
            claims.jti,
            request.service_nonce,
            expires_at=claims.expires_at,
        )
        payload = request.payload
        if (
            claims.run_id != payload.run_id
            or claims.task_id != payload.task_id
            or claims.fence_token != payload.fence_token
            or claims.profile_id != payload.profile_id
            or claims.policy_digest != payload.policy_digest
        ):
            raise ValueError("sandbox_task_token_claim_mismatch")
        expected_policy = policy_digest(self.bundle)
        signature_payload = {
            "deployment_id": self.bundle.deployment_id,
            "policy_digest": expected_policy,
        }
        if payload.policy_digest != expected_policy or not verify_payload(
            signature_payload,
            request.policy_signature,
            self.keys.policy_signature,
        ):
            raise ValueError("sandbox_policy_signature_invalid")
        profile = self.bundle.profiles.get(payload.profile_id)
        if profile is None or profile.provider not in {"docker", "bubblewrap"}:
            raise ValueError("sandbox_profile_not_runnable_by_controller")
        if request.runtime_digest != runtime_digest(profile):
            raise ValueError("sandbox_runtime_digest_mismatch")
        return profile, claims

    def _labels(self, request: ControllerRunRequest) -> dict[str, str]:
        payload = request.payload
        return {
            "com.insightforge.sandbox.deployment_id": self.bundle.deployment_id,
            "com.insightforge.sandbox.run_id": payload.run_id,
            "com.insightforge.sandbox.task_id": payload.task_id,
            "com.insightforge.sandbox.fence_token": str(payload.fence_token),
            "com.insightforge.sandbox.profile_id": payload.profile_id,
            "com.insightforge.sandbox.policy_digest": payload.policy_digest,
            "com.insightforge.sandbox.resource_kind": "worker",
            "com.insightforge.sandbox.created_at": str(time.time()),
        }

    def _authorize_service(
        self,
        request: ControllerTaskRequest | ControllerStopRequest | ControllerReconcileRequest,
        *,
        operation: str,
    ) -> None:
        """Validate one API-to-Controller control operation and its nonce."""
        validate_timestamp(request.service_timestamp)
        if not verify_payload(
            request.signed_payload(), request.service_signature, self.keys.service_auth
        ):
            raise ValueError("sandbox_service_auth_invalid")
        self.nonces.consume(
            f"controller:{operation}",
            request.service_nonce,
            expires_at=time.time() + 60,
        )
        if request.deployment_id != self.bundle.deployment_id:
            raise ValueError("sandbox_deployment_mismatch")

    @staticmethod
    def _container_labels(container: Any) -> dict[str, str]:
        return dict((container.attrs.get("Config") or {}).get("Labels") or {})

    def _owns_resource_labels(self, labels: dict[str, str], kind: str) -> bool:
        """Recheck deployment ownership after Docker returns filtered resources."""
        return (
            labels.get("com.insightforge.sandbox.deployment_id")
            == self.bundle.deployment_id
            and labels.get("com.insightforge.sandbox.resource_kind") == kind
        )

    def _owned_worker(self, container_id: str) -> Any:
        container = self.client.containers.get(container_id)
        container.reload()
        labels = self._container_labels(container)
        if (
            labels.get("com.insightforge.sandbox.deployment_id")
            != self.bundle.deployment_id
            or labels.get("com.insightforge.sandbox.resource_kind") != "worker"
        ):
            raise ValueError("sandbox_container_not_owned")
        return container

    def _profile_for_container(self, container: Any) -> SandboxProfile:
        labels = self._container_labels(container)
        profile_id = labels.get("com.insightforge.sandbox.profile_id", "")
        profile = self.bundle.profiles.get(profile_id)
        if profile is None:
            raise ValueError("sandbox_container_profile_unknown")
        return profile

    def _related_resources(self, labels: dict[str, str]) -> tuple[Any | None, Any | None]:
        network = None
        volume = None
        network_name = labels.get("com.insightforge.sandbox.network_name")
        volume_name = labels.get("com.insightforge.sandbox.input_volume_name")
        if network_name:
            with suppress(Exception):
                network = self.client.networks.get(network_name)
        if volume_name:
            with suppress(Exception):
                volume = self.client.volumes.get(volume_name)
        return network, volume

    def _cleanup_removed_worker(self, container: Any) -> None:
        """Remove a stopped Worker and its private network/input volume."""
        labels = self._container_labels(container)
        network, volume = self._related_resources(labels)
        gateway = None
        with suppress(Exception):
            gateway = self.client.containers.get(self.gateway_container)
        with suppress(Exception):
            container.remove(force=True)
        if network is not None and gateway is not None:
            with suppress(Exception):
                network.disconnect(gateway, force=True)
        if network is not None:
            with suppress(Exception):
                network.remove()
        if volume is not None:
            with suppress(Exception):
                volume.remove(force=True)

    def _terminate(
        self,
        container: Any,
        *,
        timeout_seconds: int,
        force_remove: bool = False,
        apply_retention: bool = True,
    ) -> None:
        """TERM/grace/KILL one owned Worker, then apply stopped retention."""
        profile = self._profile_for_container(container)
        container.reload()
        terminal = {"created", "exited", "dead", "removing"}
        if container.status not in terminal:
            try:
                container.stop(timeout=timeout_seconds)
            except Exception:
                pass
            container.reload()
            if container.status not in terminal:
                try:
                    container.kill(signal="SIGKILL")
                except Exception:
                    pass
                container.reload()
            if container.status not in terminal:
                raise RuntimeError("sandbox_container_termination_failed")
        if force_remove or (
            apply_retention and profile.runtime.retention == "remove"
        ):
            self._cleanup_removed_worker(container)

    def _existing_task_containers(self, request: ControllerRunRequest) -> list[Any]:
        payload = request.payload
        labels = [
            f"com.insightforge.sandbox.deployment_id={self.bundle.deployment_id}",
            "com.insightforge.sandbox.resource_kind=worker",
            f"com.insightforge.sandbox.run_id={payload.run_id}",
            f"com.insightforge.sandbox.task_id={payload.task_id}",
        ]
        return list(
            self.client.containers.list(all=True, filters={"label": labels})
        )

    @staticmethod
    def _existing_container_disposition(
        status: str,
        *,
        existing_fence: int,
        requested_fence: int,
    ) -> str:
        """Choose stale/reuse/replace without reviving stopped task epochs."""
        if existing_fence > requested_fence:
            return "stale"
        if existing_fence == requested_fence and status in {
            "created",
            "running",
            "restarting",
            "paused",
        }:
            return "reuse"
        return "replace"

    def admission_report(self) -> dict[str, Any]:
        """Return worst-case cgroup memory and current deployment occupancy."""
        maximum_tasks = max(
            1,
            min(
                self.configurable.max_concurrent_research_units,
                self.configurable.max_in_flight_tasks,
            ),
        )
        task_memory = max(
            profile.resources.memory_bytes for profile in self.bundle.profiles.values()
        )
        service_reserve = int(
            os.getenv("SANDBOX_SERVICE_MEMORY_RESERVE_BYTES", str(1_073_741_824))
        )
        available = int(self.client.info().get("MemTotal") or 0)
        required = task_memory * maximum_tasks + service_reserve
        filters = {
            "label": [
                f"com.insightforge.sandbox.deployment_id={self.bundle.deployment_id}",
                "com.insightforge.sandbox.resource_kind=worker",
            ],
            "status": ["created", "running", "restarting", "paused"],
        }
        active = len(self.client.containers.list(all=True, filters=filters))
        return {
            "ready": available >= required,
            "memory_available_bytes": available,
            "memory_required_bytes": required,
            "task_memory_limit_bytes": task_memory,
            "service_memory_reserve_bytes": service_reserve,
            "maximum_tasks": maximum_tasks,
            "active_tasks": active,
            "admit_new_task": available >= required and active < maximum_tasks,
        }

    def create(self, request: ControllerRunRequest) -> ControllerCreateResponse:
        """Create one task idempotently and return before the Worker starts."""
        profile, _claims = self._authorize(request)
        labels = self._labels(request)
        labels["com.insightforge.sandbox.deadline_at"] = str(
            time.time() + profile.resources.timeout_seconds
        )
        suffix = hashlib.sha256(
            f"{request.payload.run_id}:{request.payload.task_id}".encode()
        ).hexdigest()[:20]
        network_name = f"if-sbx-{suffix}"
        volume_name = f"if-sbx-input-{suffix}"
        labels["com.insightforge.sandbox.network_name"] = network_name
        labels["com.insightforge.sandbox.input_volume_name"] = volume_name

        for existing in self._existing_task_containers(request):
            existing.reload()
            existing_labels = self._container_labels(existing)
            existing_fence = int(
                existing_labels.get("com.insightforge.sandbox.fence_token", "0")
            )
            disposition = self._existing_container_disposition(
                existing.status,
                existing_fence=existing_fence,
                requested_fence=request.payload.fence_token,
            )
            if disposition == "stale":
                raise ValueError("stale_fence")
            if disposition == "reuse":
                return ControllerCreateResponse(
                    container_id=existing.id,
                    status=existing.status,
                )
            self._terminate(
                existing,
                timeout_seconds=profile.resources.stop_grace_seconds,
                force_remove=True,
            )

        admission = self.admission_report()
        if not admission["ready"]:
            raise RuntimeError("sandbox_admission_insufficient_memory")
        if not admission["admit_new_task"]:
            raise RuntimeError("sandbox_admission_capacity_exhausted")

        # A hard Controller crash can leave pre-Worker resources behind. Names
        # are deterministic, so clean only same-deployment orphans before reuse.
        try:
            orphan_network = self.client.networks.get(network_name)
        except Exception:
            orphan_network = None
        if orphan_network is not None:
            orphan_labels = dict(orphan_network.attrs.get("Labels") or {})
            if (
                orphan_labels.get("com.insightforge.sandbox.deployment_id")
                != self.bundle.deployment_id
            ):
                raise RuntimeError("sandbox_network_name_collision")
            for endpoint_id in list(
                (orphan_network.attrs.get("Containers") or {}).keys()
            ):
                with suppress(Exception):
                    orphan_network.disconnect(endpoint_id, force=True)
            orphan_network.remove()
        try:
            orphan_volume = self.client.volumes.get(volume_name)
        except Exception:
            orphan_volume = None
        if orphan_volume is not None:
            orphan_labels = dict(orphan_volume.attrs.get("Labels") or {})
            if (
                orphan_labels.get("com.insightforge.sandbox.deployment_id")
                != self.bundle.deployment_id
            ):
                raise RuntimeError("sandbox_volume_name_collision")
            orphan_volume.remove(force=True)

        network = None
        input_volume = None
        seed = None
        worker = None
        gateway = None
        try:
            network = self.client.networks.create(
                network_name,
                internal=True,
                check_duplicate=True,
                labels={**labels, "com.insightforge.sandbox.resource_kind": "network"},
            )
            input_volume = self.client.volumes.create(
                name=volume_name,
                labels={**labels, "com.insightforge.sandbox.resource_kind": "input"},
            )
            from docker.types import Mount

            seed = self.client.containers.create(
                image=profile.runtime.worker_image_digest,
                command=["python", "-c", "import time; time.sleep(60)"],
                detach=True,
                network_mode="none",
                mounts=[
                    Mount(
                        target="/workspace/input",
                        source=input_volume.name,
                        type="volume",
                        read_only=False,
                    )
                ],
                labels={**labels, "com.insightforge.sandbox.resource_kind": "seed"},
            )
            seed.start()
            seed.put_archive(
                "/workspace/input",
                _tar_payload(
                    "task_payload.json",
                    request.payload.model_dump_json().encode(),
                    uid=profile.runtime.uid,
                    gid=profile.runtime.gid,
                ),
            )
            seed.stop(timeout=1)
            seed.remove(force=True)
            seed = None

            gateway = self.client.containers.get(self.gateway_container)
            network.connect(gateway, aliases=["sandbox-gateway"])
            worker = self.client.containers.create(
                image=profile.runtime.worker_image_digest,
                command=["python", "-m", "open_deep_research.sandbox.worker"],
                detach=True,
                network=network.name,
                mounts=[
                    Mount(
                        target="/workspace/input",
                        source=input_volume.name,
                        type="volume",
                        read_only=True,
                    )
                ],
                tmpfs={
                    "/workspace/output": _tmpfs_options(
                        profile.resources.output_bytes,
                        uid=profile.runtime.uid,
                        gid=profile.runtime.gid,
                    ),
                    "/workspace/logs": _tmpfs_options(
                        profile.resources.log_bytes,
                        uid=profile.runtime.uid,
                        gid=profile.runtime.gid,
                    ),
                    "/workspace/artifacts": _tmpfs_options(
                        profile.resources.artifact_bytes,
                        uid=profile.runtime.uid,
                        gid=profile.runtime.gid,
                    ),
                    "/workspace/tmp": _tmpfs_options(
                        profile.resources.tmp_bytes,
                        uid=profile.runtime.uid,
                        gid=profile.runtime.gid,
                    ),
                    "/workspace/work": _tmpfs_options(
                        profile.resources.tmp_bytes,
                        uid=profile.runtime.uid,
                        gid=profile.runtime.gid,
                    ),
                },
                working_dir="/workspace",
                read_only=profile.runtime.read_only_rootfs,
                cap_drop=["ALL"],
                security_opt=["no-new-privileges:true"],
                mem_limit=profile.resources.memory_bytes,
                nano_cpus=int(profile.resources.cpu_cores * 1_000_000_000),
                pids_limit=profile.resources.pids,
                user=f"{profile.runtime.uid}:{profile.runtime.gid}",
                labels=labels,
                environment={
                    "PYTHONUNBUFFERED": "1",
                    "GET_API_KEYS_FROM_CONFIG": "false",
                    "SANDBOX_TASK_PAYLOAD_PATH": "/workspace/input/task_payload.json",
                    "SANDBOX_RESULT_PATH": "/workspace/output/result.json",
                    "SANDBOX_LOG_PATH": "/workspace/logs/worker.log",
                    "SANDBOX_GATEWAY_URL": "http://sandbox-gateway:8081",
                    "SANDBOX_TASK_TOKEN": request.task_token,
                    "SANDBOX_PROFILE_ID": request.payload.profile_id,
                    "SANDBOX_NETWORK_POLICY_MODE": profile.network.mode,
                    "INSIGHTFORGE_TASK_WORKSPACE_ROOT": "/workspace/work",
                    "TMPDIR": "/workspace/tmp",
                    "TMP": "/workspace/tmp",
                    "TEMP": "/workspace/tmp",
                },
            )
            return ControllerCreateResponse(
                container_id=worker.id,
                status="created",
            )
        except Exception:
            if worker is not None:
                with suppress(Exception):
                    worker.remove(force=True)
            if seed is not None:
                with suppress(Exception):
                    seed.remove(force=True)
            if network is not None and gateway is not None:
                with suppress(Exception):
                    network.disconnect(gateway, force=True)
            if network is not None:
                with suppress(Exception):
                    network.remove()
            if input_volume is not None:
                with suppress(Exception):
                    input_volume.remove(force=True)
            raise

    def start(self, request: ControllerTaskRequest) -> ControllerStatusResponse:
        """Start a created Worker; repeated starts are idempotent."""
        self._authorize_service(request, operation="start")
        container = self._owned_worker(request.container_id)
        if container.status == "created":
            container.start()
            container.reload()
        return self._status(container)

    def _status(self, container: Any) -> ControllerStatusResponse:
        container.reload()
        labels = self._container_labels(container)
        state = (container.attrs.get("State") or {}) if hasattr(container, "attrs") else {}
        projected_status = container.status
        exit_code = state.get("ExitCode") if container.status in {"exited", "dead"} else None
        if container.status == "running":
            try:
                ready_code = self._ready_exit_code(container)
            except (TimeoutError, SandboxWorkerProtocolError):
                profile = self._profile_for_container(container)
                self._terminate(
                    container,
                    timeout_seconds=profile.resources.stop_grace_seconds,
                    apply_retention=False,
                )
                with suppress(Exception):
                    container.reload()
                return ControllerStatusResponse(
                    container_id=container.id,
                    status=(
                        container.status
                        if container.status in {"exited", "dead", "removing"}
                        else "dead"
                    ),
                    exit_code=1,
                    timed_out=True,
                )
            if ready_code is not None:
                return ControllerStatusResponse(
                    container_id=container.id,
                    status="result_ready",
                    exit_code=ready_code,
                    timed_out=False,
                )
        profile = self._profile_for_container(container)
        deadline_at = self._deadline_at(container, labels, profile)
        timed_out = bool(deadline_at and time.time() >= deadline_at)
        if timed_out and container.status not in {"exited", "dead", "removing"}:
            self._terminate(
                container,
                timeout_seconds=profile.resources.stop_grace_seconds,
                apply_retention=False,
            )
            with suppress(Exception):
                container.reload()
        state = (container.attrs.get("State") or {}) if hasattr(container, "attrs") else {}
        projected_status = container.status
        exit_code = state.get("ExitCode") if container.status in {"exited", "dead"} else None
        return ControllerStatusResponse(
            container_id=container.id,
            status=projected_status,
            exit_code=int(exit_code) if exit_code is not None else None,
            timed_out=timed_out,
        )

    @staticmethod
    def _deadline_at(
        container: Any,
        labels: dict[str, str],
        profile: SandboxProfile,
    ) -> float:
        """Derive the task deadline from Docker's persisted start timestamp."""
        state = (container.attrs.get("State") or {}) if hasattr(container, "attrs") else {}
        started_at = str(state.get("StartedAt") or "")
        if started_at and not started_at.startswith("0001-01-01"):
            try:
                started = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
                return started.timestamp() + profile.resources.timeout_seconds
            except ValueError:
                pass
        return float(
            labels.get("com.insightforge.sandbox.deadline_at", "0") or 0
        )

    @staticmethod
    def _ready_exit_code(container: Any) -> int | None:
        """Read the bounded Worker sentinel while its output tmpfs is mounted."""
        try:
            raw = DockerControllerRuntime._read_tmpfs_file(
                container,
                WORKER_EXIT_CODE_PATH,
                max_bytes=16,
            )
            value = int(raw.decode("ascii").strip())
            if value < 0 or value > 255:
                raise ValueError("sandbox_worker_exit_code_invalid")
            return value
        except FileNotFoundError:
            return None
        except (TimeoutError, SandboxWorkerProtocolError):
            raise
        except Exception:
            return None

    @staticmethod
    def _bounded_exec(
        container: Any,
        command: list[str],
        *,
        timeout_seconds: float,
    ) -> Any:
        """Run one fixed Docker exec without letting a blocked stream stall control."""
        outcomes: queue.Queue[tuple[bool, Any]] = queue.Queue(maxsize=1)

        def invoke() -> None:
            try:
                outcomes.put(
                    (
                        True,
                        container.exec_run(command, stdout=True, stderr=False),
                    )
                )
            except BaseException as exc:  # noqa: BLE001 - transported to caller
                outcomes.put((False, exc))

        worker = threading.Thread(target=invoke, daemon=True)
        worker.start()
        worker.join(max(0.01, timeout_seconds))
        if worker.is_alive():
            # docker-py cannot cancel an exec blocked inside the daemon. The
            # daemon thread may therefore survive until Docker recovers or the
            # Controller exits; watchdog probes are independently bounded, so
            # the accepted leak rate is at most one thread per timed-out probe.
            raise TimeoutError("sandbox_container_exec_timeout")
        succeeded, value = outcomes.get_nowait()
        if not succeeded:
            raise value
        return value

    @staticmethod
    def _read_tmpfs_file(
        container: Any,
        path: str,
        *,
        max_bytes: int,
        timeout_seconds: float = CONTROLLER_EXEC_TIMEOUT_SECONDS,
    ) -> bytes:
        """Read one fixed path through a bounded, shell-free Docker exec."""
        metadata = DockerControllerRuntime._bounded_exec(
            container,
            ["/usr/bin/stat", "-c", "%f:%s", path],
            timeout_seconds=timeout_seconds,
        )
        if int(metadata.exit_code) != 0:
            raise FileNotFoundError(path)
        raw_metadata = (
            metadata.output
            if isinstance(metadata.output, bytes)
            else bytes(metadata.output)
        )
        try:
            raw_mode, raw_size = raw_metadata.decode("ascii").strip().split(":", 1)
            mode = int(raw_mode, 16)
            size = int(raw_size)
        except (UnicodeDecodeError, ValueError) as exc:
            raise SandboxWorkerProtocolError(
                "sandbox_tmpfs_stat_invalid"
            ) from exc
        if not stat.S_ISREG(mode):
            raise SandboxWorkerProtocolError("sandbox_tmpfs_file_not_regular")
        if size < 0 or size > max_bytes:
            raise SandboxWorkerProtocolError("sandbox_tmpfs_file_exceeds_limit")
        result = DockerControllerRuntime._bounded_exec(
            container,
            ["/usr/bin/head", "-c", str(max_bytes + 1), path],
            timeout_seconds=timeout_seconds,
        )
        if int(result.exit_code) != 0:
            raise FileNotFoundError(path)
        output = result.output if isinstance(result.output, bytes) else bytes(result.output)
        if len(output) > max_bytes:
            raise ValueError("sandbox_tmpfs_file_exceeds_limit")
        return output

    def status(self, request: ControllerTaskRequest) -> ControllerStatusResponse:
        """Return status without waiting for task completion."""
        self._authorize_service(request, operation="status")
        return self._status(self._owned_worker(request.container_id))

    def collect(self, request: ControllerTaskRequest) -> ControllerCollectResponse:
        """Safely collect a terminal result and then apply retention."""
        self._authorize_service(request, operation="collect")
        container = self._owned_worker(request.container_id)
        profile = self._profile_for_container(container)
        status = self._status(container)
        if status.status not in {"result_ready", "exited", "dead"}:
            raise ValueError("sandbox_task_not_terminal")
        exit_code = int(status.exit_code if status.exit_code is not None else 1)
        logs = container.logs(stdout=True, stderr=True, tail=10_000)
        if isinstance(logs, bytes):
            logs = logs[-profile.resources.log_bytes :].decode(
                "utf-8", errors="replace"
            )
        try:
            result_bytes = self._read_tmpfs_file(
                container,
                "/workspace/output/result.json",
                max_bytes=profile.resources.output_bytes,
            )
            result = SandboxTaskResultV1.model_validate_json(result_bytes)
        except Exception as exc:  # noqa: BLE001 - produce a bounded terminal failure
            labels = self._container_labels(container)
            result = SandboxTaskResultV1(
                task_id=labels.get("com.insightforge.sandbox.task_id", "unknown"),
                status="failed",
                error=(
                    "sandbox task timed out"
                    if status.timed_out
                    else f"sandbox result unavailable: {str(exc)[:500]}"
                ),
            )
        response = ControllerCollectResponse(
            container_id=container.id,
            exit_code=exit_code,
            result=result,
            logs=str(logs),
        )
        return response

    def collect_archive(self, request: ControllerTaskRequest) -> bytes:
        """Collect output/log/artifact tmpfs trees into a canonical safe tar."""
        self._authorize_service(request, operation="archive")
        container = self._owned_worker(request.container_id)
        profile = self._profile_for_container(container)
        status = self._status(container)
        if status.status not in {"result_ready", "exited", "dead"}:
            raise ValueError("sandbox_task_not_terminal")
        archives: list[tuple[str, bytes]] = []
        for prefix in ("output", "logs", "artifacts"):
            exported = container.exec_run(
                ["/bin/tar", "-C", "/workspace", "-cf", "-", prefix],
                stdout=True,
                stderr=False,
            )
            if int(exported.exit_code) != 0:
                raise RuntimeError(f"sandbox_tmpfs_export_failed:{prefix}")
            archive = (
                exported.output
                if isinstance(exported.output, bytes)
                else bytes(exported.output)
            )
            archives.append((prefix, archive))
        return repack_docker_archives(
            archives,
            limits=ArchiveLimits(
                max_bytes=(
                    profile.resources.output_bytes
                    + profile.resources.log_bytes
                    + profile.resources.artifact_bytes
                ),
                max_files=profile.resources.max_files,
            ),
        )

    def stop(self, request: ControllerStopRequest) -> None:
        """Stop only a container carrying this deployment's ownership label."""
        self._authorize_service(request, operation="stop")
        try:
            container = self._owned_worker(request.container_id)
        except Exception as exc:
            if exc.__class__.__name__ == "NotFound":
                return
            raise
        self._terminate(container, timeout_seconds=request.timeout_seconds)

    def reconcile(self, request: ControllerReconcileRequest) -> list[str]:
        """Stop only deployment-owned Workers absent from the API ownership set."""
        self._authorize_service(request, operation="reconcile")
        active = {
            (item.run_id, item.task_id): item.fence_token for item in request.active_tasks
        }
        filters = {
            "label": [
                f"com.insightforge.sandbox.deployment_id={self.bundle.deployment_id}",
                "com.insightforge.sandbox.resource_kind=worker",
            ]
        }
        stopped: list[str] = []
        for container in self.client.containers.list(all=True, filters=filters):
            container.reload()
            labels = self._container_labels(container)
            if not self._owns_resource_labels(labels, "worker"):
                continue
            identity = (
                labels.get("com.insightforge.sandbox.run_id", ""),
                labels.get("com.insightforge.sandbox.task_id", ""),
            )
            fence = int(
                labels.get("com.insightforge.sandbox.fence_token", "0") or 0
            )
            deadline = float(
                labels.get("com.insightforge.sandbox.deadline_at", "0") or 0
            )
            if active.get(identity) != fence or (deadline and time.time() >= deadline):
                profile = self._profile_for_container(container)
                self._terminate(
                    container,
                    timeout_seconds=profile.resources.stop_grace_seconds,
                )
                stopped.append(container.id)
        active_identities = set(active.items())
        seed_filters = {
            "label": [
                f"com.insightforge.sandbox.deployment_id={self.bundle.deployment_id}",
                "com.insightforge.sandbox.resource_kind=seed",
            ]
        }
        for seed in self.client.containers.list(all=True, filters=seed_filters):
            labels = self._container_labels(seed)
            if not self._owns_resource_labels(labels, "seed"):
                continue
            identity = (
                (
                    labels.get("com.insightforge.sandbox.run_id", ""),
                    labels.get("com.insightforge.sandbox.task_id", ""),
                ),
                int(labels.get("com.insightforge.sandbox.fence_token", "0") or 0),
            )
            if identity not in active_identities:
                with suppress(Exception):
                    seed.remove(force=True)
        for network in self.client.networks.list(
            filters={
                "label": [
                    f"com.insightforge.sandbox.deployment_id={self.bundle.deployment_id}",
                    "com.insightforge.sandbox.resource_kind=network",
                ]
            }
        ):
            labels = dict(network.attrs.get("Labels") or {})
            if not self._owns_resource_labels(labels, "network"):
                continue
            identity = (
                (
                    labels.get("com.insightforge.sandbox.run_id", ""),
                    labels.get("com.insightforge.sandbox.task_id", ""),
                ),
                int(labels.get("com.insightforge.sandbox.fence_token", "0") or 0),
            )
            if identity not in active_identities:
                for endpoint_id in list(
                    (network.attrs.get("Containers") or {}).keys()
                ):
                    with suppress(Exception):
                        network.disconnect(endpoint_id, force=True)
                with suppress(Exception):
                    network.remove()
        for volume in self.client.volumes.list(
            filters={
                "label": [
                    f"com.insightforge.sandbox.deployment_id={self.bundle.deployment_id}",
                    "com.insightforge.sandbox.resource_kind=input",
                ]
            }
        ):
            labels = dict(volume.attrs.get("Labels") or {})
            if not self._owns_resource_labels(labels, "input"):
                continue
            identity = (
                (
                    labels.get("com.insightforge.sandbox.run_id", ""),
                    labels.get("com.insightforge.sandbox.task_id", ""),
                ),
                int(labels.get("com.insightforge.sandbox.fence_token", "0") or 0),
            )
            if identity not in active_identities:
                with suppress(Exception):
                    volume.remove(force=True)
        return stopped

    def sweep_expired(self) -> list[str]:
        """Stop expired deployment-owned Workers even when the API is unavailable."""
        filters = {
            "label": [
                f"com.insightforge.sandbox.deployment_id={self.bundle.deployment_id}",
                "com.insightforge.sandbox.resource_kind=worker",
            ]
        }
        stopped: list[str] = []
        stopped_lock = threading.Lock()

        def inspect_one(container: Any) -> None:
            try:
                status = self._status(container)
                if not status.timed_out:
                    return
                with stopped_lock:
                    stopped.append(container.id)
                labels = self._container_labels(container)
                profile = self._profile_for_container(container)
                deadline = self._deadline_at(container, labels, profile)
                if deadline and time.time() >= deadline + 60:
                    self._terminate(
                        container,
                        timeout_seconds=profile.resources.stop_grace_seconds,
                    )
            except Exception:
                return

        workers = [
            threading.Thread(target=inspect_one, args=(container,), daemon=True)
            for container in self.client.containers.list(all=True, filters=filters)
        ]
        for worker in workers:
            worker.start()
        deadline = time.monotonic() + WATCHDOG_BATCH_TIMEOUT_SECONDS
        for worker in workers:
            worker.join(max(0.0, deadline - time.monotonic()))
        return stopped


def create_controller_app(runtime: DockerControllerRuntime) -> FastAPI:
    """Create the controller's internal-only ASGI application."""
    app = FastAPI(title="InsightForge Sandbox Controller", docs_url=None, redoc_url=None)

    async def sweep_expired_workers() -> None:
        """Enforce task deadlines even while the API process is unavailable."""
        while True:
            try:
                await asyncio.to_thread(runtime.sweep_expired)
            except Exception:
                # Readiness and reconcile surface Docker failures; the watchdog
                # must stay alive so a transient daemon restart cannot disable it.
                pass
            await asyncio.sleep(1)

    @app.on_event("startup")
    async def start_watchdog() -> None:
        app.state.sandbox_watchdog = asyncio.create_task(sweep_expired_workers())

    @app.on_event("shutdown")
    async def stop_watchdog() -> None:
        task = getattr(app.state, "sandbox_watchdog", None)
        if task is not None:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task

    @app.get("/healthz")
    async def healthz() -> dict[str, Any]:
        images = {}
        for profile_id, profile in runtime.bundle.profiles.items():
            try:
                image = await asyncio.to_thread(
                    runtime.client.images.get, profile.runtime.worker_image_digest
                )
                images[profile_id] = {"ready": True, "image_id": image.id}
            except Exception as exc:  # noqa: BLE001 - readiness must report exact failure
                images[profile_id] = {"ready": False, "error": str(exc)[:500]}
        try:
            gateway = await asyncio.to_thread(
                runtime.client.containers.get, runtime.gateway_container
            )
            gateway_ready = gateway.status == "running"
        except Exception:
            gateway_ready = False
        try:
            admission = await asyncio.to_thread(runtime.admission_report)
        except Exception as exc:  # noqa: BLE001 - readiness reports Docker failure
            admission = {"ready": False, "error": str(exc)[:500]}
        ready = (
            all(item["ready"] for item in images.values())
            and gateway_ready
            and bool(admission.get("ready"))
        )
        return {
            "status": "ok" if ready else "not_ready",
            "deployment_id": runtime.bundle.deployment_id,
            "policy_digest": policy_digest(runtime.bundle),
            "images": images,
            "gateway_ready": gateway_ready,
            "admission": admission,
        }

    @app.post("/v1/tasks/create", response_model=ControllerCreateResponse)
    async def create_task(request: ControllerRunRequest) -> ControllerCreateResponse:
        try:
            return await asyncio.to_thread(runtime.create, request)
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/v1/tasks/start", response_model=ControllerStatusResponse)
    async def start_task(request: ControllerTaskRequest) -> ControllerStatusResponse:
        try:
            return await asyncio.to_thread(runtime.start, request)
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/v1/tasks/status", response_model=ControllerStatusResponse)
    async def task_status(request: ControllerTaskRequest) -> ControllerStatusResponse:
        try:
            return await asyncio.to_thread(runtime.status, request)
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/v1/tasks/collect", response_model=ControllerCollectResponse)
    async def collect_task(request: ControllerTaskRequest) -> ControllerCollectResponse:
        try:
            return await asyncio.to_thread(runtime.collect, request)
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/v1/tasks/archive")
    async def collect_task_archive(request: ControllerTaskRequest) -> Response:
        try:
            archive = await asyncio.to_thread(runtime.collect_archive, request)
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return Response(content=archive, media_type="application/x-tar")

    @app.post("/v1/tasks/stop")
    async def stop_task(request: ControllerStopRequest) -> dict[str, str]:
        try:
            await asyncio.to_thread(runtime.stop, request)
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {"status": "stopped"}

    @app.post("/v1/tasks/reconcile")
    async def reconcile_tasks(
        request: ControllerReconcileRequest,
    ) -> dict[str, Any]:
        try:
            stopped = await asyncio.to_thread(runtime.reconcile, request)
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {"status": "reconciled", "stopped_container_ids": stopped}

    return app


def main() -> None:
    """Run the controller on its administrator-configured Unix socket."""
    import docker
    import uvicorn

    configurable = Configuration.from_runnable_config(None)
    if not configurable.sandbox_enabled:
        raise SystemExit("SANDBOX_ENABLED must be true for sandbox-controller")
    runtime = DockerControllerRuntime(
        docker_client=docker.from_env(), configurable=configurable
    )
    uvicorn.run(
        create_controller_app(runtime),
        uds=configurable.sandbox_controller_socket,
        log_level="info",
    )


if __name__ == "__main__":
    main()
