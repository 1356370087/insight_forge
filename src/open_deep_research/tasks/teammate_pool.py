"""Run-scoped persistent in-process teammate pool driven by file mailboxes."""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Coroutine

import portalocker
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from open_deep_research.configuration import Configuration
from open_deep_research.tasks.coordination import get_mailbox, publish_task_update
from open_deep_research.tasks.events import EventType
from open_deep_research.tasks.lease import LeaderLeaseManager
from open_deep_research.tasks.mailbox import (
    MailboxMessage,
    atomic_write_json,
    read_json_file,
)
from open_deep_research.tasks.registry import TaskRecord, TaskRegistry, TaskStatus
from open_deep_research.tasks.state import get_task_state_store

ExecuteResearchFn = Callable[
    [dict[str, Any], RunnableConfig],
    Coroutine[Any, Any, dict[str, Any]],
]


class TeammateDescriptor(BaseModel):
    """Durable public state for one persistent teammate."""

    teammate_id: str
    status: str = "idle"
    current_task_id: str | None = None
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)
    tasks_completed: int = 0


class TeamFile(BaseModel):
    """Run-scoped persistent teammate directory."""

    schema_version: int = 1
    run_id: str
    lead_agent_id: str = "lead"
    next_teammate_number: int = 1
    members: list[TeammateDescriptor] = Field(default_factory=list)


@dataclass
class _RuntimeTeammate:
    descriptor: TeammateDescriptor
    loop_task: asyncio.Task[None]
    active_task: asyncio.Task[None] | None = None


class TeammatePool:
    """Create teammates on demand and reuse them across clean-context tasks."""

    def __init__(
        self,
        *,
        config: RunnableConfig,
        registry: TaskRegistry,
        execute_research: ExecuteResearchFn,
    ) -> None:
        """Initialize a pool owned by one run and one Lead process."""
        self.config = config
        self.configurable = Configuration.from_runnable_config(config)
        self.run_id = str(config.get("metadata", {}).get("run_id", "default"))
        self.registry = registry
        self.execute_research = execute_research
        self.mailbox = get_mailbox(self.configurable, self.run_id)
        self.store = get_task_state_store(self.configurable)
        metadata = config.get("metadata", {})
        inherited_owner = metadata.get("run_lease_owner_id")
        inherited_token = metadata.get("run_fence_token")
        self.lease = LeaderLeaseManager(
            runs_dir=self.configurable.runs_dir,
            run_id=self.run_id,
            lease_seconds=self.configurable.leader_lease_seconds,
            lock_timeout=self.configurable.mailbox_lock_timeout_seconds,
            owner_id=str(inherited_owner) if inherited_owner else None,
        )
        self._owns_lease_lifecycle = inherited_token is None
        self.root = Path(self.configurable.runs_dir).resolve() / self.run_id / "coordination"
        self.team_path = self.root / "team.json"
        self.team_lock_path = self.root / "team.lock"
        self.consumer_prefix = f"pool-{os.getpid()}"
        self._runtimes: dict[str, _RuntimeTeammate] = {}
        self._dispatch_lock = asyncio.Lock()
        self._heartbeat_task: asyncio.Task[None] | None = None
        self.fence_token: int | None = (
            int(inherited_token) if inherited_token is not None else None
        )
        if self.fence_token is not None:
            self.lease.fence_token = self.fence_token
        self._started = False
        self._stopping = False

    def _load_team_locked(self) -> TeamFile:
        if not self.team_path.exists():
            return TeamFile(run_id=self.run_id)
        return TeamFile.model_validate(read_json_file(self.team_path))

    def _update_team_sync(self, operation):
        self.root.mkdir(parents=True, exist_ok=True)
        with portalocker.Lock(
            str(self.team_lock_path), mode="a+b", timeout=self.configurable.mailbox_lock_timeout_seconds
        ):
            team = self._load_team_locked()
            result = operation(team)
            atomic_write_json(self.team_path, team.model_dump(mode="json"))
            return result

    async def _write_descriptor(self, descriptor: TeammateDescriptor) -> None:
        def operation(team: TeamFile) -> None:
            for index, member in enumerate(team.members):
                if member.teammate_id == descriptor.teammate_id:
                    team.members[index] = descriptor
                    break
            else:
                team.members.append(descriptor)

        await asyncio.to_thread(self._update_team_sync, operation)

    async def start(self) -> None:
        """Acquire run ownership and start the lease heartbeat."""
        if self._started:
            return
        if self.fence_token is None:
            lease = await self.lease.acquire()
            self.fence_token = lease.fence_token
            self.config.setdefault("metadata", {})["run_fence_token"] = lease.fence_token
        elif not await self.lease.is_owner(expected_fence_token=self.fence_token):
            raise RuntimeError(f"Lost Lead lease for run {self.run_id}")
        self._started = True
        if self._owns_lease_lifecycle:
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        await self._restore_teammates()
        await self._restore_pending_tasks()

    async def _restore_teammates(self) -> None:
        """Restart durable teammate inbox consumers after a Lead takeover."""
        team = await asyncio.to_thread(self._load_team_locked)
        for descriptor in team.members:
            if descriptor.teammate_id in self._runtimes:
                continue
            descriptor.status = "idle"
            descriptor.current_task_id = None
            descriptor.updated_at = time.time()
            await self._write_descriptor(descriptor)
            loop_task = asyncio.create_task(self._worker_loop(descriptor))
            self._runtimes[descriptor.teammate_id] = _RuntimeTeammate(
                descriptor=descriptor,
                loop_task=loop_task,
            )

    async def _heartbeat_loop(self) -> None:
        while not self._stopping:
            await asyncio.sleep(self.configurable.leader_heartbeat_seconds)
            try:
                await self.lease.renew(expected_fence_token=self.fence_token)
            except Exception as exc:  # noqa: BLE001 - lease loss is terminal for this owner
                self._stopping = True
                for record in self.registry.list(run_id=self.run_id):
                    if record.status in {
                        TaskStatus.PENDING,
                        TaskStatus.RUNNING,
                        TaskStatus.WAITING_FOR_CONFIRMATION,
                    }:
                        record.cancelled.set()
                        self.registry.update_status(
                            record.task_id,
                            TaskStatus.FAILED,
                            error_message=f"Lead lease lost: {exc}",
                        )
                        try:
                            snapshot = await self.store.update_from_record(
                                record,
                                fence_token=self.fence_token or 0,
                            )
                            await publish_task_update(
                                self.configurable,
                                snapshot,
                                EventType.TASK_FAILED,
                            )
                        except Exception:
                            pass
                        if (
                            self.configurable.sandbox_enabled
                            and record.container_id
                        ):
                            try:
                                from open_deep_research.sandbox.controller_client import (
                                    SandboxControllerClient,
                                )
                                from open_deep_research.sandbox.schema import (
                                    load_policy_bundle,
                                )

                                bundle = load_policy_bundle(
                                    self.configurable.sandbox_policy_path
                                )
                                await SandboxControllerClient(
                                    self.configurable,
                                    bundle,
                                ).stop_task(record.container_id)
                            except Exception as stop_exc:
                                record.error_message = (
                                    f"Lead lease lost; sandbox stop failed: {stop_exc}"
                                )[:1000]
                for runtime in self._runtimes.values():
                    runtime.loop_task.cancel()
                self._started = False
                _POOLS.pop(
                    (str(Path(self.configurable.runs_dir).resolve()), self.run_id),
                    None,
                )
                return

    async def _new_teammate(self) -> _RuntimeTeammate:
        def allocate(team: TeamFile) -> TeammateDescriptor:
            teammate = TeammateDescriptor(teammate_id=f"teammate-{team.next_teammate_number}")
            team.next_teammate_number += 1
            team.members.append(teammate)
            return teammate

        descriptor = await asyncio.to_thread(self._update_team_sync, allocate)
        loop_task = asyncio.create_task(self._worker_loop(descriptor))
        runtime = _RuntimeTeammate(descriptor=descriptor, loop_task=loop_task)
        self._runtimes[descriptor.teammate_id] = runtime
        return runtime

    async def submit(self, record: TaskRecord) -> str | None:
        """Persist a pending task and dispatch it to an idle/new teammate."""
        await self.start()
        await self.store.update_from_record(
            record,
            fence_token=self.fence_token or 0,
        )
        await self._dispatch_pending()
        return record.assigned_teammate_id

    async def _restore_pending_tasks(self) -> None:
        snapshots = await self.store.list(run_id=self.run_id)
        for snapshot in snapshots:
            if self.registry.get(snapshot.task_id) is None and snapshot.status in {
                TaskStatus.PENDING,
                TaskStatus.RUNNING,
                TaskStatus.WAITING_FOR_CONFIRMATION,
            }:
                self.registry.restore(TaskRecord(
                    task_id=snapshot.task_id,
                    research_topic=snapshot.research_topic,
                    run_id=snapshot.run_id,
                    user_id=snapshot.user_id,
                    status=TaskStatus.PENDING,
                    memory_context=snapshot.memory_context,
                    requirement_ids=list(snapshot.requirement_ids),
                    coverage_contract=dict(snapshot.coverage_contract),
                    research_risk_profile=dict(
                        snapshot.research_risk_profile
                    ),
                    assignment_attempt=max(
                        snapshot.assignment_attempt,
                        1 if snapshot.assigned_teammate_id else 0,
                    ),
                    pending_update_instructions=list(snapshot.pending_update_instructions),
                    trace_parent_span_id=snapshot.trace_parent_span_id,
                    langfuse_parent_span_id=snapshot.langfuse_parent_span_id,
                    sandbox_enabled=bool(snapshot.sandbox.get("enabled")),
                    workspace_path=snapshot.sandbox.get("workspace_path"),
                    container_id=snapshot.sandbox.get("container_id"),
                    sandbox_network_mode=snapshot.sandbox.get("network_mode"),
                    output_archive_path=snapshot.sandbox.get(
                        "output_archive_path"
                    ),
                    last_sandbox_event=snapshot.sandbox.get("last_event"),
                ))
        await self._dispatch_pending()

    async def _dispatch_pending(self) -> None:
        async with self._dispatch_lock:
            if self._stopping:
                return
            pending = [
                record
                for record in self.registry.list(status_filter=TaskStatus.PENDING, run_id=self.run_id)
                if record.assigned_teammate_id is None
            ]
            for record in pending:
                idle = next(
                    (runtime for runtime in self._runtimes.values() if runtime.descriptor.status == "idle"),
                    None,
                )
                if idle is None and len(self._runtimes) < self.configurable.max_persistent_teammates:
                    idle = await self._new_teammate()
                if idle is None:
                    return
                record.assigned_teammate_id = idle.descriptor.teammate_id
                record.assignment_attempt += 1
                idle.descriptor.status = "reserved"
                idle.descriptor.current_task_id = record.task_id
                idle.descriptor.updated_at = time.time()
                await self._write_descriptor(idle.descriptor)
                await self.store.update_from_record(
                    record,
                    fence_token=self.fence_token or 0,
                )
                await self.mailbox.send(
                    recipient=idle.descriptor.teammate_id,
                    sender="lead",
                    message_type="task_assignment",
                    priority=20,
                    dedupe_key=f"{record.task_id}:assignment:{record.assignment_attempt}",
                    payload={"task_id": record.task_id},
                )

    async def send_control(
        self,
        *,
        task_id: str,
        message_type: str,
        payload: dict[str, Any],
        priority: int,
    ) -> None:
        """Send a durable control message to the task's assigned teammate."""
        record = self.registry.get(task_id)
        if (
            record is None
            or record.run_id != self.run_id
            or not record.assigned_teammate_id
        ):
            raise ValueError(f"Task {task_id} has no assigned teammate")
        await self.mailbox.send(
            recipient=record.assigned_teammate_id,
            sender="lead",
            message_type=message_type,
            payload={"task_id": task_id, **payload},
            priority=priority,
            dedupe_key=f"{task_id}:{message_type}:{payload.get('request_id', time.time_ns())}",
        )

    async def _handle_control(self, descriptor: TeammateDescriptor, message: MailboxMessage) -> bool:
        task_id = str(message.payload.get("task_id", ""))
        record = self.registry.get(task_id) if task_id else None
        if record is not None and record.run_id != self.run_id:
            record = None
        if message.type == "task_update" and record is not None:
            instruction = str(message.payload["instruction"])
            if instruction not in record.pending_update_instructions:
                record.pending_update_instructions.append(instruction)
                await self.store.update_from_record(
                    record,
                    fence_token=self.fence_token or 0,
                )
            await record.control_queue.put({"type": "update", "instruction": instruction})
        elif message.type == "cancel_request" and record is not None:
            record.cancelled.set()
        elif message.type == "shutdown_request":
            if record is not None:
                record.cancelled.set()
            await self.mailbox.send(
                recipient="lead",
                sender=descriptor.teammate_id,
                message_type="shutdown_ack",
                priority=0,
                dedupe_key=f"{descriptor.teammate_id}:shutdown_ack",
                payload={"teammate_id": descriptor.teammate_id},
            )
            return True
        return False

    async def _worker_loop(self, descriptor: TeammateDescriptor) -> None:
        consumer_id = f"{self.consumer_prefix}-{descriptor.teammate_id}"
        active: asyncio.Task[None] | None = None
        while not self._stopping:
            messages = await self.mailbox.claim(agent_id=descriptor.teammate_id, consumer_id=consumer_id)
            ack_ids: list[str] = []
            for message in messages:
                if message.type == "task_assignment" and active is None:
                    task_id = str(message.payload["task_id"])
                    record = self.registry.get(task_id)
                    if (
                        record is not None
                        and record.run_id == self.run_id
                        and record.status == TaskStatus.PENDING
                    ):
                        descriptor.status = "busy"
                        descriptor.current_task_id = task_id
                        descriptor.updated_at = time.time()
                        await self._write_descriptor(descriptor)
                        active = asyncio.create_task(self._execute_task(record))
                        runtime = self._runtimes.get(descriptor.teammate_id)
                        if runtime is not None:
                            runtime.active_task = active
                elif await self._handle_control(descriptor, message):
                    ack_ids.append(message.message_id)
                    await self.mailbox.ack(
                        agent_id=descriptor.teammate_id,
                        consumer_id=consumer_id,
                        message_ids=ack_ids,
                    )
                    if active is not None and not active.done():
                        active.cancel()
                        await asyncio.gather(active, return_exceptions=True)
                    return
                ack_ids.append(message.message_id)
            if ack_ids:
                await self.mailbox.ack(
                    agent_id=descriptor.teammate_id,
                    consumer_id=consumer_id,
                    message_ids=ack_ids,
                )
            if active is not None and active.done():
                try:
                    await active
                except asyncio.CancelledError:
                    pass
                except Exception:
                    pass
                active = None
                runtime = self._runtimes.get(descriptor.teammate_id)
                if runtime is not None:
                    runtime.active_task = None
                descriptor.status = "idle"
                descriptor.current_task_id = None
                descriptor.tasks_completed += 1
                descriptor.updated_at = time.time()
                await self._write_descriptor(descriptor)
                await self.mailbox.send(
                    recipient="lead",
                    sender=descriptor.teammate_id,
                    message_type="idle_notification",
                    priority=50,
                    dedupe_key=f"{descriptor.teammate_id}:idle:{descriptor.tasks_completed}",
                    payload={"teammate_id": descriptor.teammate_id},
                )
                await self._dispatch_pending()
            await asyncio.sleep(self.configurable.mailbox_poll_interval_ms / 1000)

    async def _execute_task(self, record: TaskRecord) -> None:
        from open_deep_research.observability import bind_span_context
        from open_deep_research.tasks.executor import run_task_with_control
        from open_deep_research.tasks.recovery import CheckpointManager

        with bind_span_context(
            record.run_id,
            record.trace_parent_span_id,
            record.langfuse_parent_span_id,
        ):
            await run_task_with_control(
                record,
                self.config,
                self.registry,
                self.execute_research,
                checkpoint_manager=CheckpointManager(runs_dir=self.configurable.runs_dir, run_id=self.run_id),
                runs_dir=self.configurable.runs_dir,
                run_id=self.run_id,
                event_log_enabled=self.configurable.event_log_enabled,
                fence_token=self.fence_token or 0,
            )

    async def shutdown(self, timeout_seconds: float = 10) -> None:
        """Request graceful teammate shutdown, then cancel stragglers."""
        active_tasks: list[asyncio.Task[None]] = []
        for runtime in self._runtimes.values():
            if runtime.descriptor.current_task_id:
                record = self.registry.get(runtime.descriptor.current_task_id)
                if record is not None:
                    record.cancelled.set()
            if runtime.active_task is not None and not runtime.active_task.done():
                runtime.active_task.cancel()
                active_tasks.append(runtime.active_task)
        if active_tasks:
            await asyncio.gather(*active_tasks, return_exceptions=True)
        for teammate_id in self._runtimes:
            await self.mailbox.send(
                recipient=teammate_id,
                sender="lead",
                message_type="shutdown_request",
                priority=0,
                dedupe_key=f"{teammate_id}:shutdown",
                payload={"teammate_id": teammate_id},
            )
        tasks = [runtime.loop_task for runtime in self._runtimes.values()]
        if tasks:
            done, pending = await asyncio.wait(tasks, timeout=timeout_seconds)
            del done
            self._stopping = True
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
        else:
            self._stopping = True
        if self._heartbeat_task is not None:
            self._heartbeat_task.cancel()
        if self._owns_lease_lifecycle and self.fence_token is not None:
            await self.lease.release(expected_fence_token=self.fence_token)
        self.fence_token = None


_POOLS: dict[tuple[str, str], TeammatePool] = {}


def get_active_teammate_pool(run_id: str) -> TeammatePool:
    """Return the unique active pool for *run_id*."""
    matches = [pool for (_runs_dir, rid), pool in _POOLS.items() if rid == run_id]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one active teammate pool for run {run_id}, found {len(matches)}")
    return matches[0]


def find_active_teammate_pool(run_id: str) -> TeammatePool | None:
    """Return an active pool when the run is owned by this process."""
    matches = [pool for (_runs_dir, rid), pool in _POOLS.items() if rid == run_id]
    return matches[0] if len(matches) == 1 else None


def get_teammate_pool(
    config: RunnableConfig,
    registry: TaskRegistry,
    execute_research: ExecuteResearchFn,
) -> TeammatePool:
    """Return the run-scoped teammate pool singleton."""
    configurable = Configuration.from_runnable_config(config)
    run_id = str(config.get("metadata", {}).get("run_id", "default"))
    key = (str(Path(configurable.runs_dir).resolve()), run_id)
    if key not in _POOLS:
        _POOLS[key] = TeammatePool(config=config, registry=registry, execute_research=execute_research)
    return _POOLS[key]


async def shutdown_teammate_pool(config: RunnableConfig) -> None:
    """Shutdown and remove one run's pool."""
    configurable = Configuration.from_runnable_config(config)
    run_id = str(config.get("metadata", {}).get("run_id", "default"))
    key = (str(Path(configurable.runs_dir).resolve()), run_id)
    pool = _POOLS.pop(key, None)
    if pool is not None:
        await pool.shutdown()
