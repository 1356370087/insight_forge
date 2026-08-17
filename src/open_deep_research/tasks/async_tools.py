"""Async SubAgent tool models and handler functions.

Defines the Pydantic tool models that replace the synchronous
``ConductResearch`` tool when ``enable_async_research`` is True, together
with handlers backed by the persistent teammate pool and file Mailbox.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Callable, Coroutine, Optional

from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableConfig

from open_deep_research.configuration import Configuration
from open_deep_research.observability import current_span_ids, get_trace_recorder
from open_deep_research.observability.telemetry import get_prometheus_metrics
from open_deep_research.public_events import (
    event_publisher_from_config,
    public_display_title,
)
from open_deep_research.sandbox.manager import stop_sandbox_container
from open_deep_research.tasks.coordination import (
    FileDomainDecisionStore,
    publish_task_update,
)
from open_deep_research.tasks.events import EventType, JSONLEventWriter, ResearchEvent
from open_deep_research.tasks.registry import (
    TaskRecord,
    TaskRegistry,
    TaskStatus,
)
from open_deep_research.tasks.state import (
    TaskSnapshot,
    TaskStateStore,
    get_task_state_store,
)


def _run_fence_token(config: RunnableConfig | None) -> int:
    """Read the current run ownership epoch from propagated metadata."""
    value = ((config or {}).get("metadata") or {}).get("run_fence_token", 0)
    return int(value or 0)

# Type alias for the function that actually launches a background researcher.
LaunchTaskFn = Callable[
    [TaskRecord, RunnableConfig],
    Coroutine[Any, Any, None],
]


# ---------------------------------------------------------------------------
# Handler functions
# ---------------------------------------------------------------------------


def _sandbox_summary(record: TaskRecord) -> str:
    """Return a compact sandbox status block for tool output."""
    if not record.sandbox_enabled:
        return ""
    container = record.container_id[:12] if record.container_id else "pending"
    lines = [
        f"Sandbox: enabled ({record.sandbox_network_mode or 'unknown'})",
        f"Workspace: {record.workspace_path or 'pending'}",
        f"Container: {container}",
    ]
    if record.output_archive_path:
        lines.append(f"Output archive: {record.output_archive_path}")
    if record.last_sandbox_event:
        lines.append(f"Last sandbox event: {record.last_sandbox_event}")
    return "\n".join(lines) + "\n"


def _snapshot_sandbox_summary(snapshot: TaskSnapshot) -> str:
    """Return a compact sandbox status block for a serializable snapshot."""
    sandbox = snapshot.sandbox
    if not sandbox.get("enabled"):
        return ""
    container_id = sandbox.get("container_id")
    container = container_id[:12] if container_id else "pending"
    lines = [
        f"Sandbox: enabled ({sandbox.get('network_mode') or 'unknown'})",
        f"Workspace: {sandbox.get('workspace_path') or 'pending'}",
        f"Container: {container}",
    ]
    if sandbox.get("output_archive_path"):
        lines.append(f"Output archive: {sandbox['output_archive_path']}")
    if sandbox.get("last_event"):
        lines.append(f"Last sandbox event: {sandbox['last_event']}")
    return "\n".join(lines) + "\n"


def _record_to_snapshot(record: TaskRecord) -> TaskSnapshot:
    """Bridge legacy in-memory records into shared-state snapshots."""
    return TaskSnapshot.from_record(record)


async def _get_snapshot_with_registry_fallback(
    store: TaskStateStore,
    registry: TaskRegistry,
    task_id: str,
    *,
    run_id: str = "",
) -> Optional[TaskSnapshot]:
    record = registry.get(task_id)
    if record is not None and run_id and record.run_id not in {"", run_id}:
        return None
    if record is not None and not record.run_id:
        return _record_to_snapshot(record)
    snapshot = await store.get(task_id, run_id=run_id or None)
    if snapshot is not None:
        return snapshot
    record = registry.get(task_id)
    if record is None or (run_id and record.run_id not in {"", run_id}):
        return None
    snapshot = _record_to_snapshot(record)
    await store.upsert(snapshot)
    return snapshot


async def _list_snapshots_with_registry_fallback(
    store: TaskStateStore,
    registry: TaskRegistry,
    *,
    status_filter: Optional[TaskStatus] = None,
    run_id: Optional[str] = None,
) -> list[TaskSnapshot]:
    if not run_id:
        return [
            _record_to_snapshot(record)
            for record in registry.list(status_filter=status_filter)
        ]
    snapshots = await store.list(status_filter=status_filter, run_id=run_id)
    if snapshots:
        return snapshots
    records = registry.list(status_filter=status_filter, run_id=run_id)
    bridged = [_record_to_snapshot(record) for record in records]
    for snapshot in bridged:
        await store.upsert(snapshot)
    return bridged


def format_task_snapshot_for_context(snapshot: TaskSnapshot) -> str:
    """Format a task snapshot for lead-agent context."""
    metrics = snapshot.metrics
    if snapshot.status == TaskStatus.COMPLETED:
        result = snapshot.result or {}
        compressed = result.get("compressed_research", "(no findings)")
        return (
            f"### {snapshot.task_id} - COMPLETED\n"
            f"Topic: {snapshot.research_topic}\n"
            f"Duration: {snapshot.elapsed_seconds:.1f}s\n"
            f"Queries: {metrics.get('query_count', 0)} | "
            f"Sources: {metrics.get('source_count', 0)}\n\n"
            f"{_snapshot_sandbox_summary(snapshot)}"
            f"{compressed}\n"
        )
    if snapshot.status == TaskStatus.RUNNING:
        return (
            f"### {snapshot.task_id} - RUNNING\n"
            f"Topic: {snapshot.research_topic}\n"
            f"Phase: {snapshot.phase.value}\n"
            f"Elapsed: {snapshot.elapsed_seconds:.1f}s\n"
            f"Queries so far: {metrics.get('query_count', 0)} | "
            f"Sources so far: {metrics.get('source_count', 0)}\n"
            f"{_snapshot_sandbox_summary(snapshot)}"
        )
    if snapshot.status == TaskStatus.FAILED:
        return (
            f"### {snapshot.task_id} - FAILED\n"
            f"Topic: {snapshot.research_topic}\n"
            f"Error: {snapshot.error_message or 'unknown error'}\n"
            f"{_snapshot_sandbox_summary(snapshot)}"
            f"Consider re-launching with StartResearchTask if this topic is still needed.\n"
        )
    if snapshot.status == TaskStatus.CANCELLED:
        return (
            f"### {snapshot.task_id} - CANCELLED\n"
            f"Topic: {snapshot.research_topic}\n"
            f"{_snapshot_sandbox_summary(snapshot)}"
        )
    if snapshot.status == TaskStatus.TIMED_OUT:
        return (
            f"### {snapshot.task_id} - TIMED OUT\n"
            f"Topic: {snapshot.research_topic}\n"
            f"{_snapshot_sandbox_summary(snapshot)}"
            f"Consider re-launching with a narrower scope.\n"
        )
    if snapshot.status == TaskStatus.WAITING_FOR_CONFIRMATION:
        domain = snapshot.pending_domain or "(unknown)"
        return (
            f"### {snapshot.task_id} - WAITING FOR DOMAIN APPROVAL\n"
            f"Topic: {snapshot.research_topic}\n"
            f"Phase: {snapshot.phase.value}\n"
            f"Elapsed: {snapshot.elapsed_seconds:.1f}s\n"
            f"Domain awaiting approval: {domain}\n"
            f"Use ApproveResearchDomain(task_id={snapshot.task_id}, "
            f"domain='{domain}', allow=True/False) to allow or deny. "
            f"The task is paused until you decide.\n"
        )
    return (
        f"### {snapshot.task_id} - {snapshot.status.value.upper()}\n"
        f"Topic: {snapshot.research_topic}\n"
    )


async def _publish_snapshot_update(
    configurable: Configuration,
    snapshot: TaskSnapshot,
    event_type: EventType,
    event_writer: Optional[JSONLEventWriter],
) -> None:
    try:
        await publish_task_update(configurable, snapshot, event_type)
    except Exception as exc:
        if event_writer is not None:
            event_writer.write(ResearchEvent(
                event_type=EventType.TASK_MAILBOX_DELIVERY_FAILED,
                task_id=snapshot.task_id,
                run_id=snapshot.run_id,
                phase=snapshot.phase.value,
                data={"error": str(exc)},
            ))
        # The durable snapshot remains authoritative and can be fetched with
        # CheckResearchTask even if the push notification is unavailable.


async def handle_start_research_task(
    tool_call: dict[str, Any],
    config: RunnableConfig,
    registry: TaskRegistry,
    launch_task: LaunchTaskFn,
    event_writer: Optional[JSONLEventWriter] = None,
    memory_context: Optional[str] = None,
    coverage_contract: Optional[dict[str, Any]] = None,
    research_risk_profile: Optional[dict[str, Any]] = None,
) -> ToolMessage:
    """Create a task record, spawn a background researcher, and return the task_id."""
    configurable = Configuration.from_runnable_config(config)
    state_store = get_task_state_store(configurable)

    # Extract run-level identifiers for isolation
    run_id = config.get("metadata", {}).get("run_id", "default")
    user_id = (
        config.get("configurable", {}).get("memory_user_id")
        or config.get("metadata", {}).get("user_id")
    )

    # Respect concurrency limit (scoped to this run). A task paused for domain
    # confirmation (WAITING_FOR_CONFIRMATION) still holds a slot, so count_active
    # is used instead of count_running.
    active_count = max(
        registry.count_active(run_id=run_id),
        await state_store.count_active(run_id=run_id),
    )
    if active_count >= configurable.max_in_flight_tasks:
        get_trace_recorder(config).active_span().record_outcome(
            error_type="task_capacity_exceeded",
        )
        return ToolMessage(
            content=(
                f"Cannot start new task: already at maximum in-flight tasks "
                f"({configurable.max_in_flight_tasks}). Wait for some tasks to "
                f"complete, then try again."
            ),
            name="StartResearchTask",
            tool_call_id=tool_call["id"],
        )

    research_topic = tool_call["args"]["research_topic"]
    requirement_ids = [
        str(item)
        for item in tool_call.get("args", {}).get("requirement_ids", [])
        if str(item).strip()
    ]
    display_title = public_display_title(
        str(tool_call.get("args", {}).get("display_title") or research_topic)
    )
    wave_id = str(config.get("metadata", {}).get("research_wave_id") or "wave-1")
    record = registry.create(
        research_topic=research_topic,
        run_id=run_id,
        user_id=user_id,
        display_title=display_title,
        wave_id=wave_id,
        plan_task_id=tool_call["id"],
        requirement_ids=requirement_ids,
        coverage_contract=dict(coverage_contract or {}),
        research_risk_profile=dict(research_risk_profile or {}),
    )
    _trace_run_id, trace_parent_span_id = current_span_ids()
    record.trace_parent_span_id = trace_parent_span_id
    record.langfuse_parent_span_id = getattr(
        get_trace_recorder(config).active_span(),
        "langfuse_observation_id",
        None,
    )
    if memory_context:
        record.memory_context = memory_context
    if configurable.enable_docker_sandbox:
        record.sandbox_enabled = True
        record.sandbox_network_mode = configurable.sandbox_network_mode

    # Emit creation event
    if event_writer is not None:
        event_writer.write(ResearchEvent(
            event_type=EventType.TASK_CREATED,
            task_id=record.task_id,
            run_id=event_writer.run_id,
            data={"research_topic": research_topic},
        ))
    snapshot = await state_store.update_from_record(
        record,
        fence_token=_run_fence_token(config),
    )
    metrics = get_prometheus_metrics(configurable)
    if metrics is not None:
        metrics.observe_task_transition(record, EventType.TASK_CREATED)
        pending = await state_store.list(status_filter=TaskStatus.PENDING, run_id=run_id)
        metrics.set_task_counts(len(pending), await state_store.count_active(run_id=run_id))
    await _publish_snapshot_update(
        configurable, snapshot, EventType.TASK_CREATED, event_writer
    )
    publisher = event_publisher_from_config(config)
    common_public = {
        "task_id": record.task_id,
        "wave_id": record.wave_id,
        "plan_task_id": record.plan_task_id,
        "title": record.display_title,
        "mode": "async",
        "status": "pending",
        "phase": record.phase.value,
    }
    await publisher.publish(
        "plan.task.added",
        stage="researching",
        payload=common_public,
        dedupe_key=f"plan:task:{record.task_id}:added",
    )
    await publisher.publish(
        "research.task.created",
        stage="researching",
        payload=common_public,
        dedupe_key=f"task:{record.task_id}:created",
    )

    # Submit to the persistent pool. Assignment is durable before this returns.
    # If pool admission itself fails, persist a terminal state so a task created
    # moments earlier cannot remain PENDING forever.
    try:
        await launch_task(record, config)
    except Exception as exc:
        registry.update_status(
            record.task_id,
            TaskStatus.FAILED,
            error_message=f"Unable to launch task: {exc}",
        )
        snapshot = await state_store.update_from_record(
            record,
            fence_token=_run_fence_token(config),
        )
        await _publish_snapshot_update(
            configurable, snapshot, EventType.TASK_FAILED, event_writer
        )
        await publisher.publish(
            "research.task.failed",
            stage="researching",
            payload={
                "task_id": record.task_id,
                "wave_id": record.wave_id,
                "mode": "async",
                "status": "failed",
                "phase": record.phase.value,
                "elapsed_ms": int(record.elapsed_seconds * 1000),
                "error_code": "task_launch_failed",
                "message": "The research task could not be started.",
            },
            dedupe_key=f"task:{record.task_id}:failed",
        )
        raise

    return ToolMessage(
        content=(
            f"Research task launched successfully.\n"
            f"task_id: {record.task_id}\n"
            f"topic: {research_topic}\n"
            f"{_sandbox_summary(record)}"
            f"The orchestrator will report state changes automatically; "
            f"use CheckResearchTask for an on-demand refresh."
        ),
        name="StartResearchTask",
        tool_call_id=tool_call["id"],
    )


async def handle_check_research_task(
    tool_call: dict[str, Any],
    registry: TaskRegistry,
    event_writer: Optional[JSONLEventWriter] = None,
    state_store: Optional[TaskStateStore] = None,
    *,
    run_id: str = "",
) -> ToolMessage:
    """Poll one or more tasks and return their status / results."""
    store = state_store or get_task_state_store(Configuration.from_runnable_config(None))
    task_ids: list[str] = tool_call["args"]["task_ids"]
    parts: list[str] = []

    for task_id in task_ids:
        snapshot = await _get_snapshot_with_registry_fallback(
            store, registry, task_id, run_id=run_id
        )
        if snapshot is None:
            parts.append(f"### {task_id}\nStatus: **UNKNOWN** — no such task.\n")
            continue
        parts.append(format_task_snapshot_for_context(snapshot))

    content = "\n---\n".join(parts) if parts else "No tasks found."
    return ToolMessage(
        content=content,
        name="CheckResearchTask",
        tool_call_id=tool_call["id"],
    )


async def handle_list_research_tasks(
    tool_call: dict[str, Any],
    registry: TaskRegistry,
    *,
    run_id: str = "",
    state_store: Optional[TaskStateStore] = None,
) -> ToolMessage:
    """Return a summary table of all (or filtered) tasks scoped to *run_id*."""
    store = state_store or get_task_state_store(Configuration.from_runnable_config(None))
    status_filter_str: Optional[str] = tool_call["args"].get("status_filter")
    status_filter = None
    if status_filter_str:
        try:
            status_filter = TaskStatus(status_filter_str)
        except ValueError:
            pass  # invalid filter → return all

    records = await _list_snapshots_with_registry_fallback(
        store, registry, status_filter=status_filter, run_id=run_id
    )

    if not records:
        return ToolMessage(
            content="No research tasks found.",
            name="ListResearchTasks",
            tool_call_id=tool_call["id"],
        )

    lines = [
        "| task_id | topic | status | phase | elapsed | queries | sources | sandbox | container |",
        "|---------|-------|--------|-------|---------|---------|---------|---------|-----------|",
    ]
    for r in records:
        topic_short = r.research_topic[:60] + "..." if len(r.research_topic) > 60 else r.research_topic
        container_id = r.sandbox.get("container_id")
        container = container_id[:12] if container_id else ""
        sandbox = r.sandbox.get("network_mode") or ("enabled" if r.sandbox.get("enabled") else "")
        lines.append(
            f"| {r.task_id[-8:]} | {topic_short} | {r.status.value} | {r.phase.value} "
            f"| {r.elapsed_seconds:.0f}s | {r.metrics.get('query_count', 0)} | "
            f"{r.metrics.get('source_count', 0)} | {sandbox} | {container} |"
        )

    summary = (
        f"Total: {len(records)} tasks | "
        f"Running: {sum(record.status == TaskStatus.RUNNING for record in records)} | "
        f"Max in-flight: (see configuration)\n\n"
        + "\n".join(lines)
    )

    return ToolMessage(
        content=summary,
        name="ListResearchTasks",
        tool_call_id=tool_call["id"],
    )


async def handle_update_research_task(
    tool_call: dict[str, Any],
    registry: TaskRegistry,
    event_writer: Optional[JSONLEventWriter] = None,
    state_store: Optional[TaskStateStore] = None,
    *,
    run_id: str = "",
    fence_token: int = 0,
) -> ToolMessage:
    """Queue an update instruction on the task's control queue."""
    task_id: str = tool_call["args"]["task_id"]
    instruction: str = tool_call["args"]["instruction"]

    record = registry.get(task_id)
    if record is None or (run_id and record.run_id not in {"", run_id}):
        return ToolMessage(
            content=f"Task {task_id} not found.",
            name="UpdateResearchTask",
            tool_call_id=tool_call["id"],
        )

    if record.status != TaskStatus.RUNNING:
        return ToolMessage(
            content=(
                f"Task {task_id} is {record.status.value}, not RUNNING. "
                f"Updates can only be sent to running tasks."
            ),
            name="UpdateResearchTask",
            tool_call_id=tool_call["id"],
        )

    if record.run_id:
        from open_deep_research.tasks.teammate_pool import find_active_teammate_pool

        pool = find_active_teammate_pool(record.run_id)
        if pool is not None:
            await pool.send_control(
                task_id=task_id,
                message_type="task_update",
                payload={"instruction": instruction},
                priority=10,
            )
        else:
            if instruction not in record.pending_update_instructions:
                record.pending_update_instructions.append(instruction)
                if state_store is not None:
                    await state_store.update_from_record(
                        record,
                        fence_token=fence_token,
                    )
            await record.control_queue.put({"type": "update", "instruction": instruction})
    else:
        await record.control_queue.put({"type": "update", "instruction": instruction})

    if event_writer is not None:
        event_writer.write(ResearchEvent(
            event_type=EventType.TASK_UPDATED,
            task_id=task_id,
            run_id=event_writer.run_id,
            phase=record.phase.value,
            data={"instruction": instruction},
        ))

    return ToolMessage(
        content=(
            f"Update instruction queued for task {task_id}.\n"
            f"The task will incorporate this guidance at its next iteration.\n"
            f"Instruction: {instruction}"
        ),
        name="UpdateResearchTask",
        tool_call_id=tool_call["id"],
    )


async def handle_cancel_research_task(
    tool_call: dict[str, Any],
    registry: TaskRegistry,
    event_writer: Optional[JSONLEventWriter] = None,
    state_store: Optional[TaskStateStore] = None,
    configurable: Optional[Configuration] = None,
    *,
    run_id: str = "",
    fence_token: int = 0,
) -> ToolMessage:
    """Signal cancellation for one or more tasks."""
    effective_config = configurable or Configuration.from_runnable_config(None)
    store = state_store or get_task_state_store(effective_config)
    task_ids: list[str] = tool_call["args"]["task_ids"]
    reason: str = tool_call["args"].get("reason", "No reason provided")
    results: list[str] = []

    for task_id in task_ids:
        record = registry.get(task_id)
        if record is None or (run_id and record.run_id not in {"", run_id}):
            results.append(f"- {task_id}: not found")
            continue

        if record.status not in (
            TaskStatus.PENDING,
            TaskStatus.RUNNING,
            TaskStatus.WAITING_FOR_CONFIRMATION,
        ):
            results.append(f"- {task_id}: already {record.status.value}")
            continue

        if record.run_id:
            from open_deep_research.tasks.teammate_pool import find_active_teammate_pool

            pool = find_active_teammate_pool(record.run_id)
            if pool is not None:
                await pool.send_control(
                    task_id=task_id,
                    message_type="cancel_request",
                    payload={"reason": reason},
                    priority=0,
                )
            else:
                record.cancelled.set()
        else:
            record.cancelled.set()
        registry.update_status(task_id, TaskStatus.CANCELLED, completed_at=time.time())
        stop_error = None
        if record.container_id:
            try:
                await asyncio.to_thread(stop_sandbox_container, record.container_id)
            except Exception as exc:
                stop_error = str(exc)

        if event_writer is not None:
            event_writer.write(ResearchEvent(
                event_type=EventType.TASK_CANCELLED,
                task_id=task_id,
                run_id=event_writer.run_id,
                phase=record.phase.value,
                data={"reason": reason, "sandbox_stop_error": stop_error},
            ))
        snapshot = await store.update_from_record(
            record,
            fence_token=fence_token,
        )
        metrics = get_prometheus_metrics(effective_config)
        if metrics is not None:
            metrics.observe_task_transition(record, EventType.TASK_CANCELLED)
            pending = await store.list(
                status_filter=TaskStatus.PENDING,
                run_id=record.run_id or None,
            )
            metrics.set_task_counts(
                len(pending),
                await store.count_active(run_id=record.run_id or None),
            )
        await _publish_snapshot_update(
            effective_config, snapshot, EventType.TASK_CANCELLED, event_writer
        )
        if record.run_id and registry.count_active(run_id=record.run_id) == 0:
            from open_deep_research.tasks.domain_approvals import (
                get_domain_approval_registry,
            )

            get_domain_approval_registry().clear_run(record.run_id)

        if stop_error:
            results.append(f"- {task_id}: cancelled; sandbox stop failed: {stop_error}")
        else:
            results.append(f"- {task_id}: cancelled")

    return ToolMessage(
        content="Cancellation results:\n" + "\n".join(results),
        name="CancelResearchTask",
        tool_call_id=tool_call["id"],
    )


async def handle_approve_research_domain(
    tool_call: dict[str, Any],
    config: RunnableConfig,
    registry: TaskRegistry,
    event_writer: Optional[JSONLEventWriter] = None,
    state_store: Optional[TaskStateStore] = None,
) -> ToolMessage:
    """Record the supervisor's allow/deny decision for a paused task's domain.

    Resolves the pending ``asyncio.Future`` that the in-process governance layer
    is awaiting, flips the task back to ``RUNNING``, and caches the decision for
    the rest of the run. Also pushes a ``domain_decision`` marker onto the
    control queue (informational; the future is the real resume signal).
    """
    task_id: str = tool_call["args"]["task_id"]
    domain: str = tool_call["args"]["domain"]
    allow: bool = tool_call["args"]["allow"]

    current_run_id = str(config.get("metadata", {}).get("run_id", "default"))
    record = registry.get(task_id)
    if record is None or record.run_id not in {"", current_run_id}:
        return ToolMessage(
            content=f"Task {task_id} not found.",
            name="ApproveResearchDomain",
            tool_call_id=tool_call["id"],
        )

    if record.status not in (TaskStatus.WAITING_FOR_CONFIRMATION, TaskStatus.RUNNING):
        return ToolMessage(
            content=(
                f"Task {task_id} is {record.status.value}; only waiting or running "
                f"tasks accept domain decisions."
            ),
            name="ApproveResearchDomain",
            tool_call_id=tool_call["id"],
        )

    configurable = Configuration.from_runnable_config(config)
    if record.run_id:
        await FileDomainDecisionStore(configurable, record.run_id).record(domain, allow)
        from open_deep_research.tasks.teammate_pool import find_active_teammate_pool

        pool = find_active_teammate_pool(record.run_id)
        if pool is not None:
            await pool.send_control(
                task_id=task_id,
                message_type="domain_decision",
                payload={"domain": domain.lower(), "allow": allow},
                priority=0,
            )
        else:
            from open_deep_research.tasks.domain_approvals import (
                get_domain_approval_registry,
            )

            get_domain_approval_registry().record_decision(record.run_id, domain, allow)
    else:
        from open_deep_research.tasks.domain_approvals import (
            get_domain_approval_registry,
        )

        get_domain_approval_registry().record_decision(record.run_id, domain, allow)

    if record.status == TaskStatus.WAITING_FOR_CONFIRMATION:
        registry.update_status(task_id, TaskStatus.RUNNING)
    record.pending_domain = None
    record.pending_domain_tool = None

    if event_writer is not None:
        event_writer.write(ResearchEvent(
            event_type=EventType.TASK_DOMAIN_DECISION,
            task_id=task_id,
            run_id=event_writer.run_id,
            phase=record.phase.value,
            data={"domain": domain.lower(), "allow": allow},
        ))

    store = state_store or get_task_state_store(configurable)
    snapshot = await store.update_from_record(
        record,
        fence_token=_run_fence_token(config),
    )
    await _publish_snapshot_update(
        configurable, snapshot, EventType.TASK_DOMAIN_DECISION, event_writer
    )

    return ToolMessage(
        content=(
            f"Domain '{domain}' {'approved' if allow else 'denied'} for task "
            f"{task_id} (run {record.run_id}). The task will "
            f"{'resume' if allow else 'skip that fetch'}."
        ),
        name="ApproveResearchDomain",
        tool_call_id=tool_call["id"],
    )


async def collect_completed_task_outputs(
    registry: TaskRegistry,
    *,
    run_id: str = "",
    state_store: Optional[TaskStateStore] = None,
) -> list[dict[str, Any]]:
    """Gather outputs from completed tasks for *run_id* for the final report.

    Called when ResearchComplete is invoked in async mode.  The *run_id*
    filter prevents cross-run contamination from the global registry.
    """
    store = state_store or get_task_state_store(Configuration.from_runnable_config(None))
    completed = await _list_snapshots_with_registry_fallback(
        store, registry, status_filter=TaskStatus.COMPLETED, run_id=run_id
    )
    return [
        {
            "research_topic": r.research_topic,
            "compressed_research": (r.result or {}).get("compressed_research", ""),
            "raw_notes": (r.result or {}).get("raw_notes", []),
            "candidate_registry": (r.result or {}).get("candidate_registry", []),
            "document_registry": (r.result or {}).get("document_registry", []),
            "evidence_registry": (r.result or {}).get("evidence_registry", []),
            "web_research_iterations": (r.result or {}).get("web_research_iterations", []),
            "task_id": r.task_id,
            "query_count": r.metrics.get("query_count", 0),
            "source_count": r.metrics.get("source_count", 0),
            "citation_count": r.metrics.get("citation_count", 0),
            "elapsed_seconds": r.elapsed_seconds,
        }
        for r in completed
    ]
