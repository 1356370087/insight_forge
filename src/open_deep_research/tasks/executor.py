"""Background executor for async SubAgent research tasks.

Runs compiled researcher subgraphs as ``asyncio.Task`` instances so the
supervisor is never blocked waiting for a single researcher to finish.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Callable, Coroutine, Optional

from langchain_core.messages import HumanMessage, message_to_dict
from langchain_core.runnables import RunnableConfig

from open_deep_research.configuration import Configuration
from open_deep_research.observability.telemetry import get_prometheus_metrics
from open_deep_research.public_events import (
    event_publisher_from_config,
    extract_public_sources,
    summarize_public_findings,
)
from open_deep_research.public_task_activity import publish_task_activity
from open_deep_research.run_context import RunContextStore
from open_deep_research.sandbox.manager import DockerSandboxManager
from open_deep_research.tasks.coordination import publish_task_update
from open_deep_research.tasks.domain_approvals import get_domain_approval_registry
from open_deep_research.tasks.events import EventType, JSONLEventWriter, ResearchEvent
from open_deep_research.tasks.recovery import CheckpointManager, ResearcherCheckpoint
from open_deep_research.tasks.registry import (
    TaskPhase,
    TaskRecord,
    TaskRegistry,
    TaskStatus,
)
from open_deep_research.tasks.state import get_task_state_store

# A callable that invokes the researcher subgraph and returns its output dict.
ExecuteResearchFn = Callable[
    [dict[str, Any], RunnableConfig],
    Coroutine[Any, Any, dict[str, Any]],
]


# ---------------------------------------------------------------------------
# Event helper — open-append-close per event (safe for background tasks)
# ---------------------------------------------------------------------------


def _emit_event(
    event: ResearchEvent,
    runs_dir: str,
    run_id: str,
    enabled: bool = True,
) -> None:
    """Write a single event to the run's JSONL file using a short-lived writer.

    This pattern is safe for background ``asyncio.Task`` instances that
    outlive the supervisor's writer handle.
    """
    if not enabled:
        return
    writer = JSONLEventWriter(run_id=run_id, runs_dir=runs_dir)
    try:
        writer.write(event)
    finally:
        writer.close()


def _run_fence_token(config: RunnableConfig) -> int:
    """Return the ownership epoch propagated by the run orchestrator."""
    value = (config.get("metadata") or {}).get("run_fence_token", 0)
    return int(value or 0)


async def _emit_state_change(
    task_record: TaskRecord,
    config: RunnableConfig,
    *,
    event_type: EventType,
    runs_dir: str,
    run_id: str,
    event_log_enabled: bool,
    phase: Optional[str] = None,
    data: Optional[dict[str, Any]] = None,
    notify: bool = True,
) -> None:
    """Append JSONL, persist latest state, then notify the Lead mailbox."""
    event = ResearchEvent(
        event_type=event_type,
        task_id=task_record.task_id,
        run_id=run_id,
        phase=phase or task_record.phase.value,
        data=data or {},
    )
    _emit_event(event, runs_dir=runs_dir, run_id=run_id, enabled=event_log_enabled)

    configurable = Configuration.from_runnable_config(config)
    metrics = get_prometheus_metrics(configurable)
    if metrics is not None:
        metrics.observe_task_transition(task_record, event_type)
    store = get_task_state_store(configurable)
    snapshot = await store.update_from_record(
        task_record,
        fence_token=_run_fence_token(config),
    )
    activity_type, activity_kind, activity_phase, activity_status, activity_title = {
        EventType.TASK_STARTED: (
            "task.started",
            "lifecycle",
            "initializing",
            "running",
            "Subagent 已启动",
        ),
        EventType.TASK_COMPLETED: (
            "task.completed",
            "lifecycle",
            "terminal",
            "success",
            "Subagent 已完成",
        ),
        EventType.TASK_FAILED: (
            "task.failed",
            "error",
            "terminal",
            "error",
            "Subagent 执行失败",
        ),
        EventType.TASK_TIMED_OUT: (
            "task.timed_out",
            "error",
            "terminal",
            "error",
            "Subagent 执行超时",
        ),
        EventType.TASK_CANCELLED: (
            "task.cancelled",
            "lifecycle",
            "terminal",
            "cancelled",
            "Subagent 已取消",
        ),
        EventType.CHECKPOINT_LOADED: (
            "checkpoint.loaded",
            "checkpoint",
            "initializing",
            "success",
            "已恢复研究检查点",
        ),
        EventType.CHECKPOINT_SAVED: (
            "checkpoint.saved",
            "checkpoint",
            "evidence_review",
            "success",
            "已保存研究检查点",
        ),
        EventType.TASK_DOMAIN_CONFIRMATION_REQUESTED: (
            "security.blocked",
            "security",
            "tool_execution",
            "warning",
            "等待域名访问确认",
        ),
        EventType.TASK_DOMAIN_DECISION: (
            "control.received",
            "control",
            "tool_execution",
            "success",
            "已应用域名访问决定",
        ),
        EventType.TASK_UPDATED: (
            "control.received",
            "control",
            "reasoning",
            "success",
            "已接收任务控制更新",
        ),
        EventType.TASK_PHASE_CHANGE: (
            "task.phase.changed",
            "lifecycle",
            "reasoning",
            "running",
            "研究阶段已更新",
        ),
    }.get(event_type, (None, None, None, None, None))
    if event_type == EventType.TASK_COMPLETED and configurable.quality_evaluation_enabled:
        activity_type = "task.phase.changed"
        activity_kind = "lifecycle"
        activity_phase = "handoff"
        activity_status = "running"
        activity_title = "等待交接质量复核"
    if activity_type is not None:
        safe_payload: dict[str, Any] = {
            "mode": "async",
            "wave_id": task_record.wave_id,
            "source_count": task_record.source_count,
        }
        event_data = data or {}
        if event_type == EventType.TASK_COMPLETED and configurable.quality_evaluation_enabled:
            safe_payload["activity_label"] = "质量交接"
        if event_type in {EventType.CHECKPOINT_LOADED, EventType.CHECKPOINT_SAVED}:
            safe_payload["completed_queries"] = int(
                event_data.get("completed_queries", 0) or 0
            )
        if event_type in {
            EventType.TASK_UPDATED,
            EventType.TASK_DOMAIN_DECISION,
        }:
            instruction = str(
                event_data.get("instruction")
                or event_data.get("decision")
                or "Task control update received."
            )
            safe_payload["instruction_summary"] = instruction[:240]
        if event_type == EventType.TASK_DOMAIN_CONFIRMATION_REQUESTED:
            safe_payload["domain"] = str(
                event_data.get("domain")
                or task_record.pending_domain
                or "unknown"
            )[:253]
        await publish_task_activity(
            config,
            task_id=task_record.task_id,
            event_type=activity_type,
            kind=activity_kind,
            phase=activity_phase,
            status=activity_status,
            title=activity_title,
            summary=(
                "该事件来自异步 Researcher 的持久化执行状态。"
                if activity_status != "error"
                else "任务已进入异常终态，请查看安全错误码和运行警告。"
            ),
            iteration=None,
            duration_ms=int(task_record.elapsed_seconds * 1000),
            payload=safe_payload,
            dedupe_key=f"async:{task_record.task_id}:{event.event_id}",
            update_run_summary=True,
        )
    public_type = {
        EventType.TASK_STARTED: "research.task.started",
        EventType.TASK_COMPLETED: "research.task.completed",
        EventType.TASK_FAILED: "research.task.failed",
        EventType.TASK_CANCELLED: "research.task.cancelled",
        EventType.TASK_TIMED_OUT: "research.task.timed_out",
        EventType.TASK_UPDATED: "research.task.progress",
        EventType.TASK_PHASE_CHANGE: "research.task.progress",
        EventType.TASK_DOMAIN_CONFIRMATION_REQUESTED: "research.task.progress",
        EventType.TASK_DOMAIN_DECISION: "research.task.progress",
    }.get(event_type)
    if event_type == EventType.TASK_COMPLETED and configurable.quality_evaluation_enabled:
        public_type = "research.task.progress"
    if public_type is not None:
        status = task_record.status.value
        if event_type == EventType.TASK_COMPLETED and configurable.quality_evaluation_enabled:
            status = "running"
        public_payload: dict[str, Any] = {
            "task_id": task_record.task_id,
            "wave_id": task_record.wave_id,
            "plan_task_id": task_record.plan_task_id,
            "title": task_record.display_title,
            "mode": "async",
            "status": status,
            "phase": (
                "handoff"
                if event_type == EventType.TASK_COMPLETED
                and configurable.quality_evaluation_enabled
                else phase or task_record.phase.value
            ),
            "elapsed_ms": int(task_record.elapsed_seconds * 1000),
            "source_count": task_record.source_count,
        }
        if public_type in {"research.task.failed", "research.task.timed_out"}:
            public_payload.update({
                "error_code": "research_task_timed_out" if event_type == EventType.TASK_TIMED_OUT else "research_task_failed",
                "message": "The research task timed out." if event_type == EventType.TASK_TIMED_OUT else "The research task failed.",
            })
        if public_type == "research.task.completed":
            public_payload["admission_status"] = task_record.admission_status
            if not configurable.quality_evaluation_enabled:
                public_payload["summary_status"] = "pending"
        publisher = event_publisher_from_config(config)
        transition = event_type.value.removeprefix("task.")
        dedupe_suffix = (
            f"{transition}:{snapshot.version}"
            if public_type == "research.task.progress"
            else transition
        )
        await publisher.publish(
            public_type,
            stage="researching",
            payload=public_payload,
            dedupe_key=f"task:{task_record.task_id}:{dedupe_suffix}",
        )
        if (
            event_type == EventType.TASK_COMPLETED
            and not configurable.quality_evaluation_enabled
            and task_record.result
        ):
            summary = await summarize_public_findings(task_record.result, config)
            sources = extract_public_sources(
                task_record.result,
                limit=configurable.public_event_source_limit,
            )
            for source in sources:
                await publisher.publish(
                    "research.source.discovered",
                    stage="researching",
                    payload={"task_id": task_record.task_id, **source},
                    dedupe_key=f"source:{source['source_id']}",
                )
            if summary:
                await publisher.publish(
                    "findings.updated",
                    stage="researching",
                    payload={
                        "task_id": task_record.task_id,
                        "wave_id": task_record.wave_id,
                        "summary": summary,
                        "sources": sources,
                        "source_count": len(sources),
                    },
                    dedupe_key=f"task:{task_record.task_id}:findings",
                )
            else:
                await publisher.publish(
                    "research.task.completed",
                    stage="researching",
                    payload={
                        **public_payload,
                        "source_count": max(task_record.source_count, len(sources)),
                        "summary_status": "unavailable",
                        "message": "Research summary is temporarily unavailable.",
                    },
                    dedupe_key=f"task:{task_record.task_id}:summary:unavailable",
                )
    if metrics is not None:
        pending = await store.list(status_filter=TaskStatus.PENDING, run_id=run_id)
        metrics.set_task_counts(len(pending), await store.count_active(run_id=run_id))
    if not notify:
        return
    try:
        await publish_task_update(configurable, snapshot, event_type)
    except Exception as exc:
        _emit_event(
            ResearchEvent(
                event_type=EventType.TASK_MAILBOX_DELIVERY_FAILED,
                task_id=task_record.task_id,
                run_id=run_id,
                phase=phase or task_record.phase.value,
                data={"error": str(exc)},
            ),
            runs_dir=runs_dir,
            run_id=run_id,
            enabled=event_log_enabled,
        )
        # The task snapshot is authoritative. A transient notification failure
        # must not roll back or strand the task lifecycle.


async def emit_task_state_change(
    task_record: TaskRecord,
    config: RunnableConfig,
    *,
    event_type: EventType,
    runs_dir: str,
    run_id: str,
    event_log_enabled: bool,
    phase: Optional[str] = None,
    data: Optional[dict[str, Any]] = None,
    notify: bool = True,
) -> None:
    """Public wrapper over :func:`_emit_state_change` for cross-module callers.

    The tool-governance layer uses this to emit domain-confirmation transitions
    (JSONL event + state-store snapshot + Mailbox notification) without duplicating
    the plumbing.
    """
    await _emit_state_change(
        task_record,
        config,
        event_type=event_type,
        runs_dir=runs_dir,
        run_id=run_id,
        event_log_enabled=event_log_enabled,
        phase=phase,
        data=data,
        notify=notify,
    )


def _config_with_task_id(config: RunnableConfig, task_id: str) -> RunnableConfig:
    """Return a shallow copy of *config* with ``metadata["task_id"]`` set.

    Lets the tool-governance egress check locate the exact ``TaskRecord`` to
    pause for a domain decision, without mutating the caller's config.
    """
    new_config: RunnableConfig = dict(config)  # type: ignore[assignment]
    new_config["metadata"] = {
        **(config.get("metadata") or {}),
        "task_id": task_id,
        "research_mode": "async",
    }
    return new_config


def _clear_run_approvals_if_idle(registry: TaskRegistry, run_id: str) -> None:
    """Release run-scoped approval state only after its last active task ends."""
    if registry.count_active(run_id=run_id) == 0:
        get_domain_approval_registry().clear_run(run_id)


# ---------------------------------------------------------------------------
# Simple executor (no control queue, no checkpoints)
# ---------------------------------------------------------------------------


async def run_task(
    task_record: TaskRecord,
    config: RunnableConfig,
    registry: TaskRegistry,
    execute_research: ExecuteResearchFn,
    *,
    runs_dir: str = ".runs",
    run_id: str = "default",
    event_log_enabled: bool = True,
) -> None:
    """Execute a single research task in the background (simple version)."""
    configurable = Configuration.from_runnable_config(config)
    timeout = configurable.task_timeout_seconds
    registry.update_status(
        task_record.task_id,
        TaskStatus.RUNNING,
        started_at=time.time(),
    )

    try:
        await _emit_state_change(
            task_record,
            config,
            event_type=EventType.TASK_STARTED,
            runs_dir=runs_dir,
            run_id=run_id,
            event_log_enabled=event_log_enabled,
            data={"research_topic": task_record.research_topic},
        )
    except Exception as exc:  # noqa: BLE001 - state must not remain RUNNING
        registry.update_status(
            task_record.task_id,
            TaskStatus.FAILED,
            error_message=f"Unable to publish task start: {exc}",
            completed_at=time.time(),
        )
        try:
            configurable = Configuration.from_runnable_config(config)
            snapshot = await get_task_state_store(configurable).update_from_record(
                task_record,
                fence_token=_run_fence_token(config),
            )
            await publish_task_update(configurable, snapshot, EventType.TASK_FAILED)
        except Exception:
            pass
        return

    try:
        if task_record.cancelled.is_set():
            return

        researcher_input = {
            "researcher_messages": [
                HumanMessage(content=task_record.research_topic)
            ],
            "research_topic": task_record.research_topic,
            "requirement_ids": list(task_record.requirement_ids),
            "coverage_contract": dict(task_record.coverage_contract),
            "research_risk_profile": dict(
                task_record.research_risk_profile
            ),
        }

        if configurable.enable_docker_sandbox:
            sandbox_result = await DockerSandboxManager().run_researcher_task(
                task_record,
                config,
                researcher_input,
                runs_dir=runs_dir,
                run_id=run_id,
                event_log_enabled=event_log_enabled,
            )
            result = sandbox_result.result
        else:
            result = await asyncio.wait_for(
                execute_research(researcher_input, _config_with_task_id(config, task_record.task_id)),
                timeout=timeout,
            )

        if task_record.cancelled.is_set():
            return

        registry.update_status(
            task_record.task_id,
            TaskStatus.COMPLETED,
            phase=TaskPhase.COMPLETED,
            completed_at=time.time(),
            result=result,
            source_count=result.get("metrics", {}).get("sources_read", task_record.source_count),
        )

        await _emit_state_change(
            task_record,
            config,
            event_type=EventType.TASK_COMPLETED,
            runs_dir=runs_dir,
            run_id=run_id,
            event_log_enabled=event_log_enabled,
            phase="completed",
            data={
                "elapsed_seconds": task_record.elapsed_seconds,
                "compressed_length": len(result.get("compressed_research", "")),
            },
        )
        _clear_run_approvals_if_idle(registry, run_id)

    except asyncio.TimeoutError:
        registry.update_status(
            task_record.task_id,
            TaskStatus.TIMED_OUT,
            error_message=f"Task exceeded {timeout}s timeout.",
            completed_at=time.time(),
        )
        await _emit_state_change(
            task_record,
            config,
            event_type=EventType.TASK_TIMED_OUT,
            runs_dir=runs_dir,
            run_id=run_id,
            event_log_enabled=event_log_enabled,
            data={"timeout_seconds": timeout},
        )
        _clear_run_approvals_if_idle(registry, run_id)

    except Exception as exc:
        registry.update_status(
            task_record.task_id,
            TaskStatus.FAILED,
            error_message=str(exc),
            completed_at=time.time(),
        )
        await _emit_state_change(
            task_record,
            config,
            event_type=EventType.TASK_FAILED,
            runs_dir=runs_dir,
            run_id=run_id,
            event_log_enabled=event_log_enabled,
            data={"error": str(exc)},
        )
        _clear_run_approvals_if_idle(registry, run_id)


# ---------------------------------------------------------------------------
# Controlled executor (control queue + checkpoints)
# ---------------------------------------------------------------------------


async def run_task_with_control(
    task_record: TaskRecord,
    config: RunnableConfig,
    registry: TaskRegistry,
    execute_research: ExecuteResearchFn,
    *,
    checkpoint_manager: Optional[CheckpointManager] = None,
    runs_dir: str = ".runs",
    run_id: str = "default",
    event_log_enabled: bool = True,
    fence_token: int = 0,
) -> None:
    """Execute a research task with control-queue polling and checkpoint support."""
    configurable = Configuration.from_runnable_config(config)
    timeout = configurable.task_timeout_seconds
    registry.update_status(
        task_record.task_id,
        TaskStatus.RUNNING,
        started_at=time.time(),
    )

    try:
        await _emit_state_change(
            task_record,
            config,
            event_type=EventType.TASK_STARTED,
            runs_dir=runs_dir,
            run_id=run_id,
            event_log_enabled=event_log_enabled,
            data={"research_topic": task_record.research_topic},
        )
    except Exception as exc:  # noqa: BLE001 - state must not remain RUNNING
        registry.update_status(
            task_record.task_id,
            TaskStatus.FAILED,
            error_message=f"Unable to publish task start: {exc}",
            completed_at=time.time(),
        )
        try:
            snapshot = await get_task_state_store(configurable).update_from_record(
                task_record,
                fence_token=fence_token,
            )
            await publish_task_update(configurable, snapshot, EventType.TASK_FAILED)
        except Exception:
            pass
        return

    # --- Resume from checkpoint if available ---------------------------------
    existing_checkpoint = None
    if checkpoint_manager is not None and configurable.task_checkpoint_enabled:
        existing_checkpoint = checkpoint_manager.load(task_record.task_id)
        if existing_checkpoint is not None:
            await _emit_state_change(
                task_record,
                config,
                event_type=EventType.CHECKPOINT_LOADED,
                runs_dir=runs_dir,
                run_id=run_id,
                    event_log_enabled=event_log_enabled,
                    phase=existing_checkpoint.phase,
                    data={"completed_queries": len(existing_checkpoint.completed_queries)},
                    notify=False,
            )

    # Build initial or restored researcher state
    if existing_checkpoint is not None:
        from langchain_core.messages import messages_from_dict
        researcher_state: dict[str, Any] = {
            "researcher_messages": messages_from_dict(existing_checkpoint.messages_snapshot),
            "research_topic": (
                existing_checkpoint.research_topic
                or task_record.research_topic
            ),
            "tool_call_iterations": existing_checkpoint.tool_call_iterations,
            "memory_context": existing_checkpoint.memory_context or task_record.memory_context,
            "requirement_ids": list(
                existing_checkpoint.requirement_ids
                or task_record.requirement_ids
            ),
            "coverage_contract": dict(
                existing_checkpoint.coverage_contract
                or task_record.coverage_contract
            ),
            "research_risk_profile": dict(
                existing_checkpoint.research_risk_profile
                or task_record.research_risk_profile
            ),
            "next_step": existing_checkpoint.next_step,
            "fence_token": existing_checkpoint.fence_token,
            "committed_tool_call_ids": list(
                existing_checkpoint.committed_tool_call_ids
            ),
            "artifact_refs": list(existing_checkpoint.artifact_refs),
            "completion_decision": dict(existing_checkpoint.completion_decision),
            "compressed_research": existing_checkpoint.compressed_research,
            "raw_notes": list(existing_checkpoint.raw_notes),
            "pending_tool_results": list(
                existing_checkpoint.pending_tool_results
            ),
            "research_complete_requested": (
                existing_checkpoint.research_complete_requested
            ),
            "research_complete_succeeded": (
                existing_checkpoint.research_complete_succeeded
            ),
            "result_assessment": dict(
                existing_checkpoint.result_assessment
            ),
            "permission_denials": list(
                existing_checkpoint.permission_denials
            ),
            "candidate_registry": list(
                existing_checkpoint.candidate_registry
            ),
            "document_registry": list(
                existing_checkpoint.document_registry
            ),
            "evidence_registry": list(
                existing_checkpoint.evidence_registry
            ),
            "web_research_iterations": list(
                existing_checkpoint.web_research_iterations
            ),
            "applied_query_event_ids": list(
                existing_checkpoint.applied_query_event_ids
            ),
            "query_state_snapshot": existing_checkpoint.query_state,
        }
        task_record.phase = (
            TaskPhase.COMPRESSING
            if existing_checkpoint.phase == "compressing"
            else TaskPhase.RESEARCHING
        )
    else:
        researcher_state = {
            "researcher_messages": [
                HumanMessage(content=task_record.research_topic)
            ],
            "research_topic": task_record.research_topic,
            "memory_context": task_record.memory_context,
            "requirement_ids": list(task_record.requirement_ids),
            "coverage_contract": dict(task_record.coverage_contract),
            "research_risk_profile": dict(
                task_record.research_risk_profile
            ),
        }

    # --- Helper: save checkpoint -------------------------------------------
    async def _save_checkpoint(phase_override: Optional[str] = None) -> None:
        if checkpoint_manager is None or not configurable.task_checkpoint_enabled:
            return
        phase = phase_override or task_record.phase.value
        messages_dicts = [
            message_to_dict(m)
            for m in researcher_state.get("researcher_messages", [])
        ]
        cp = ResearcherCheckpoint(
            task_id=task_record.task_id,
            phase=phase,
            next_step=str(researcher_state.get("next_step", "model")),
            fence_token=fence_token,
            committed_tool_call_ids=list(
                researcher_state.get("committed_tool_call_ids", [])
            ),
            artifact_refs=list(researcher_state.get("artifact_refs", [])),
            completion_decision=dict(
                researcher_state.get("completion_decision", {})
            ),
            messages_snapshot=messages_dicts,
            tool_call_iterations=researcher_state.get("tool_call_iterations", 0),
            research_topic=task_record.research_topic,
            run_id=task_record.run_id,
            user_id=task_record.user_id,
            memory_context=task_record.memory_context,
            query_state=researcher_state.get("query_state_snapshot"),
            requirement_ids=list(
                researcher_state.get("requirement_ids", [])
                or task_record.requirement_ids
            ),
            coverage_contract=dict(
                researcher_state.get("coverage_contract", {})
                or task_record.coverage_contract
            ),
            research_risk_profile=dict(
                researcher_state.get("research_risk_profile", {})
                or task_record.research_risk_profile
            ),
            raw_notes=list(researcher_state.get("raw_notes", [])),
            pending_tool_results=list(
                researcher_state.get("pending_tool_results", [])
            ),
            research_complete_requested=bool(
                researcher_state.get("research_complete_requested", False)
            ),
            research_complete_succeeded=bool(
                researcher_state.get("research_complete_succeeded", False)
            ),
            result_assessment=dict(
                researcher_state.get("result_assessment", {})
            ),
            permission_denials=list(
                researcher_state.get("permission_denials", [])
            ),
            candidate_registry=list(
                researcher_state.get("candidate_registry", [])
            ),
            document_registry=list(
                researcher_state.get("document_registry", [])
            ),
            evidence_registry=list(
                researcher_state.get("evidence_registry", [])
            ),
            web_research_iterations=list(
                researcher_state.get("web_research_iterations", [])
            ),
            applied_query_event_ids=list(
                researcher_state.get("applied_query_event_ids", [])
            ),
        )
        checkpoint_manager.save(cp)
        await _emit_state_change(
            task_record,
            config,
            event_type=EventType.CHECKPOINT_SAVED,
            runs_dir=runs_dir,
            run_id=run_id,
            event_log_enabled=event_log_enabled,
            phase=phase,
            data={},
            notify=False,
        )

    async def _save_query_state(query_state: Any) -> None:
        """Bridge inner Query checkpoints into the task checkpoint file."""
        researcher_state["query_state_snapshot"] = (
            query_state.to_snapshot()
        )
        researcher_state["tool_call_iterations"] = query_state.turn
        pending_batch = query_state.pending_tool_batch
        researcher_state["committed_tool_call_ids"] = (
            list(pending_batch.committed_tool_call_ids)
            if pending_batch is not None
            else []
        )
        researcher_state["next_step"] = query_state.phase.value
        await _save_checkpoint()

    researcher_state["_query_checkpoint_callback"] = _save_query_state

    try:
        max_iterations = 3  # safety valve for update loops
        for iteration in range(max_iterations):
            if task_record.cancelled.is_set():
                await _save_checkpoint()
                return

            # Drain any pending update instructions
            instructions = list(task_record.pending_update_instructions)
            task_record.pending_update_instructions.clear()
            while not task_record.control_queue.empty():
                try:
                    msg = task_record.control_queue.get_nowait()
                    if msg.get("type") == "update" and msg["instruction"] not in instructions:
                        instructions.append(msg["instruction"])
                except asyncio.QueueEmpty:
                    break

            for instr in instructions:
                researcher_state["researcher_messages"].append(
                    HumanMessage(content=f"[Supervisor Instruction] {instr}")
                )

            await _save_checkpoint()

            for instr in instructions:
                await _emit_state_change(
                    task_record,
                    config,
                    event_type=EventType.TASK_UPDATED,
                    runs_dir=runs_dir,
                    run_id=run_id,
                    event_log_enabled=event_log_enabled,
                    data={"instruction": instr},
                    notify=False,
                )

            async def invoke_researcher() -> dict[str, Any]:
                if configurable.enable_docker_sandbox:
                    sandbox_result = await DockerSandboxManager().run_researcher_task(
                        task_record,
                        config,
                        researcher_state,
                        runs_dir=runs_dir,
                        run_id=run_id,
                        event_log_enabled=event_log_enabled,
                    )
                    return sandbox_result.result
                return await asyncio.wait_for(
                    execute_research(
                        researcher_state,
                        _config_with_task_id(config, task_record.task_id),
                    ),
                    timeout=timeout,
                )

            research_task = asyncio.create_task(invoke_researcher())
            cancellation_wait = asyncio.create_task(task_record.cancelled.wait())
            control_wait = asyncio.create_task(task_record.control_queue.get())
            done, pending = await asyncio.wait(
                {research_task, cancellation_wait, control_wait},
                return_when=asyncio.FIRST_COMPLETED,
            )

            if cancellation_wait in done:
                research_task.cancel()
                await asyncio.gather(research_task, return_exceptions=True)
                control_wait.cancel()
                await asyncio.gather(control_wait, return_exceptions=True)
                await _save_checkpoint()
                return

            if control_wait in done:
                control = control_wait.result()
                cancellation_wait.cancel()
                await asyncio.gather(cancellation_wait, return_exceptions=True)
                if control.get("type") == "update":
                    research_task.cancel()
                    await asyncio.gather(research_task, return_exceptions=True)
                    instruction = str(control["instruction"])
                    try:
                        task_record.pending_update_instructions.remove(instruction)
                    except ValueError:
                        pass
                    researcher_state["researcher_messages"].append(
                        HumanMessage(content=f"[Supervisor Instruction] {instruction}")
                    )
                    await _save_checkpoint()
                    await _emit_state_change(
                        task_record,
                        config,
                        event_type=EventType.TASK_UPDATED,
                        runs_dir=runs_dir,
                        run_id=run_id,
                        event_log_enabled=event_log_enabled,
                        data={"instruction": instruction},
                        notify=False,
                    )
                    continue
                # Domain-decision markers are informational; the governance
                # future has already resumed the active researcher.
                result = await research_task
            else:
                result = await research_task

            for task in pending:
                if task is not research_task:
                    task.cancel()
            await asyncio.gather(
                *(task for task in pending if task is not research_task),
                return_exceptions=True,
            )

            if task_record.cancelled.is_set():
                await _save_checkpoint()
                return

            # --- Success ----------------------------------------------------
            durable_result = {
                "schema_version": 2,
                "task_id": task_record.task_id,
                "research_topic": task_record.research_topic,
                "requirement_ids": list(task_record.requirement_ids),
                "coverage_contract": dict(task_record.coverage_contract),
                "research_risk_profile": dict(
                    task_record.research_risk_profile
                ),
                "compressed_research": result.get("compressed_research", ""),
                "raw_notes": result.get("raw_notes", []),
                "metrics": result.get("metrics", {}),
                "candidate_registry": result.get("candidate_registry", []),
                "document_registry": result.get("document_registry", []),
                "evidence_registry": result.get("evidence_registry", []),
                "web_research_iterations": result.get("web_research_iterations", []),
            }
            registry.update_status(
                task_record.task_id,
                TaskStatus.COMPLETED,
                phase=TaskPhase.COMPLETED,
                completed_at=time.time(),
                result=durable_result,
                source_count=durable_result.get("metrics", {}).get(
                    "sources_read", task_record.source_count
                ),
            )

            context_store = RunContextStore(
                run_id,
                runs_dir=configurable.runs_dir,
                inline_content_max_chars=configurable.query_journal_inline_content_max_chars,
            )
            lease_owner_id = (config.get("metadata") or {}).get(
                "run_lease_owner_id"
            )
            if fence_token and lease_owner_id:
                context_store.bind_fence_token(
                    fence_token,
                    str(lease_owner_id),
                    advance_manifest=False,
                )
            digest = context_store.persist_task_result(task_record.task_id, durable_result)
            task_record.result_artifact_path = (
                f"context/artifacts/research_tasks/{task_record.task_id}.json"
            )
            task_record.result_artifact_sha256 = digest

            if checkpoint_manager is not None:
                checkpoint_manager.delete(
                    task_record.task_id,
                    fence_token=fence_token,
                )

            await _emit_state_change(
                task_record,
                config,
                event_type=EventType.TASK_COMPLETED,
                runs_dir=runs_dir,
                run_id=run_id,
                event_log_enabled=event_log_enabled,
                phase="completed",
                data={
                    "elapsed_seconds": task_record.elapsed_seconds,
                    "iterations": iteration + 1,
                },
            )
            _clear_run_approvals_if_idle(registry, run_id)
            return

        raise RuntimeError(
            f"Task exceeded {max_iterations} supervisor-update restarts"
        )

    except asyncio.TimeoutError:
        registry.update_status(
            task_record.task_id,
            TaskStatus.TIMED_OUT,
            error_message=f"Task exceeded {timeout}s timeout.",
            completed_at=time.time(),
        )
        await _save_checkpoint()
        await _emit_state_change(
            task_record,
            config,
            event_type=EventType.TASK_TIMED_OUT,
            runs_dir=runs_dir,
            run_id=run_id,
            event_log_enabled=event_log_enabled,
            data={"timeout_seconds": timeout},
        )
        _clear_run_approvals_if_idle(registry, run_id)

    except Exception as exc:
        registry.update_status(
            task_record.task_id,
            TaskStatus.FAILED,
            error_message=str(exc),
            completed_at=time.time(),
        )
        await _save_checkpoint()
        await _emit_state_change(
            task_record,
            config,
            event_type=EventType.TASK_FAILED,
            runs_dir=runs_dir,
            run_id=run_id,
            event_log_enabled=event_log_enabled,
            data={"error": str(exc)},
        )
        _clear_run_approvals_if_idle(registry, run_id)
