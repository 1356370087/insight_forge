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
    store = get_task_state_store(configurable)
    snapshot = await store.update_from_record(task_record)
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
        raise


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
    new_config["metadata"] = {**(config.get("metadata") or {}), "task_id": task_id}
    return new_config


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

    await _emit_state_change(
        task_record,
        config,
        event_type=EventType.TASK_STARTED,
        runs_dir=runs_dir,
        run_id=run_id,
        event_log_enabled=event_log_enabled,
        data={"research_topic": task_record.research_topic},
    )

    try:
        if task_record.cancelled.is_set():
            return

        researcher_input = {
            "researcher_messages": [
                HumanMessage(content=task_record.research_topic)
            ],
            "research_topic": task_record.research_topic,
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
        get_domain_approval_registry().clear_run(run_id)

    except asyncio.TimeoutError:
        registry.update_status(
            task_record.task_id,
            TaskStatus.TIMED_OUT,
            error_message=f"Task exceeded {timeout}s timeout.",
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
        get_domain_approval_registry().clear_run(run_id)

    except Exception as exc:
        registry.update_status(
            task_record.task_id,
            TaskStatus.FAILED,
            error_message=str(exc),
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
        get_domain_approval_registry().clear_run(run_id)


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
) -> None:
    """Execute a research task with control-queue polling and checkpoint support."""
    configurable = Configuration.from_runnable_config(config)
    timeout = configurable.task_timeout_seconds
    registry.update_status(
        task_record.task_id,
        TaskStatus.RUNNING,
        started_at=time.time(),
    )

    await _emit_state_change(
        task_record,
        config,
        event_type=EventType.TASK_STARTED,
        runs_dir=runs_dir,
        run_id=run_id,
        event_log_enabled=event_log_enabled,
        data={"research_topic": task_record.research_topic},
    )

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
            "research_topic": task_record.research_topic,
            "tool_call_iterations": existing_checkpoint.tool_call_iterations,
            "memory_context": task_record.memory_context,
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
            messages_snapshot=messages_dicts,
            tool_call_iterations=researcher_state.get("tool_call_iterations", 0),
            research_topic=task_record.research_topic,
            run_id=task_record.run_id,
            user_id=task_record.user_id,
            memory_context=task_record.memory_context,
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

    try:
        max_iterations = 3  # safety valve for update loops
        for iteration in range(max_iterations):
            if task_record.cancelled.is_set():
                await _save_checkpoint()
                return

            # Drain any pending update instructions
            instructions: list[str] = []
            while not task_record.control_queue.empty():
                try:
                    msg = task_record.control_queue.get_nowait()
                    if msg.get("type") == "update":
                        instructions.append(msg["instruction"])
                except asyncio.QueueEmpty:
                    break

            for instr in instructions:
                researcher_state["researcher_messages"].append(
                    HumanMessage(content=f"[Supervisor Instruction] {instr}")
                )
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

            await _save_checkpoint()

            if configurable.enable_docker_sandbox:
                sandbox_result = await DockerSandboxManager().run_researcher_task(
                    task_record,
                    config,
                    researcher_state,
                    runs_dir=runs_dir,
                    run_id=run_id,
                    event_log_enabled=event_log_enabled,
                )
                result = sandbox_result.result
            else:
                result = await asyncio.wait_for(
                    execute_research(researcher_state, _config_with_task_id(config, task_record.task_id)),
                    timeout=timeout,
                )

            if task_record.cancelled.is_set():
                await _save_checkpoint()
                return

            # --- Success ----------------------------------------------------
            durable_result = {
                "task_id": task_record.task_id,
                "research_topic": task_record.research_topic,
                "compressed_research": result.get("compressed_research", ""),
                "raw_notes": result.get("raw_notes", []),
                "metrics": result.get("metrics", {}),
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
            digest = context_store.persist_task_result(task_record.task_id, durable_result)
            task_record.result_artifact_path = (
                f"context/artifacts/research_tasks/{task_record.task_id}.json"
            )
            task_record.result_artifact_sha256 = digest

            if checkpoint_manager is not None:
                checkpoint_manager.delete(task_record.task_id)

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
            get_domain_approval_registry().clear_run(run_id)
            return

    except asyncio.TimeoutError:
        registry.update_status(
            task_record.task_id,
            TaskStatus.TIMED_OUT,
            error_message=f"Task exceeded {timeout}s timeout.",
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
        get_domain_approval_registry().clear_run(run_id)

    except Exception as exc:
        registry.update_status(
            task_record.task_id,
            TaskStatus.FAILED,
            error_message=str(exc),
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
        get_domain_approval_registry().clear_run(run_id)
