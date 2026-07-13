"""Outer session engines for Deep Research.

``QueryEngine`` owns the session/protocol layer. The lower-level model/tool loop
lives in :mod:`open_deep_research.agents.query`.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage, get_buffer_string
from langchain_core.runnables import RunnableConfig

from open_deep_research.configuration import Configuration
from open_deep_research.observability import get_trace_recorder
from open_deep_research.run_context import (
    JournalCorruptedError,
    ResearchBriefPersistenceError,
    RunContextStore,
)
from open_deep_research.runtime import (
    END,
    RuntimeCommand,
    apply_update_to_state,
    coerce_command,
    normalize_messages,
)
from open_deep_research.security.inputs import validate_client_messages


def _ensure_config(config: RunnableConfig | None, fallback: RunnableConfig | None = None) -> RunnableConfig:
    merged: RunnableConfig = {"configurable": {}, "metadata": {}}
    for source in (fallback, config):
        if not source:
            continue
        merged["configurable"].update(source.get("configurable", {}))
        merged["metadata"].update(source.get("metadata", {}))
    merged["metadata"].setdefault("run_id", merged["configurable"].get("thread_id") or str(uuid.uuid4()))
    merged["configurable"].setdefault("thread_id", merged["metadata"]["run_id"])
    return merged

def _message_text(messages: list[Any]) -> str:
    normalized = [m for m in normalize_messages(messages) if isinstance(m, BaseMessage)]
    return get_buffer_string(normalized)

def _config_user_id(config: RunnableConfig) -> str | None:
    metadata = config.get("metadata", {})
    configurable = config.get("configurable", {})
    return metadata.get("user_id") or metadata.get("owner") or configurable.get("user_id")

class QueryEngine:
    """Conversation and protocol shell for the lead research agent."""

    def __init__(self, config: RunnableConfig | None = None):
        """Create a lead-agent session engine."""
        self.config = _ensure_config(config)
        self.run_id = self.config["metadata"]["run_id"]
        self.messages: list[Any] = []
        self.transcript: list[dict[str, Any]] = []
        self.total_usage: dict[str, Any] = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "retry_count": 0,
            "rate_limited_count": 0,
            "rate_429": 0.0,
            "total_llm_tool_calls": 0,
            "cache_hit_rate": 0.0,
            "cache_input_ratio": 0.0,
            "llm_output_tokens_per_second": 0.0,
            "tool_success_rate": 0.0,
        }
        self.permission_denials: list[dict[str, Any]] = []
        self.cancelled = False
        self.status = "pending"
        self.pending_human_action: dict[str, Any] | None = None
        self.human_feedback: list[dict[str, Any]] = []
        self._feedback_cursor = 0
        self._pending_action_future: asyncio.Future[dict[str, Any]] | None = None
        self._pending_action_loop: asyncio.AbstractEventLoop | None = None
        self.final_state: dict[str, Any] | None = None
        self.context_store: RunContextStore | None = None
        self.persistence_degraded = False
        self._configure_context_store()

    def _configure_context_store(self) -> None:
        """Configure the optional run-scoped persistence store."""
        configurable = Configuration.from_runnable_config(self.config)
        if not configurable.query_session_persistence_enabled:
            self.context_store = None
            return
        self.context_store = RunContextStore(
            self.run_id,
            runs_dir=configurable.runs_dir,
            inline_content_max_chars=configurable.query_journal_inline_content_max_chars,
        )

    def _clear_run_resources(self) -> None:
        """Best-effort cleanup of process-local resources owned by this run."""
        try:
            from open_deep_research.tasks.domain_approvals import (
                get_domain_approval_registry,
            )

            get_domain_approval_registry().clear_run(self.run_id)
        except Exception:
            # Cleanup must never replace the terminal result of a completed or
            # already-failed research run.
            pass

    @classmethod
    def load(
        cls,
        run_id: str,
        *,
        runs_dir: str = ".runs",
        config: RunnableConfig | None = None,
    ) -> QueryEngine:
        """Load a persisted run shell for explicit recovery."""
        bootstrap_store = RunContextStore(run_id, runs_dir=runs_dir)
        manifest = bootstrap_store.load_manifest()
        if manifest.coordination_backend != "file_mailbox":
            raise JournalCorruptedError("legacy_coordination_backend_not_resumable")
        persisted = manifest.config
        persisted_configurable = {
            key: value
            for key, value in persisted.get("configurable", {}).items()
            if value != "[REDACTED]"
        }
        persisted_metadata = {
            key: value
            for key, value in persisted.get("metadata", {}).items()
            if value != "[REDACTED]"
        }
        supplied: RunnableConfig = {
            "configurable": {
                **persisted_configurable,
                **((config or {}).get("configurable", {})),
                "thread_id": run_id,
                "runs_dir": runs_dir,
            },
            "metadata": {
                **persisted_metadata,
                **((config or {}).get("metadata", {})),
                "run_id": run_id,
            },
        }
        engine = cls(supplied)
        if engine.context_store is None:
            raise JournalCorruptedError("run_context_persistence_disabled")
        return engine

    async def resume(self) -> dict[str, Any]:
        """Resume a persisted run and return its final state."""
        async for _event in self.stream_resume():
            pass
        return self.final_state or {}

    async def _persist_update(
        self,
        *,
        channel: str,
        stage: str,
        update: dict[str, Any],
        scope: str = "main",
        record_type: str = "state_delta",
        extra: dict[str, Any] | None = None,
    ) -> None:
        if self.context_store is None:
            return
        try:
            payload = {"scope": scope, "update": update, **(extra or {})}
            await self.context_store.append(
                channel=channel,  # type: ignore[arg-type]
                record_type=record_type,  # type: ignore[arg-type]
                stage=stage,
                payload=payload,
            )
        except Exception as exc:  # noqa: BLE001 - journal is fail-open after brief
            self.persistence_degraded = True
            self.context_store.mark_persistence_degraded(exc)

    async def _persist_checkpoint(
        self,
        stage: str,
        next_stage: str,
        *,
        status: str = "running",
        channel: str = "lead",
        payload: dict[str, Any] | None = None,
    ) -> None:
        if self.context_store is None:
            return
        try:
            await self.context_store.checkpoint(
                stage,
                next_stage,
                status=status,
                channel=channel,  # type: ignore[arg-type]
                payload=payload,
            )
        except Exception as exc:  # noqa: BLE001 - checkpoint journal is fail-open
            self.persistence_degraded = True
            self.context_store.mark_persistence_degraded(exc)

    async def _persist_artifact_event(self, stage: str, path: str, sha256: str) -> None:
        if self.context_store is None:
            return
        try:
            await self.context_store.append(
                channel="lead",
                record_type="artifact_committed",
                stage=stage,
                payload={"path": path, "sha256": sha256},
            )
        except Exception as exc:  # noqa: BLE001 - artifact exists even if journal fails
            self.persistence_degraded = True
            self.context_store.mark_persistence_degraded(exc)

    async def _write_optional_text_artifact(self, stage: str, path: str, content: str) -> None:
        """Write a non-brief artifact using the configured fail-open policy."""
        if self.context_store is None or not content:
            return
        try:
            digest = self.context_store.write_text_atomic(path, content)
            await self._persist_artifact_event(stage, path, digest)
        except Exception as exc:  # noqa: BLE001 - only research_brief.md is strict
            self.persistence_degraded = True
            self.context_store.mark_persistence_degraded(exc)

    def interrupt(self) -> None:
        """Request cancellation before the next step starts."""
        self.cancelled = True
        self.status = "cancelled"

    async def submit_message(
        self,
        messages: list[Any],
        config: RunnableConfig | None = None,
    ) -> dict[str, Any]:
        """Run a complete research request and return the final state."""
        async for _event in self.stream_message(messages, config):
            pass
        return self.final_state or {}

    def _hitl_config(self) -> Configuration:
        return Configuration.from_runnable_config(self.config)

    def _hitl_enabled(self) -> bool:
        return self._hitl_config().enable_human_in_loop

    def _latest_revision_message(self) -> str | None:
        for item in reversed(self.human_feedback):
            if item.get("type") in {"plan_revision", "outline_revision"}:
                return str(item.get("message") or "")
        return None

    def _draft_research_plan(self, state: dict[str, Any]) -> str:
        brief = str(state.get("research_brief") or _message_text(state.get("messages", []))).strip()
        memory = str(state.get("memory_context") or "").strip()
        revision = self._latest_revision_message()
        parts = [
            "# Research Plan",
            "## Research objective",
            brief or "Clarify and answer the user's research request with grounded evidence.",
            "## Subquestions",
            "- What are the core claims, entities, and decision criteria in the request?",
            "- Which primary or high-authority sources can verify each important claim?",
            "- Where do sources disagree, and what uncertainty should be preserved in the report?",
            "## Expected task split",
            "- Gather authoritative background and primary sources.",
            "- Cross-check disputed or high-impact claims with independent evidence.",
            "- Synthesize findings into a report-ready evidence map.",
            "## Evidence to prioritize",
            "- Primary documentation, official data, direct publications, and timestamped source pages.",
            "- Source metadata sufficient to support citations in the final report.",
            "## Risks and uncertainties",
            "- Search APIs may miss JS-rendered, gated, or interaction-heavy pages.",
            "- Claims without direct evidence should be marked as uncertain instead of over-stated.",
        ]
        if memory:
            parts.extend(["## Memory context", memory])
        if revision:
            parts.extend(["## User-requested revision", revision])
        return "\n".join(parts)

    def _draft_report_outline(self, state: dict[str, Any]) -> str:
        brief = str(state.get("research_brief") or "Research findings").strip()
        notes = state.get("notes") or []
        evidence_count = len(notes)
        return "\n".join([
            "# Report Outline",
            "## 1. Executive answer",
            brief,
            "## 2. Key findings and evidence",
            f"Summarize and cite {evidence_count} collected research note(s).",
            "## 3. Uncertainties and conflicting evidence",
            "Call out unresolved questions, weak sources, and areas requiring caution.",
            "## 4. Recommendations or next steps",
            "Translate the evidence into the user's requested decision or deliverable.",
        ])

    def _open_human_action(self, action_type: str, payload: dict[str, Any]) -> dict[str, Any]:
        loop = asyncio.get_running_loop()
        self._pending_action_loop = loop
        self._pending_action_future = loop.create_future()
        self.pending_human_action = {
            "action_id": str(uuid.uuid4()),
            "type": action_type,
            "payload": payload,
            "created_at": time.time(),
        }
        return self.pending_human_action

    async def _wait_for_human_action(self) -> dict[str, Any]:
        if self._pending_action_future is None:
            raise RuntimeError("No pending human action")
        decision = await self._pending_action_future
        self.pending_human_action = None
        self._pending_action_future = None
        self._pending_action_loop = None
        return decision

    def handle_human_action(self, action_id: str, action: str, message: str = "") -> dict[str, Any]:
        """Resolve a pending plan or outline approval action."""
        if action not in {"approve", "revise", "cancel"}:
            raise ValueError("Unsupported human action")
        pending = self.pending_human_action
        if not pending or pending.get("action_id") != action_id:
            raise ValueError("No matching pending human action")
        if self._pending_action_future is None or self._pending_action_future.done():
            raise ValueError("Pending human action is not awaiting input")
        decision = {"action": action, "message": message or "", "action_id": action_id}

        def set_result() -> None:
            if self._pending_action_future and not self._pending_action_future.done():
                self._pending_action_future.set_result(decision)

        if self._pending_action_loop and self._pending_action_loop.is_running():
            self._pending_action_loop.call_soon_threadsafe(set_result)
        else:
            set_result()
        return {"status": "accepted", "action": action}

    async def submit_feedback(self, feedback: dict[str, Any]) -> dict[str, Any]:
        """Accept user direction or evidence questions while a run is active."""
        feedback_type = str(feedback.get("type") or "direction")
        if feedback_type not in {"direction", "evidence_question"}:
            raise ValueError("Unsupported feedback type")
        message = str(feedback.get("message") or "").strip()
        if not message:
            raise ValueError("Feedback message is required")
        entry = {
            "feedback_id": str(uuid.uuid4()),
            "type": feedback_type,
            "message": message,
            "task_id": feedback.get("task_id"),
            "source_url": feedback.get("source_url"),
            "claim_text": feedback.get("claim_text"),
            "created_at": time.time(),
        }
        self.human_feedback.append(entry)
        await self._queue_task_feedback_if_possible(entry)
        return {"status": "accepted", "feedback_id": entry["feedback_id"]}

    async def _queue_task_feedback_if_possible(self, entry: dict[str, Any]) -> None:
        cfg = self._hitl_config()
        task_id = entry.get("task_id")
        if cfg.hitl_feedback_mode != "task_queue" or not task_id:
            return
        try:
            from open_deep_research.tasks.registry import TaskStatus, get_task_registry
        except Exception:  # noqa: BLE001 - task registry is optional for sync research paths
            return
        record = get_task_registry().get(str(task_id))
        if (
            record is None
            or record.run_id != self.run_id
            or record.status != TaskStatus.RUNNING
        ):
            return
        await record.control_queue.put({"type": "update", "instruction": self._format_feedback_instruction(entry)})
        entry["queued_to_task"] = True

    def _format_feedback_instruction(self, entry: dict[str, Any]) -> str:
        label = "Evidence question" if entry.get("type") == "evidence_question" else "User feedback"
        details = [f"{label}: {entry.get('message', '')}"]
        if entry.get("source_url"):
            details.append(f"Source URL: {entry['source_url']}")
        if entry.get("claim_text"):
            details.append(f"Claim text: {entry['claim_text']}")
        return "\n".join(details)

    def _drain_human_feedback(self, supervisor_state: dict[str, Any]) -> None:
        new_feedback = self.human_feedback[self._feedback_cursor:]
        if not new_feedback:
            return
        messages = list(supervisor_state.get("supervisor_messages", []))
        for entry in new_feedback:
            prefix = "[Evidence Question]" if entry.get("type") == "evidence_question" else "[User Feedback]"
            messages.append(HumanMessage(content=f"{prefix} {self._format_feedback_instruction(entry)}"))
        supervisor_state["supervisor_messages"] = messages
        supervisor_state["human_feedback"] = list(self.human_feedback)
        self._feedback_cursor = len(self.human_feedback)

    async def _maybe_await_plan_approval(self, state: dict[str, Any]) -> AsyncIterator[dict[str, Any]]:
        if not self._hitl_enabled():
            return
        cfg = self._hitl_config()
        revisions = 0
        if not cfg.hitl_require_plan_approval:
            plan = self._draft_research_plan(state)
            state["research_plan"] = plan
            state["approved_research_plan"] = plan
            state["human_feedback"] = list(self.human_feedback)
            return
        while True:
            plan = self._draft_research_plan(state)
            state["research_plan"] = plan
            state["human_feedback"] = list(self.human_feedback)
            self.status = "awaiting_plan_approval"
            pending = self._open_human_action("plan_approval", {"research_plan": plan})
            yield self._event(
                "hitl.plan_pending",
                {
                    "run_id": self.run_id,
                    "status": self.status,
                    "research_plan": plan,
                    "pending_human_action": pending,
                },
            )
            decision = await self._wait_for_human_action()
            if decision["action"] == "approve":
                state["approved_research_plan"] = plan
                state["pending_human_action"] = None
                self.status = "running"
                yield self._event("hitl.plan_approved", {"run_id": self.run_id, "status": self.status})
                return
            if decision["action"] == "cancel":
                self.cancelled = True
                self.status = "cancelled"
                yield self._event("hitl.cancelled", {"run_id": self.run_id, "status": self.status})
                return
            revisions += 1
            if revisions > cfg.hitl_max_plan_revisions:
                raise RuntimeError("Research plan revision limit exceeded")
            self.human_feedback.append({
                "feedback_id": str(uuid.uuid4()),
                "type": "plan_revision",
                "message": decision.get("message", ""),
                "created_at": time.time(),
            })
            state["human_feedback"] = list(self.human_feedback)
            self.status = "running"
            yield self._event(
                "hitl.plan_revised",
                {"run_id": self.run_id, "status": self.status, "revision_count": revisions},
            )

    async def _maybe_await_outline_approval(self, state: dict[str, Any]) -> AsyncIterator[dict[str, Any]]:
        if not self._hitl_enabled():
            return
        cfg = self._hitl_config()
        outline = self._draft_report_outline(state)
        state["report_outline"] = outline
        if not cfg.hitl_require_outline_approval:
            return
        while True:
            self.status = "awaiting_outline_approval"
            pending = self._open_human_action("outline_approval", {"report_outline": outline})
            yield self._event(
                "hitl.outline_pending",
                {
                    "run_id": self.run_id,
                    "status": self.status,
                    "report_outline": outline,
                    "pending_human_action": pending,
                },
            )
            decision = await self._wait_for_human_action()
            if decision["action"] == "approve":
                state["pending_human_action"] = None
                self.status = "running"
                yield self._event("hitl.outline_approved", {"run_id": self.run_id, "status": self.status})
                return
            if decision["action"] == "cancel":
                self.cancelled = True
                self.status = "cancelled"
                yield self._event("hitl.cancelled", {"run_id": self.run_id, "status": self.status})
                return
            self.human_feedback.append({
                "feedback_id": str(uuid.uuid4()),
                "type": "outline_revision",
                "message": decision.get("message", ""),
                "created_at": time.time(),
            })
            state["human_feedback"] = list(self.human_feedback)
            outline = self._draft_report_outline(state)
            if decision.get("message"):
                outline = f"{outline}\n## User-requested outline revision\n{decision['message']}"
            state["report_outline"] = outline
            self.status = "running"

    def _cancelled_state(self, state: dict[str, Any]) -> dict[str, Any]:
        result = {
            "status": "cancelled",
            "usage": self._usage_subset(self.total_usage),
            "metrics": self._metrics_subset(self.total_usage),
            "permission_denials": self.permission_denials,
        }
        self.final_state = {**state, "result": result}
        return self.final_state

    @staticmethod
    def _usage_subset(total: dict[str, Any]) -> dict[str, Any]:
        """Token-only view of a finish_run summary (backward-compatible usage payload)."""
        return {
            "input_tokens": total.get("input_tokens", 0),
            "output_tokens": total.get("output_tokens", 0),
            "total_tokens": total.get("total_tokens", 0),
            "cached_input_tokens": total.get("cached_input_tokens", 0),
            "cache_creation_input_tokens": total.get("cache_creation_input_tokens", 0),
            "reasoning_tokens": total.get("reasoning_tokens", 0),
            "estimated_cost_usd": total.get("estimated_cost_usd", 0.0),
        }

    @staticmethod
    def _metrics_subset(total: dict[str, Any]) -> dict[str, Any]:
        """Retry/429 view of a finish_run summary (the new metrics payload)."""
        return {
            "retry_count": total.get("retry_count", 0),
            "rate_limit_events": total.get("rate_limit_events", 0),
            "rate_limited_count": total.get("rate_limited_count", 0),
            "terminal_rate_limited_count": total.get("terminal_rate_limited_count", 0),
            "rate_429": total.get("rate_429", 0.0),
            "total_llm_tool_calls": total.get("total_llm_tool_calls", 0),
            "attempt_count": total.get("attempt_count", 0),
            "llm_call_count": total.get("llm_call_count", 0),
            "cache_eligible_count": total.get("cache_eligible_count", 0),
            "cache_hit_count": total.get("cache_hit_count", 0),
            "cache_hit_rate": total.get("cache_hit_rate", 0.0),
            "cache_input_ratio": total.get("cache_input_ratio", 0.0),
            "llm_output_input_ratio": total.get("llm_output_input_ratio", 0.0),
            "llm_reasoning_output_ratio": total.get(
                "llm_reasoning_output_ratio", 0.0
            ),
            "llm_output_tokens_per_second": total.get(
                "llm_output_tokens_per_second", 0.0
            ),
            "tool_call_count": total.get("tool_call_count", 0),
            "tool_success_count": total.get("tool_success_count", 0),
            "tool_success_rate": total.get("tool_success_rate", 0.0),
            "empty_tool_result_count": total.get("empty_tool_result_count", 0),
            "zero_source_search_count": total.get("zero_source_search_count", 0),
        }

    async def stream_message(
        self,
        messages: list[Any],
        config: RunnableConfig | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream protocol events for a complete research request."""
        validate_client_messages(messages)
        self.config = _ensure_config(config, self.config)
        self.run_id = self.config["metadata"]["run_id"]
        self._configure_context_store()
        self.status = "running"
        state: dict[str, Any] = {
            "messages": normalize_messages(messages),
            "human_feedback": list(self.human_feedback),
        }
        self.messages = state["messages"]
        if self.context_store is not None:
            try:
                self.context_store.initialize(_config_user_id(self.config), self.config)
                await self._persist_update(
                    channel="lead",
                    stage="received",
                    update={
                        "messages": {"type": "override", "value": state["messages"]},
                        "human_feedback": {"type": "override", "value": state["human_feedback"]},
                    },
                    extra={
                        "owner_id": _config_user_id(self.config),
                        "config": self.context_store._safe_config(self.config),  # noqa: SLF001
                    },
                )
                await self._persist_checkpoint("received", "summarize_messages")
            except Exception as exc:  # noqa: BLE001 - brief persistence remains the strict gate
                self.persistence_degraded = True
                self.context_store.mark_persistence_degraded(exc)
        async for event in self._stream_execution(state, "summarize_messages"):
            yield event

    async def stream_resume(self) -> AsyncIterator[dict[str, Any]]:
        """Replay a persisted Query run and continue from its stable checkpoint."""
        if self.context_store is None:
            raise JournalCorruptedError("run_context_persistence_disabled")
        replay = self.context_store.replay()
        expected_owner = replay.manifest.owner_id
        current_owner = _config_user_id(self.config)
        if expected_owner and current_owner != expected_owner:
            raise PermissionError("run_owner_mismatch")
        if replay.manifest.status == "completed":
            raise RuntimeError("run_already_completed")
        if replay.manifest.status == "cancelled" or replay.manifest.next_stage == "cancelled":
            raise RuntimeError("run_not_recoverable")
        self.persistence_degraded = replay.manifest.persistence_degraded
        state = replay.state
        self.messages = list(state.get("messages", []))
        self.human_feedback = list(state.get("human_feedback", []))
        self._feedback_cursor = len(self.human_feedback)
        self.status = "running"
        if state.get("enable_async_research"):
            from open_deep_research.agents import deep_researcher as graph

            await graph.restore_async_research_tasks(self.config)
        try:
            self.context_store._update_manifest(  # noqa: SLF001 - same persistence boundary
                status="running",
                recovered_from_degraded_persistence=replay.manifest.persistence_degraded,
            )
        except Exception as exc:  # noqa: BLE001
            self.persistence_degraded = True
            self.context_store.mark_persistence_degraded(exc)
        async for event in self._stream_execution(
            state,
            replay.manifest.next_stage,
            restored_supervisor_state=replay.supervisor_state or None,
            recovered=True,
        ):
            yield event

    async def _stream_execution(
        self,
        state: dict[str, Any],
        start_stage: str,
        *,
        restored_supervisor_state: dict[str, Any] | None = None,
        recovered: bool = False,
    ) -> AsyncIterator[dict[str, Any]]:
        """Execute or resume the outer Query pipeline from a stable stage."""
        recorder = get_trace_recorder(self.config)
        run_metadata = {
            "thread_id": self.config.get("configurable", {}).get("thread_id"),
            "message_count": len(state["messages"]),
            "recovered": recovered,
        }

        with recorder.start_run(
            self.run_id,
            name="lead.run",
            user_id=_config_user_id(self.config),
            metadata=run_metadata,
            input_payload=state.get("messages", []),
        ):
            event_name = "run.resumed" if recovered else "run.started"
            yield self._event(event_name, {"run_id": self.run_id, "recovered": recovered})
            try:
                from open_deep_research.agents import deep_researcher as graph
                stage = start_stage

                if stage == "summarize_messages":
                    await self._run_node("summarize_messages", graph.summarize_messages, state)
                    await self._persist_checkpoint("messages_compacted", "memory_recall")
                    yield self._event("lead.message", {"stage": "summarize_messages"})
                    stage = "memory_recall"

                if stage == "memory_recall":
                    await self._run_node("memory_recall", graph.memory_recall, state)
                    await self._persist_checkpoint("memory_recalled", "clarify_with_user")
                    yield self._event("lead.message", {"stage": "memory_recall"})
                    stage = "clarify_with_user"

                if stage == "clarify_with_user":
                    clarify = await self._run_node("clarify_with_user", graph.clarify_with_user, state)
                    yield self._event("lead.message", {"stage": "clarify_with_user", "goto": clarify.goto})
                    if clarify.goto == END:
                        await self._persist_checkpoint("clarified", "completed", status="completed")
                        result_text = _message_text(state.get("messages", []))
                        async for event in self._finish_success(state, recorder, result_text):
                            yield event
                        return
                    await self._persist_checkpoint("clarified", "write_research_brief")
                    stage = "write_research_brief"

                if stage == "write_research_brief":
                    await self._run_node("write_research_brief", graph.write_research_brief, state)
                    if self.context_store is not None:
                        # Brief persistence is intentionally the only strict local write.
                        try:
                            if not self.context_store.manifest_path.exists():
                                self.context_store.initialize(_config_user_id(self.config), self.config)
                            digest = self.context_store.persist_research_brief(str(state.get("research_brief", "")))
                        except ResearchBriefPersistenceError:
                            raise
                        except Exception as exc:
                            raise ResearchBriefPersistenceError("research_brief_persistence_failed") from exc
                        await self._persist_artifact_event("research_brief_written", "research_brief.md", digest)
                    await self._persist_checkpoint("research_brief_written", "plan_approval")
                    yield self._event("lead.message", {"stage": "write_research_brief"})
                    stage = "plan_approval"

                if self.context_store is not None and self.context_store.brief_path.exists():
                    state["research_brief"] = self.context_store.load_research_brief()

                if stage == "plan_approval":
                    with recorder.start_span(
                        name="node.plan_approval",
                        kind="agent",
                        agent_role="lead",
                        attributes={"hitl_enabled": self._hitl_enabled()},
                    ):
                        async for hitl_event in self._maybe_await_plan_approval(state):
                            yield hitl_event
                    if self.cancelled:
                        await self._persist_checkpoint("cancelled", "cancelled", status="cancelled")
                        self.total_usage = recorder.finish_run(self.run_id, "cancelled")
                        self._cancelled_state(state)
                        yield self._event(
                            "run.cancelled",
                            {
                                "run_id": self.run_id,
                                "usage": self._usage_subset(self.total_usage),
                                "metrics": self._metrics_subset(self.total_usage),
                            },
                        )
                        return
                    approved_plan = str(state.get("approved_research_plan") or "")
                    await self._write_optional_text_artifact(
                        "plan_approved", "approved_plan.md", approved_plan
                    )
                    await self._persist_update(
                        channel="lead",
                        stage="plan_approved",
                        update={
                            "research_plan": state.get("research_plan"),
                            "approved_research_plan": state.get("approved_research_plan"),
                            "human_feedback": {"type": "override", "value": list(self.human_feedback)},
                        },
                    )
                    await self._persist_checkpoint("plan_approved", "supervisor.supervisor")
                    stage = "supervisor.supervisor"

                if stage.startswith("supervisor."):
                    supervisor_step = stage.split(".", 1)[1]
                    if restored_supervisor_state is None and supervisor_step == "supervisor":
                        # Preserve the original overridable method contract used by embedders/tests.
                        supervisor_update = await self._run_supervisor(state)
                    else:
                        supervisor_update = await self._run_supervisor(
                            state,
                            restored_state=restored_supervisor_state,
                            start_step=supervisor_step,
                        )
                    apply_update_to_state(state, supervisor_update)
                    await self._persist_update(
                        channel="lead",
                        stage="research_complete",
                        update=supervisor_update,
                    )
                    state["human_feedback"] = list(self.human_feedback)
                    await self._persist_task_outputs(state)
                    await self._persist_checkpoint("research_complete", "outline_approval")
                    yield self._event("lead.message", {"stage": "research_supervisor"})
                    stage = "outline_approval"

                if stage == "outline_approval":
                    with recorder.start_span(
                        name="node.outline_approval",
                        kind="agent",
                        agent_role="lead",
                        attributes={"hitl_enabled": self._hitl_enabled()},
                    ):
                        async for hitl_event in self._maybe_await_outline_approval(state):
                            yield hitl_event
                    if self.cancelled:
                        await self._persist_checkpoint("cancelled", "cancelled", status="cancelled")
                        self.total_usage = recorder.finish_run(self.run_id, "cancelled")
                        self._cancelled_state(state)
                        yield self._event(
                            "run.cancelled",
                            {
                                "run_id": self.run_id,
                                "usage": self._usage_subset(self.total_usage),
                                "metrics": self._metrics_subset(self.total_usage),
                            },
                        )
                        return
                    outline = str(state.get("report_outline") or "")
                    await self._write_optional_text_artifact(
                        "outline_approved", "report_outline.md", outline
                    )
                    await self._persist_update(
                        channel="lead",
                        stage="outline_approved",
                        update={"report_outline": state.get("report_outline")},
                    )
                    await self._persist_checkpoint("outline_approved", "final_report_generation")
                    stage = "final_report_generation"

                if stage == "final_report_generation":
                    if self.context_store is not None:
                        state["research_brief"] = self.context_store.load_research_brief()
                    with recorder.start_span(name="node.final_report_generation", kind="agent", agent_role="lead"):
                        report_update = await graph.final_report_generation(state, self.config)
                    apply_update_to_state(state, report_update)
                    await self._persist_update(
                        channel="lead",
                        stage="report_generated",
                        update=report_update,
                    )
                    await self._write_optional_text_artifact(
                        "report_generated", "final_report.md", str(state.get("final_report", ""))
                    )
                    await self._persist_checkpoint("report_generated", "memory_extract_and_write")
                    yield self._event("report.completed", {"run_id": self.run_id})
                    stage = "memory_extract_and_write"

                if stage == "memory_extract_and_write":
                    with recorder.start_span(name="node.memory_extract_and_write", kind="agent", agent_role="lead"):
                        memory_cmd = coerce_command(
                            await graph.memory_extract_and_write(state, self.config),
                            default_goto=END,
                        )
                    apply_update_to_state(state, memory_cmd.update)
                    await self._persist_update(
                        channel="lead",
                        stage="memory_written",
                        update=memory_cmd.update,
                    )
                    await self._persist_checkpoint("memory_written", "completed")

                async for event in self._finish_success(state, recorder, str(state.get("final_report", ""))):
                    yield event
                return
            except Exception as exc:
                from open_deep_research.tasks.teammate_pool import (
                    shutdown_teammate_pool,
                )

                try:
                    await shutdown_teammate_pool(self.config)
                except Exception:
                    pass
                self._clear_run_resources()
                if self.cancelled:
                    self.status = "cancelled"
                    await self._persist_checkpoint("cancelled", "cancelled", status="cancelled")
                    self.total_usage = recorder.finish_run(self.run_id, "cancelled")
                    self._cancelled_state(state)
                    yield self._event(
                        "run.cancelled",
                        {
                            "run_id": self.run_id,
                            "usage": self._usage_subset(self.total_usage),
                            "metrics": self._metrics_subset(self.total_usage),
                        },
                    )
                    return
                self.status = "failed"
                if self.context_store is not None:
                    try:
                        await self.context_store.append(
                            channel="lead",
                            record_type="run_status",
                            stage="failed",
                            payload={"status": "failed", "error": str(exc)},
                        )
                        # Preserve the last stable next_stage for explicit recovery.
                        self.context_store._update_manifest(  # noqa: SLF001
                            status="failed",
                            result={"status": "error", "error": str(exc)},
                        )
                    except Exception:
                        pass
                self.total_usage = recorder.finish_run(self.run_id, "error", str(exc))
                self.final_state = {
                    **state,
                    "result": {
                        "status": "error",
                        "error": str(exc),
                        "usage": self._usage_subset(self.total_usage),
                        "metrics": self._metrics_subset(self.total_usage),
                        "permission_denials": self.permission_denials,
                        "persistence_degraded": self.persistence_degraded,
                    },
                }
                yield self._event(
                    "run.failed",
                    {
                        "run_id": self.run_id,
                        "error": str(exc),
                        "usage": self._usage_subset(self.total_usage),
                        "metrics": self._metrics_subset(self.total_usage),
                    },
                )

    async def _finish_success(
        self,
        state: dict[str, Any],
        recorder: Any,
        result_text: str,
    ) -> AsyncIterator[dict[str, Any]]:
        """Finish a successful run, persist terminal status, and emit its event."""
        from open_deep_research.tasks.teammate_pool import shutdown_teammate_pool

        await shutdown_teammate_pool(self.config)
        self._clear_run_resources()
        recorder.active_span().set_output(result_text)
        self.total_usage = recorder.finish_run(self.run_id, "success")
        result = {
            "status": "success",
            "result": result_text,
            "usage": self._usage_subset(self.total_usage),
            "metrics": self._metrics_subset(self.total_usage),
            "permission_denials": self.permission_denials,
            "persistence_degraded": self.persistence_degraded,
        }
        report_artifacts = state.get("report_artifacts")
        if report_artifacts:
            result["artifacts"] = report_artifacts
        self.status = "completed"
        self.final_state = {**state, "result": result}
        await self._persist_checkpoint("completed", "completed", status="completed")
        if self.context_store is not None:
            try:
                self.context_store._update_manifest(  # noqa: SLF001
                    status="completed",
                    result={"status": "success"},
                    final_artifacts={"final_report": "final_report.md"} if state.get("final_report") else {},
                )
            except Exception as exc:  # noqa: BLE001
                self.persistence_degraded = True
                self.context_store.mark_persistence_degraded(exc)
        yield self._event(
            "run.completed",
            {
                "run_id": self.run_id,
                "status": "success",
                "result": result,
                "usage": self._usage_subset(self.total_usage),
                "metrics": self._metrics_subset(self.total_usage),
            },
        )

    async def _persist_task_outputs(self, state: dict[str, Any]) -> None:
        """Persist completed async task outputs and their findings manifest."""
        if self.context_store is None:
            return
        outputs = list(state.get("completed_task_outputs", []))
        refs: list[dict[str, Any]] = []
        for index, output in enumerate(outputs):
            task_id = str(output.get("task_id") or f"task-{index}")
            try:
                digest = self.context_store.persist_task_result(task_id, output)
                refs.append({"task_id": task_id, "path": f"artifacts/research_tasks/{task_id}.json", "sha256": digest})
            except Exception as exc:  # noqa: BLE001
                self.persistence_degraded = True
                self.context_store.mark_persistence_degraded(exc)
        if refs:
            try:
                self.context_store.write_json_atomic("findings_manifest.json", {"tasks": refs})
            except Exception as exc:  # noqa: BLE001
                self.persistence_degraded = True
                self.context_store.mark_persistence_degraded(exc)

    async def _run_node(self, name: str, node: Any, state: dict[str, Any]) -> RuntimeCommand:
        if self.cancelled:
            raise RuntimeError("Run cancelled")
        recorder = get_trace_recorder(self.config)
        with recorder.start_span(name=f"node.{name}", kind="agent", agent_role="lead"):
            result = await node(state, self.config)
        command = coerce_command(result)
        apply_update_to_state(state, command.update)
        record_type = "context_compacted" if "conversation_summary" in command.update else "state_delta"
        await self._persist_update(
            channel="lead",
            stage=name,
            update=command.update,
            record_type=record_type,
            extra={"summary": command.update.get("conversation_summary")} if record_type == "context_compacted" else None,
        )
        self._record("node", {"name": name, "goto": command.goto})
        return command

    async def _run_supervisor(
        self,
        main_state: dict[str, Any],
        *,
        restored_state: dict[str, Any] | None = None,
        start_step: str = "supervisor",
    ) -> dict[str, Any]:
        from open_deep_research.agents import deep_researcher as graph

        base_state: dict[str, Any] = {
            "supervisor_messages": list(main_state.get("supervisor_messages", [])),
            "research_brief": main_state.get("research_brief", ""),
            "notes": list(main_state.get("notes", [])),
            "research_iterations": 0,
            "raw_notes": list(main_state.get("raw_notes", [])),
            "candidate_registry": list(main_state.get("candidate_registry", [])),
            "document_registry": list(main_state.get("document_registry", [])),
            "evidence_registry": list(main_state.get("evidence_registry", [])),
            "web_research_iterations": list(main_state.get("web_research_iterations", [])),
            "enable_async_research": main_state.get("enable_async_research", False),
            "memory_context": main_state.get("memory_context"),
            "approved_research_plan": main_state.get("approved_research_plan"),
            "human_feedback": list(self.human_feedback),
            "processed_mailbox_message_ids": list(
                main_state.get("processed_mailbox_message_ids", [])
            ),
        }
        supervisor_state = {**base_state, **(restored_state or {})}
        # The authoritative brief always wins over journal/state caches.
        if self.context_store is not None and self.context_store.brief_path.exists():
            supervisor_state["research_brief"] = self.context_store.load_research_brief()
        if not restored_state:
            await self._persist_update(
                channel="supervisor",
                stage="supervisor_running",
                scope="supervisor",
                update={
                    "supervisor_messages": {
                        "type": "override",
                        "value": supervisor_state["supervisor_messages"],
                    },
                    "research_brief": supervisor_state["research_brief"],
                    "research_iterations": 0,
                    "enable_async_research": supervisor_state["enable_async_research"],
                    "memory_context": supervisor_state.get("memory_context"),
                    "approved_research_plan": supervisor_state.get("approved_research_plan"),
                },
            )
        next_step = END if start_step == END else start_step
        recorder = get_trace_recorder(self.config)
        while next_step != END:
            if self.cancelled:
                raise RuntimeError("Run cancelled")
            with recorder.start_span(
                name=f"supervisor.{next_step}",
                kind="agent",
                agent_role="supervisor",
                attributes={"iteration": supervisor_state.get("research_iterations", 0)},
            ):
                if next_step == "supervisor":
                    self._drain_human_feedback(supervisor_state)
                    compacted = await graph.compact_query_context(
                        list(supervisor_state.get("supervisor_messages", [])),
                        research_brief=str(supervisor_state.get("research_brief", "")),
                        channel="supervisor",
                        config=self.config,
                    )
                    if compacted is not None:
                        compact_update = {
                            "supervisor_messages": {
                                "type": "override",
                                "value": compacted["messages"],
                            }
                        }
                        apply_update_to_state(supervisor_state, compact_update)
                        await self._persist_update(
                            channel="supervisor",
                            stage="supervisor_running",
                            scope="supervisor",
                            update=compact_update,
                            record_type="context_compacted",
                            extra={
                                "summary": compacted["summary"],
                                "recent_messages": compacted["recent_messages"],
                            },
                        )
                    command = coerce_command(
                        await graph.supervisor(supervisor_state, self.config),
                        default_goto="supervisor_tools",
                    )
                elif next_step == "supervisor_tools":
                    command = coerce_command(
                        await graph.supervisor_tools(supervisor_state, self.config),
                        default_goto="supervisor",
                    )
                else:
                    raise RuntimeError(f"Unknown supervisor step: {next_step}")
            apply_update_to_state(supervisor_state, command.update)
            await self._persist_update(
                channel="supervisor",
                stage="supervisor_running",
                scope="supervisor",
                update=command.update,
            )
            pending_acks = list(command.update.get("pending_mailbox_acks", []))
            if pending_acks:
                from open_deep_research.configuration import Configuration
                from open_deep_research.tasks.coordination import ack_lead_updates

                configurable = Configuration.from_runnable_config(self.config)
                for ack in pending_acks:
                    await ack_lead_updates(
                        configurable,
                        run_id=str(ack["run_id"]),
                        consumer_id=str(ack["consumer_id"]),
                        message_ids=list(ack["message_ids"]),
                    )
                supervisor_state["pending_mailbox_acks"] = []
            self._record("supervisor", {"step": next_step, "goto": command.goto})
            next_step = command.goto
            await self._persist_checkpoint(
                "supervisor_running",
                f"supervisor.{next_step}",
                channel="supervisor",
                payload={"supervisor_step": next_step},
            )
        return {
            "supervisor_messages": {"type": "override", "value": supervisor_state.get("supervisor_messages", [])},
            "notes": {"type": "override", "value": supervisor_state.get("notes", [])},
            "raw_notes": {"type": "override", "value": supervisor_state.get("raw_notes", [])},
            "candidate_registry": {"type": "override", "value": supervisor_state.get("candidate_registry", [])},
            "document_registry": {"type": "override", "value": supervisor_state.get("document_registry", [])},
            "evidence_registry": {"type": "override", "value": supervisor_state.get("evidence_registry", [])},
            "web_research_iterations": {"type": "override", "value": supervisor_state.get("web_research_iterations", [])},
            "completed_task_outputs": {
                "type": "override",
                "value": supervisor_state.get("completed_task_outputs", []),
            },
            "processed_mailbox_message_ids": {
                "type": "override",
                "value": supervisor_state.get("processed_mailbox_message_ids", []),
            },
            "research_brief": supervisor_state.get("research_brief", main_state.get("research_brief", "")),
            "approved_research_plan": supervisor_state.get("approved_research_plan"),
            "human_feedback": {"type": "override", "value": list(self.human_feedback)},
        }

    def _event(self, event_type: str, data: dict[str, Any]) -> dict[str, Any]:
        payload = {"event": event_type, "data": data}
        self._record(event_type, data)
        return payload

    def _record(self, event_type: str, data: dict[str, Any]) -> None:
        record = {"ts": time.time(), "event": event_type, "data": data}
        self.transcript.append(record)
        configurable = Configuration.from_runnable_config(self.config)
        if not configurable.event_log_enabled:
            return
        path = Path(configurable.runs_dir) / self.run_id / "transcript.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

class ResearcherQueryEngine:
    """Runs a single focused Researcher with a clean context window."""

    def __init__(self, config: RunnableConfig | None = None):
        """Create a researcher engine."""
        self.config = _ensure_config(config)

    async def run_topic(
        self,
        research_topic: str,
        *,
        memory_context: str | None = None,
        config: RunnableConfig | None = None,
    ) -> dict[str, Any]:
        """Run one focused topic in a clean researcher context."""
        from langchain_core.messages import HumanMessage

        state = {
            "researcher_messages": [HumanMessage(content=research_topic)],
            "research_topic": research_topic,
            "memory_context": memory_context,
        }
        return await self.ainvoke(state, config)

    async def ainvoke(
        self,
        state: dict[str, Any],
        config: RunnableConfig | None = None,
    ) -> dict[str, Any]:
        """Invoke the researcher loop with an explicit researcher state."""
        from open_deep_research.agents import deep_researcher as graph

        cfg = _ensure_config(config, self.config)
        researcher_state = dict(state)
        researcher_state["researcher_messages"] = normalize_messages(
            researcher_state.get("researcher_messages", []),
        )
        recorder = get_trace_recorder(cfg)
        topic = str(researcher_state.get("research_topic", ""))
        with recorder.start_span(
            name="researcher.topic",
            kind="agent",
            agent_role="researcher",
            attributes={"research_topic": topic[:500]},
        ):
            next_step = "researcher"
            while next_step != END:
                task_id = str(cfg.get("metadata", {}).get("task_id", ""))
                if task_id:
                    from langchain_core.messages import HumanMessage

                    from open_deep_research.tasks.registry import get_task_registry

                    task_record = get_task_registry().get(task_id)
                    expected_run_id = str(cfg.get("metadata", {}).get("run_id", "default"))
                    if task_record is not None and task_record.run_id == expected_run_id:
                        if task_record.cancelled.is_set():
                            return {
                                **researcher_state,
                                "cancelled": True,
                                "compressed_research": "",
                                "raw_notes": [],
                            }
                        while not task_record.control_queue.empty():
                            control = task_record.control_queue.get_nowait()
                            if control.get("type") == "update":
                                researcher_state.setdefault("researcher_messages", []).append(
                                    HumanMessage(
                                        content=f"[Supervisor Instruction] {control['instruction']}"
                                    )
                                )
                if next_step == "researcher":
                    command = coerce_command(
                        await graph.researcher(researcher_state, cfg),
                        default_goto="researcher_tools",
                    )
                elif next_step == "researcher_tools":
                    command = coerce_command(
                        await graph.researcher_tools(researcher_state, cfg),
                        default_goto="researcher",
                    )
                elif next_step == "assess_research_results":
                    command = coerce_command(
                        await graph.assess_research_results(researcher_state, cfg),
                        default_goto="researcher",
                    )
                elif next_step == "compress_research":
                    update = await graph.compress_research(researcher_state, cfg)
                    apply_update_to_state(researcher_state, update)
                    return researcher_state
                else:
                    raise RuntimeError(f"Unknown researcher step: {next_step}")
                apply_update_to_state(researcher_state, command.update)
                next_step = command.goto
        return researcher_state
