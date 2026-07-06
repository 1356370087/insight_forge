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
from open_deep_research.runtime import (
    END,
    RuntimeCommand,
    apply_update_to_state,
    coerce_command,
    normalize_messages,
)


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
        if record is None or record.status != TaskStatus.RUNNING:
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
        result = {"status": "cancelled", "permission_denials": self.permission_denials}
        self.final_state = {**state, "result": result}
        return self.final_state

    @staticmethod
    def _usage_subset(total: dict[str, Any]) -> dict[str, int]:
        """Token-only view of a finish_run summary (backward-compatible usage payload)."""
        return {
            "input_tokens": total.get("input_tokens", 0),
            "output_tokens": total.get("output_tokens", 0),
            "total_tokens": total.get("total_tokens", 0),
        }

    @staticmethod
    def _metrics_subset(total: dict[str, Any]) -> dict[str, Any]:
        """Retry/429 view of a finish_run summary (the new metrics payload)."""
        return {
            "retry_count": total.get("retry_count", 0),
            "rate_limited_count": total.get("rate_limited_count", 0),
            "rate_429": total.get("rate_429", 0.0),
            "total_llm_tool_calls": total.get("total_llm_tool_calls", 0),
        }

    async def stream_message(
        self,
        messages: list[Any],
        config: RunnableConfig | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream protocol events for a complete research request."""
        self.config = _ensure_config(config, self.config)
        self.run_id = self.config["metadata"]["run_id"]
        self.status = "running"
        state: dict[str, Any] = {
            "messages": normalize_messages(messages),
            "human_feedback": list(self.human_feedback),
        }
        self.messages = state["messages"]
        recorder = get_trace_recorder(self.config)
        run_metadata = {
            "thread_id": self.config.get("configurable", {}).get("thread_id"),
            "message_count": len(state["messages"]),
        }

        with recorder.start_run(
            self.run_id,
            name="lead.run",
            user_id=_config_user_id(self.config),
            metadata=run_metadata,
        ):
            yield self._event("run.started", {"run_id": self.run_id})
            try:
                from open_deep_research.agents import deep_researcher as graph

                await self._run_node("summarize_messages", graph.summarize_messages, state)
                yield self._event("lead.message", {"stage": "summarize_messages"})
                await self._run_node("memory_recall", graph.memory_recall, state)
                yield self._event("lead.message", {"stage": "memory_recall"})

                clarify = await self._run_node("clarify_with_user", graph.clarify_with_user, state)
                yield self._event("lead.message", {"stage": "clarify_with_user", "goto": clarify.goto})
                if clarify.goto == END:
                    self.total_usage = recorder.finish_run(self.run_id, "success")
                    self.status = "completed"
                    self.final_state = {
                        **state,
                        "result": {
                            "status": "success",
                            "result": _message_text(state.get("messages", [])),
                            "usage": self._usage_subset(self.total_usage),
                            "metrics": self._metrics_subset(self.total_usage),
                            "permission_denials": self.permission_denials,
                        },
                    }
                    yield self._event(
                        "run.completed",
                        {
                            "run_id": self.run_id,
                            "status": "success",
                            "usage": self._usage_subset(self.total_usage),
                            "metrics": self._metrics_subset(self.total_usage),
                        },
                    )
                    return

                await self._run_node("write_research_brief", graph.write_research_brief, state)
                yield self._event("lead.message", {"stage": "write_research_brief"})

                async for hitl_event in self._maybe_await_plan_approval(state):
                    yield hitl_event
                if self.cancelled:
                    self._cancelled_state(state)
                    return

                supervisor_update = await self._run_supervisor(state)
                apply_update_to_state(state, supervisor_update)
                state["human_feedback"] = list(self.human_feedback)
                yield self._event("lead.message", {"stage": "research_supervisor"})

                async for hitl_event in self._maybe_await_outline_approval(state):
                    yield hitl_event
                if self.cancelled:
                    self._cancelled_state(state)
                    return

                with recorder.start_span(name="node.final_report_generation", kind="agent", agent_role="lead"):
                    report_update = await graph.final_report_generation(state, self.config)
                apply_update_to_state(state, report_update)
                yield self._event("report.completed", {"run_id": self.run_id})

                with recorder.start_span(name="node.memory_extract_and_write", kind="agent", agent_role="lead"):
                    memory_cmd = coerce_command(
                        await graph.memory_extract_and_write(state, self.config),
                        default_goto=END,
                    )
                apply_update_to_state(state, memory_cmd.update)
                self.total_usage = recorder.finish_run(self.run_id, "success")
                result = {
                    "status": "success",
                    "result": state.get("final_report", ""),
                    "usage": self._usage_subset(self.total_usage),
                    "metrics": self._metrics_subset(self.total_usage),
                    "permission_denials": self.permission_denials,
                }
                report_artifacts = state.get("report_artifacts")
                if report_artifacts:
                    result["artifacts"] = report_artifacts
                self.status = "completed"
                self.final_state = {**state, "result": result}
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
            except Exception as exc:
                self.status = "failed"
                self.total_usage = recorder.finish_run(self.run_id, "error", str(exc))
                self.final_state = {
                    **state,
                    "result": {
                        "status": "error",
                        "error": str(exc),
                        "usage": self._usage_subset(self.total_usage),
                        "metrics": self._metrics_subset(self.total_usage),
                        "permission_denials": self.permission_denials,
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

    async def _run_node(self, name: str, node: Any, state: dict[str, Any]) -> RuntimeCommand:
        if self.cancelled:
            raise RuntimeError("Run cancelled")
        recorder = get_trace_recorder(self.config)
        with recorder.start_span(name=f"node.{name}", kind="agent", agent_role="lead"):
            result = await node(state, self.config)
        command = coerce_command(result)
        apply_update_to_state(state, command.update)
        self._record("node", {"name": name, "goto": command.goto})
        return command

    async def _run_supervisor(self, main_state: dict[str, Any]) -> dict[str, Any]:
        from open_deep_research.agents import deep_researcher as graph

        supervisor_state: dict[str, Any] = {
            "supervisor_messages": list(main_state.get("supervisor_messages", [])),
            "research_brief": main_state.get("research_brief", ""),
            "notes": list(main_state.get("notes", [])),
            "research_iterations": 0,
            "raw_notes": list(main_state.get("raw_notes", [])),
            "enable_async_research": main_state.get("enable_async_research", False),
            "memory_context": main_state.get("memory_context"),
            "approved_research_plan": main_state.get("approved_research_plan"),
            "human_feedback": list(self.human_feedback),
        }
        next_step = "supervisor"
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
            self._record("supervisor", {"step": next_step, "goto": command.goto})
            next_step = command.goto
        return {
            "supervisor_messages": {"type": "override", "value": supervisor_state.get("supervisor_messages", [])},
            "notes": {"type": "override", "value": supervisor_state.get("notes", [])},
            "raw_notes": {"type": "override", "value": supervisor_state.get("raw_notes", [])},
            "completed_task_outputs": {
                "type": "override",
                "value": supervisor_state.get("completed_task_outputs", []),
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
                elif next_step == "compress_research":
                    update = await graph.compress_research(researcher_state, cfg)
                    apply_update_to_state(researcher_state, update)
                    return researcher_state
                else:
                    raise RuntimeError(f"Unknown researcher step: {next_step}")
                apply_update_to_state(researcher_state, command.update)
                next_step = command.goto
        return researcher_state
