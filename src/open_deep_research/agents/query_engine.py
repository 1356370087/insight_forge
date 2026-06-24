"""Outer session engines for Deep Research.

``QueryEngine`` owns the session/protocol layer. The lower-level model/tool loop
lives in :mod:`open_deep_research.agents.query`.
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from langchain_core.messages import BaseMessage, get_buffer_string
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
        self.total_usage: dict[str, int] = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        self.permission_denials: list[dict[str, Any]] = []
        self.cancelled = False
        self.final_state: dict[str, Any] | None = None

    def interrupt(self) -> None:
        """Request cancellation before the next step starts."""
        self.cancelled = True

    async def submit_message(
        self,
        messages: list[Any],
        config: RunnableConfig | None = None,
    ) -> dict[str, Any]:
        """Run a complete research request and return the final state."""
        async for _event in self.stream_message(messages, config):
            pass
        return self.final_state or {}

    async def stream_message(
        self,
        messages: list[Any],
        config: RunnableConfig | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream protocol events for a complete research request."""
        self.config = _ensure_config(config, self.config)
        self.run_id = self.config["metadata"]["run_id"]
        state: dict[str, Any] = {"messages": normalize_messages(messages)}
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
                    self.final_state = {
                        **state,
                        "result": {
                            "status": "success",
                            "result": _message_text(state.get("messages", [])),
                            "usage": self.total_usage,
                            "permission_denials": self.permission_denials,
                        },
                    }
                    yield self._event(
                        "run.completed",
                        {"run_id": self.run_id, "status": "success", "usage": self.total_usage},
                    )
                    return

                await self._run_node("write_research_brief", graph.write_research_brief, state)
                yield self._event("lead.message", {"stage": "write_research_brief"})

                supervisor_update = await self._run_supervisor(state)
                apply_update_to_state(state, supervisor_update)
                yield self._event("lead.message", {"stage": "research_supervisor"})

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
                    "usage": self.total_usage,
                    "permission_denials": self.permission_denials,
                }
                self.final_state = {**state, "result": result}
                yield self._event("run.completed", {"run_id": self.run_id, "status": "success", "result": result})
            except Exception as exc:
                self.total_usage = recorder.finish_run(self.run_id, "error", str(exc))
                self.final_state = {
                    **state,
                    "result": {
                        "status": "error",
                        "error": str(exc),
                        "usage": self.total_usage,
                        "permission_denials": self.permission_denials,
                    },
                }
                yield self._event("run.failed", {"run_id": self.run_id, "error": str(exc)})

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
