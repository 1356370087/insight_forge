"""Small runtime primitives that replace LangGraph graph scheduling."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
)

END = "__end__"
REMOVE_ALL_MESSAGES = "__remove_all__"
GotoT = TypeVar("GotoT")

@dataclass
class RuntimeCommand(Generic[GotoT]):
    """A tiny command object used by the hand-written runtime.

    Existing node functions return a ``goto`` target plus an ``update`` payload.
    The hand-written runtimes consume this object directly.
    """

    goto: str = END
    update: dict[str, Any] = field(default_factory=dict)

    def __class_getitem__(cls, _item: Any) -> type[RuntimeCommand]:
        """Allow legacy annotations such as ``Command[Literal[...]]``."""
        return cls

def message_from_dict(message: Mapping[str, Any]) -> BaseMessage:
    """Convert an API-style message dict into a LangChain message."""
    role = message.get("role") or message.get("type")
    content = message.get("content", "")
    if role in {"user", "human"}:
        return HumanMessage(content=content)
    if role in {"assistant", "ai"}:
        return AIMessage(content=content)
    if role == "system":
        return SystemMessage(content=content)
    if role == "tool":
        return ToolMessage(
            content=content,
            name=str(message.get("name", "")),
            tool_call_id=str(message.get("tool_call_id", "")),
        )
    raise ValueError(f"Unsupported message role: {role!r}")

def normalize_messages(messages: list[Any]) -> list[Any]:
    """Normalize API dictionaries while preserving existing message objects."""
    normalized: list[Any] = []
    for message in messages:
        if isinstance(message, BaseMessage):
            normalized.append(message)
        elif isinstance(message, Mapping):
            normalized.append(message_from_dict(message))
        else:
            normalized.append(message)
    return normalized

def _merge_sequence(current: Any, new_value: Any) -> list[Any]:
    base = list(current or [])
    values = new_value if isinstance(new_value, list) else [new_value]
    if any(
        isinstance(item, RemoveMessage) and item.id == REMOVE_ALL_MESSAGES
        for item in values
    ):
        values = [
            item
            for item in values
            if not (isinstance(item, RemoveMessage) and item.id == REMOVE_ALL_MESSAGES)
        ]
        base = []
    return base + normalize_messages(values)

def _merge_reducer_value(current: Any, new_value: Any) -> Any:
    if isinstance(new_value, dict) and new_value.get("type") == "override":
        return new_value.get("value")
    if isinstance(current, list) or isinstance(new_value, list):
        return list(current or []) + list(new_value or [])
    return new_value

MESSAGE_KEYS = {"messages", "supervisor_messages", "researcher_messages"}
REDUCER_KEYS = {
    "raw_notes",
    "notes",
    "completed_task_outputs",
    "memory_candidates",
    "human_feedback",
    "candidate_registry",
    "document_registry",
    "evidence_registry",
    "web_research_iterations",
}

def apply_update_to_state(state: dict[str, Any], update: Mapping[str, Any] | None) -> dict[str, Any]:
    """Apply a node update using the reducer semantics the graph used to own."""
    if not update:
        return state
    for key, value in update.items():
        if key in MESSAGE_KEYS:
            if isinstance(value, dict) and value.get("type") == "override":
                state[key] = normalize_messages(list(value.get("value") or []))
            else:
                state[key] = _merge_sequence(state.get(key), value)
        elif key in REDUCER_KEYS:
            state[key] = _merge_reducer_value(state.get(key), value)
        else:
            if isinstance(value, dict) and value.get("type") == "override":
                state[key] = value.get("value")
            else:
                state[key] = value
    return state

def coerce_command(result: Any, default_goto: str = END) -> RuntimeCommand:
    """Normalize plain dict node outputs and command outputs."""
    if isinstance(result, RuntimeCommand):
        return result
    if isinstance(result, dict):
        return RuntimeCommand(goto=default_goto, update=result)
    return RuntimeCommand(goto=default_goto, update={})
