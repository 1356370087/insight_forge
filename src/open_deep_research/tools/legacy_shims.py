"""Compatibility helpers that are not executable tools."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from langchain_core.messages import (
    AIMessage,
    MessageLikeRepresentation,
    filter_messages,
)
from langchain_core.runnables import RunnableConfig

from open_deep_research.models.resolution import resolve_api_key, resolve_base_url


def clear_run_web_budget(run_id: str) -> None:
    """Clear process-local web-pipeline fetch counters for one run."""
    from open_deep_research.tools.web_research.pipeline import (
        clear_run_web_budget as clear,
    )

    clear(run_id)


def get_notes_from_tool_calls(
    messages: list[MessageLikeRepresentation],
) -> list[Any]:
    """Extract accepted research handoffs and evidence reads as report notes."""
    notes: list[Any] = []
    for tool_msg in filter_messages(messages, include_types="tool"):
        if getattr(tool_msg, "name", None) not in {
            "ConductResearch",
            "ReadResearchArtifact",
        }:
            continue
        content = str(tool_msg.content)
        lowered = content.lower()
        if (
            "rejected_by_supervisor_quality_gate" in lowered
            or lowered.startswith("error:")
        ):
            continue
        try:
            payload = json.loads(content)
        except (TypeError, ValueError):
            payload = None
        if isinstance(payload, dict) and "error_type" in payload:
            continue
        notes.append(tool_msg.content)
    return notes


def remove_up_to_last_ai_message(
    messages: list[MessageLikeRepresentation],
) -> list[MessageLikeRepresentation]:
    """Remove the latest assistant turn and everything after it."""
    for index in range(len(messages) - 1, -1, -1):
        if isinstance(messages[index], AIMessage):
            return messages[:index]
    return messages


def get_today_str() -> str:
    """Return the current date in the prompt-facing legacy format."""
    now = datetime.now()
    return f"{now:%a} {now:%b} {now.day}, {now:%Y}"


def get_config_value(value: Any) -> Any:
    """Extract a primitive value from configuration enums."""
    if value is None or isinstance(value, str | dict):
        return value
    return value.value


def get_api_key_for_model(model_name: str, config: RunnableConfig) -> str | None:
    """Compatibility shim for shared credential resolution."""
    return resolve_api_key(model_name, config)


def get_base_url_for_model(model_name: str) -> str | None:
    """Compatibility shim for shared endpoint resolution."""
    return resolve_base_url(model_name)


def get_model_connection_kwargs(
    model_name: str,
    config: RunnableConfig,
) -> dict[str, str | None]:
    """Return provider credentials and an optional compatible base URL."""
    connection: dict[str, str | None] = {
        "api_key": get_api_key_for_model(model_name, config)
    }
    if base_url := get_base_url_for_model(model_name):
        connection["base_url"] = base_url
    return connection


__all__ = [
    "clear_run_web_budget",
    "get_api_key_for_model",
    "get_base_url_for_model",
    "get_config_value",
    "get_model_connection_kwargs",
    "get_notes_from_tool_calls",
    "get_today_str",
    "remove_up_to_last_ai_message",
]
