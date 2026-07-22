"""Evidence-aware Researcher context envelope and artifact offload."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import ToolMessage

from open_deep_research.run_context import RunContextStore


@dataclass(frozen=True, slots=True)
class ResearchContextEnvelope:
    """Compute the request tokens available after mandatory reservations."""

    model_context_tokens: int
    reserved_output_tokens: int
    tool_schema_tokens: int
    fixed_prompt_tokens: int
    safety_margin_tokens: int

    @property
    def available_input_tokens(self) -> int:
        """Return the non-negative remaining input budget."""
        return max(
            0,
            self.model_context_tokens
            - self.reserved_output_tokens
            - self.tool_schema_tokens
            - self.fixed_prompt_tokens
            - self.safety_margin_tokens,
        )


def response_was_truncated(response: Any) -> bool:
    """Return whether provider metadata reports an output-length stop."""
    metadata = getattr(response, "response_metadata", {}) or {}
    reason = str(
        metadata.get("finish_reason")
        or metadata.get("stop_reason")
        or metadata.get("stop_sequence")
        or ""
    ).lower()
    return reason in {
        "length",
        "max_tokens",
        "max_output_tokens",
        "model_length",
    }


def offload_tool_message(
    message: ToolMessage,
    *,
    store: RunContextStore,
    task_id: str,
    max_inline_chars: int,
) -> ToolMessage:
    """Persist a large accepted result and return a compact paired ToolMessage."""
    content = str(message.content)
    if len(content) <= max_inline_chars:
        return message.model_copy(deep=True)
    relative_path = f"artifacts/evidence/{task_id}/{message.tool_call_id}.json"
    artifact = {
        "schema_version": 1,
        "task_id": task_id,
        "tool_call_id": str(message.tool_call_id),
        "tool_name": message.name,
        "content": content,
    }
    digest = store.write_json_atomic(relative_path, artifact)
    target = store.context_dir / relative_path
    preview = content[:max_inline_chars]
    payload = {
        "status": "artifact_offloaded",
        "tool_call_id": str(message.tool_call_id),
        "tool_name": message.name,
        "preview": preview,
        "content_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        "artifact_ref": {
            "path": relative_path,
            "sha256": digest,
            "content_bytes": target.stat().st_size,
        },
    }
    return ToolMessage(
        content=json.dumps(payload, ensure_ascii=False, sort_keys=True),
        name=message.name,
        tool_call_id=str(message.tool_call_id),
    )
