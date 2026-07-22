from __future__ import annotations

import json

from langchain_core.messages import ToolMessage

from open_deep_research.agents.research_context import (
    ResearchContextEnvelope,
    offload_tool_message,
    response_was_truncated,
)
from open_deep_research.run_context import RunContextStore


def test_context_envelope_reserves_output_tools_and_margin():
    envelope = ResearchContextEnvelope(
        model_context_tokens=100_000,
        reserved_output_tokens=10_000,
        tool_schema_tokens=5_000,
        fixed_prompt_tokens=4_000,
        safety_margin_tokens=1_000,
    )

    assert envelope.available_input_tokens == 80_000


def test_large_tool_result_is_offloaded_with_hash_verified_reference(tmp_path):
    store = RunContextStore("context-run", runs_dir=str(tmp_path))
    store.initialize(None, {})
    message = ToolMessage(
        content=json.dumps({
            "claim": "supported conclusion",
            "counterexample": "important exception",
            "source_url": "https://example.com/source",
            "raw": "x" * 20_000,
        }),
        name="web_research",
        tool_call_id="tool-1",
    )

    compact = offload_tool_message(
        message,
        store=store,
        task_id="researcher-1",
        max_inline_chars=2_000,
    )
    payload = json.loads(str(compact.content))

    assert payload["artifact_ref"]["sha256"]
    assert payload["tool_call_id"] == "tool-1"
    artifact = store.load_evidence_artifact(
        payload["artifact_ref"]["path"],
        expected_sha256=payload["artifact_ref"]["sha256"],
    )
    assert "important exception" in artifact["content"]
    assert "https://example.com/source" in artifact["content"]


def test_response_was_truncated_checks_provider_metadata():
    class Response:
        response_metadata = {"finish_reason": "length"}
        usage_metadata = {}

    assert response_was_truncated(Response()) is True
