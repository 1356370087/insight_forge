"""Phase 1 parity tests for build_report (default profile).

Asserts that routing the default report type through the new registry-based
``build_report`` reproduces the original single-call synthesis behavior:
``final_report`` equals the model output, notes/task-outputs are cleared via the
override reducer, and no extra artifact keys are produced.
"""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from open_deep_research.report import assembly as assembly_module
from open_deep_research.report import build_report
from open_deep_research.run_context import RunContextStore


def _config(**configurable: Any) -> RunnableConfig:
    return {"configurable": configurable, "metadata": {"run_id": "report-test"}}


@pytest.mark.asyncio
async def test_legacy_report_is_byte_identical_to_model_output(monkeypatch):
    # Other evaluation-test modules load the developer .env during collection;
    # isolate this default-value assertion from that process-global side effect.
    monkeypatch.delenv("FINAL_REPORT_MODEL", raising=False)
    monkeypatch.setenv("WEB_PIPELINE_MODE", "legacy")
    fake_content = "# Report\n\nBody with [link](https://example.com)."
    captured: dict[str, Any] = {}

    async def fake_invoke(model, messages, config, *, span_name, agent_role=None, model_name=None, **_kw):
        captured["span_name"] = span_name
        captured["model_name"] = model_name
        return AIMessage(content=fake_content)

    monkeypatch.setattr(
        assembly_module, "invoke_model_with_retry_observability", fake_invoke
    )

    state = {
        "messages": [],
        "research_brief": "the brief",
        "notes": ["finding one", "finding two"],
        "completed_task_outputs": [],
    }
    update = await build_report(state, _config(web_pipeline_mode="legacy"))

    # final_report is exactly the model output
    assert update["final_report"] == fake_content
    # messages carries the writer AIMessage
    assert update["messages"][0].content == fake_content
    # notes / completed_task_outputs cleared via override reducer (unchanged behavior)
    assert update["notes"] == {"type": "override", "value": []}
    assert update["completed_task_outputs"] == {"type": "override", "value": []}
    # Default profile adds no optional presentation artifacts. The evaluation
    # snapshot is a state contract and does not alter the report payload.
    assert "report_artifacts" not in update
    assert "sources" not in update
    # observability span/model wired exactly as before
    assert captured["span_name"] == "lead.final_report"
    assert captured["model_name"] == "openai:gpt-4.1"


@pytest.mark.asyncio
async def test_default_report_aggregates_completed_task_outputs(monkeypatch):
    captured: dict[str, str] = {}

    async def fake_invoke(model, messages, config, *, span_name, agent_role=None, model_name=None, **_kw):
        captured["prompt"] = messages[0].content
        return AIMessage(content="ok")

    monkeypatch.setattr(
        assembly_module, "invoke_model_with_retry_observability", fake_invoke
    )

    state = {
        "messages": [],
        "research_brief": "the brief",
        "notes": ["a supervisor note"],
        "completed_task_outputs": [
            {"research_topic": "Topic A", "compressed_research": "compressed A"},
        ],
    }
    await build_report(state, _config())

    # Findings must include the Research Task header built from completed_task_outputs
    assert "## Research Task: Topic A" in captured["prompt"]
    assert "compressed A" in captured["prompt"]
    assert "a supervisor note" in captured["prompt"]


@pytest.mark.asyncio
async def test_report_persists_evaluation_snapshot_before_clearing_runtime_state(
    monkeypatch,
):
    async def fake_invoke(
        model,
        messages,
        config,
        *,
        span_name,
        agent_role=None,
        model_name=None,
        **_kw,
    ):
        return AIMessage(content="ok")

    monkeypatch.setattr(
        assembly_module, "invoke_model_with_retry_observability", fake_invoke
    )
    state = {
        "messages": [],
        "research_brief": "compare A and B",
        "notes": ["accepted finding"],
        "supervisor_messages": [
            {
                "type": "ai",
                "tool_calls": [
                    {
                        "name": "ConductResearch",
                        "args": {
                            "research_topic": "Topic A",
                            "api_key": "must-not-persist",
                            "nested": {
                                "authorization": "Bearer must-not-persist",
                            },
                            "long_query": "x" * 1_200,
                        },
                        "id": "call-1",
                    }
                ],
            },
            {
                "type": "tool",
                "name": "ConductResearch",
                "tool_call_id": "call-1",
                "content": (
                    "task completed\n"
                    "Authorization: Bearer must-not-persist\n"
                    "api_key=must-not-persist"
                ),
            },
        ],
        "completed_task_outputs": [
            {
                "task_id": "task-1",
                "research_topic": "Topic A",
                "query_count": 3,
                "source_count": 2,
                "citation_count": 2,
                "elapsed_seconds": 4.5,
            }
        ],
        "evidence_registry": [
            {
                "evidence_id": "ev-accepted",
                "claim": "Supported claim",
                "source_url": "https://accepted.example/source",
                "security_status": "accepted",
            },
            {
                "evidence_id": "ev-quarantined",
                "claim": "Untrusted claim",
                "source_url": "https://quarantined.example/source",
                "security_status": "quarantined",
            },
        ],
    }

    update = await build_report(state, _config())

    assert update["completed_task_outputs"] == {"type": "override", "value": []}
    snapshot = update["evaluation_snapshot"]
    assert snapshot["schema_version"] == "1.0"
    assert [item["evidence_id"] for item in snapshot["evidence_registry"]] == [
        "ev-accepted"
    ]
    assert snapshot["tool_trace"]["completed_task_metrics"] == [
        {
            "task_id": "task-1",
            "research_topic": "Topic A",
            "query_count": 3,
            "source_count": 2,
            "citation_count": 2,
            "elapsed_seconds": 4.5,
        }
    ]
    assert snapshot["tool_trace"]["supervisor_tool_calls"][0]["name"] == (
        "ConductResearch"
    )
    projected_args = snapshot["tool_trace"]["supervisor_tool_calls"][0]["args"]
    assert projected_args["api_key"] == "[REDACTED]"
    assert projected_args["nested"]["authorization"] == "[REDACTED]"
    assert len(projected_args["long_query"]) == 1_000
    assert snapshot["tool_trace"]["supervisor_tool_results"][0]["tool_call_id"] == (
        "call-1"
    )
    result_preview = snapshot["tool_trace"]["supervisor_tool_results"][0][
        "content_preview"
    ]
    assert "must-not-persist" not in result_preview
    assert "[REDACTED]" in result_preview


@pytest.mark.asyncio
async def test_report_snapshot_recovers_researcher_trace_from_artifact_refs(
    monkeypatch,
    tmp_path,
) -> None:
    async def fake_invoke(
        model,
        messages,
        config,
        *,
        span_name,
        agent_role=None,
        model_name=None,
        **_kw,
    ):
        return AIMessage(content="ok")

    monkeypatch.setattr(
        assembly_module,
        "invoke_model_with_retry_observability",
        fake_invoke,
    )
    run_id = "report-artifact-trace"
    store = RunContextStore(run_id, runs_dir=str(tmp_path))
    task_id = "research-task-1"
    digest = store.persist_task_result(task_id, {
        "task_id": task_id,
        "researcher_messages": [
            {
                "type": "ai",
                "data": {
                    "tool_calls": [{
                        "name": "fetch_url",
                        "args": {"url": "https://primary.example/source"},
                        "id": "fetch-1",
                    }],
                },
            },
            {
                "type": "tool",
                "data": {
                    "type": "tool",
                    "name": "fetch_url",
                    "tool_call_id": "fetch-1",
                    "status": "success",
                },
            },
        ],
    })
    state = {
        "messages": [],
        "research_brief": "Read one primary source.",
        "notes": ["accepted finding"],
        "supervisor_messages": [],
        "research_artifact_refs": {
            task_id: {
                "path": f"context/artifacts/research_tasks/{task_id}.json",
                "sha256": digest,
            },
        },
    }
    config: RunnableConfig = {
        "configurable": {"runs_dir": str(tmp_path)},
        "metadata": {"run_id": run_id},
    }

    update = await build_report(state, config)

    trace = update["evaluation_snapshot"]["tool_trace"]
    assert trace["availability"]["researcher_tool_names_retained"] is True
    assert trace["researcher_tool_calls"] == [{
        "task_id": task_id,
        "name": "fetch_url",
        "args": {"url": "https://primary.example/source"},
        "id": "fetch-1",
    }]
    assert trace["researcher_tool_results"] == [{
        "task_id": task_id,
        "name": "fetch_url",
        "tool_call_id": "fetch-1",
        "status": "success",
    }]


@pytest.mark.asyncio
async def test_unknown_report_type_falls_back_to_default(monkeypatch):
    async def fake_invoke(model, messages, config, *, span_name, agent_role=None, model_name=None, **_kw):
        return AIMessage(content="fallback body")

    monkeypatch.setattr(
        assembly_module, "invoke_model_with_retry_observability", fake_invoke
    )

    state = {
        "messages": [],
        "research_brief": "b",
        "notes": ["n"],
        "completed_task_outputs": [],
    }
    update = await build_report(state, _config(report_type="totally-unknown"))
    assert update["final_report"] == "fallback body"
    assert "report_artifacts" not in update


@pytest.mark.asyncio
async def test_bibtex_reference_style_replaces_sources_section(monkeypatch):
    body = (
        "# Report\n\nSome finding from a source.\n\n"
        "### Sources\n[1] Old Source: https://old.example.com\n"
    )

    async def fake_invoke(model, messages, config, *, span_name, agent_role=None, model_name=None, **_kw):
        return AIMessage(content=body)

    monkeypatch.setattr(
        assembly_module, "invoke_model_with_retry_observability", fake_invoke
    )

    state = {
        "messages": [],
        "research_brief": "b",
        "notes": ["--- SOURCE 1: Old Source ---\nURL: https://old.example.com"],
        "completed_task_outputs": [],
    }
    update = await build_report(state, _config(reference_style="bibtex_like"))

    # The Sources section is re-rendered as BibTeX-like entries
    assert "@misc{ref1," in update["final_report"]
    assert "howpublished = {https://old.example.com}" in update["final_report"]
    assert "[1] Old Source:" not in update["final_report"]
    # The report body outside Sources is preserved
    assert "Some finding from a source." in update["final_report"]
    # Structured sources are surfaced into state
    assert update["sources"]["type"] == "override"
    assert update["sources"]["value"][0]["url"] == "https://old.example.com"


@pytest.mark.asyncio
async def test_artifact_markdown_matches_reference_rewritten_final_report(monkeypatch):
    body = "# Report\n\n### Sources\n[1] Source: https://example.com\n"

    async def fake_invoke(
        model, messages, config, *, span_name, agent_role=None, model_name=None, **_kw
    ):
        return AIMessage(content=body)

    monkeypatch.setattr(
        assembly_module, "invoke_model_with_retry_observability", fake_invoke
    )
    state = {
        "messages": [],
        "research_brief": "b",
        "notes": ["[Source](https://example.com)"],
        "completed_task_outputs": [],
    }

    update = await build_report(
        state,
        _config(output_format="structured_json", reference_style="bibtex_like"),
    )

    assert update["report_artifacts"]["markdown"] == update["final_report"]
