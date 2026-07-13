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
    # default profile adds NO extra artifact keys (byte-identical to original node)
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
