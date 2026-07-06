"""Integration tests for non-default report genres (Phase 3)."""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from open_deep_research.report import assembly as assembly_module
from open_deep_research.report import build_report
from open_deep_research.report.models import StructuredReport
from open_deep_research.report.profiles import get_profile


def _config(**configurable: Any) -> RunnableConfig:
    return {"configurable": configurable, "metadata": {"run_id": "genre-test"}}


@pytest.mark.asyncio
async def test_faq_genre_produces_structured_json_artifact(monkeypatch):
    body = (
        "# FAQ: Topic\n\n"
        "Intro line.\n\n"
        "## Q: What is X?\n\nA: X is a thing. See [src](https://x.io).\n\n"
        "## Q: Why?\n\nA: Because.\n\n"
        "### Sources\n[1] src: https://x.io\n"
    )

    async def fake_invoke(model, messages, config, *, span_name, agent_role=None, model_name=None, **_kw):
        return AIMessage(content=body)

    monkeypatch.setattr(
        assembly_module, "invoke_model_with_retry_observability", fake_invoke
    )

    state = {
        "messages": [],
        "research_brief": "brief",
        "notes": ["finding"],
        "completed_task_outputs": [],
    }
    update = await build_report(state, _config(report_type="faq"))

    # final_report is still the markdown body (backward-compatible primary output)
    assert update["final_report"] == body
    # A structured_json artifact is surfaced
    assert update["report_artifacts"]["format"] == "structured_json"
    structured_json = update["report_artifacts"]["structured_json"]
    # It validates against the StructuredReport schema
    validated = StructuredReport(**structured_json)
    assert validated.title == "FAQ: Topic"
    assert [s.name for s in validated.sections] == ["Q: What is X?", "Q: Why?"]
    # Sources recovered into state
    assert update["sources"]["value"][0]["url"] == "https://x.io"


@pytest.mark.asyncio
async def test_each_genre_resolves_a_distinct_prompt(monkeypatch):
    """Each registered one-shot genre formats its own prompt template."""
    seen: dict[str, str] = {}

    async def fake_invoke(model, messages, config, *, span_name, agent_role=None, model_name=None, **_kw):
        # The prompt is the single HumanMessage content; capture a signature line.
        prompt = messages[0].content
        # The genre prompts each contain a distinctive instruction.
        for marker, key in [
            ("EXECUTIVE SUMMARY", "executive_summary"),
            ("DECISION BRIEF", "decision_brief"),
            ("FAQ", "faq"),
        ]:
            if marker in prompt:
                seen[key] = prompt
        return AIMessage(content="ok")

    monkeypatch.setattr(
        assembly_module, "invoke_model_with_retry_observability", fake_invoke
    )

    state = {"messages": [], "research_brief": "b", "notes": ["n"], "completed_task_outputs": []}
    for genre in ("executive_summary", "decision_brief", "faq"):
        await build_report(state, _config(report_type=genre))

    assert set(seen.keys()) == {"executive_summary", "decision_brief", "faq"}
    # And each genre's profile points at the matching prompt constant name
    assert get_profile("executive_summary").prompt_template == "executive_summary_prompt"
    assert get_profile("decision_brief").prompt_template == "decision_brief_prompt"
    assert get_profile("faq").prompt_template == "faq_prompt"


@pytest.mark.asyncio
async def test_markdown_genre_does_not_emit_artifacts(monkeypatch):
    """executive_summary (markdown default format) adds no artifacts key."""
    async def fake_invoke(model, messages, config, *, span_name, agent_role=None, model_name=None, **_kw):
        return AIMessage(content="# Summary\n\nbody")

    monkeypatch.setattr(
        assembly_module, "invoke_model_with_retry_observability", fake_invoke
    )

    state = {"messages": [], "research_brief": "b", "notes": ["n"], "completed_task_outputs": []}
    update = await build_report(state, _config(report_type="executive_summary"))
    assert update["final_report"] == "# Summary\n\nbody"
    assert "report_artifacts" not in update
    assert "sources" not in update
