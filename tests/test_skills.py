"""Tests for Agent Skills (Phase 5)."""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from open_deep_research.configuration import Configuration
from open_deep_research.report import assembly as assembly_module
from open_deep_research.report import build_report
from open_deep_research.report.models import ReportOutline, SectionSpec
from open_deep_research.skills import (
    BUILTIN_SKILLS,
    get_skill,
    get_skill_report_context,
    get_skill_researcher_context,
    load_skill_tools,
)


def _config(**configurable: Any) -> RunnableConfig:
    return {"configurable": configurable, "metadata": {"run_id": "skills-test"}}


def test_builtin_skills_registered():
    assert set(BUILTIN_SKILLS) == {"medical", "legal", "finance"}
    assert get_skill("medical").key == "medical"
    assert get_skill("unknown") is None  # never raises


def test_researcher_context_concatenates_only_enabled_skills():
    ctx = get_skill_researcher_context(["medical", "legal"])
    assert "MEDICAL" in ctx
    assert "LEGAL" in ctx
    # finance was not enabled
    assert "FINANCE" not in ctx


def test_context_empty_when_no_skills():
    assert get_skill_researcher_context(None) == ""
    assert get_skill_researcher_context([]) == ""
    assert get_skill_report_context(["unknown"]) == ""


@pytest.mark.asyncio
async def test_load_skill_tools_returns_empty_for_context_only():
    tools = await load_skill_tools({"configurable": {"skills": ["medical"]}}, set())
    assert tools == []


@pytest.mark.asyncio
async def test_skill_report_context_injected_into_writer_prompt(monkeypatch):
    captured: dict[str, Any] = {}

    async def fake_invoke(model, messages, config, *, span_name, agent_role=None, model_name=None, **_kw):
        captured["prompt"] = messages[0].content
        return AIMessage(content="ok")

    monkeypatch.setattr(
        assembly_module, "invoke_model_with_retry_observability", fake_invoke
    )

    state = {"messages": [], "research_brief": "b", "notes": ["n"], "completed_task_outputs": []}
    await build_report(state, _config(skills=["medical"]))

    # The medical report-context (disclaimer) is prepended to the writer prompt.
    assert "not medical advice" in captured["prompt"].lower()


@pytest.mark.asyncio
async def test_no_skill_context_for_default_config(monkeypatch):
    """Skills unset -> no skill context leaks into the writer prompt (parity)."""
    captured: dict[str, Any] = {}

    async def fake_invoke(model, messages, config, *, span_name, agent_role=None, model_name=None, **_kw):
        captured["prompt"] = messages[0].content
        return AIMessage(content="ok")

    monkeypatch.setattr(
        assembly_module, "invoke_model_with_retry_observability", fake_invoke
    )

    state = {"messages": [], "research_brief": "b", "notes": ["n"], "completed_task_outputs": []}
    await build_report(state, _config())  # no skills

    assert "MEDICAL" not in captured["prompt"]
    assert "not medical advice" not in captured["prompt"].lower()


@pytest.mark.asyncio
async def test_skill_report_context_injected_into_all_sectioned_writer_prompts(monkeypatch):
    prompts: list[str] = []

    class FakeStructuredModel:
        def with_structured_output(self, _schema, **_kwargs):
            return self

    async def fake_invoke(
        model, messages, config, *, span_name, agent_role=None, model_name=None, **_kw
    ):
        prompts.append(messages[0].content)
        if span_name == "lead.report_outline":
            return ReportOutline(title="T", sections=[SectionSpec(name="Findings")])
        return AIMessage(content="body")

    monkeypatch.setattr(
        assembly_module, "invoke_model_with_retry_observability", fake_invoke
    )
    monkeypatch.setattr(
        assembly_module, "init_chat_model", lambda **_kw: FakeStructuredModel()
    )
    state = {
        "messages": [],
        "research_brief": "b",
        "notes": ["n"],
        "completed_task_outputs": [],
    }

    await build_report(
        state,
        _config(report_type="comparison_matrix", skills=["medical"]),
    )

    assert len(prompts) == 4
    assert all("not medical advice" in prompt.lower() for prompt in prompts)


def test_skills_environment_variable_accepts_comma_separated_values(monkeypatch):
    monkeypatch.setenv("SKILLS", "medical, legal")

    assert Configuration.from_runnable_config().skills == ["medical", "legal"]

    monkeypatch.setenv("SKILLS", '["finance", "medical"]')

    assert Configuration.from_runnable_config().skills == ["finance", "medical"]
