from open_deep_research.agents.deep_researcher import build_researcher_system_prompt
from open_deep_research.configuration import Configuration
from open_deep_research.prompts import (
    lead_researcher_async_prompt,
    lead_researcher_prompt,
)


def _render_supervisor_prompt(template: str) -> str:
    return template.format(
        date="2026-07-14",
        max_concurrent_research_units=5,
        max_researcher_iterations=6,
        max_react_tool_calls=12,
    )


def _assert_delegation_and_scaling_contract(prompt: str) -> None:
    for required_field in (
        "Objective",
        "Deliverable",
        "Scope and boundaries",
        "Tool and source guidance",
        "Effort budget and stopping rule",
    ):
        assert required_field in prompt

    assert "3-10 evidence-gathering tool calls" in prompt
    assert "10-15 evidence-gathering tool calls" in prompt
    assert "more than 10" in prompt
    assert "waves of at most 5" in prompt
    assert "12 Researcher iterations" in prompt
    assert "semiconductor shortage" in prompt
    assert "explicitly excludes" in prompt or "excludes the other three scopes" in prompt


def test_sync_supervisor_prompt_teaches_delegation_and_scaling() -> None:
    prompt = _render_supervisor_prompt(lead_researcher_prompt)

    _assert_delegation_and_scaling_contract(prompt)
    assert "Every `research_topic` must be a standalone contract" in prompt
    assert "This is not a per-agent tool-call budget" in prompt


def test_async_supervisor_prompt_teaches_delegation_and_scaling() -> None:
    prompt = _render_supervisor_prompt(lead_researcher_async_prompt)

    _assert_delegation_and_scaling_contract(prompt)
    assert "Every StartResearchTask `research_topic`" in prompt
    assert "Respect the 6-iteration Supervisor limit" in prompt


def test_researcher_prompt_accepts_delegated_effort_budget() -> None:
    prompt = build_researcher_system_prompt(Configuration(max_react_tool_calls=12))

    assert "Delegated budget wins" in prompt
    assert "3-10 evidence-gathering tool calls" in prompt
    assert "10-15 evidence-gathering tool calls" in prompt
    assert "`max_react_tool_calls` Researcher-iteration cap" in prompt
    assert "After 5 search tool calls" not in prompt
