"""Tests for the progressive broad-to-narrow research strategy."""

import pytest

from open_deep_research.prompts import research_system_prompt
from open_deep_research.state import ConductResearch
from open_deep_research.tasks.async_tools import StartResearchTask
from open_deep_research.tools.utils import tavily_search


def test_research_prompt_requires_progressive_search_within_task_contract() -> None:
    rendered = research_system_prompt.format(mcp_prompt="", date="July 7, 2026")
    normalized = " ".join(rendered.split())

    assert "Treat the delegated task contract as binding" in normalized
    assert "If the task says to use only `fetch_url`, never call `web_research`" in normalized
    assert "Start broad only when unconstrained" in normalized
    assert "1-3 short, broad queries" in normalized
    assert "Map the information landscape" in normalized
    assert "Add only the necessary dimension" in normalized
    assert (
        "rephrase it more broadly without crossing any explicit tool, URL, source"
        in normalized
    )


def test_research_prompt_is_search_backend_agnostic() -> None:
    rendered = research_system_prompt.format(mcp_prompt="", date="July 7, 2026")

    assert "Configured search tool" in rendered
    assert "selected search backend" in rendered
    assert "Do not call think_tool with a search tool" in rendered
    assert "tavily_search" not in rendered


def test_research_task_schemas_distinguish_objectives_from_queries() -> None:
    for task_model in (ConductResearch, StartResearchTask):
        description = task_model.model_fields["research_topic"].description or ""

        assert "complete, self-contained research objective" in description
        assert "not a search-engine query to copy verbatim" in description
        assert "at least a paragraph" not in description


@pytest.mark.asyncio
async def test_tavily_schema_exposes_progressive_query_guidance() -> None:
    schema_description = tavily_search.input_schema.model_json_schema()["description"]
    tool_description = await tavily_search.description()

    assert "Short, search-engine-ready queries" in schema_description
    assert "Begin broad on the first call" in schema_description
    assert "evidence-backed gaps" in schema_description
    assert "Start with short, broad queries" in tool_description
