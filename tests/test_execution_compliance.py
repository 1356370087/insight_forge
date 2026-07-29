"""Offline regressions for content/execution quality separation."""

from open_deep_research.evaluation.execution import (
    content_coverage_requirements,
    evaluate_execution_compliance,
    extract_execution_constraints,
)
from open_deep_research.report.coverage import derive_coverage_checklist
from open_deep_research.tool_taxonomy import classify_tool_name

R17_QUESTION = (
    "请核验 PEP 8 与 PEP 257 的规则，最后给出三条审查项。"
    "必须只创建一个 ConductResearch 子任务，由同一个研究员在该任务中分别使用 "
    "fetch_url 读取 https://peps.python.org/pep-0008/ 与 "
    "https://peps.python.org/pep-0257/；不得拆分为单来源子任务，"
    "不得使用搜索或二手来源，最终报告必须引用这两个官方 URL。"
)


def _trace(*, include_search: bool = False) -> dict:
    researcher_calls = [
        {
            "task_id": "task-1",
            "name": "fetch_url",
            "args": {"url": "https://peps.python.org/pep-0008/"},
            "id": "fetch-8",
        },
        {
            "task_id": "task-1",
            "name": "fetch_url",
            "args": {"url": "https://peps.python.org/pep-0257/"},
            "id": "fetch-257",
        },
        {
            "task_id": "task-1",
            "name": "ResearchComplete",
            "args": {},
            "id": "done-1",
        },
    ]
    if include_search:
        researcher_calls.append(
            {
                "task_id": "task-1",
                "name": "web_search",
                "args": {"query": "PEP 257"},
                "id": "search-1",
            }
        )
    return {
        "supervisor_tool_calls": [
            {"name": "ConductResearch", "args": {}, "id": "task-1"},
            {"name": "ResearchComplete", "args": {}, "id": "done"},
        ],
        "researcher_tool_calls": researcher_calls,
        "availability": {"researcher_tool_names_retained": True},
    }


def _evidence() -> list[dict]:
    return [
        {
            "source_url": "https://peps.python.org/pep-0008/",
            "security_status": "accepted",
        },
        {
            "source_url": "https://peps.python.org/pep-0257/",
            "security_status": "accepted",
        },
    ]


def test_r16_and_r17_equivalent_traces_have_identical_compliance() -> None:
    r16 = R17_QUESTION.replace(" PEP", "PEP").replace(" URL", "URL")
    r17 = R17_QUESTION

    first = evaluate_execution_compliance(r16, _trace(), _evidence())
    second = evaluate_execution_compliance(r17, _trace(), _evidence())

    assert first.status == second.status == "passed"
    assert first.score == second.score == 1.0
    assert first.reason_codes == second.reason_codes == []


def test_r17_original_wording_requires_exactly_one_conduct_research() -> None:
    constraints = extract_execution_constraints(
        "必须只从一个ConductResearch子任务中分别使用fetch_url读取两个URL。"
    )

    assert constraints.conduct_research_count == 1


def test_r17_adjacent_chinese_conjunction_keeps_urls_separate() -> None:
    constraints = extract_execution_constraints(
        "必须只从一个ConductResearch子任务中，分别使用fetch_url读取"
        "https://peps.python.org/pep-0008/与https://peps.python.org/pep-0257/，"
        "不得拆分为单来源子任务，不得使用搜索或二手来源。"
    )

    assert constraints.required_urls == [
        "https://peps.python.org/pep-0008",
        "https://peps.python.org/pep-0257",
    ]


def test_r17_constraints_are_removed_from_content_completeness() -> None:
    requirements = derive_coverage_checklist(R17_QUESTION)
    content = content_coverage_requirements(requirements)

    assert any("三条审查项" in item for item in content)
    assert any("最终报告必须引用" in item for item in content)
    assert not any("ConductResearch" in item for item in content)
    assert not any("fetch_url" in item for item in content)
    assert not any("不得使用搜索" in item for item in content)


def test_execution_compliance_detects_forbidden_search() -> None:
    result = evaluate_execution_compliance(
        R17_QUESTION,
        _trace(include_search=True),
        _evidence(),
    )

    assert result.status == "failed"
    assert result.score == 0.0
    assert "forbidden_tool_used" in result.reason_codes


def test_execution_constraints_without_researcher_trace_are_evaluator_error() -> None:
    result = evaluate_execution_compliance(
        R17_QUESTION,
        {
            "supervisor_tool_calls": [
                {"name": "ConductResearch", "args": {}, "id": "task-1"}
            ],
            "researcher_tool_calls": [],
            "availability": {"researcher_tool_names_retained": False},
        },
        _evidence(),
    )

    assert result.status == "evaluator_error"
    assert result.reason_codes == ["researcher_tool_trace_missing"]


def test_exact_tool_taxonomy_does_not_treat_research_complete_as_search() -> None:
    assert classify_tool_name("ResearchComplete") == "control"
    assert classify_tool_name("ReadResearchArtifact") == "control"
    assert classify_tool_name("web_search") == "search"
    assert classify_tool_name("fetch_url") == "fetch"


def test_unrecognized_workflow_constraint_is_unverifiable() -> None:
    constraints = extract_execution_constraints(
        "必须让研究员在月相合适时执行内部仪式，然后再生成报告。"
    )

    assert constraints.unverifiable_clauses


def test_negative_english_tool_constraint_is_not_required() -> None:
    constraints = extract_execution_constraints(
        "Use only these URLs: https://example.com/a and "
        "https://example.com/b; do not use search."
    )

    assert constraints.required_tools == []
    assert constraints.forbidden_categories == ["search"]
    assert constraints.allowed_urls == [
        "https://example.com/a",
        "https://example.com/b",
    ]


def test_negative_fetch_url_constraint_is_not_required() -> None:
    constraints = extract_execution_constraints(
        "Do not use fetch_url; use web_search for the current result."
    )

    assert "fetch_url" not in constraints.required_tools
    assert constraints.required_tools == ["web_search"]
