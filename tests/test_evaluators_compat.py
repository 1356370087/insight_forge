from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, ToolMessage

from tests import evaluators
from tests.supervisor_parallel_evaluation import right_parallelism_evaluator


class _StructuredRunner:
    def __init__(self, schema: type) -> None:
        self.schema = schema

    def with_retry(self, **_kwargs: Any) -> "_StructuredRunner":
        return self

    def invoke(self, _messages: list[dict[str, Any]]) -> Any:
        values: dict[str, Any] = {
            "OverallQualityScore": {
                "research_depth": 4,
                "source_quality": 4,
                "analytical_rigor": 4,
                "practical_value": 4,
                "balance_and_objectivity": 4,
                "writing_quality": 4,
            },
            "RelevanceScore": {"reasoning": "relevant", "score": 4},
            "StructureScore": {"reasoning": "structured", "score": 4},
            "CorrectnessScore": {"reasoning": "correct", "score": 4},
            "GroundednessScore": {
                "claims": [{"claim": "Claim A", "grounded": True}],
            },
            "CompletenessScore": {"reasoning": "complete", "score": 4},
            "CitationAccuracyScore": {
                "citations": [
                    {
                        "claim": "Claim A",
                        "citation": "https://primary.example/paper",
                        "supported": True,
                    }
                ],
                "reasoning": "citation supports the claim",
            },
            "ToolEfficiencyScore": {
                "tool_selection_score": 5,
                "call_efficiency_score": 4,
                "reasoning": "appropriate tools and bounded calls",
            },
        }
        return self.schema.model_validate(values[self.schema.__name__])


class _FakeJudge:
    def with_structured_output(self, schema: type, **_kwargs: Any) -> _StructuredRunner:
        return _StructuredRunner(schema)


def _outputs() -> dict[str, Any]:
    return {
        "final_report": "Claim A [source](https://primary.example/paper)",
        "raw_notes": ["Claim A. URL: https://primary.example/paper"],
        "research_brief": "Research claim A.",
        "supervisor_messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "ConductResearch",
                        "args": {"research_topic": "claim A"},
                        "id": "call-1",
                    }
                ],
            ),
            ToolMessage(content="research returned", name="ConductResearch", tool_call_id="call-1"),
        ],
        "result": {
            "status": "success",
            "metrics": {"total_llm_tool_calls": 8, "retry_count": 0},
        },
        "evaluation_metadata": {
            "search_api": "tavily",
            "max_researcher_iterations": 6,
            "max_react_tool_calls": 10,
        },
    }


def test_evidence_evaluators_accept_notes_fallback(monkeypatch) -> None:
    monkeypatch.setattr(evaluators, "_get_eval_model", lambda: _FakeJudge())
    inputs = {"messages": [{"role": "user", "content": "Research claim A"}]}
    outputs = _outputs()
    outputs["notes"] = outputs.pop("raw_notes")

    assert evaluators.eval_groundedness(inputs, outputs)[0]["score"] == 1
    assert evaluators.eval_citation_accuracy(inputs, outputs)["score"] == 1


def test_required_quality_metrics_are_emitted(monkeypatch) -> None:
    monkeypatch.setattr(evaluators, "_get_eval_model", lambda: _FakeJudge())
    inputs = {"messages": [{"role": "user", "content": "Research claim A"}]}
    outputs = _outputs()

    grounded = evaluators.eval_groundedness(inputs, outputs)
    citation = evaluators.eval_citation_accuracy(inputs, outputs)
    completeness = evaluators.eval_completeness(inputs, outputs)
    overall = evaluators.eval_overall_quality(inputs, outputs)
    efficiency = evaluators.eval_tool_efficiency(inputs, outputs)

    keys = {
        item["key"]
        for result in (grounded, citation, completeness, overall, efficiency)
        for item in (result if isinstance(result, list) else [result])
    }
    assert {
        "factual_accuracy_score",
        "citation_accuracy_score",
        "completeness_score",
        "source_quality_score",
        "tool_efficiency_score",
    } <= keys


def test_failed_run_returns_zero_scores_instead_of_key_errors(monkeypatch) -> None:
    monkeypatch.setattr(evaluators, "_get_eval_model", lambda: _FakeJudge())
    inputs = {"messages": [{"role": "user", "content": "Research claim A"}]}
    outputs = {"result": {"status": "error", "error": "provider unavailable"}}

    assert evaluators.eval_citation_accuracy(inputs, outputs)["score"] == 0
    assert evaluators.eval_completeness(inputs, outputs)["score"] == 0
    assert evaluators.eval_tool_efficiency(inputs, outputs)["score"] == 0
    grounded = evaluators.eval_groundedness(inputs, outputs)
    assert all(item["score"] == 0 for item in grounded)


def test_empty_claim_and_citation_lists_do_not_divide_by_zero(monkeypatch) -> None:
    class EmptyRunner(_StructuredRunner):
        def invoke(self, _messages: list[dict[str, Any]]) -> Any:
            if self.schema.__name__ == "GroundednessScore":
                return self.schema(claims=[])
            return self.schema(citations=[], reasoning="no citations")

    class EmptyJudge:
        def with_structured_output(self, schema: type, **_kwargs: Any) -> EmptyRunner:
            return EmptyRunner(schema)

    monkeypatch.setattr(evaluators, "_get_eval_model", lambda: EmptyJudge())
    inputs = {"messages": [{"role": "user", "content": "Research claim A"}]}

    assert evaluators.eval_groundedness(inputs, _outputs())[0]["score"] == 0
    assert evaluators.eval_citation_accuracy(inputs, _outputs())["score"] == 0


def test_parallelism_evaluator_reads_current_top_level_state() -> None:
    result = right_parallelism_evaluator(_outputs(), {"parallel": 1})

    assert result["score"] is True
    assert "observed 1" in result["comment"]
