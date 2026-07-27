"""Strict benchmark export validation tests."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from open_deep_research.evaluation.benchmark_export import (
    ExportValidationError,
    build_benchmark_records,
    validate_benchmark_records,
)


def _example(example_id: int, prompt: str = "prompt") -> SimpleNamespace:
    return SimpleNamespace(
        id=f"ref-{example_id}",
        metadata={"id": example_id},
        inputs={"messages": [{"role": "user", "content": prompt}]},
    )


def _run(
    example_id: int,
    *,
    report: str | None = "report",
    prompt: str = "prompt",
) -> SimpleNamespace:
    return SimpleNamespace(
        id=f"run-{example_id}",
        reference_example_id=f"ref-{example_id}",
        inputs={"messages": [{"role": "user", "content": prompt}]},
        outputs={"final_report": report} if report is not None else {},
        error=None,
    )


def test_export_fails_instead_of_silently_skipping_missing_run() -> None:
    examples = [_example(1), _example(2)]

    with pytest.raises(ExportValidationError, match=r"missing.*2"):
        build_benchmark_records(examples, [_run(1)])


def test_export_fails_for_duplicate_or_incomplete_runs() -> None:
    examples = [_example(1), _example(2)]
    runs = [_run(1), _run(1), _run(2, report=None)]

    with pytest.raises(ExportValidationError) as error:
        build_benchmark_records(examples, runs)

    message = str(error.value)
    assert "duplicate" in message
    assert "no non-empty final_report" in message


def test_valid_export_is_complete_and_stably_sorted() -> None:
    examples = [_example(10, "ten"), _example(2, "two")]
    records = build_benchmark_records(
        examples,
        [_run(10, report="article ten", prompt="ten"), _run(2, report="article two", prompt="two")],
    )

    assert records == [
        {"id": 2, "prompt": "two", "article": "article two"},
        {"id": 10, "prompt": "ten", "article": "article ten"},
    ]


def test_record_validator_reports_exact_missing_ids() -> None:
    records = [
        {"id": item_id, "prompt": f"p{item_id}", "article": f"a{item_id}"}
        for item_id in range(1, 101)
        if item_id not in {57, 98}
    ]

    with pytest.raises(ExportValidationError) as error:
        validate_benchmark_records(records, expected_ids=range(1, 101))

    assert error.value.missing_ids == (57, 98)
