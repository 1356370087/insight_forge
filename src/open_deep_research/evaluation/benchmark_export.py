"""Strict, deterministic benchmark export validation and persistence."""

from __future__ import annotations

import json
import os
import re
import uuid
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class ExportValidationError(ValueError):
    """Raised when an export cannot prove one valid record per dataset example."""

    def __init__(
        self,
        message: str,
        *,
        missing_ids: Iterable[Any] = (),
        duplicate_ids: Iterable[Any] = (),
        invalid_ids: Iterable[Any] = (),
        unexpected_ids: Iterable[Any] = (),
    ) -> None:
        """Store machine-readable diagnostics alongside the error message."""
        super().__init__(message)
        self.missing_ids = tuple(missing_ids)
        self.duplicate_ids = tuple(duplicate_ids)
        self.invalid_ids = tuple(invalid_ids)
        self.unexpected_ids = tuple(unexpected_ids)


def _value(item: object, key: str, default: Any = None) -> Any:
    if isinstance(item, Mapping):
        return item.get(key, default)
    return getattr(item, key, default)


def _sort_key(value: Any) -> tuple[int, Any]:
    if isinstance(value, int) and not isinstance(value, bool):
        return (0, value)
    if isinstance(value, str) and value.isdecimal():
        return (0, int(value))
    return (1, str(value))


def _format_ids(values: Iterable[Any]) -> str:
    return ", ".join(str(value) for value in sorted(values, key=_sort_key))


def _extract_prompt(item: object) -> str | None:
    payload = _value(item, "inputs", {})
    if not isinstance(payload, Mapping):
        return None
    nested = payload.get("inputs")
    if isinstance(nested, Mapping):
        payload = nested
    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        return None
    content = _value(messages[0], "content")
    if not isinstance(content, str) or not content.strip():
        return None
    return content


def _dataset_id(example: object) -> Any:
    metadata = _value(example, "metadata", {})
    return metadata.get("id") if isinstance(metadata, Mapping) else None


def validate_benchmark_records(
    records: Iterable[Mapping[str, Any]],
    *,
    expected_ids: Iterable[Any],
) -> list[dict[str, Any]]:
    """Require exact ID coverage and non-empty prompt/article payloads."""
    materialized = [dict(record) for record in records]
    expected = list(expected_ids)
    expected_counts = Counter(expected)
    duplicate_expected = [
        item_id for item_id, count in expected_counts.items() if count > 1
    ]
    record_ids = [record.get("id") for record in materialized]
    record_counts = Counter(record_ids)
    duplicate_ids = [
        item_id for item_id, count in record_counts.items() if count > 1
    ]
    expected_set = set(expected)
    record_set = set(record_ids)
    missing_ids = sorted(expected_set - record_set, key=_sort_key)
    unexpected_ids = sorted(record_set - expected_set, key=_sort_key)
    invalid_ids = [
        record.get("id")
        for record in materialized
        if record.get("id") is None
        or not isinstance(record.get("prompt"), str)
        or not str(record.get("prompt", "")).strip()
        or not isinstance(record.get("article"), str)
        or not str(record.get("article", "")).strip()
    ]
    issues: list[str] = []
    if duplicate_expected:
        issues.append(f"duplicate expected ids: {_format_ids(duplicate_expected)}")
    if missing_ids:
        issues.append(f"missing ids: {_format_ids(missing_ids)}")
    if duplicate_ids:
        issues.append(f"duplicate record ids: {_format_ids(duplicate_ids)}")
    if unexpected_ids:
        issues.append(f"unexpected ids: {_format_ids(unexpected_ids)}")
    if invalid_ids:
        issues.append(f"invalid records: {_format_ids(invalid_ids)}")
    if issues:
        raise ExportValidationError(
            "benchmark export validation failed: " + "; ".join(issues),
            missing_ids=missing_ids,
            duplicate_ids=duplicate_ids,
            invalid_ids=invalid_ids,
            unexpected_ids=unexpected_ids,
        )
    return sorted(materialized, key=lambda record: _sort_key(record["id"]))


def build_benchmark_records(
    examples: Iterable[object],
    runs: Iterable[object],
) -> list[dict[str, Any]]:
    """Build one benchmark record per example, failing on every ambiguity."""
    example_list = list(examples)
    run_list = list(runs)
    examples_by_reference: dict[Any, object] = {}
    data_ids: list[Any] = []
    issues: list[str] = []
    invalid_ids: list[Any] = []
    for example in example_list:
        reference_id = _value(example, "id")
        data_id = _dataset_id(example)
        if reference_id is None or data_id is None:
            issues.append("dataset example is missing reference id or metadata.id")
            invalid_ids.append(data_id)
            continue
        if reference_id in examples_by_reference:
            issues.append(f"duplicate dataset reference id: {reference_id}")
            invalid_ids.append(data_id)
            continue
        examples_by_reference[reference_id] = example
        data_ids.append(data_id)

    runs_by_reference: dict[Any, list[object]] = defaultdict(list)
    unexpected_run_ids: list[Any] = []
    for run in run_list:
        reference_id = _value(run, "reference_example_id")
        if reference_id not in examples_by_reference:
            unexpected_run_ids.append(_value(run, "id", reference_id))
            continue
        runs_by_reference[reference_id].append(run)

    missing_ids: list[Any] = []
    duplicate_ids: list[Any] = []
    records: list[dict[str, Any]] = []
    for reference_id, example in examples_by_reference.items():
        data_id = _dataset_id(example)
        matching_runs = runs_by_reference.get(reference_id, [])
        if not matching_runs:
            missing_ids.append(data_id)
            continue
        if len(matching_runs) > 1:
            duplicate_ids.append(data_id)
            continue
        run = matching_runs[0]
        outputs = _value(run, "outputs", {})
        report = (
            outputs.get("final_report") if isinstance(outputs, Mapping) else None
        )
        run_error = _value(run, "error")
        if run_error or not isinstance(report, str) or not report.strip():
            invalid_ids.append(data_id)
            issues.append(f"id {data_id}: no non-empty final_report")
            continue
        dataset_prompt = _extract_prompt(example)
        run_prompt = _extract_prompt(run)
        if dataset_prompt is None or run_prompt is None:
            invalid_ids.append(data_id)
            issues.append(f"id {data_id}: missing prompt messages")
            continue
        if dataset_prompt != run_prompt:
            invalid_ids.append(data_id)
            issues.append(f"id {data_id}: run prompt differs from dataset prompt")
            continue
        records.append(
            {
                "id": data_id,
                "prompt": dataset_prompt,
                "article": report,
            }
        )

    if missing_ids:
        issues.append(f"missing ids: {_format_ids(missing_ids)}")
    if duplicate_ids:
        issues.append(f"duplicate runs for ids: {_format_ids(duplicate_ids)}")
    if unexpected_run_ids:
        issues.append(
            "runs with unknown reference ids: "
            + _format_ids(unexpected_run_ids)
        )
    if issues:
        raise ExportValidationError(
            "benchmark export validation failed: " + "; ".join(issues),
            missing_ids=sorted(missing_ids, key=_sort_key),
            duplicate_ids=sorted(duplicate_ids, key=_sort_key),
            invalid_ids=sorted(invalid_ids, key=_sort_key),
            unexpected_ids=sorted(unexpected_run_ids, key=_sort_key),
        )
    return validate_benchmark_records(records, expected_ids=data_ids)


def benchmark_output_path(
    output_dir: Path,
    *,
    dataset_name: str,
    model_name: str,
) -> Path:
    """Resolve a traversal-safe output filename beneath ``output_dir``."""
    for label, value in (
        ("dataset_name", dataset_name),
        ("model_name", model_name),
    ):
        if value in {".", ".."} or _SAFE_COMPONENT.fullmatch(value) is None:
            raise ValueError(f"unsafe {label}: {value!r}")
    return output_dir / f"{dataset_name}_{model_name}.jsonl"


def write_benchmark_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    """Atomically replace one JSONL artifact after validation has succeeded."""
    materialized = [dict(record) for record in records]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            for record in materialized:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
