#!/usr/bin/env python3
"""Strictly export one complete benchmark record per LangSmith example."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langsmith import Client

from open_deep_research.evaluation.benchmark_export import (
    benchmark_output_path,
    build_benchmark_records,
    write_benchmark_jsonl,
)

load_dotenv()


def extract_langsmith_data(
    project_name: str,
    model_name: str,
    dataset_name: str,
    api_key: str | None,
    *,
    client: Any | None = None,
    output_dir: Path = Path("tests/expt_results"),
) -> str:
    """Validate a LangSmith experiment completely before writing JSONL."""
    print(f"Extracting data from LangSmith project: {project_name}")  # noqa: T201
    print(f"Using dataset: {dataset_name}")  # noqa: T201
    if client is None and not api_key:
        raise ValueError(
            "API key must be provided via --api-key or LANGSMITH_API_KEY"
        )
    effective_client = client or Client(api_key=api_key)
    project_data = effective_client.read_project(project_name=project_name)
    reference_dataset_id = getattr(project_data, "reference_dataset_id", None)
    if reference_dataset_id is None:
        raise ValueError("LangSmith project has no reference dataset")
    examples = list(
        effective_client.list_examples(dataset_id=reference_dataset_id)
    )
    runs = list(
        effective_client.list_runs(project_name=project_name, is_root=True)
    )
    records = build_benchmark_records(examples, runs)
    output_path = benchmark_output_path(
        output_dir,
        dataset_name=dataset_name,
        model_name=model_name,
    )
    write_benchmark_jsonl(output_path, records)
    print(f"Data written to {output_path}")  # noqa: T201
    print(f"Total records: {len(records)}")  # noqa: T201
    return str(output_path)


def main() -> None:
    """Run the strict LangSmith export CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-name",
        required=True,
        help="LangSmith project name",
    )
    parser.add_argument(
        "--model-name",
        required=True,
        help="Model name for output filename",
    )
    parser.add_argument(
        "--dataset-name",
        required=True,
        help="Dataset name for output filename",
    )
    parser.add_argument(
        "--api-key",
        help="LangSmith API key (defaults to LANGSMITH_API_KEY)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("tests/expt_results"),
        help="Validated JSONL destination directory",
    )
    args = parser.parse_args()
    extract_langsmith_data(
        project_name=args.project_name,
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        api_key=args.api_key or os.getenv("LANGSMITH_API_KEY"),
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
