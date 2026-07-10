#!/usr/bin/env python3
"""Extract data from LangSmith and save to JSONL file with configurable dataset."""

import argparse
import json
import os

from dotenv import load_dotenv
from langsmith import Client

load_dotenv()


def extract_langsmith_data(project_name, model_name, dataset_name, api_key):
    """Extract data from LangSmith and save to JSONL file."""
    print(f"Extracting data from LangSmith project: {project_name}")  # noqa: T201
    print(f"Using dataset: {dataset_name}")  # noqa: T201
    
    client = Client(api_key=api_key)
    
    # Read project to get reference dataset id
    project_data = client.read_project(project_name=project_name)
    
    # Read reference dataset to get examples
    examples_gen = client.list_examples(dataset_id=project_data.reference_dataset_id)
    examples = []
    for example in examples_gen:
        examples.append(example)
    examples_dict = {example.id: example for example in examples}
    
    # Read project runs to get runs inputs and outputs and reference_example_ids
    output_runs = client.list_runs(
        project_name=project_name,
        is_root=True
    )
    
    runs = []
    for run in output_runs:
        if run.outputs is not None and run.outputs.get("final_report") is not None:
            runs.append(run)
    
    output_jsonl = []
    for run in runs:
        run_inputs = run.inputs.get("inputs", run.inputs)
        messages = run_inputs.get("messages", [])
        example = examples_dict.get(run.reference_example_id)
        if not messages or example is None:
            continue
        output_jsonl.append({
            "id": example.metadata["id"],
            "prompt": messages[0]["content"],
            "article": run.outputs["final_report"],
        })
    
    # Write output_jsonl to JSONL file in tests/expt_results directory
    output_file_path = f"tests/expt_results/{dataset_name}_{model_name}.jsonl"
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    with open(output_file_path, 'w', encoding='utf-8') as f:
        for item in output_jsonl:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"Data written to {output_file_path}")  # noqa: T201
    print(f"Total records: {len(output_jsonl)}")  # noqa: T201
    return output_file_path


def main():
    parser = argparse.ArgumentParser(description='Extract data from LangSmith project')
    parser.add_argument('--project-name', required=True, help='LangSmith project name')
    parser.add_argument('--model-name', required=True, help='Model name for output filename')
    parser.add_argument('--dataset-name', required=True, help='Dataset name for output filename')
    parser.add_argument('--api-key', help='LangSmith API key (defaults to LANGSMITH_API_KEY env var)')
    
    args = parser.parse_args()
    
    api_key = args.api_key or os.getenv('LANGSMITH_API_KEY')
    if not api_key:
        raise ValueError("API key must be provided via --api-key or LANGSMITH_API_KEY environment variable")
    
    extract_langsmith_data(
        project_name=args.project_name,
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        api_key=api_key
    )


if __name__ == "__main__":
    main()
