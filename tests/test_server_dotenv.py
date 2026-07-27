"""Regression tests for loading project configuration at service startup."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_server_import_loads_dotenv_before_building_configuration(tmp_path) -> None:
    """The documented uvicorn command must honor a colocated .env file."""
    (tmp_path / ".env").write_text(
        "\n".join([
            "QUALITY_EVALUATION_ENABLED=true",
            "QUALITY_EVALUATION_MODEL=openai:qwen3.7-plus",
            "QUALITY_EVALUATION_BASE_URL=https://workspace.example.test/v1",
        ]),
        encoding="utf-8",
    )
    child_env = os.environ.copy()
    for name in (
        "QUALITY_EVALUATION_ENABLED",
        "QUALITY_EVALUATION_MODEL",
        "QUALITY_EVALUATION_BASE_URL",
    ):
        child_env.pop(name, None)
    python_path = str(ROOT / "src")
    if child_env.get("PYTHONPATH"):
        python_path += os.pathsep + child_env["PYTHONPATH"]
    child_env["PYTHONPATH"] = python_path
    code = """
import json
import open_deep_research.server
from open_deep_research.configuration import Configuration
configuration = Configuration.from_runnable_config(None)
print(json.dumps({
    "enabled": configuration.quality_evaluation_enabled,
    "model": configuration.quality_evaluation_model,
    "base_url": configuration.quality_evaluation_base_url,
}))
"""

    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env=child_env,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    resolved = json.loads(completed.stdout.strip().splitlines()[-1])

    assert resolved == {
        "enabled": True,
        "model": "openai:qwen3.7-plus",
        "base_url": "https://workspace.example.test/v1",
    }
