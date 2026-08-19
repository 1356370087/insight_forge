"""Shared pytest isolation for project-local dotenv configuration."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from dotenv import dotenv_values

from open_deep_research.model_circuit import _reset_model_circuit_registry

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_ENV_AT_PYTEST_START = frozenset(os.environ)
_PROJECT_DOTENV_KEYS = frozenset(
    str(key)
    for key in dotenv_values(_PROJECT_ROOT / ".env")
    if key and key not in _ENV_AT_PYTEST_START
)


@pytest.fixture(autouse=True)
def isolate_project_dotenv_from_unit_tests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prevent server-import dotenv values from overriding per-test config."""
    for key in _PROJECT_DOTENV_KEYS:
        monkeypatch.delenv(key, raising=False)


@pytest.fixture(autouse=True)
def isolate_model_circuit_registry() -> None:
    """Prevent process-local model health from leaking between tests."""
    _reset_model_circuit_registry()
    yield
    _reset_model_circuit_registry()
