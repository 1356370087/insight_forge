"""Regression coverage for the shared model-resolution boundary."""

from __future__ import annotations

from collections import OrderedDict

import pytest

from open_deep_research.model_resolution import (
    build_model_config,
    get_configurable_model_template,
    parse_model_spec,
    resolve_api_key,
    resolve_base_url,
    resolve_compatibility_kwargs,
)
from open_deep_research.tools import model_limits


@pytest.mark.parametrize(
    ("model_spec", "expected"),
    [
        ("anthropic:claude-sonnet-4", ("anthropic", "claude-sonnet-4")),
        ("azure_openai:gpt-4.1", ("azure_openai", "gpt-4.1")),
        ("google_genai:gemini-2.5-pro", ("google_genai", "gemini-2.5-pro")),
        ("openai:qwen3.7-max", ("openai", "qwen3.7-max")),
        ("claude-sonnet-4", ("anthropic", "claude-sonnet-4")),
        ("gemini-2.5-pro", ("google_genai", "gemini-2.5-pro")),
        ("gemma-3-27b", ("google_genai", "gemma-3-27b")),
        ("deepseek-chat", ("deepseek", "deepseek-chat")),
        ("gpt-4.1", ("openai", "gpt-4.1")),
    ],
)
def test_parse_model_spec_matrix(
    model_spec: str,
    expected: tuple[str, str],
) -> None:
    assert parse_model_spec(model_spec) == expected


@pytest.mark.parametrize("model_spec", ["", "  ", ":gpt-4.1", "openai:"])
def test_parse_model_spec_rejects_incomplete_values(model_spec: str) -> None:
    with pytest.raises(ValueError):
        parse_model_spec(model_spec)


def test_resolve_api_key_uses_role_override_before_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GET_API_KEYS_FROM_CONFIG", "false")
    monkeypatch.setenv("OPENAI_API_KEY", "provider-key")
    assert (
        resolve_api_key("openai:gpt-4.1", override_key="role-key")
        == "role-key"
    )


def test_resolve_api_key_uses_dashscope_before_openai(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GET_API_KEYS_FROM_CONFIG", "false")
    monkeypatch.setenv("DASHSCOPE_API_KEY", "dashscope-key")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    assert resolve_api_key("openai:qwen-plus") == "dashscope-key"


def test_resolve_api_key_prefers_deepseek_for_compatible_openai_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GET_API_KEYS_FROM_CONFIG", "false")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-key")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    assert resolve_api_key("openai:deepseek-chat") == "deepseek-key"


def test_resolve_api_key_honors_config_only_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GET_API_KEYS_FROM_CONFIG", "true")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "environment-key")
    config = {
        "configurable": {"apiKeys": {"ANTHROPIC_API_KEY": "config-key"}}
    }
    assert resolve_api_key("anthropic:claude-sonnet-4", config) == "config-key"


def test_resolve_base_url_includes_dashscope_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("DASHSCOPE_BASE_URL", raising=False)
    assert resolve_base_url("openai:qwen-plus") == (
        "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    assert (
        resolve_base_url("openai:qwen-plus", configured="https://example.test/v1")
        == "https://example.test/v1"
    )


def test_resolve_compatibility_kwargs_covers_deepseek_and_qwen() -> None:
    assert resolve_compatibility_kwargs("openai:deepseek-v4-pro") == {
        "extra_body": {"thinking": {"type": "disabled"}}
    }
    assert resolve_compatibility_kwargs("openai:qwen-plus") == {
        "extra_body": {"enable_thinking": False}
    }


def test_build_model_config_assembles_shared_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GET_API_KEYS_FROM_CONFIG", "false")
    monkeypatch.setenv("DASHSCOPE_API_KEY", "dashscope-key")
    config = build_model_config(
        "openai:qwen3.7-max",
        12_345,
        {},
        role="quality_evaluation",
    )
    assert config == {
        "model": "openai:qwen3.7-max",
        "max_retries": 0,
        "api_key": "dashscope-key",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "tags": ["langsmith:nostream"],
        "extra_body": {
            "enable_thinking": True,
            "thinking_budget": 12_345,
        },
    }


def test_configurable_model_template_is_lazy_singleton() -> None:
    assert get_configurable_model_template() is get_configurable_model_template()


def test_token_limit_prefers_exact_then_longest_match(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    limits = OrderedDict(
        [
            ("openai:example", 100),
            ("openai:example[large]", 1_000),
        ]
    )
    monkeypatch.setattr(model_limits, "MODEL_TOKEN_LIMITS", limits)
    assert model_limits.get_model_token_limit("openai:example") == 100
    assert model_limits.get_model_token_limit("openai:example[large]") == 1_000
    assert model_limits.get_model_token_limit("proxy/openai:example[large]-v2") == 1_000
