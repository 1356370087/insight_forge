"""Provider-aware evaluation Judge construction tests."""

from __future__ import annotations

import pytest

from open_deep_research.evaluation import judge


def test_judge_falls_back_to_runtime_quality_model_configuration(
    monkeypatch,
) -> None:
    monkeypatch.delenv("EVALUATION_MODEL", raising=False)
    monkeypatch.delenv("EVALUATION_BASE_URL", raising=False)
    monkeypatch.delenv("EVALUATION_API_KEY", raising=False)
    monkeypatch.delenv("QUALITY_EVALUATION_API_KEY", raising=False)
    monkeypatch.setenv("QUALITY_EVALUATION_MODEL", "openai:qwen3.7-plus")
    monkeypatch.setenv(
        "QUALITY_EVALUATION_BASE_URL",
        "https://workspace.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
    )
    monkeypatch.setenv("DASHSCOPE_API_KEY", "dashscope-key")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")

    resolved = judge.JudgeConfig.from_env()

    assert resolved.provider == "openai"
    assert resolved.model == "qwen3.7-plus"
    assert resolved.api_key == "dashscope-key"
    assert (
        resolved.base_url
        == "https://workspace.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
    )


def test_qwen_judge_disables_thinking_without_deepseek_options(monkeypatch) -> None:
    captured: dict = {}
    monkeypatch.setattr(
        judge,
        "ChatOpenAI",
        lambda **kwargs: captured.update(kwargs) or object(),
    )
    config = judge.JudgeConfig(
        provider="openai",
        model="qwen3.7-plus",
        api_key="dashscope-key",
        base_url=(
            "https://workspace.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
        ),
    )

    judge.build_judge_model(config)

    assert captured["extra_body"] == {"enable_thinking": False}


def test_qwen_max_judge_enables_required_thinking(monkeypatch) -> None:
    captured: dict = {}
    monkeypatch.setattr(
        judge,
        "ChatOpenAI",
        lambda **kwargs: captured.update(kwargs) or object(),
    )
    config = judge.JudgeConfig(
        provider="openai",
        model="qwen3.7-max-2026-05-17",
        api_key="dashscope-key",
        base_url=(
            "https://workspace.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
        ),
    )

    judge.build_judge_model(config)

    assert captured["extra_body"] == {
        "enable_thinking": True,
        "thinking_budget": config.max_tokens,
    }
    assert "max_tokens" not in captured


def test_deepseek_judge_uses_only_deepseek_compatible_configuration(
    monkeypatch,
) -> None:
    captured: dict = {}
    monkeypatch.setenv("EVALUATION_MODEL", "deepseek:deepseek-v4-flash")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-key")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://deepseek.example/v1")
    monkeypatch.setattr(
        judge,
        "ChatOpenAI",
        lambda **kwargs: captured.update(kwargs) or object(),
    )

    judge.build_judge_model(judge.JudgeConfig.from_env())

    assert captured["model"] == "deepseek-v4-flash"
    assert captured["api_key"] == "deepseek-key"
    assert captured["base_url"] == "https://deepseek.example/v1"
    assert captured["extra_body"] == {"thinking": {"type": "disabled"}}


def test_anthropic_judge_uses_native_client_without_openai_extra_body(
    monkeypatch,
) -> None:
    captured: dict = {}
    monkeypatch.setenv("EVALUATION_MODEL", "anthropic:claude-sonnet-4-5")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-key")
    monkeypatch.setenv("EVALUATION_BASE_URL", "https://anthropic.example")
    monkeypatch.setattr(
        judge,
        "ChatAnthropic",
        lambda **kwargs: captured.update(kwargs) or object(),
    )

    judge.build_judge_model(judge.JudgeConfig.from_env())

    assert captured == {
        "model": "claude-sonnet-4-5",
        "api_key": "anthropic-key",
        "base_url": "https://anthropic.example",
        "max_tokens": 8192,
        "max_retries": 2,
        "temperature": 0,
    }


def test_unknown_judge_provider_is_rejected(monkeypatch) -> None:
    monkeypatch.setenv("EVALUATION_MODEL", "unknown:model")

    with pytest.raises(ValueError, match="Unsupported evaluation provider"):
        judge.JudgeConfig.from_env()
