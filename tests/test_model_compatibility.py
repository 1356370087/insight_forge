"""Tests for provider-specific model request compatibility."""

from open_deep_research.configuration import get_model_compatibility_kwargs


def test_openai_deepseek_disables_thinking():
    """OpenAI-compatible DeepSeek models must disable thinking for tool choice."""
    assert get_model_compatibility_kwargs("openai:deepseek-v4-pro") == {
        "extra_body": {"thinking": {"type": "disabled"}}
    }


def test_openai_deepseek_match_is_case_insensitive():
    """Provider/model matching should tolerate configuration casing."""
    assert get_model_compatibility_kwargs(" OpenAI:DeepSeek-V4-Flash") == {
        "extra_body": {"thinking": {"type": "disabled"}}
    }


def test_other_models_are_unchanged():
    """Compatibility options must not leak into ordinary OpenAI models."""
    assert get_model_compatibility_kwargs("openai:gpt-4.1") == {}
