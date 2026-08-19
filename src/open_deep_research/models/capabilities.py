"""Small provider/model capability rules shared across model factories."""

from __future__ import annotations


def dashscope_qwen_enable_thinking(model_spec: str) -> bool:
    """Return the thinking flag required by one DashScope Qwen model.

    Qwen 3.7 Max endpoints reject ``enable_thinking=False``. Other Qwen models
    retain the established non-thinking JSON-evaluator behavior.
    """
    _provider, separator, model = model_spec.partition(":")
    normalized = (model if separator else model_spec).strip().lower()
    return normalized.startswith("qwen3.7-max")
