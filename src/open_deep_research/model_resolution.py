"""Shared provider-neutral model parsing and configuration helpers.

This module intentionally depends only on LangChain, the standard library, and
the small :mod:`model_capabilities` rule set.  Keeping it below configuration,
agents, tools, and report assembly makes the model boundary safe to reuse from
all of those layers without creating import cycles.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from functools import lru_cache
from typing import Any

from langchain.chat_models import init_chat_model

from open_deep_research.model_capabilities import (
    dashscope_qwen_enable_thinking,
)

_DASHSCOPE_DEFAULT_BASE_URL = (
    "https://dashscope.aliyuncs.com/compatible-mode/v1"
)
_CONFIGURABLE_MODEL_FIELDS = (
    "model",
    "max_tokens",
    "max_retries",
    "api_key",
    "base_url",
    "default_headers",
    "headers",
    "extra_body",
)

_PROVIDER_KEY_CANDIDATES: dict[str, tuple[str, ...]] = {
    "anthropic": ("ANTHROPIC_API_KEY",),
    "azure_openai": ("AZURE_OPENAI_API_KEY", "OPENAI_API_KEY"),
    "cohere": ("COHERE_API_KEY",),
    "deepseek": ("DEEPSEEK_API_KEY", "OPENAI_API_KEY"),
    "google": ("GOOGLE_API_KEY",),
    "google_genai": ("GOOGLE_API_KEY",),
    "google_vertexai": ("GOOGLE_API_KEY",),
    "groq": ("GROQ_API_KEY",),
    "mistralai": ("MISTRAL_API_KEY",),
    "openai": ("OPENAI_API_KEY",),
    "xai": ("XAI_API_KEY",),
}


def parse_model_spec(model_spec: str) -> tuple[str, str]:
    """Parse a model identifier into its normalized provider and model name.

    Explicit ``provider:model`` identifiers preserve any non-empty provider.
    Unprefixed identifiers use the historic runtime heuristics shared by the
    quality path, with OpenAI as the compatibility default.
    """
    normalized = str(model_spec or "").strip()
    if not normalized:
        raise ValueError("model specification must not be empty")

    provider, separator, model = normalized.partition(":")
    if separator:
        provider = provider.strip().lower()
        model = model.strip()
        if not provider:
            raise ValueError("model specification must include a provider")
        if not model:
            raise ValueError("model specification must include a model name")
        return provider, model

    lowered = normalized.lower()
    if lowered.startswith("claude"):
        return "anthropic", normalized
    if lowered.startswith(("gemini", "gemma")):
        return "google_genai", normalized
    if "deepseek" in lowered:
        return "deepseek", normalized
    return "openai", normalized


def is_dashscope_qwen(
    model_spec: str,
    base_url: str | None = None,
) -> bool:
    """Return whether a model uses DashScope's OpenAI-compatible Qwen API."""
    provider, model = parse_model_spec(model_spec)
    lowered_url = (base_url or "").lower()
    return provider == "openai" and (
        model.lower().startswith("qwen")
        or "dashscope.aliyuncs.com" in lowered_url
        or ".maas.aliyuncs.com" in lowered_url
    )


def _key_source(config: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if os.getenv("GET_API_KEYS_FROM_CONFIG", "false").lower() != "true":
        return os.environ
    configurable = (config or {}).get("configurable", {})
    if not isinstance(configurable, Mapping):
        return {}
    api_keys = configurable.get("apiKeys", {})
    return api_keys if isinstance(api_keys, Mapping) else {}


def _first_value(
    source: Mapping[str, Any],
    names: tuple[str, ...],
) -> str | None:
    for name in names:
        value = source.get(name)
        if value:
            return str(value)
    return None


def resolve_api_key(
    model_spec: str,
    config: Mapping[str, Any] | None = None,
    *,
    override_key: str | None = None,
    base_url: str | None = None,
) -> str | None:
    """Resolve a model credential using the shared isolation and priority rules.

    The ``GET_API_KEYS_FROM_CONFIG`` switch remains authoritative: when true,
    only ``configurable.apiKeys`` is inspected; otherwise only the environment
    is inspected.  A supplied role-specific key always wins.
    """
    if override_key:
        return str(override_key)

    source = _key_source(config)
    provider, model = parse_model_spec(model_spec)
    if is_dashscope_qwen(model_spec, base_url):
        dashscope_key = _first_value(source, ("DASHSCOPE_API_KEY",))
        if dashscope_key:
            return dashscope_key
    if provider == "deepseek" or (
        provider == "openai" and model.lower().startswith("deepseek")
    ):
        return _first_value(source, _PROVIDER_KEY_CANDIDATES["deepseek"])
    candidates = _PROVIDER_KEY_CANDIDATES.get(
        provider,
        (f"{provider.upper()}_API_KEY",),
    )
    return _first_value(source, candidates)


def resolve_base_url(
    model_spec: str,
    *,
    configured: str | None = None,
) -> str | None:
    """Resolve an explicitly configured or provider-default model endpoint."""
    if configured:
        return configured
    provider, model = parse_model_spec(model_spec)
    if is_dashscope_qwen(model_spec):
        return os.getenv("DASHSCOPE_BASE_URL") or _DASHSCOPE_DEFAULT_BASE_URL
    if provider == "deepseek" or (
        provider == "openai" and model.lower().startswith("deepseek")
    ):
        return os.getenv("DEEPSEEK_BASE_URL") or "https://api.deepseek.com"
    if provider == "openai":
        return os.getenv("OPENAI_BASE_URL")
    if provider == "anthropic":
        return os.getenv("ANTHROPIC_BASE_URL")
    return os.getenv(f"{provider.upper()}_BASE_URL")


def resolve_compatibility_kwargs(model_spec: str) -> dict[str, Any]:
    """Return request options required by an OpenAI-compatible model."""
    provider, model = parse_model_spec(model_spec)
    lowered_model = model.lower()
    if provider == "deepseek" or (
        provider == "openai" and lowered_model.startswith("deepseek")
    ):
        return {"extra_body": {"thinking": {"type": "disabled"}}}
    if is_dashscope_qwen(model_spec):
        return {
            "extra_body": {
                "enable_thinking": dashscope_qwen_enable_thinking(model_spec),
            }
        }
    return {}


def _configured_role_value(
    config: Mapping[str, Any] | None,
    role: str,
    suffix: str,
) -> str | None:
    configurable = (config or {}).get("configurable", {})
    if not isinstance(configurable, Mapping):
        return None
    value = configurable.get(f"{role}_{suffix}")
    return str(value) if value else None


def build_model_config(
    model_spec: str,
    max_tokens: int,
    config: Mapping[str, Any] | None,
    *,
    role: str,
    tags: bool = True,
    configured_base_url: str | None = None,
) -> dict[str, Any]:
    """Build the shared model, token, connection, tag, and compatibility set."""
    source = _key_source(config)
    role_key_name = f"{role.upper()}_API_KEY"
    role_override = _first_value(source, (role_key_name,))
    configured_base_url = configured_base_url or _configured_role_value(
        config,
        role,
        "base_url",
    )
    api_key = resolve_api_key(
        model_spec,
        config,
        override_key=role_override,
        base_url=configured_base_url,
    )
    model_config: dict[str, Any] = {
        "model": model_spec,
        "max_tokens": max_tokens,
        "max_retries": 0,
        "api_key": api_key,
    }
    if base_url := resolve_base_url(
        model_spec,
        configured=configured_base_url,
    ):
        model_config["base_url"] = base_url
    if tags:
        model_config["tags"] = ["langsmith:nostream"]
    compatibility = resolve_compatibility_kwargs(model_spec)
    model_config.update(compatibility)
    extra_body = model_config.get("extra_body")
    if (
        isinstance(extra_body, dict)
        and extra_body.get("enable_thinking") is True
    ):
        extra_body["thinking_budget"] = max_tokens
        model_config.pop("max_tokens", None)
    return model_config


@lru_cache(maxsize=1)
def get_configurable_model_template() -> Any:
    """Return the process-wide lazily initialized configurable chat template."""
    return init_chat_model(configurable_fields=_CONFIGURABLE_MODEL_FIELDS)
