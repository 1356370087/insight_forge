"""Declarative availability predicates shared by built-in tool definitions."""

from __future__ import annotations

from langchain_core.runnables import RunnableConfig

from open_deep_research.configuration import Configuration, SearchAPI
from open_deep_research.sandbox.policy import network_policy_mode


def _configuration(config: RunnableConfig) -> Configuration:
    return Configuration.from_runnable_config(config)


def provider_search_enabled(config: RunnableConfig, provider: SearchAPI) -> bool:
    """Return whether a legacy/shadow provider search tool is enabled."""
    configurable = _configuration(config)
    return (
        network_policy_mode(configurable) != "offline"
        and configurable.web_pipeline_mode in {"legacy", "shadow"}
        and SearchAPI(configurable.search_api) is provider
    )


def legacy_fetch_enabled(config: RunnableConfig) -> bool:
    """Return whether the legacy webpage fetcher is enabled."""
    return _configuration(config).web_pipeline_mode in {"legacy", "shadow"}


def enforced_pipeline_enabled(config: RunnableConfig) -> bool:
    """Return whether enforced pipeline tools are enabled."""
    return _configuration(config).web_pipeline_mode == "enforced"
