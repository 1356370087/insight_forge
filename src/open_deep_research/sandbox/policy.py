"""Shared egress-domain policy for the sandbox and tool-governance layers.

These functions are pure (no Docker dependency) so the tool-governance layer can
enforce the same domain allowlist the sandbox derives, without importing the
Docker SDK. :func:`allowed_domains` mirrors the former
``DockerSandboxManager._allowed_domains`` derivation.
"""

from __future__ import annotations

from urllib.parse import urlparse

from open_deep_research.configuration import Configuration, SearchAPI

#: Network modes in which the tool-layer egress check actively enforces.
ENFORCE_NETWORK_MODES = {"allow-search-only", "allowlist-domain"}

#: Network modes in which egress is unrestricted (or fully offline).
PASS_THROUGH_NETWORK_MODES = {"no-network", "open-network"}


def allowed_domains(configurable: Configuration) -> list[str]:
    """Return the sorted set of egress hosts derived from configuration.

    Domains come from ``sandbox_allowed_domains`` plus the provider model hosts,
    the search-API host, and the MCP server URL host. Mirrors the logic that used
    to live on :class:`DockerSandboxManager`.
    """
    domains: set[str] = set(configurable.sandbox_allowed_domains or [])
    for model_name in (
        configurable.research_model,
        configurable.compression_model,
        configurable.summarization_model,
    ):
        model = (model_name or "").lower()
        if model.startswith("openai:"):
            domains.add("api.openai.com")
        elif model.startswith("anthropic:"):
            domains.add("api.anthropic.com")
        elif model.startswith("google:") or model.startswith("gemini:"):
            domains.add("generativelanguage.googleapis.com")
        elif model.startswith("groq:"):
            domains.add("api.groq.com")
        elif model.startswith("deepseek:"):
            domains.add("api.deepseek.com")

    search_api = configurable.search_api
    search_value = search_api.value if isinstance(search_api, SearchAPI) else str(search_api)
    if search_value == SearchAPI.TAVILY.value:
        domains.add("api.tavily.com")
    elif search_value == SearchAPI.OPENAI.value:
        domains.add("api.openai.com")
    elif search_value == SearchAPI.ANTHROPIC.value:
        domains.add("api.anthropic.com")

    if configurable.mcp_config and configurable.mcp_config.url:
        host = urlparse(configurable.mcp_config.url).hostname
        if host:
            domains.add(host)

    if configurable.browser_mcp_enabled and configurable.browser_mcp_config and configurable.browser_mcp_config.url:
        host = urlparse(configurable.browser_mcp_config.url).hostname
        if host:
            domains.add(host)

    return sorted(domain for domain in domains if domain)


def egress_host_from_url(url: str) -> str | None:
    """Extract a normalized lowercase hostname from a URL, or ``None`` if unparseable."""
    try:
        host = urlparse(url).hostname
    except (ValueError, TypeError):
        return None
    return host.lower() if host else None


def is_enforced_mode(configurable: Configuration) -> bool:
    """Return ``True`` iff the tool-layer egress check should actively enforce."""
    return configurable.sandbox_network_mode in ENFORCE_NETWORK_MODES
