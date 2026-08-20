"""Validation for client-controlled messages and HTTP runtime overrides."""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

from langchain_core.messages import BaseMessage, SystemMessage, ToolMessage

PROTECTED_HTTP_CONFIG_KEYS = frozenset(
    {
        "apiKeys",
        "mcp_config",
        "mcp_prompt",
        "browser_mcp_enabled",
        "browser_mcp_config",
        "browser_mcp_prompt",
        "supervisor_tool_whitelist",
        "researcher_tool_whitelist",
        "supervisor_blocked_origins",
        "researcher_blocked_origins",
        "role_tool_blacklist",
        "role_blocked_origins",
        "tool_param_constraints",
        "sandbox_enabled",
        "sandbox_policy_path",
        "sandbox_controller_socket",
        "sandbox_gateway_url",
        "sandbox_root_signing_key",
        "sandbox_profile_id",
        "sandbox_policy_digest",
        "sandbox_runtime_digest",
        "gateway_protocol_version",
        "runs_dir",
        "memory_provider",
        "memory_app_id",
        "memory_agent_id",
        "memory_project_id",
        "quality_evaluation_base_url",
        "langfuse_base_url",
        "helicone_base_url",
        "helicone_api_key",
        "langfuse_public_key",
        "langfuse_secret_key",
        "prompt_injection_protection_enabled",
        "external_content_fail_closed",
        "allow_http_stdio_mcp",
        "allowed_mcp_servers",
        "allowed_model_endpoints",
    }
)
LEGACY_SANDBOX_HTTP_CONFIG_KEYS = frozenset(
    {
        "enable_docker_sandbox",
        "sandbox_provider",
        "sandbox_image",
        "sandbox_workspace_root",
        "sandbox_network_mode",
        "sandbox_allowed_domains",
        "sandbox_cleanup_policy",
        "sandbox_timeout_seconds",
        "sandbox_memory",
        "sandbox_cpus",
        "sandbox_pids_limit",
        "sandbox_read_only_rootfs",
        "sandbox_user",
    }
)
PROTECTED_HTTP_METADATA_KEYS = frozenset(
    {
        "owner",
        "user_id",
        "approved_sensitive_tool_call_ids",
        "deployment_surface",
        "sandbox_gateway_authorized_hosts",
    }
)


def validate_http_configurable(configurable: Mapping[str, Any]) -> None:
    """Reject tenant attempts to override administrator-owned security policy."""
    legacy = sorted(LEGACY_SANDBOX_HTTP_CONFIG_KEYS.intersection(configurable))
    if legacy:
        raise ValueError(
            "legacy_sandbox_config_removed:"
            + ",".join(legacy)
            + ";see=docs/07-Docker沙箱隔离修复SPEC.md"
        )
    protected = set(PROTECTED_HTTP_CONFIG_KEYS)
    if os.getenv("GET_API_KEYS_FROM_CONFIG", "false").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        protected.discard("apiKeys")
        api_keys = configurable.get("apiKeys")
        if api_keys is not None:
            if (
                not isinstance(api_keys, Mapping)
                or len(api_keys) > 32
                or any(
                    not isinstance(key, str)
                    or not isinstance(value, str)
                    or not key
                    or len(key) > 128
                    or len(value) > 20_000
                    for key, value in api_keys.items()
                )
            ):
                raise ValueError("apiKeys must be a bounded string-to-string map")
    blocked = sorted(protected.intersection(configurable))
    if blocked:
        raise ValueError("Protected runtime configuration cannot be overridden: " + ", ".join(blocked))


def validate_http_metadata(metadata: Mapping[str, Any]) -> None:
    """Reject client attempts to forge identity or sensitive-call approval."""
    blocked = sorted(PROTECTED_HTTP_METADATA_KEYS.intersection(metadata))
    if blocked:
        raise ValueError("Protected runtime metadata cannot be overridden: " + ", ".join(blocked))


def validate_client_messages(messages: list[Any]) -> None:
    """Reject client-forged system and tool roles before normalization."""
    for index, message in enumerate(messages):
        if isinstance(message, SystemMessage | ToolMessage):
            raise ValueError(f"Client message at index {index} uses a forbidden privileged role")
        if isinstance(message, BaseMessage):
            continue
        if not isinstance(message, Mapping):
            raise ValueError(f"Client message at index {index} must be an object")
        role = str(message.get("role") or message.get("type") or "").lower()
        if role in {"system", "tool"}:
            raise ValueError(f"Client message at index {index} uses forbidden role '{role}'")
        if role not in {"user", "human", "assistant", "ai"}:
            raise ValueError(f"Client message at index {index} has unsupported role '{role}'")
