"""Versioned administrator policy for sandbox execution."""

from __future__ import annotations

import hashlib
import json
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Literal

import tomllib
from pydantic import BaseModel, ConfigDict, Field, model_validator


class FilesystemPolicy(BaseModel):
    """Filesystem roots visible to a sandboxed operation."""

    model_config = ConfigDict(extra="forbid")

    read_roots: list[str] = Field(default_factory=lambda: ["/workspace/input"])
    write_roots: list[str] = Field(
        default_factory=lambda: [
            "/workspace/output",
            "/workspace/logs",
            "/workspace/artifacts",
            "/workspace/tmp",
        ]
    )
    deny_read: list[str] = Field(default_factory=list)
    deny_write: list[str] = Field(default_factory=list)


class NetworkPolicy(BaseModel):
    """Gateway-enforced outbound network policy."""

    model_config = ConfigDict(extra="forbid")

    mode: Literal["offline", "gateway-only", "allowlist"] = "gateway-only"
    allow_domains: list[str] = Field(default_factory=list)
    deny_domains: list[str] = Field(default_factory=list)
    allow_ports: list[int] = Field(default_factory=lambda: [80, 443])
    allow_http_methods: list[str] = Field(
        default_factory=lambda: ["GET", "HEAD", "OPTIONS"]
    )
    allow_private_destinations: bool = False
    unknown_target: Literal["ask", "deny"] = "deny"

    @model_validator(mode="after")
    def validate_network_values(self) -> NetworkPolicy:
        """Normalize and validate ports, methods and domain spellings."""
        if any(port < 1 or port > 65535 for port in self.allow_ports):
            raise ValueError("sandbox network ports must be between 1 and 65535")
        self.allow_http_methods = sorted(
            {method.strip().upper() for method in self.allow_http_methods if method.strip()}
        )
        self.allow_domains = sorted(
            {domain.strip().lower() for domain in self.allow_domains if domain.strip()}
        )
        self.deny_domains = sorted(
            {domain.strip().lower() for domain in self.deny_domains if domain.strip()}
        )
        return self


class CommandPolicy(BaseModel):
    """Command patterns available to the optional developer shell tool."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    allow: list[str] = Field(default_factory=list)
    ask: list[str] = Field(default_factory=list)
    deny: list[str] = Field(default_factory=list)


class ToolPolicy(BaseModel):
    """Tool-effect policy evaluated before a tool crosses a trust boundary."""

    model_config = ConfigDict(extra="forbid")

    allow_effects: list[str] = Field(default_factory=lambda: ["read_only"])
    ask_effects: list[str] = Field(
        default_factory=lambda: ["sensitive_read", "external_write", "local_write"]
    )
    deny_effects: list[str] = Field(default_factory=lambda: ["destructive"])
    deny_tools: list[str] = Field(default_factory=list)


class ResourcePolicy(BaseModel):
    """Bounded task resources; tmpfs limits are maxima, not reservations."""

    model_config = ConfigDict(extra="forbid")

    memory_bytes: int = Field(default=1_073_741_824, ge=64 * 1024 * 1024)
    cpu_cores: float = Field(default=1.0, gt=0.0, le=64.0)
    pids: int = Field(default=256, ge=16, le=4096)
    timeout_seconds: int = Field(default=900, ge=1, le=86_400)
    approval_timeout_seconds: int = Field(default=900, ge=1, le=86_400)
    output_bytes: int = Field(default=64 * 1024 * 1024, ge=1024)
    log_bytes: int = Field(default=16 * 1024 * 1024, ge=1024)
    artifact_bytes: int = Field(default=256 * 1024 * 1024, ge=1024)
    tmp_bytes: int = Field(default=256 * 1024 * 1024, ge=1024)
    max_files: int = Field(default=10_000, ge=1, le=1_000_000)
    stop_grace_seconds: int = Field(default=5, ge=1, le=60)


class RuntimePolicy(BaseModel):
    """Immutable runtime identity and hardening requirements."""

    model_config = ConfigDict(extra="forbid")

    worker_image_digest: str
    browser_gateway_image_digest: str | None = None
    read_only_rootfs: bool = True
    uid: int = Field(default=65_532, ge=1)
    gid: int = Field(default=65_532, ge=1)
    seccomp_profile: str | None = None
    retention: Literal["remove", "retain_stopped"] = "remove"

    @model_validator(mode="after")
    def require_immutable_image(self) -> RuntimePolicy:
        """Require an immutable Docker image reference."""
        if "@sha256:" not in self.worker_image_digest and not self.worker_image_digest.startswith("sha256:"):
            raise ValueError("sandbox worker_image_digest must use an immutable sha256 reference")
        return self


class SandboxProfile(BaseModel):
    """One complete, administrator-owned sandbox capability profile."""

    model_config = ConfigDict(extra="forbid")

    provider: Literal["docker", "bubblewrap"] = "docker"
    approval_policy: Literal["on_request", "never"] = "never"
    filesystem: FilesystemPolicy = Field(default_factory=FilesystemPolicy)
    network: NetworkPolicy = Field(default_factory=NetworkPolicy)
    commands: CommandPolicy = Field(default_factory=CommandPolicy)
    tools: ToolPolicy = Field(default_factory=ToolPolicy)
    resources: ResourcePolicy = Field(default_factory=ResourcePolicy)
    runtime: RuntimePolicy


PolicyDecision = Literal["deny", "ask", "allow"]


def domain_matches(pattern: str, host: str) -> bool:
    """Match exact, one-label wildcard, or recursive wildcard domains."""
    normalized_pattern = pattern.lower().strip().rstrip(".")
    normalized_host = host.lower().strip().rstrip(".")
    if normalized_pattern == normalized_host:
        return True
    if normalized_pattern.startswith("**."):
        suffix = normalized_pattern[3:]
        return normalized_host == suffix or normalized_host.endswith("." + suffix)
    if normalized_pattern.startswith("*."):
        suffix = normalized_pattern[2:]
        return (
            normalized_host != suffix
            and normalized_host.endswith("." + suffix)
            and normalized_host.count(".") == suffix.count(".") + 1
        )
    return False


def network_target_decision(
    policy: NetworkPolicy,
    host: str,
    port: int,
) -> PolicyDecision:
    """Apply the global deny > ask > allow priority to one network target."""
    if policy.mode == "offline" or port not in policy.allow_ports:
        return "deny"
    if any(domain_matches(pattern, host) for pattern in policy.deny_domains):
        return "deny"
    allowed = any(domain_matches(pattern, host) for pattern in policy.allow_domains)
    if policy.unknown_target == "ask" and not allowed:
        return "ask"
    return "allow" if allowed else "deny"


def tool_policy_decision(
    profile: SandboxProfile,
    *,
    tool_name: str,
    effect: str,
) -> PolicyDecision:
    """Apply Profile tool deny/ask/allow sets without trusting Worker metadata."""
    normalized_name = tool_name.lower()
    if any(
        fnmatchcase(normalized_name, pattern.lower())
        for pattern in profile.tools.deny_tools
    ) or effect in profile.tools.deny_effects:
        return "deny"
    if effect in profile.tools.ask_effects:
        return "ask"
    if effect in profile.tools.allow_effects:
        return "allow"
    return "ask" if profile.approval_policy == "on_request" else "deny"


def command_policy_decision(
    profile: SandboxProfile,
    command: str,
) -> PolicyDecision:
    """Evaluate an administrator command pattern with deny > ask > allow."""
    if not profile.commands.enabled:
        return "deny"
    if any(fnmatchcase(command, pattern) for pattern in profile.commands.deny):
        return "deny"
    if any(fnmatchcase(command, pattern) for pattern in profile.commands.ask):
        return "ask"
    if any(fnmatchcase(command, pattern) for pattern in profile.commands.allow):
        return "allow"
    return "ask" if profile.approval_policy == "on_request" else "deny"


def filesystem_path_allowed(
    profile: SandboxProfile,
    path: str,
    *,
    write: bool,
) -> bool:
    """Check a workspace-relative file path against Profile roots and denies."""
    normalized = path.replace("\\", "/").strip()
    parts = [part for part in normalized.split("/") if part not in {"", "."}]
    if normalized.startswith("/") or ".." in parts:
        return False
    denied = profile.filesystem.deny_write if write else profile.filesystem.deny_read
    if any(fnmatchcase(normalized, pattern) for pattern in denied):
        return False
    roots = profile.filesystem.write_roots if write else profile.filesystem.read_roots
    return any(
        root.rstrip("/") in {"/workspace", "/workspace/work"}
        or root.rstrip("/").startswith("/workspace/work/")
        for root in roots
    )


class SandboxPolicyBundle(BaseModel):
    """Top-level, versioned policy document loaded from administrator TOML."""

    model_config = ConfigDict(extra="forbid")

    version: Literal[1] = 1
    deployment_id: str = Field(min_length=1, max_length=128)
    role_priority: list[str] = Field(
        default_factory=lambda: ["admin", "developer", "researcher", "viewer"]
    )
    profile_by_role: dict[str, str] = Field(default_factory=dict)
    default_profile: str = "research-gateway-only"
    profiles: dict[str, SandboxProfile]

    @model_validator(mode="after")
    def validate_profile_references(self) -> SandboxPolicyBundle:
        """Reject missing default or role-mapped Profile references."""
        if self.default_profile not in self.profiles:
            raise ValueError("sandbox default_profile does not exist")
        missing = sorted(set(self.profile_by_role.values()) - set(self.profiles))
        if missing:
            raise ValueError("sandbox role profile does not exist: " + ",".join(missing))
        if not self.role_priority:
            raise ValueError("sandbox role_priority must not be empty")
        return self

    def select_profile(self, roles: set[str] | frozenset[str]) -> tuple[str, SandboxProfile]:
        """Select the first administrator-mapped role by explicit priority."""
        normalized = {str(role).strip().lower() for role in roles}
        for role in self.role_priority:
            if role.lower() in normalized and role in self.profile_by_role:
                profile_id = self.profile_by_role[role]
                return profile_id, self.profiles[profile_id]
        return self.default_profile, self.profiles[self.default_profile]


def load_policy_bundle(path: str | Path) -> SandboxPolicyBundle:
    """Load and strictly validate one administrator TOML policy bundle."""
    policy_path = Path(path).resolve()
    if not policy_path.is_file():
        raise ValueError(f"sandbox_policy_not_found:{policy_path}")
    with policy_path.open("rb") as handle:
        return SandboxPolicyBundle.model_validate(tomllib.load(handle))


def policy_digest(bundle: SandboxPolicyBundle) -> str:
    """Return a stable SHA-256 digest of canonical, validated policy data."""
    encoded = json.dumps(
        bundle.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def runtime_digest(profile: SandboxProfile) -> str:
    """Return the immutable runtime subset digest for the run contract."""
    payload = {
        "provider": profile.provider,
        "resources": profile.resources.model_dump(mode="json"),
        "runtime": profile.runtime.model_dump(mode="json"),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def resolve_profile(
    configurable: object,
    *,
    roles: set[str] | frozenset[str] | None = None,
) -> tuple[SandboxPolicyBundle, str, SandboxProfile]:
    """Resolve an administrator profile from a Configuration-like object."""
    path = getattr(configurable, "sandbox_policy_path")
    bundle = load_policy_bundle(path)
    profile_id, profile = bundle.select_profile(roles or set())
    pinned = str(getattr(configurable, "sandbox_profile_id", "") or "")
    if pinned and pinned != "research-gateway-only" and pinned in bundle.profiles:
        profile_id, profile = pinned, bundle.profiles[pinned]
    return bundle, profile_id, profile
