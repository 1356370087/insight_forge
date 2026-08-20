"""Versioned, secret-free wire contracts between API, Worker and Gateway."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

GATEWAY_PROTOCOL_VERSION = 1
WORKER_EXIT_CODE_PATH = "/workspace/output/.worker-exit-code"
ResearchStage = Literal[
    "preparing",
    "planning",
    "researching",
    "synthesizing",
    "writing",
    "finalizing",
]


class TaskTokenClaimsV1(BaseModel):
    """Capability claims bound to one task and ownership epoch."""

    model_config = ConfigDict(extra="forbid")

    version: Literal[1] = 1
    run_id: str
    task_id: str
    fence_token: int = Field(ge=1)
    profile_id: str
    policy_digest: str
    issued_at: float
    expires_at: float
    jti: str

    @model_validator(mode="after")
    def validate_lifetime(self) -> TaskTokenClaimsV1:
        """Require a positive task capability lifetime."""
        if self.expires_at <= self.issued_at:
            raise ValueError("sandbox task token lifetime is invalid")
        return self

_RESEARCHER_STATE_KEYS = frozenset(
    {
        "researcher_messages",
        "tool_call_iterations",
        "research_topic",
        "requirement_ids",
        "coverage_contract",
        "research_risk_profile",
        "compressed_research",
        "raw_notes",
        "memory_context",
        "pending_tool_results",
        "research_complete_requested",
        "research_complete_succeeded",
        "result_assessment",
        "completion_decision",
        "permission_denials",
        "candidate_registry",
        "document_registry",
        "evidence_registry",
        "web_research_iterations",
        "applied_query_event_ids",
        "query_state_snapshot",
        "committed_tool_call_ids",
        "next_step",
        "cancelled",
        "metrics",
    }
)


def _assert_json_safe(value: Any, path: str = "$") -> None:
    if value is None or isinstance(value, str | int | float | bool):
        return
    if isinstance(value, list | tuple):
        for index, item in enumerate(value):
            _assert_json_safe(item, f"{path}[{index}]")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"sandbox payload key is not a string at {path}")
            _assert_json_safe(item, f"{path}.{key}")
        return
    raise ValueError(f"sandbox payload contains non-JSON value at {path}: {type(value).__name__}")


class SandboxTaskPayloadV1(BaseModel):
    """Minimal Researcher input accepted by an untrusted Worker."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    gateway_protocol_version: Literal[1] = GATEWAY_PROTOCOL_VERSION
    task_id: str = Field(min_length=1, max_length=256)
    run_id: str = Field(min_length=1, max_length=256)
    research_topic: str
    researcher_state: dict[str, Any]
    runtime_config: dict[str, Any]
    profile_id: str
    policy_digest: str
    fence_token: int = Field(ge=1)

    @field_validator("researcher_state")
    @classmethod
    def validate_researcher_state(cls, value: dict[str, Any]) -> dict[str, Any]:
        """Reject unrecognized or non-JSON Researcher state."""
        unexpected = sorted(set(value) - _RESEARCHER_STATE_KEYS)
        if unexpected:
            raise ValueError("sandbox researcher_state contains unsupported keys: " + ",".join(unexpected))
        _assert_json_safe(value, "$.researcher_state")
        return value

    @field_validator("runtime_config")
    @classmethod
    def validate_runtime_config(cls, value: dict[str, Any]) -> dict[str, Any]:
        """Reject credential-shaped keys and non-JSON runtime values."""
        _assert_json_safe(value, "$.runtime_config")
        forbidden = {
            key for key in value
            if key.lower() in {"apikeys", "api_key", "password", "credential"}
            or key.lower().endswith(
                ("_api_key", "_secret_key", "_auth_token", "_password", "_credential")
            )
        }
        if forbidden:
            raise ValueError("sandbox runtime_config contains credential-shaped keys: " + ",".join(sorted(forbidden)))
        return value


class SandboxTaskResultV1(BaseModel):
    """Complete, bounded Researcher result returned to the host."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    task_id: str
    status: Literal["completed", "failed"]
    compressed_research: str = ""
    raw_notes: list[str] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)
    candidate_registry: list[dict[str, Any]] = Field(default_factory=list)
    document_registry: list[dict[str, Any]] = Field(default_factory=list)
    evidence_registry: list[dict[str, Any]] = Field(default_factory=list)
    web_research_iterations: list[dict[str, Any]] = Field(default_factory=list)
    permission_denials: list[dict[str, Any]] = Field(default_factory=list)
    coverage_ledger: dict[str, Any] = Field(default_factory=dict)
    completion_decision: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None

    @model_validator(mode="after")
    def validate_status(self) -> SandboxTaskResultV1:
        """Require failed results to carry a bounded error signal."""
        if self.status == "failed" and not self.error:
            raise ValueError("failed sandbox result requires error")
        _assert_json_safe(self.model_dump(mode="json"))
        return self


class GatewayModelRequestV1(BaseModel):
    """One logical model operation issued by a Worker."""

    model_config = ConfigDict(extra="forbid")

    protocol_version: Literal[1] = GATEWAY_PROTOCOL_VERSION
    run_id: str
    task_id: str
    role: str
    stage: ResearchStage
    logical_operation_id: str = Field(min_length=1)
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]] = Field(default_factory=list)
    structured_schema: dict[str, Any] | None = None
    model_kwargs: dict[str, Any] = Field(default_factory=dict)

    @field_validator("model_kwargs")
    @classmethod
    def validate_model_kwargs(cls, value: dict[str, Any]) -> dict[str, Any]:
        """Allow only non-credential provider response-shaping options."""
        unexpected = sorted(set(value) - {"response_format"})
        if unexpected:
            raise ValueError(
                "sandbox model kwargs contain unsupported keys: "
                + ",".join(unexpected)
            )
        _assert_json_safe(value, "$.model_kwargs")
        return value


class GatewayModelOutcomeV1(BaseModel):
    """Idempotent terminal outcome for a logical model operation."""

    model_config = ConfigDict(extra="forbid")

    protocol_version: Literal[1] = GATEWAY_PROTOCOL_VERSION
    logical_operation_id: str
    physical_attempt_id: str
    status: Literal["completed", "failed", "uncertain"]
    message: dict[str, Any] | None = None
    usage: dict[str, int] = Field(default_factory=dict)
    fallback_events: list[dict[str, Any]] = Field(default_factory=list)
    provider_ttft_ms: float | None = None
    error_type: str | None = None
    error_message: str | None = None


class GatewayOperationLookupRequestV1(BaseModel):
    """Lookup one previously journaled logical model operation."""

    model_config = ConfigDict(extra="forbid")

    protocol_version: Literal[1] = GATEWAY_PROTOCOL_VERSION
    run_id: str
    task_id: str
    stage: ResearchStage
    logical_operation_id: str = Field(min_length=1)


class GatewayOperationLookupOutcomeV1(BaseModel):
    """Journal lookup result without triggering a new physical attempt."""

    model_config = ConfigDict(extra="forbid")

    found: bool
    outcome: GatewayModelOutcomeV1 | None = None


class GatewayToolRequestV1(BaseModel):
    """One authoritative tool operation delegated by a Worker."""

    model_config = ConfigDict(extra="forbid")

    protocol_version: Literal[1] = GATEWAY_PROTOCOL_VERSION
    run_id: str
    task_id: str
    role: str
    stage: ResearchStage
    execution_zone: Literal["sandbox_local", "gateway"]
    logical_operation_id: str = Field(min_length=1)
    tool_call_id: str = Field(min_length=1)
    tool_name: str = Field(min_length=1)
    arguments: dict[str, Any]


class GatewayToolOutcomeV1(BaseModel):
    """Bounded tool result or normalized governance error."""

    model_config = ConfigDict(extra="forbid")

    protocol_version: Literal[1] = GATEWAY_PROTOCOL_VERSION
    logical_operation_id: str
    tool_call_id: str
    status: Literal["completed", "failed", "approval_required"]
    output: Any = None
    error: dict[str, Any] | None = None
    approval_id: str | None = None


class GatewayToolCatalogRequestV1(BaseModel):
    """Request the authoritative dynamic Gateway tool schema catalog."""

    model_config = ConfigDict(extra="forbid")

    protocol_version: Literal[1] = GATEWAY_PROTOCOL_VERSION
    run_id: str
    task_id: str
    role: str
    stage: ResearchStage = "researching"


class GatewayCatalogToolV1(BaseModel):
    """One schema-only tool projection safe to return to a Worker."""

    model_config = ConfigDict(extra="forbid")

    name: str
    definition: dict[str, Any]
    prompt: str | None = None
    origin: str
    effect: str
    execution_zone: Literal["gateway"] = "gateway"
    retryable: bool = False
    concurrency_safe: bool = False
    max_output_chars: int | None = None


class GatewayToolCatalogOutcomeV1(BaseModel):
    """Authoritative schema catalog for one role/run/fence."""

    model_config = ConfigDict(extra="forbid")

    tools: list[GatewayCatalogToolV1] = Field(default_factory=list)
