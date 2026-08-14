"""Configuration management for the Open Deep Research system."""

import hashlib
import json
import os
import uuid
from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field, field_validator, model_validator

from open_deep_research.quality_policy import (
    QualityEvaluationRigor,
    get_quality_rigor_policy,
    rigor_from_legacy_min_score,
)

RUN_CONFIG_SCHEMA_VERSION = 4
QUALITY_POLICY_VERSION = "quality-gate-v4"
RUN_CONFIG_FROZEN_FIELDS = (
    "max_structured_output_retries",
    "model_transport_max_attempts",
    "context_recovery_max_attempts",
    "output_token_escalation_enabled",
    "output_continuation_max_attempts",
    "model_fallbacks",
    "model_context_window_overrides",
    "model_max_output_tokens_overrides",
    "unknown_model_context_window_tokens",
    "run_deadline_seconds",
    "max_run_model_calls",
    "max_run_tool_calls",
    "max_run_input_tokens",
    "max_run_output_tokens",
    "max_run_cost_micro_usd",
    "max_concurrent_tool_calls",
    "max_tool_batch_size",
    "model_call_timeout_seconds",
    "tool_call_timeout_seconds",
    "research_tool_call_timeout_seconds",
    "hook_timeout_seconds",
    "max_concurrent_research_units",
    "search_api",
    "max_researcher_iterations",
    "max_react_tool_calls",
    "summarization_model",
    "summarization_model_max_tokens",
    "research_model",
    "research_model_max_tokens",
    "compression_model",
    "compression_model_max_tokens",
    "final_report_model",
    "final_report_model_max_tokens",
    "web_pipeline_mode",
    "web_rerank_model",
    "web_evidence_model",
    "message_summary_model",
    "message_summary_model_max_tokens",
    "quality_evaluation_enabled",
    "quality_evaluation_model",
    "quality_evaluation_model_max_tokens",
    "quality_evaluation_base_url",
    "quality_evaluation_fail_open",
    "quality_evaluation_rigor",
    "quality_evaluation_min_sources",
    "quality_evaluation_max_input_chars",
    "quality_risk_mode",
    "quality_caveat_admission_enabled",
    "quality_gap_recovery_max_attempts",
    "sandbox_network_mode",
    "task_timeout_seconds",
    "sandbox_timeout_seconds",
)
_QUALITY_V4_FROZEN_FIELDS = {
    "quality_risk_mode",
    "quality_caveat_admission_enabled",
    "quality_gap_recovery_max_attempts",
}
RUN_CONFIG_FROZEN_FIELDS_V3 = tuple(
    field_name
    for field_name in RUN_CONFIG_FROZEN_FIELDS
    if field_name not in _QUALITY_V4_FROZEN_FIELDS
)
_MODEL_RECOVERY_FROZEN_FIELDS = {
    "output_token_escalation_enabled",
    "output_continuation_max_attempts",
    "model_fallbacks",
    "model_context_window_overrides",
    "model_max_output_tokens_overrides",
    "unknown_model_context_window_tokens",
}
RUN_CONFIG_FROZEN_FIELDS_V2 = tuple(
    field_name
    for field_name in RUN_CONFIG_FROZEN_FIELDS_V3
    if field_name not in _MODEL_RECOVERY_FROZEN_FIELDS
)
RUN_CONFIG_FROZEN_FIELDS_V1 = tuple(
    "quality_evaluation_min_score"
    if field_name == "quality_evaluation_rigor"
    else field_name
    for field_name in RUN_CONFIG_FROZEN_FIELDS_V2
)


class SearchAPI(Enum):
    """Enumeration of available search API providers."""
    
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    TAVILY = "tavily"
    NONE = "none"


def _resolve_quality_rigor(
    config: RunnableConfig,
    *,
    prefer_configurable: bool = False,
) -> tuple[QualityEvaluationRigor, dict[str, Any] | None]:
    """Resolve the new rigor option and migrate the removed score option."""
    configurable = config.get("configurable", {})
    metadata = config.get("metadata", {})
    frozen = metadata.get("runtime_config_frozen") is True
    configured_rigor = configurable.get("quality_evaluation_rigor")
    configured_legacy = configurable.get("quality_evaluation_min_score")

    if frozen or prefer_configurable:
        explicit_rigor = configured_rigor
        legacy_score = configured_legacy
    else:
        explicit_rigor = os.environ.get(
            "QUALITY_EVALUATION_RIGOR", configured_rigor
        )
        legacy_score = os.environ.get(
            "QUALITY_EVALUATION_MIN_SCORE", configured_legacy
        )

    if explicit_rigor is not None and str(explicit_rigor).strip():
        rigor = (
            explicit_rigor
            if isinstance(explicit_rigor, QualityEvaluationRigor)
            else QualityEvaluationRigor(str(explicit_rigor).strip().lower())
        )
        warning = (
            {
                "code": "legacy_quality_min_score_ignored",
                "resolved_rigor": rigor.value,
            }
            if legacy_score is not None and str(legacy_score).strip()
            else None
        )
        return rigor, warning
    if legacy_score is not None and str(legacy_score).strip():
        rigor = rigor_from_legacy_min_score(legacy_score)
        return rigor, {
            "code": "legacy_quality_min_score_mapped",
            "resolved_rigor": rigor.value,
        }
    return QualityEvaluationRigor.BALANCED, None


def get_model_compatibility_kwargs(model_name: str) -> dict[str, Any]:
    """Return provider-specific request options for a configured model.

    DeepSeek V4 enables thinking by default. The handwritten agent runtime uses
    forced tool choices for structured output, which DeepSeek rejects while
    thinking is enabled, so OpenAI-compatible DeepSeek models run in
    non-thinking mode.
    """
    if model_name.strip().lower().startswith("openai:deepseek"):
        return {"extra_body": {"thinking": {"type": "disabled"}}}
    return {}

class MCPConfig(BaseModel):
    """Configuration for Model Context Protocol (MCP) servers."""
    
    url: Optional[str] = Field(
        default=None,
        optional=True,
    )
    """The URL of the MCP server"""
    tools: Optional[List[str]] = Field(
        default=None,
        optional=True,
    )
    """The tools to make available to the LLM"""
    auth_required: Optional[bool] = Field(
        default=False,
        optional=True,
    )
    """Whether the MCP server requires authentication"""
    tool_effects: Dict[str, Literal[
        "read_only",
        "sensitive_read",
        "external_write",
        "local_write",
        "destructive",
    ]] = Field(default_factory=dict)
    """Explicit side-effect classification for every exposed MCP tool."""


class BrowserMCPConfig(BaseModel):
    """Configuration for browser-level MCP exploration tools."""

    transport: Literal["stdio", "streamable_http", "sse", "websocket"] = Field(
        default="stdio",
        optional=True,
    )
    """MCP transport used by the browser server."""
    url: Optional[str] = Field(
        default=None,
        optional=True,
    )
    """Exact MCP endpoint URL for HTTP/SSE/WebSocket transports."""
    command: Optional[str] = Field(
        default="npx",
        optional=True,
    )
    """Executable used for stdio transport."""
    args: Optional[List[str]] = Field(
        default_factory=lambda: ["@playwright/mcp@latest"],
        optional=True,
    )
    """Command arguments used for stdio transport."""
    env: Optional[Dict[str, str]] = Field(
        default=None,
        optional=True,
    )
    """Optional environment variables for stdio transport."""
    tools: Optional[List[str]] = Field(
        default=None,
        optional=True,
    )
    """Required browser tool allowlist. None exposes no browser tools."""
    tool_effects: Dict[str, Literal[
        "read_only",
        "sensitive_read",
        "external_write",
        "local_write",
        "destructive",
    ]] = Field(default_factory=dict)
    """Explicit side-effect classification for every exposed browser tool."""

class Configuration(BaseModel):
    """Main configuration class for the Deep Research agent."""
    
    # General Configuration
    max_structured_output_retries: int = Field(
        default=3,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 3,
                "min": 1,
                "max": 10,
                "description": "Maximum number of retries for structured output calls from models"
            }
        }
    )
    model_transport_max_attempts: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum attempts for retryable model transport failures.",
    )
    context_recovery_max_attempts: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum reactive context recovery attempts per model stage.",
    )
    output_token_escalation_enabled: bool = Field(
        default=True,
        description=(
            "Allow one bounded max-output-token escalation before continuation."
        ),
    )
    output_continuation_max_attempts: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Maximum continuation requests after a truncated response.",
    )
    model_fallbacks: dict[str, list[str]] = Field(
        default_factory=dict,
        description=(
            "Optional frozen model fallback chains keyed by model role. "
            "An empty object disables fallback."
        ),
    )
    model_context_window_overrides: dict[str, int] = Field(
        default_factory=dict,
        description="Explicit context-window overrides keyed by model id.",
    )
    model_max_output_tokens_overrides: dict[str, int] = Field(
        default_factory=dict,
        description="Explicit maximum output-token overrides keyed by model id.",
    )
    unknown_model_context_window_tokens: int = Field(
        default=32_768,
        ge=4_096,
        description=(
            "Conservative context window used when a model has no known profile."
        ),
    )
    run_deadline_seconds: Optional[int] = Field(
        default=None,
        ge=1,
        description="Optional wall-clock deadline for one complete research run.",
    )
    max_run_model_calls: Optional[int] = Field(default=None, ge=1)
    max_run_tool_calls: Optional[int] = Field(default=None, ge=1)
    max_run_input_tokens: Optional[int] = Field(default=None, ge=1)
    max_run_output_tokens: Optional[int] = Field(default=None, ge=1)
    max_run_cost_micro_usd: Optional[int] = Field(default=None, ge=1)
    max_concurrent_tool_calls: int = Field(default=8, ge=1, le=128)
    max_tool_batch_size: int = Field(default=32, ge=1, le=512)
    model_call_timeout_seconds: float = Field(default=180, gt=0)
    tool_call_timeout_seconds: float = Field(default=120, gt=0)
    research_tool_call_timeout_seconds: float = Field(
        default=300,
        gt=0,
        description=(
            "Timeout for researcher tools that may include search, fetching, reranking, "
            "and evidence extraction."
        ),
    )
    hook_timeout_seconds: float = Field(default=120, gt=0)
    allow_clarification: bool = Field(
        default=True,
        metadata={
            "x_oap_ui_config": {
                "type": "boolean",
                "default": True,
                "description": "Whether to allow the researcher to ask the user clarifying questions before starting research"
            }
        }
    )
    # Observability / Operations
    observability_enabled: bool = Field(
        default=True,
        metadata={
            "x_oap_ui_config": {
                "type": "boolean",
                "default": True,
                "description": "Master switch for SQLite, Langfuse, and Prometheus observability.",
            }
        },
    )
    sqlite_observability_enabled: bool = Field(
        default=True,
        metadata={
            "x_oap_ui_config": {
                "type": "boolean",
                "default": True,
                "description": "Keep the local SQLite trace store as a development/failover backend.",
            }
        },
    )
    trace_store_path: str = Field(
        default=".runs/traces.sqlite3",
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": ".runs/traces.sqlite3",
                "description": "SQLite path used for run/span/usage observability data.",
            }
        },
    )
    trace_payload_mode: Literal["none", "preview", "full"] = Field(
        default="preview",
        metadata={
            "x_oap_ui_config": {
                "type": "select",
                "default": "preview",
                "description": "How much prompt/output payload to persist in trace spans.",
                "options": [
                    {"label": "None", "value": "none"},
                    {"label": "Preview", "value": "preview"},
                    {"label": "Full", "value": "full"},
                ],
            }
        },
    )
    trace_preview_chars: int = Field(
        default=2000,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 2000,
                "description": "Maximum characters stored for trace payload previews.",
            }
        },
    )
    trace_redaction_enabled: bool = Field(
        default=True,
        description="Redact common credentials and bearer tokens from trace payloads.",
    )
    model_costs_per_million: Dict[str, Dict[str, float]] = Field(
        default_factory=dict,
        description=(
            "Optional per-model USD rates per million tokens. Supported keys are "
            "input, output, cached_input, cache_creation_input, and reasoning."
        ),
    )
    langfuse_enabled: bool = Field(
        default=False,
        metadata={
            "x_oap_ui_config": {
                "type": "boolean",
                "default": False,
                "description": "Mirror TraceRecorder runs/spans to Langfuse.",
            }
        },
    )
    langfuse_public_key: Optional[str] = Field(default=None)
    langfuse_secret_key: Optional[str] = Field(default=None)
    langfuse_base_url: str = Field(default="https://cloud.langfuse.com")
    langfuse_environment: str = Field(default="development")
    langfuse_release: Optional[str] = Field(default=None)
    langfuse_sample_rate: float = Field(default=1.0, ge=0.0, le=1.0)
    langfuse_flush_on_run_end: bool = Field(
        default=False,
        description="Synchronously flush Langfuse after every run (useful for short-lived jobs).",
    )
    langfuse_langchain_callback_enabled: bool = Field(
        default=False,
        description=(
            "Attach Langfuse's LangChain CallbackHandler to model calls as supplemental "
            "instrumentation. Disabled by default because TraceRecorder already records generations."
        ),
    )
    prometheus_enabled: bool = Field(
        default=False,
        metadata={
            "x_oap_ui_config": {
                "type": "boolean",
                "default": False,
                "description": "Publish low-cardinality aggregate metrics for Prometheus/Grafana.",
            }
        },
    )
    prometheus_metrics_path: str = Field(default="/metrics")
    prometheus_namespace: str = Field(default="open_deep_research")
    helicone_enabled: bool = Field(
        default=False,
        metadata={
            "x_oap_ui_config": {
                "type": "boolean",
                "default": False,
                "description": "Enable optional Helicone headers/base URL enrichment for LLM calls.",
            }
        },
    )
    helicone_api_key: Optional[str] = Field(
        default=None,
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "",
                "description": "Helicone API key used when helicone_enabled is true.",
            }
        },
    )
    helicone_base_url: Optional[str] = Field(
        default=None,
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "",
                "description": "Optional Helicone provider gateway base URL.",
            }
        },
    )
    helicone_headers_enabled: bool = Field(
        default=True,
        metadata={
            "x_oap_ui_config": {
                "type": "boolean",
                "default": True,
                "description": "Attach Helicone session/property headers to model configs.",
            }
        },
    )
    max_concurrent_research_units: int = Field(
        default=5,
        metadata={
            "x_oap_ui_config": {
                "type": "slider",
                "default": 5,
                "min": 1,
                "max": 20,
                "step": 1,
                "description": "Maximum number of research units to run concurrently. This will allow the researcher to use multiple sub-agents to conduct research. Note: with more concurrency, you may run into rate limits."
            }
        }
    )
    # Research Configuration
    search_api: SearchAPI = Field(
        default=SearchAPI.TAVILY,
        metadata={
            "x_oap_ui_config": {
                "type": "select",
                "default": "tavily",
                "description": "Search API to use for research. NOTE: Make sure your Researcher Model supports the selected search API.",
                "options": [
                    {"label": "Tavily", "value": SearchAPI.TAVILY.value},
                    {"label": "OpenAI Native Web Search", "value": SearchAPI.OPENAI.value},
                    {"label": "Anthropic Native Web Search", "value": SearchAPI.ANTHROPIC.value},
                    {"label": "None", "value": SearchAPI.NONE.value}
                ]
            }
        }
    )
    max_researcher_iterations: int = Field(
        default=6,
        metadata={
            "x_oap_ui_config": {
                "type": "slider",
                "default": 6,
                "min": 1,
                "max": 10,
                "step": 1,
                "description": "Maximum number of research iterations for the Research Supervisor. This is the number of times the Research Supervisor will reflect on the research and ask follow-up questions."
            }
        }
    )
    max_react_tool_calls: int = Field(
        default=10,
        metadata={
            "x_oap_ui_config": {
                "type": "slider",
                "default": 10,
                "min": 1,
                "max": 30,
                "step": 1,
                "description": "Maximum number of tool calling iterations to make in a single researcher step."
            }
        }
    )
    # Human-in-the-loop collaboration
    enable_human_in_loop: bool = Field(
        default=False,
        metadata={
            "x_oap_ui_config": {
                "type": "boolean",
                "default": False,
                "description": "Enable human approval and feedback checkpoints during research runs.",
            }
        },
    )
    hitl_require_plan_approval: bool = Field(
        default=True,
        metadata={
            "x_oap_ui_config": {
                "type": "boolean",
                "default": True,
                "description": "Pause after drafting a research plan until the user approves, revises, or cancels.",
            }
        },
    )
    hitl_require_outline_approval: bool = Field(
        default=True,
        metadata={
            "x_oap_ui_config": {
                "type": "boolean",
                "default": True,
                "description": "Pause before final report generation until the user approves, revises, or cancels the outline.",
            }
        },
    )
    hitl_max_plan_revisions: int = Field(
        default=3,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 3,
                "min": 0,
                "max": 10,
                "description": "Maximum number of user-requested research plan revisions before failing the run.",
            }
        },
    )
    hitl_feedback_mode: Literal["safe_points", "task_queue"] = Field(
        default="safe_points",
        metadata={
            "x_oap_ui_config": {
                "type": "select",
                "default": "safe_points",
                "description": "How mid-run feedback is applied: at supervisor safe points or queued to running research tasks when possible.",
                "options": [
                    {"label": "Safe points", "value": "safe_points"},
                    {"label": "Task queue", "value": "task_queue"},
                ],
            }
        },
    )

    # Model Configuration
    summarization_model: str = Field(
        default="openai:gpt-4.1-mini",
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "openai:gpt-4.1-mini",
                "description": "Model for summarizing research results from Tavily search results"
            }
        }
    )
    summarization_model_max_tokens: int = Field(
        default=8192,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 8192,
                "description": "Maximum output tokens for summarization model"
            }
        }
    )
    max_content_length: int = Field(
        default=50000,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 50000,
                "min": 1000,
                "max": 200000,
                "description": "Maximum character length for webpage content before summarization"
            }
        }
    )
    research_model: str = Field(
        default="openai:gpt-4.1",
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "openai:gpt-4.1",
                "description": "Model for conducting research. NOTE: Make sure your Researcher Model supports the selected search API."
            }
        }
    )
    research_model_max_tokens: int = Field(
        default=10000,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 10000,
                "description": "Maximum output tokens for research model"
            }
        }
    )
    compression_model: str = Field(
        default="openai:gpt-4.1",
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "openai:gpt-4.1",
                "description": "Model for compressing research findings from sub-agents. NOTE: Make sure your Compression Model supports the selected search API."
            }
        }
    )
    compression_model_max_tokens: int = Field(
        default=8192,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 8192,
                "description": "Maximum output tokens for compression model"
            }
        }
    )
    final_report_model: str = Field(
        default="openai:gpt-4.1",
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "openai:gpt-4.1",
                "description": "Model for writing the final report from all research findings"
            }
        }
    )
    final_report_model_max_tokens: int = Field(
        default=10000,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 10000,
                "description": "Maximum output tokens for final report model"
            }
        }
    )
    # Report Output Configuration
    report_type: str = Field(
        default="default",
        metadata={
            "x_oap_ui_config": {
                "type": "select",
                "default": "default",
                "description": "Report product form / genre. Unknown values fall back to 'default'.",
                "options": [
                    {"label": "Default (comprehensive report)", "value": "default"},
                    {"label": "Executive Summary", "value": "executive_summary"},
                    {"label": "Decision Brief", "value": "decision_brief"},
                    {"label": "FAQ", "value": "faq"},
                    {"label": "Comparison Matrix (sectioned)", "value": "comparison_matrix"},
                    {"label": "Pros & Cons (sectioned)", "value": "pros_cons"},
                    {"label": "Literature Review (sectioned)", "value": "literature_review"},
                ],
            }
        },
    )
    output_format: Optional[str] = Field(
        default=None,
        metadata={
            "x_oap_ui_config": {
                "type": "select",
                "default": None,
                "description": "Output format for the report deliverable. When unset, inherits the report type's default. Unknown values fall back to that profile default.",
                "options": [
                    {"label": "Markdown", "value": "markdown"},
                    {"label": "Structured JSON", "value": "structured_json"},
                    {"label": "Slides", "value": "slides"},
                    {"label": "One-pager", "value": "one_pager"},
                ],
            }
        },
    )
    reference_style: Optional[str] = Field(
        default=None,
        metadata={
            "x_oap_ui_config": {
                "type": "select",
                "default": None,
                "description": "Citation/reference rendering style. When unset, inherits the report type's default.",
                "options": [
                    {"label": "Numbered ([1] Title: URL)", "value": "numbered"},
                    {"label": "BibTeX-like (@misc{...})", "value": "bibtex_like"},
                ],
            }
        },
    )
    # MCP server configuration
    mcp_config: Optional[MCPConfig] = Field(
        default=None,
        optional=True,
        metadata={
            "x_oap_ui_config": {
                "type": "mcp",
                "description": "MCP server configuration"
            }
        }
    )
    mcp_prompt: Optional[str] = Field(
        default=None,
        optional=True,
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "description": "Any additional instructions to pass along to the Agent regarding the MCP tools that are available to it."
            }
        }
    )
    browser_mcp_enabled: bool = Field(
        default=False,
        metadata={
            "x_oap_ui_config": {
                "type": "boolean",
                "default": False,
                "description": "Enable optional browser-level exploration tools via Playwright MCP.",
            }
        },
    )
    """Whether to load browser-level MCP tools for researcher agents."""
    browser_mcp_config: Optional[BrowserMCPConfig] = Field(
        default=None,
        optional=True,
        metadata={
            "x_oap_ui_config": {
                "type": "object",
                "description": "Playwright MCP connection configuration. Defaults to stdio npx @playwright/mcp@latest when enabled.",
            }
        },
    )
    """Connection details for the optional Playwright MCP browser tool server."""
    browser_mcp_prompt: Optional[str] = Field(
        default=None,
        optional=True,
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "description": "Additional researcher instructions for using browser MCP tools.",
            }
        },
    )
    """Additional prompt guidance for browser MCP tool usage."""
    # Agent Skills: domain context packs for research/report orchestration.
    skills: Optional[List[str]] = Field(
        default=None,
        metadata={
            "x_oap_ui_config": {
                "type": "array",
                "default": None,
                "description": "Domain skill packs to enable (e.g. medical, legal, finance). Adds curated research/report context. Unknown keys are ignored.",
                "options": [
                    {"label": "Medical", "value": "medical"},
                    {"label": "Legal", "value": "legal"},
                    {"label": "Finance", "value": "finance"},
                ],
            }
        },
    )
    """Enabled domain skill keys (context-only builtins in v1)."""
    # Tool Governance Configuration
    # Per-role tool name whitelists. None = backward compatible (all assembled tools allowed).
    supervisor_tool_whitelist: Optional[List[str]] = Field(
        default=None,
        optional=True,
        metadata={
            "x_oap_ui_config": {
                "type": "array",
                "default": None,
                "description": "If set, only these tool names may be used by the supervisor (main graph). None = all assembled tools allowed."
            }
        }
    )
    """Whitelist of tool names the supervisor is permitted to call. None disables the whitelist (all assembled tools allowed)."""
    researcher_tool_whitelist: Optional[List[str]] = Field(
        default=None,
        optional=True,
        metadata={
            "x_oap_ui_config": {
                "type": "array",
                "default": None,
                "description": "If set, only these tool names may be used by researchers (subgraph). None = all assembled tools allowed."
            }
        }
    )
    """Whitelist of tool names a researcher is permitted to call. None disables the whitelist (all assembled tools allowed)."""
    # Per-role origin filters. Values: "system", "search", "mcp", "provider_native". None/empty = no origin blocked.
    supervisor_blocked_origins: Optional[List[str]] = Field(
        default=None,
        optional=True,
        metadata={
            "x_oap_ui_config": {
                "type": "array",
                "default": None,
                "description": "Tool origins blocked for the supervisor. Values: system, search, mcp, provider_native. None = none blocked."
            }
        }
    )
    """Tool origins the supervisor is forbidden from using (system/search/mcp/provider_native). None = no origin blocked."""
    researcher_blocked_origins: Optional[List[str]] = Field(
        default=None,
        optional=True,
        metadata={
            "x_oap_ui_config": {
                "type": "array",
                "default": None,
                "description": "Tool origins blocked for researchers. Values: system, search, mcp, provider_native. None = none blocked."
            }
        }
    )
    """Tool origins a researcher is forbidden from using (system/search/mcp/provider_native). None = no origin blocked."""
    # Tool execution retry (applied on the researcher path for network-bound tools).
    max_tool_retries: int = Field(
        default=3,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 3,
                "min": 0,
                "max": 10,
                "description": "Max retries for retryable tool errors (network/timeout/429/503) using exponential backoff. 0 = no retry. Retry assumes tool idempotency."
            }
        }
    )
    """Maximum retry attempts for retryable tool errors (network/timeout/429/503). 0 disables retry."""
    tool_retry_base_delay: float = Field(
        default=1.0,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 1.0,
                "min": 0,
                "max": 60,
                "description": "Base delay in seconds for exponential backoff between tool retries."
            }
        }
    )
    """Base delay (seconds) for exponential backoff between tool retries."""
    tool_retry_max_delay: float = Field(
        default=30.0,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 30.0,
                "min": 1,
                "max": 300,
                "description": "Maximum delay cap in seconds for tool retry backoff."
            }
        }
    )
    """Maximum delay cap (seconds) for tool retry backoff."""
    # Optional per-tool parameter constraints, layered on top of JSON-Schema checks.
    # Shape: {tool_name: {param_name: {minItems, maxItems, minLength, maxLength, minimum, maximum}}}
    tool_param_constraints: Optional[Dict[str, Dict[str, Dict[str, Any]]]] = Field(
        default=None,
        optional=True,
        metadata={
            "x_oap_ui_config": {
                "type": "object",
                "default": None,
                "description": "Per-tool parameter constraints layered on top of schema checks. Shape: {tool_name: {param: {minItems, maxItems, minLength, maxLength, minimum, maximum}}}. None = no extra constraints."
            }
        }
    )
    """Optional per-tool parameter bounds (e.g. queries maxItems, per-query maxLength) applied after JSON-Schema validation."""
    # User-role blacklists (deny). Applied on top of agent-scope whitelists.
    # Maps a user role (from JWT app_metadata) to blocked tool names / origins.
    role_tool_blacklist: Optional[Dict[str, List[str]]] = Field(
        default=None,
        optional=True,
        metadata={
            "x_oap_ui_config": {
                "type": "object",
                "default": None,
                "description": "Map of user role -> blocked tool names. Applied on top of agent-scope whitelists. None = no role-based tool blocking."
            }
        }
    )
    """Map of user role (from JWT app_metadata) to blocked tool names. None = no role-based blocking."""
    role_blocked_origins: Optional[Dict[str, List[str]]] = Field(
        default=None,
        optional=True,
        metadata={
            "x_oap_ui_config": {
                "type": "object",
                "default": None,
                "description": "Map of user role -> blocked tool origins. Applied on top of agent-scope origin filters. None = no role-based origin blocking."
            }
        }
    )
    """Map of user role (from JWT app_metadata) to blocked tool origins. None = no role-based blocking."""
    # Async SubAgent Configuration
    enable_async_research: bool = Field(
        default=False,
        metadata={
            "x_oap_ui_config": {
                "type": "boolean",
                "default": False,
                "description": "Enable async SubAgent research with StartResearchTask/CheckResearchTask tools instead of synchronous ConductResearch."
            }
        }
    )
    runs_dir: str = Field(
        default=".runs",
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": ".runs",
                "description": "Directory for run event logs, task outputs, and checkpoints."
            }
        }
    )
    task_timeout_seconds: int = Field(
        default=600,
        metadata={
            "x_oap_ui_config": {
                "type": "slider",
                "default": 600,
                "min": 60,
                "max": 3600,
                "step": 60,
                "description": "Maximum seconds a single SubAgent task can run before timeout."
            }
        }
    )
    max_in_flight_tasks: int = Field(
        default=10,
        metadata={
            "x_oap_ui_config": {
                "type": "slider",
                "default": 10,
                "min": 1,
                "max": 50,
                "step": 1,
                "description": "Maximum number of concurrently running async research tasks."
            }
        }
    )
    task_checkpoint_enabled: bool = Field(
        default=True,
        metadata={
            "x_oap_ui_config": {
                "type": "boolean",
                "default": True,
                "description": "Enable checkpoint-based recovery for failed/cancelled async tasks."
            }
        }
    )
    event_log_enabled: bool = Field(
        default=True,
        metadata={
            "x_oap_ui_config": {
                "type": "boolean",
                "default": True,
                "description": "Enable structured JSONL event logging per run."
            }
        }
    )
    public_event_summary_max_chars: int = Field(
        default=1200,
        ge=200,
        le=10_000,
        description="Maximum characters exposed in one public findings summary.",
    )
    public_event_source_limit: int = Field(
        default=10,
        ge=0,
        le=100,
        description="Maximum source references exposed by one public event.",
    )
    sse_heartbeat_seconds: float = Field(
        default=15,
        ge=1,
        le=120,
        description="Seconds between SSE keep-alive comments.",
    )
    sse_poll_interval_ms: int = Field(
        default=250,
        ge=25,
        le=5000,
        description="Polling interval used to tail public events across workers.",
    )
    task_state_backend: Literal["file", "memory"] = Field(
        default="file",
        metadata={
            "x_oap_ui_config": {
                "type": "select",
                "default": "file",
                "description": "File-backed production state or test-only process memory.",
                "options": [
                    {"label": "File Mailbox", "value": "file"},
                    {"label": "Memory", "value": "memory"},
                ],
            }
        },
    )
    web_pipeline_mode: Literal["legacy", "shadow", "enforced"] = Field(
        default="enforced",
        metadata={
            "x_oap_ui_config": {
                "type": "select",
                "default": "enforced",
                "description": "Web evidence pipeline rollout mode.",
                "options": [
                    {"label": "Legacy", "value": "legacy"},
                    {"label": "Shadow", "value": "shadow"},
                    {"label": "Enforced", "value": "enforced"},
                ],
            }
        },
    )
    web_pipeline_shadow_sample_rate: float = Field(default=0.1, ge=0.0, le=1.0)
    fetch_backend_order: List[str] = Field(
        default_factory=lambda: ["local", "playwright", "tavily_extract", "firecrawl"]
    )
    external_extract_backends: List[str] = Field(
        default_factory=lambda: ["tavily_extract", "firecrawl"]
    )
    fetch_top_k: int = Field(default=5, ge=3, le=8)
    web_min_source_authority: float = Field(
        default=0.65,
        ge=0.0,
        le=1.0,
        description=(
            "Minimum authority score required before a web candidate may be fetched "
            "and promoted to report evidence."
        ),
    )
    search_candidate_limit: int = Field(default=20, ge=1, le=100)
    max_fetches_per_researcher: int = Field(default=12, ge=1, le=100)
    max_fetches_per_run: int = Field(default=40, ge=1, le=500)
    fetch_global_concurrency: int = Field(default=4, ge=1, le=32)
    fetch_per_host_concurrency: int = Field(default=2, ge=1, le=8)
    web_rerank_model: str = Field(default="openai:gpt-4.1-mini")
    web_evidence_model: str = Field(default="openai:gpt-4.1-mini")
    html_max_bytes: int = Field(default=2 * 1024 * 1024, ge=1024)
    pdf_max_bytes: int = Field(default=20 * 1024 * 1024, ge=1024)
    pdf_max_pages: int = Field(default=100, ge=1, le=1000)
    respect_robots_txt: bool = Field(default=True)
    browser_render_fallback_enabled: bool = Field(default=True)
    max_persistent_teammates: int = Field(default=5, ge=1, le=50)
    mailbox_poll_interval_ms: int = Field(default=500, ge=50, le=10000)
    mailbox_lock_timeout_seconds: float = Field(default=5, gt=0, le=60)
    mailbox_claim_lease_seconds: float = Field(default=30, gt=0, le=3600)
    mailbox_max_delivery_attempts: int = Field(default=5, ge=1, le=100)
    mailbox_acked_retention_seconds: int = Field(default=86400, ge=0)
    mailbox_compaction_threshold: int = Field(default=1000, ge=10)
    leader_heartbeat_seconds: float = Field(default=5, gt=0, le=300)
    leader_lease_seconds: float = Field(default=15, gt=1, le=3600)
    # Main Graph Message Summarization
    enable_message_summarization: bool = Field(
        default=False,
        metadata={
            "x_oap_ui_config": {
                "type": "boolean",
                "default": False,
                "description": "Summarize long main-graph message histories and keep only recent raw messages."
            }
        }
    )
    message_summary_trigger_tokens: int = Field(
        default=24000,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 24000,
                "description": "Approximate message-token threshold that triggers conversation summarization."
            }
        }
    )
    message_summary_keep_last: int = Field(
        default=8,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 8,
                "description": "Number of recent raw messages to keep after summarization."
            }
        }
    )
    message_summary_model: Optional[str] = Field(
        default=None,
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "",
                "description": "Optional model for conversation summarization. Defaults to summarization_model."
            }
        }
    )
    message_summary_model_max_tokens: int = Field(
        default=1024,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 1024,
                "description": "Maximum output tokens for conversation summaries."
            }
        }
    )
    # Runtime quality gates
    quality_evaluation_enabled: bool = Field(
        default=False,
        description="Evaluate researcher tool results and subagent handoffs at runtime.",
    )
    quality_evaluation_model: str = Field(
        default="openai:qwen3.7-plus",
        description=(
            "Provider:model identifier used by runtime quality gates, for example "
            "openai:qwen3.7-plus or anthropic:claude-sonnet-4-5."
        ),
    )
    quality_evaluation_model_max_tokens: int = Field(default=2048, ge=256)
    quality_evaluation_base_url: Optional[str] = Field(
        default=None,
        description=(
            "Optional provider endpoint override. DashScope Qwen falls back to its "
            "public OpenAI-compatible endpoint when this is unset."
        ),
    )
    quality_evaluation_fail_open: bool = Field(
        default=True,
        description=(
            "Continue on evaluator transport or parsing failures; deterministic "
            "evidence admission checks still apply."
        ),
    )
    quality_evaluation_rigor: QualityEvaluationRigor = Field(
        default=QualityEvaluationRigor.BALANCED,
        description=(
            "Semantic approval rigor for runtime and offline quality Judges. "
            "Safety, execution compliance, and evidence integrity remain hard gates."
        ),
        metadata={
            "x_oap_ui_config": {
                "type": "select",
                "default": "balanced",
                "options": [
                    {"label": "极宽松", "value": "very_relaxed"},
                    {"label": "宽松", "value": "relaxed"},
                    {"label": "均衡", "value": "balanced"},
                    {"label": "严格", "value": "strict"},
                    {"label": "极严格", "value": "very_strict"},
                ],
                "description": "质量审批严格程度；默认使用均衡模式。",
            }
        },
    )
    quality_evaluation_min_sources: int = Field(default=2, ge=0, le=20)
    quality_evaluation_max_input_chars: int = Field(
        default=30000,
        ge=1000,
        description=(
            "Hard character limit for the complete serialized JSON payload sent "
            "to a runtime quality evaluator. Oversized semantic fields are "
            "truncated with an explicit input_truncated marker."
        ),
    )
    quality_risk_mode: Literal["auto", "high", "standard"] = Field(
        default="auto",
        description=(
            "Classify caveat-admission risk automatically from versioned "
            "medical/legal/finance rules, or force high/standard behavior."
        ),
    )
    quality_caveat_admission_enabled: bool = Field(
        default=True,
        description=(
            "Allow coverage-complete, non-high-risk handoffs with only soft "
            "uncertainties to enter report synthesis."
        ),
    )
    quality_gap_recovery_max_attempts: int = Field(
        default=1,
        ge=0,
        le=3,
        description=(
            "Additional bounded Researcher turns reserved for explicit "
            "quality-assessment gaps after the normal tool-call limit."
        ),
    )
    # File-backed Query session context
    query_session_persistence_enabled: bool = Field(
        default=True,
        metadata={
            "x_oap_ui_config": {
                "type": "boolean",
                "default": True,
                "description": "Persist run-scoped Query state and authoritative context artifacts under runs_dir.",
            }
        },
    )
    query_context_compaction_enabled: Optional[bool] = Field(
        default=None,
        optional=True,
        metadata={
            "x_oap_ui_config": {
                "type": "boolean",
                "default": True,
                "description": "Compact Lead and Supervisor histories into a durable summary plus recent raw window.",
            }
        },
    )
    query_context_trigger_ratio: float = Field(
        default=0.75,
        ge=0.1,
        le=0.95,
    )
    query_context_recent_window_ratio: float = Field(
        default=0.25,
        ge=0.05,
        le=0.75,
    )
    query_context_summary_max_tokens: int = Field(default=8000, ge=128)
    query_journal_inline_content_max_chars: int = Field(default=32768, ge=1024)
    # Prompt-injection and external-content protection. These are administrator
    # settings on the HTTP surface and are rejected when supplied by a run.
    prompt_injection_protection_enabled: bool = Field(default=True)
    external_content_fail_closed: bool = Field(default=True)
    allow_http_stdio_mcp: bool = Field(default=False)
    require_sensitive_tool_approval: bool = Field(default=True)
    max_external_content_bytes: int = Field(default=1_000_000, ge=1024)
    max_mcp_description_chars: int = Field(default=2_000, ge=64)
    max_mcp_output_chars: int = Field(default=30_000, ge=256)
    allowed_mcp_servers: list[str] = Field(default_factory=list)
    allowed_model_endpoints: list[str] = Field(default_factory=list)
    # Docker Sandbox Configuration
    enable_docker_sandbox: bool = Field(
        default=False,
        metadata={
            "x_oap_ui_config": {
                "type": "boolean",
                "default": False,
                "description": "Run async Researcher SubAgents in isolated Docker containers."
            }
        }
    )
    sandbox_provider: Literal["docker"] = Field(
        default="docker",
        metadata={
            "x_oap_ui_config": {
                "type": "select",
                "default": "docker",
                "description": "Sandbox provider for async Researcher isolation.",
                "options": [{"label": "Docker", "value": "docker"}],
            }
        }
    )
    sandbox_image: str = Field(
        default="open-deep-research-sandbox:latest",
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "open-deep-research-sandbox:latest",
                "description": "Docker image used to run isolated Researcher workers."
            }
        }
    )
    sandbox_workspace_root: Optional[str] = Field(
        default=None,
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "",
                "description": "Optional root directory for sandbox workspaces. Defaults to runs_dir/<run_id>/workspaces."
            }
        }
    )
    sandbox_network_mode: Literal[
        "no-network",
        "allow-search-only",
        "allowlist-domain",
        "open-network",
    ] = Field(
        default="allow-search-only",
        metadata={
            "x_oap_ui_config": {
                "type": "select",
                "default": "allow-search-only",
                "description": "Network isolation policy for sandbox containers.",
                "options": [
                    {"label": "No network", "value": "no-network"},
                    {"label": "Allow search only", "value": "allow-search-only"},
                    {"label": "Allowlist domain", "value": "allowlist-domain"},
                    {"label": "Open network", "value": "open-network"},
                ],
            }
        }
    )
    sandbox_allowed_domains: list[str] = Field(
        default_factory=list,
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "",
                "description": "Additional allowed domains for proxy-backed sandbox network policies."
            }
        }
    )
    sandbox_cleanup_policy: Literal["always", "on_success", "never"] = Field(
        default="always",
        metadata={
            "x_oap_ui_config": {
                "type": "select",
                "default": "always",
                "description": "When to remove sandbox containers and temporary files.",
                "options": [
                    {"label": "Always", "value": "always"},
                    {"label": "On success", "value": "on_success"},
                    {"label": "Never", "value": "never"},
                ],
            }
        }
    )
    sandbox_timeout_seconds: Optional[int] = Field(
        default=None,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": "",
                "description": "Optional sandbox-specific task timeout. Defaults to task_timeout_seconds."
            }
        }
    )
    sandbox_memory: str = Field(
        default="1g",
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "1g",
                "description": "Memory limit for each sandbox container."
            }
        }
    )
    sandbox_cpus: float = Field(
        default=1.0,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 1.0,
                "description": "CPU quota for each sandbox container."
            }
        }
    )
    sandbox_pids_limit: int = Field(
        default=256,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 256,
                "description": "Maximum process count for each sandbox container."
            }
        }
    )
    sandbox_read_only_rootfs: bool = Field(
        default=True,
        metadata={
            "x_oap_ui_config": {
                "type": "boolean",
                "default": True,
                "description": "Mount the sandbox container root filesystem read-only."
            }
        }
    )
    sandbox_user: str = Field(
        default="1000:1000",
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "1000:1000",
                "description": "User id and group id used inside sandbox containers."
            }
        }
    )
    # Mem0 Long-Term Memory Configuration
    enable_memory: bool = Field(
        default=False,
        metadata={
            "x_oap_ui_config": {
                "type": "boolean",
                "default": False,
                "description": "Enable mem0 long-term memory for user preferences, domain profile, and project memory across research sessions."
            }
        }
    )
    memory_provider: Literal["platform", "oss"] = Field(
        default="platform",
        metadata={
            "x_oap_ui_config": {
                "type": "select",
                "default": "platform",
                "description": "Mem0 backend: 'platform' uses AsyncMemoryClient (cloud), 'oss' uses open-source Memory.",
                "options": [
                    {"label": "Platform (Cloud)", "value": "platform"},
                    {"label": "OSS (Open Source)", "value": "oss"},
                ]
            }
        }
    )
    memory_app_id: Optional[str] = Field(
        default=None,
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "",
                "description": "Mem0 app_id for multi-tenancy. Falls back to org_id or 'default'."
            }
        }
    )
    memory_agent_id: Optional[str] = Field(
        default=None,
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "",
                "description": "Mem0 agent_id. Falls back to 'deep_researcher'."
            }
        }
    )
    memory_project_id: Optional[str] = Field(
        default=None,
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "",
                "description": "Mem0 project ID for scoping memories. Override with MEM0_MEMORY_PROJECT_ID env var."
            }
        }
    )
    memory_top_k: int = Field(
        default=8,
        metadata={
            "x_oap_ui_config": {
                "type": "slider",
                "default": 8,
                "min": 1,
                "max": 50,
                "step": 1,
                "description": "Maximum number of memories to recall per query."
            }
        }
    )
    memory_min_confidence: float = Field(
        default=0.75,
        metadata={
            "x_oap_ui_config": {
                "type": "slider",
                "default": 0.75,
                "min": 0.0,
                "max": 1.0,
                "step": 0.05,
                "description": "Minimum confidence threshold for extracting memory candidates."
            }
        }
    )
    memory_auto_write: bool = Field(
        default=True,
        metadata={
            "x_oap_ui_config": {
                "type": "boolean",
                "default": True,
                "description": "Whether to automatically extract and write memory candidates after report generation."
            }
        }
    )
    memory_write_after_report: bool = Field(
        default=True,
        metadata={
            "x_oap_ui_config": {
                "type": "boolean",
                "default": True,
                "description": "If True, memory extraction happens after the final report. If False, memory is only recalled on entry."
            }
        }
    )
    memory_fail_open: bool = Field(
        default=True,
        metadata={
            "x_oap_ui_config": {
                "type": "boolean",
                "default": True,
                "description": "If True, memory write failures are logged but never block the final report."
            }
        }
    )
    memory_advanced_enabled: bool = Field(
        default=False,
        description="Enable schema-v2 observations, explainable reranking, reflection, profiles, and forgetting.",
    )
    memory_decay_enabled: bool = Field(
        default=True,
        description="Enable Mem0 Platform v3 project-level access decay when advanced memory is active.",
    )
    memory_reflection_enabled: bool = Field(default=True)
    memory_profile_enabled: bool = Field(default=True)
    memory_soft_forgetting_enabled: bool = Field(default=True)
    memory_verified_insights_enabled: bool = Field(default=True)
    memory_search_threshold: float = Field(default=0.1, ge=0.0, le=1.0)
    memory_search_rerank: bool = Field(default=True)
    memory_importance_weight: float = Field(default=0.25, ge=0.0, le=1.0)
    memory_relevance_weight: float = Field(default=0.55, ge=0.0, le=1.0)
    memory_recency_weight: float = Field(default=0.20, ge=0.0, le=1.0)
    memory_reflection_observation_threshold: int = Field(default=5, ge=1)
    memory_reflection_importance_threshold: int = Field(default=25, ge=1)
    memory_reflection_max_age_hours: int = Field(default=24, ge=1)
    memory_profile_max_chars: int = Field(default=4000, ge=512, le=4000)
    memory_v2_app_suffix: str = Field(default=".v2", min_length=1, max_length=32)
    memory_half_life_days: dict[str, int] = Field(
        default_factory=lambda: {
            "user_research_preference": 180,
            "domain_profile": 180,
            "project_memory": 90,
            "verified_research_insight": 30,
            "reflection": 90,
        }
    )

    @field_validator("memory_half_life_days", mode="before")
    @classmethod
    def parse_memory_half_lives(cls, value: Any) -> dict[str, int]:
        """Accept a JSON object from environment-based configuration."""
        if isinstance(value, str):
            value = json.loads(value)
        if not isinstance(value, dict):
            raise ValueError("memory_half_life_days must be an object")
        parsed = {str(key): int(days) for key, days in value.items()}
        if any(days < 1 for days in parsed.values()):
            raise ValueError("memory half-lives must be at least one day")
        return parsed

    @field_validator(
        "model_fallbacks",
        "model_context_window_overrides",
        "model_max_output_tokens_overrides",
        mode="before",
    )
    @classmethod
    def parse_model_recovery_maps(cls, value: Any) -> dict[str, Any]:
        """Accept JSON objects for model recovery configuration."""
        if value is None or value == "":
            return {}
        if isinstance(value, str):
            value = json.loads(value)
        if not isinstance(value, dict):
            raise ValueError("model recovery configuration must be an object")
        return value

    @field_validator("model_fallbacks")
    @classmethod
    def validate_model_fallbacks(
        cls,
        value: dict[str, list[str]],
    ) -> dict[str, list[str]]:
        """Reject unknown roles and empty model identifiers."""
        allowed_roles = {
            "supervisor",
            "researcher",
            "summarization",
            "compression",
            "final_report",
            "message_summary",
            "quality_evaluation",
        }
        unknown = set(value) - allowed_roles
        if unknown:
            raise ValueError(
                "unknown model fallback roles: " + ",".join(sorted(unknown))
            )
        normalized: dict[str, list[str]] = {}
        for role, candidates in value.items():
            if not isinstance(candidates, list):
                raise ValueError(
                    f"model fallback chain for {role} must be an array"
                )
            normalized[role] = [
                str(candidate).strip()
                for candidate in candidates
                if str(candidate).strip()
            ]
        return normalized

    @field_validator(
        "model_context_window_overrides",
        "model_max_output_tokens_overrides",
    )
    @classmethod
    def validate_model_token_overrides(
        cls,
        value: dict[str, int],
    ) -> dict[str, int]:
        """Require positive model capability overrides."""
        normalized = {
            str(model): int(tokens) for model, tokens in value.items()
        }
        if any(tokens < 1 for tokens in normalized.values()):
            raise ValueError("model token overrides must be positive")
        return normalized

    @field_validator("sandbox_allowed_domains", mode="before")
    @classmethod
    def parse_sandbox_allowed_domains(cls, value: Any) -> list[str]:
        """Accept comma-separated env values for sandbox domain allowlists."""
        if value is None or value == "":
            return []
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    @field_validator(
        "supervisor_tool_whitelist",
        "researcher_tool_whitelist",
        "supervisor_blocked_origins",
        "researcher_blocked_origins",
        mode="before",
    )
    @classmethod
    def parse_optional_tool_policy_list(cls, value: Any) -> list[str] | None:
        """Accept JSON arrays or comma-separated tool policy environment values."""
        if value is None or value == "":
            return None
        if isinstance(value, str):
            if value.lstrip().startswith("["):
                parsed = json.loads(value)
                if not isinstance(parsed, list):
                    raise ValueError("tool policy JSON must be an array")
                value = parsed
            else:
                value = value.split(",")
        if not isinstance(value, list | tuple | set):
            raise ValueError("tool policy must be a list or comma-separated string")
        return [str(item).strip() for item in value if str(item).strip()]

    @field_validator("mcp_config", "browser_mcp_config", mode="before")
    @classmethod
    def parse_mcp_config_json(cls, value: Any) -> Any:
        """Allow administrator-owned MCP configuration to come from JSON env vars."""
        if isinstance(value, str):
            parsed = json.loads(value)
            if not isinstance(parsed, dict):
                raise ValueError("MCP configuration JSON must be an object")
            return parsed
        return value

    @field_validator("allowed_mcp_servers", "allowed_model_endpoints", mode="before")
    @classmethod
    def parse_security_allowlists(cls, value: Any) -> list[str]:
        """Accept JSON arrays or comma-separated administrator allowlists."""
        if value is None or value == "":
            return []
        if isinstance(value, str):
            if value.lstrip().startswith("["):
                parsed = json.loads(value)
                if not isinstance(parsed, list):
                    raise ValueError("security allowlist JSON value must be an array")
                return [str(item).strip() for item in parsed if str(item).strip()]
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    @field_validator("skills", mode="before")
    @classmethod
    def parse_skills(cls, value: Any) -> Optional[List[str]]:
        """Accept JSON arrays or comma-separated values for enabled skills."""
        if value is None or value == "":
            return None
        if isinstance(value, str):
            if value.lstrip().startswith("["):
                parsed = json.loads(value)
                if not isinstance(parsed, list):
                    raise ValueError("skills JSON value must be an array")
                return parsed
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    @model_validator(mode="after")
    def validate_leader_lease_timing(self) -> "Configuration":
        """Ensure a live leader renews well before its lease can expire."""
        if self.leader_heartbeat_seconds * 3 > self.leader_lease_seconds:
            raise ValueError(
                "leader_heartbeat_seconds must be no more than one third of "
                "leader_lease_seconds"
            )
        memory_weight_sum = (
            self.memory_relevance_weight
            + self.memory_importance_weight
            + self.memory_recency_weight
        )
        if abs(memory_weight_sum - 1.0) > 1e-9:
            raise ValueError("advanced memory retrieval weights must sum to 1.0")
        return self

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "Configuration":
        """Create a Configuration instance from a RunnableConfig."""
        configurable = config.get("configurable", {}) if config else {}
        metadata = config.get("metadata", {}) if config else {}
        frozen = metadata.get("runtime_config_frozen") is True
        field_names = list(cls.model_fields.keys())
        values: dict[str, Any] = {
            field_name: (
                configurable.get(field_name)
                if frozen
                and field_name in RUN_CONFIG_FROZEN_FIELDS
                and field_name in configurable
                else os.environ.get(
                    field_name.upper(), configurable.get(field_name)
                )
            )
            for field_name in field_names
        }
        rigor, _warning = _resolve_quality_rigor(config or {})
        values["quality_evaluation_rigor"] = rigor
        # Explicit env-var overrides for fields with non-standard env names
        if os.environ.get("MEM0_PROVIDER") and "memory_provider" not in (configurable or {}):
            values["memory_provider"] = os.environ["MEM0_PROVIDER"]
        if os.environ.get("MEM0_MEMORY_PROJECT_ID") and "memory_project_id" not in (configurable or {}):
            values["memory_project_id"] = os.environ["MEM0_MEMORY_PROJECT_ID"]
        return cls(**{k: v for k, v in values.items() if v is not None})

    class Config:
        """Pydantic configuration."""
        
        arbitrary_types_allowed = True


def frozen_run_config_values(config: RunnableConfig) -> dict[str, Any]:
    """Return the canonical, non-secret values covered by the run contract."""
    configurable = config.get("configurable", {})
    schema_version = int(
        config.get("metadata", {}).get(
            "run_config_schema_version", RUN_CONFIG_SCHEMA_VERSION
        )
    )
    frozen_fields = (
        RUN_CONFIG_FROZEN_FIELDS_V1
        if schema_version == 1
        else RUN_CONFIG_FROZEN_FIELDS_V2
        if schema_version == 2
        else RUN_CONFIG_FROZEN_FIELDS_V3
        if schema_version == 3
        else RUN_CONFIG_FROZEN_FIELDS
    )
    return {
        field_name: configurable[field_name]
        for field_name in frozen_fields
        if field_name in configurable
    }


def run_config_fingerprint(config: RunnableConfig) -> str:
    """Hash the frozen run contract without credential-bearing configuration."""
    payload = {
        "schema_version": int(
            config.get("metadata", {}).get(
                "run_config_schema_version", RUN_CONFIG_SCHEMA_VERSION
            )
        ),
        "policy_version": str(
            config.get("metadata", {}).get(
                "quality_policy_version", QUALITY_POLICY_VERSION
            )
        ),
        "config": frozen_run_config_values(config),
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def freeze_run_config(
    config: RunnableConfig,
    *,
    prefer_configurable: bool = False,
) -> RunnableConfig:
    """Resolve and pin model, policy, timeout, and budget values for one run."""
    frozen: RunnableConfig = {
        **config,
        "configurable": dict(config.get("configurable", {})),
        "metadata": dict(config.get("metadata", {})),
    }
    metadata = frozen["metadata"]
    if metadata.get("runtime_config_frozen") is True:
        schema_version = int(
            metadata.get("run_config_schema_version", 1)
        )
        frozen_fields = (
            RUN_CONFIG_FROZEN_FIELDS_V1
            if schema_version == 1
            else RUN_CONFIG_FROZEN_FIELDS_V2
            if schema_version == 2
            else RUN_CONFIG_FROZEN_FIELDS_V3
            if schema_version == 3
            else RUN_CONFIG_FROZEN_FIELDS
        )
        missing = [
            field_name
            for field_name in frozen_fields
            if field_name not in frozen["configurable"]
        ]
        if missing:
            raise ValueError(
                "frozen_run_config_incomplete:" + ",".join(sorted(missing))
            )
        expected = str(metadata.get("run_config_fingerprint") or "")
        actual = run_config_fingerprint(frozen)
        if expected and expected != actual:
            raise ValueError("run_config_fingerprint_mismatch")
        metadata["run_config_fingerprint"] = actual
        return frozen

    rigor, warning = _resolve_quality_rigor(
        frozen,
        prefer_configurable=prefer_configurable,
    )
    if prefer_configurable:
        configured_values = dict(frozen["configurable"])
        configured_values.pop("quality_evaluation_min_score", None)
        configured_values["quality_evaluation_rigor"] = rigor
        resolved_config = Configuration(**configured_values)
    else:
        resolved_config = Configuration.from_runnable_config(frozen)
    resolved = resolved_config.model_dump(mode="json")
    for field_name in RUN_CONFIG_FROZEN_FIELDS:
        frozen["configurable"][field_name] = resolved[field_name]
    frozen["configurable"].pop("quality_evaluation_min_score", None)
    if warning:
        warnings = list(metadata.get("quality_configuration_warnings", []))
        if warning not in warnings:
            warnings.append(warning)
        metadata["quality_configuration_warnings"] = warnings
    rigor_policy = get_quality_rigor_policy(rigor)
    metadata.update(
        {
            "runtime_config_frozen": True,
            "run_config_schema_version": RUN_CONFIG_SCHEMA_VERSION,
            "quality_policy_version": QUALITY_POLICY_VERSION,
            "quality_rigor_policy": rigor_policy.as_dict(),
            "quality_evaluation_epoch": str(
                metadata.get("quality_evaluation_epoch") or uuid.uuid4()
            ),
        }
    )
    metadata["run_config_fingerprint"] = run_config_fingerprint(frozen)
    return frozen
