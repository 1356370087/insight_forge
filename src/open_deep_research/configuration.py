"""Configuration management for the Open Deep Research system."""

import json
import os
from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field, field_validator, model_validator


class SearchAPI(Enum):
    """Enumeration of available search API providers."""
    
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    TAVILY = "tavily"
    NONE = "none"


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
        default="shadow",
        metadata={
            "x_oap_ui_config": {
                "type": "select",
                "default": "shadow",
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
        description="OpenAI-compatible Qwen model used by the runtime quality gates.",
    )
    quality_evaluation_model_max_tokens: int = Field(default=2048, ge=256)
    quality_evaluation_base_url: Optional[str] = Field(
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
        description="OpenAI-compatible DashScope endpoint for the quality model.",
    )
    quality_evaluation_fail_open: bool = Field(
        default=True,
        description="Allow research to continue if the quality evaluator is unavailable.",
    )
    quality_evaluation_min_score: int = Field(default=3, ge=1, le=5)
    quality_evaluation_min_sources: int = Field(default=2, ge=0, le=20)
    quality_evaluation_max_input_chars: int = Field(default=30000, ge=1000)
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

    @field_validator("sandbox_allowed_domains", mode="before")
    @classmethod
    def parse_sandbox_allowed_domains(cls, value: Any) -> list[str]:
        """Accept comma-separated env values for sandbox domain allowlists."""
        if value is None or value == "":
            return []
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

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
        return self

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "Configuration":
        """Create a Configuration instance from a RunnableConfig."""
        configurable = config.get("configurable", {}) if config else {}
        field_names = list(cls.model_fields.keys())
        values: dict[str, Any] = {
            field_name: os.environ.get(field_name.upper(), configurable.get(field_name))
            for field_name in field_names
        }
        # Explicit env-var overrides for fields with non-standard env names
        if os.environ.get("MEM0_PROVIDER") and "memory_provider" not in (configurable or {}):
            values["memory_provider"] = os.environ["MEM0_PROVIDER"]
        if os.environ.get("MEM0_MEMORY_PROJECT_ID") and "memory_project_id" not in (configurable or {}):
            values["memory_project_id"] = os.environ["MEM0_MEMORY_PROJECT_ID"]
        return cls(**{k: v for k, v in values.items() if v is not None})

    class Config:
        """Pydantic configuration."""
        
        arbitrary_types_allowed = True
