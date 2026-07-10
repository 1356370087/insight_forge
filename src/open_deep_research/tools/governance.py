"""Tool governance: permission control, validation, egress policy, and retry.

This module centralizes all cross-cutting concerns for *how* tools are invoked
by the LangGraph agents in this project:

* **Origin policy** -- using origin declared on the project ``Tool`` Interface.
* **Permission control** -- a per-role gate combining a tool-name whitelist with
  an origin blocklist, plus an MCP auth-required token presence check.
* **Parameter validation** -- configured JSON-Schema constraints followed by
  Pydantic model validation before any side effect.
* **Retry with exponential backoff** -- for retryable tool errors (network,
  timeout, rate-limit/429, service-unavailable/503), returning a structured
  ``ToolError`` (machine-readable JSON) to the model when retries are exhausted.

The public entry point is :func:`execute_governed_tool_call`. It returns both a
transport message and the original ``ToolResult`` while containing failures so
one call cannot abort a concurrent batch.
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Optional

import aiohttp
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import ToolException
from pydantic import BaseModel, Field, ValidationError

from open_deep_research.configuration import Configuration
from open_deep_research.observability import get_trace_recorder
from open_deep_research.sandbox.policy import (
    allowed_domains,
    egress_host_from_url,
    is_enforced_mode,
)
from open_deep_research.tasks.domain_approvals import get_domain_approval_registry
from open_deep_research.tasks.events import EventType
from open_deep_research.tasks.executor import emit_task_state_change
from open_deep_research.tasks.registry import (
    TaskRecord,
    TaskStatus,
    get_task_registry,
)
from open_deep_research.tools.base import (
    Tool,
    ToolContext,
    ToolOrigin,
    ToolResult,
    serialize_tool_output,
)

# Native OpenAI/Anthropic SDKs are optional direct dependencies. They are guarded
# so this module imports cleanly even when the SDKs are absent; classification
# just falls back to the shared error classifier in that case.
try:  # pragma: no cover - import guard
    import openai as _openai_sdk
except ImportError:  # pragma: no cover
    _openai_sdk = None
try:  # pragma: no cover - import guard
    import anthropic as _anthropic_sdk
except ImportError:  # pragma: no cover
    _anthropic_sdk = None

logger = logging.getLogger(__name__)

##########################
# Enums
##########################


class AgentRole(str, Enum):
    """Which graph is invoking the tool -- drives whitelist/origin resolution."""

    SUPERVISOR = "supervisor"
    RESEARCHER = "researcher"


class ToolErrorType(str, Enum):
    """Stable, LLM-parseable error categories emitted in structured errors."""

    permission_denied = "permission_denied"
    validation_error = "validation_error"
    rate_limited = "rate_limited"
    timeout = "timeout"
    network_error = "network_error"
    service_unavailable = "service_unavailable"
    max_retries_exceeded = "max_retries_exceeded"
    tool_not_found = "tool_not_found"
    egress_domain_denied = "egress_domain_denied"
    egress_domain_pending = "egress_domain_pending"
    unknown = "unknown"


##########################
# Origin labeling
##########################

def get_tool_retryable(tool: Tool) -> bool:
    """Return retry policy declared directly by the Tool Interface."""
    return tool.retryable


def get_tool_origin(tool: Tool) -> ToolOrigin:
    """Return origin declared directly by the Tool Interface."""
    return tool.origin


##########################
# Structured error
##########################


class ToolError(BaseModel):
    """A structured, LLM-parseable tool failure rendered into a ``ToolMessage``."""

    error_type: ToolErrorType
    tool_name: str
    message: str
    attempts: Optional[int] = None
    """Number of execution attempts made (set on retryable/max_retries_exceeded)."""
    retryable: bool = False
    """Whether a retry could plausibly help (hint for the model)."""
    detail: dict[str, Any] = Field(default_factory=dict)
    """Machine-readable context, e.g. ``{"status": 503}`` or ``{"missing": ["queries"]}``."""

    def to_tool_message(self, tool_call_id: str) -> ToolMessage:
        """Render this error as a ``ToolMessage`` whose content is the JSON payload."""
        return ToolMessage(
            content=self.model_dump_json(),
            name=self.tool_name,
            tool_call_id=tool_call_id,
        )


@dataclass(frozen=True, slots=True)
class GovernedToolCallResult:
    """Transport message plus the original typed result of one invocation."""

    message: ToolMessage
    result: Optional[ToolResult[Any]] = None
    error: Optional[ToolError] = None


def _error_result(error: ToolError, tool_call_id: str) -> GovernedToolCallResult:
    """Build a governed outcome for a structured failure."""
    return GovernedToolCallResult(
        message=error.to_tool_message(tool_call_id),
        error=error,
    )


def _safe_exc_str(exc: BaseException) -> str:
    """Render an exception to a string, tolerating exceptions whose ``__str__`` itself raises."""
    try:
        text = str(exc)
    except Exception:  # noqa: BLE001 -- e.g. aiohttp ClientResponseError with request_info=None
        text = f"{type(exc).__name__} (message unavailable)"
    return text or type(exc).__name__


def _safe_status(exc: BaseException) -> Optional[int]:
    """Best-effort extraction of an HTTP status from an exception."""
    status = getattr(exc, "status", None)
    if isinstance(status, int):
        return status
    return None


class ToolExecutionFailure(Exception):
    """Internal signal raised by :func:`invoke_tool_with_retry` when retries are exhausted.

    Carries the classified :class:`ToolErrorType` and attempt count so the entry
    point can render a single structured error without re-classifying.
    """

    def __init__(self, error_type: ToolErrorType, attempts: int, inner: BaseException) -> None:
        """Store the classified type, attempt count, and original exception."""
        super().__init__(f"{error_type.value} after {attempts} attempt(s): {_safe_exc_str(inner)}")
        self.error_type = error_type
        self.attempts = attempts
        self.inner = inner


##########################
# Error classification
##########################


def classify_retryable_error(exc: BaseException) -> tuple[ToolErrorType, bool]:
    """Classify an exception into a (error_type, retryable) pair.

    Retryable categories: timeout, network errors, rate-limit (429),
    service-unavailable (503 and other 5xx). Non-retryable for client 4xx.
    Recurses into ``__cause__`` / ``__context__`` and ``ExceptionGroup`` so that
    errors wrapped by libraries (e.g. Tavily, MCP adapters) are still detected.
    """
    # 1. Timeouts.
    if isinstance(exc, asyncio.TimeoutError):
        return ToolErrorType.timeout, True

    # 2. aiohttp HTTP response errors carrying a status code.
    if isinstance(exc, aiohttp.ClientResponseError):
        status = getattr(exc, "status", None)
        if status == 429:
            return ToolErrorType.rate_limited, True
        if status == 503 or (status is not None and status >= 500):
            return ToolErrorType.service_unavailable, True
        if status in (408, 425):
            return ToolErrorType.rate_limited, True
        return ToolErrorType.unknown, False

    # 3. Generic connection / network errors.
    if isinstance(exc, aiohttp.ClientError | ConnectionError | OSError):
        return ToolErrorType.network_error, True

    # 4. LangChain ToolException carrying HTTP hints in its message.
    if isinstance(exc, ToolException):
        msg = str(exc).lower()
        if "429" in msg or "rate" in msg:
            return ToolErrorType.rate_limited, True
        if "503" in msg or "service unavailable" in msg or "service unavailable" in msg:
            return ToolErrorType.service_unavailable, True
        if "timeout" in msg or "timed out" in msg:
            return ToolErrorType.timeout, True
        return ToolErrorType.unknown, False

    # 5. Recurse into the cause chain (errors wrapped by libraries).
    cause = exc.__cause__ or exc.__context__
    if cause is not None and cause is not exc:
        return classify_retryable_error(cause)

    # 6. ExceptionGroup (Python 3.11+) -- check each sub-exception.
    if hasattr(exc, "exceptions"):
        for sub in exc.exceptions:
            et, retryable = classify_retryable_error(sub)
            if retryable:
                return et, retryable

    return ToolErrorType.unknown, False


def _classify_http_status(status: Any) -> Optional[tuple[ToolErrorType, bool]]:
    """Map an HTTP status to (error_type, retryable); None if not retryable-coded."""
    if not isinstance(status, int):
        return None
    if status == 429:
        return ToolErrorType.rate_limited, True
    if status == 503 or status >= 500:
        return ToolErrorType.service_unavailable, True
    if status in (408, 425):
        return ToolErrorType.rate_limited, True
    return None


def _is_llm_parse_failure(exc: BaseException) -> bool:
    """Return whether a non-transient schema or parse failure occurred."""
    if isinstance(exc, ValidationError):
        return True
    if isinstance(exc, OutputParserException):
        return True
    return False


def _classify_openai_error(exc: BaseException) -> Optional[tuple[ToolErrorType, bool]]:
    """Classify a native OpenAI SDK exception, or None if it is not one."""
    if _openai_sdk is None:
        return None
    if isinstance(exc, _openai_sdk.RateLimitError):
        return ToolErrorType.rate_limited, True
    if isinstance(exc, _openai_sdk.APITimeoutError):
        return ToolErrorType.timeout, True
    if isinstance(exc, _openai_sdk.APIConnectionError):
        return ToolErrorType.network_error, True
    if isinstance(exc, _openai_sdk.APIStatusError):
        mapped = _classify_http_status(getattr(exc, "status_code", None))
        if mapped is not None:
            return mapped
        return ToolErrorType.unknown, False
    return None


def _classify_anthropic_error(exc: BaseException) -> Optional[tuple[ToolErrorType, bool]]:
    """Classify a native Anthropic SDK exception, or None if it is not one."""
    if _anthropic_sdk is None:
        return None
    if isinstance(exc, _anthropic_sdk.RateLimitError):
        return ToolErrorType.rate_limited, True
    if isinstance(exc, _anthropic_sdk.APITimeoutError):
        return ToolErrorType.timeout, True
    if isinstance(exc, _anthropic_sdk.APIConnectionError):
        return ToolErrorType.network_error, True
    if isinstance(exc, _anthropic_sdk.APIStatusError):
        mapped = _classify_http_status(getattr(exc, "status_code", None))
        if mapped is not None:
            return mapped
        return ToolErrorType.unknown, False
    return None


def classify_llm_retryable_error(exc: BaseException) -> tuple[ToolErrorType, bool]:
    """Classify an LLM / native-SDK exception into (error_type, retryable).

    Extends :func:`classify_retryable_error` with explicit branches for the
    native OpenAI/Anthropic SDK exception types so that LLM 429s and transient
    transport errors are retried (and recorded by the observability retry loop).
    Schema/parse failures (``OutputParserException``, pydantic ``ValidationError``)
    are explicitly non-retryable -- retrying them would spin. Anything not matched
    here falls through to the shared classifier, which handles aiohttp status
    codes, ``ConnectionError``/``TimeoutError``, ``ToolException`` keyword hints,
    and the ``__cause__``/``__context__``/``ExceptionGroup`` chains.
    """
    # 1. Schema/parse failures are not transient.
    if _is_llm_parse_failure(exc):
        return ToolErrorType.unknown, False

    # 2. Native OpenAI SDK errors.
    result = _classify_openai_error(exc)
    if result is not None:
        return result

    # 3. Native Anthropic SDK errors.
    result = _classify_anthropic_error(exc)
    if result is not None:
        return result

    # 4. Fall back to the shared classifier (aiohttp/connection/timeout/cause chain).
    return classify_retryable_error(exc)


##########################
# Parameter validation (dependency-free JSON-Schema subset)
##########################


# JSON-Schema types mapped to Python type checks.
_PY_TYPE_CHECKS: dict[str, Callable[[Any], bool]] = {
    "string": lambda v: isinstance(v, str),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": lambda v: isinstance(v, int | float) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
    "array": lambda v: isinstance(v, list),
    "object": lambda v: isinstance(v, dict),
}


def _resolve_spec(spec: dict[str, Any]) -> dict[str, Any]:
    """Resolve a property spec, unwrapping ``anyOf`` (first branch) for type checks."""
    if "anyOf" in spec and isinstance(spec["anyOf"], list) and spec["anyOf"]:
        first = spec["anyOf"][0]
        if isinstance(first, dict):
            merged = dict(first)
            merged.update({k: v for k, v in spec.items() if k != "anyOf"})
            return merged
    return spec


def _check_value(name: str, value: Any, spec: dict[str, Any]) -> Optional[ToolError]:
    """Validate a single argument value against its JSON-Schema property spec.

    Covers the subset emitted by LangChain's ``tool.args``: type, enum, array
    item type, and the standard numeric/string/array bound constraints
    (``minimum``/``maximum``, ``minLength``/``maxLength``, ``minItems``/
    ``maxItems``). Lenient on anything unrecognized -- returns ``None`` rather
    than raising.
    """
    spec = _resolve_spec(spec)
    type_name = spec.get("type")

    # Enum constraint (checked before/after type -- order tolerant).
    if "enum" in spec and value not in spec["enum"]:
        return ToolError(
            error_type=ToolErrorType.validation_error,
            tool_name="",
            message=f"Argument '{name}' must be one of {spec['enum']}; got {value!r}",
            detail={"argument": name, "allowed": spec["enum"], "got": value},
        )

    if type_name:
        checker = _PY_TYPE_CHECKS.get(type_name)
        if checker is not None and not checker(value):
            return ToolError(
                error_type=ToolErrorType.validation_error,
                tool_name="",
                message=f"Argument '{name}' expected type '{type_name}'; got {type(value).__name__}",
                detail={"argument": name, "expected_type": type_name, "got_type": type(value).__name__},
            )

    # Numeric bounds (integer/number).
    if type_name in ("integer", "number") and isinstance(value, int | float) and not isinstance(value, bool):
        if "minimum" in spec and value < spec["minimum"]:
            return ToolError(
                error_type=ToolErrorType.validation_error, tool_name="",
                message=f"Argument '{name}' must be >= {spec['minimum']}; got {value}",
                detail={"argument": name, "constraint": "minimum", "minimum": spec["minimum"], "got": value},
            )
        if "maximum" in spec and value > spec["maximum"]:
            return ToolError(
                error_type=ToolErrorType.validation_error, tool_name="",
                message=f"Argument '{name}' must be <= {spec['maximum']}; got {value}",
                detail={"argument": name, "constraint": "maximum", "maximum": spec["maximum"], "got": value},
            )

    # String length bounds.
    if type_name == "string" and isinstance(value, str):
        if "minLength" in spec and len(value) < spec["minLength"]:
            return ToolError(
                error_type=ToolErrorType.validation_error, tool_name="",
                message=f"Argument '{name}' length must be >= {spec['minLength']}; got length {len(value)}",
                detail={"argument": name, "constraint": "minLength", "minLength": spec["minLength"], "got_length": len(value)},
            )
        if "maxLength" in spec and len(value) > spec["maxLength"]:
            return ToolError(
                error_type=ToolErrorType.validation_error, tool_name="",
                message=f"Argument '{name}' length must be <= {spec['maxLength']}; got length {len(value)}",
                detail={"argument": name, "constraint": "maxLength", "maxLength": spec["maxLength"], "got_length": len(value)},
            )

    # Array bounds and element type.
    if type_name == "array" and isinstance(value, list):
        if "minItems" in spec and len(value) < spec["minItems"]:
            return ToolError(
                error_type=ToolErrorType.validation_error, tool_name="",
                message=f"Argument '{name}' must have >= {spec['minItems']} items; got {len(value)}",
                detail={"argument": name, "constraint": "minItems", "minItems": spec["minItems"], "got_items": len(value)},
            )
        if "maxItems" in spec and len(value) > spec["maxItems"]:
            return ToolError(
                error_type=ToolErrorType.validation_error, tool_name="",
                message=f"Argument '{name}' must have <= {spec['maxItems']} items; got {len(value)}",
                detail={"argument": name, "constraint": "maxItems", "maxItems": spec["maxItems"], "got_items": len(value)},
            )
        if "items" in spec and isinstance(spec["items"], dict):
            item_spec = spec["items"]
            item_type = item_spec.get("type")
            if item_type:
                item_checker = _PY_TYPE_CHECKS.get(item_type)
                if item_checker is not None:
                    for i, item in enumerate(value):
                        if not item_checker(item):
                            return ToolError(
                                error_type=ToolErrorType.validation_error, tool_name="",
                                message=f"Argument '{name}[{i}]' expected type '{item_type}'; got {type(item).__name__}",
                                detail={"argument": name, "index": i, "expected_type": item_type, "got_type": type(item).__name__},
                            )
                # Per-element string-length bounds (e.g. per-query max length).
                if item_type == "string":
                    for i, item in enumerate(value):
                        if isinstance(item, str):
                            err = _check_value(f"{name}[{i}]", item, item_spec)
                            if err is not None:
                                return err
    return None


def validate_tool_args(
    tool: Tool, args: dict[str, Any], config: Optional[RunnableConfig] = None,
) -> Optional[ToolError]:
    """Validate ``args`` against the tool's LLM-facing input schema.

    Uses ``tool.input_schema.model_json_schema()``, which yields a standard
    JSON-Schema object with ``properties`` and ``required`` keys. Returns
    ``None`` on success, or a :class:`ToolError` describing the first problem.
    Never raises.

    ``InjectedToolArg`` parameters (e.g. ``config``, ``max_results`` for
    ``tavily_search``) are excluded from the LLM-facing schema by LangChain and
    are *not* in ``required``, so the validator never demands them. They may
    still appear in ``properties``; since they are injected at runtime rather
    than emitted by the model, we ignore any properties not present in ``args``.

    When ``config`` is provided and ``tool_param_constraints`` is set, per-tool
    parameter bounds (``minItems``/``maxItems``/``minLength``/``maxLength``/
    ``minimum``/``maximum``) are layered on top of the schema's own constraints.
    """
    try:
        schema = tool.input_schema.model_json_schema()
    except Exception:  # noqa: BLE001 -- malformed adapters fail leniently here
        return None
    if not isinstance(schema, dict):
        return None  # Unknown schema shape -- lenient.

    required = schema.get("required", []) or []
    properties = schema.get("properties", {}) or {}
    additional_props_false = schema.get("additionalProperties") is False

    # Missing required arguments.
    missing = [k for k in required if k not in args]
    if missing:
        return ToolError(
            error_type=ToolErrorType.validation_error,
            tool_name=tool.name,
            message=f"Missing required arguments: {missing}",
            detail={"missing": missing},
        )

    # Resolve per-tool configured constraints (layered on top of the schema).
    extra_constraints: dict[str, dict[str, Any]] = {}
    if config is not None:
        configurable = Configuration.from_runnable_config(config)
        if configurable.tool_param_constraints:
            extra_constraints = configurable.tool_param_constraints.get(tool.name, {}) or {}

    for key, val in args.items():
        spec = properties.get(key)
        if spec is None:
            if additional_props_false:
                return ToolError(
                    error_type=ToolErrorType.validation_error,
                    tool_name=tool.name,
                    message=f"Unexpected argument '{key}'; tool does not accept it",
                    detail={"unexpected": [key]},
                )
            continue  # Lenient: ignore unknown args by default (LLMs add noise).
        if not isinstance(spec, dict):
            continue
        # Merge configured constraints into the spec copy for this argument.
        # Top-level bounds (minItems/maxItems/minLength/maxLength/minimum/maximum)
        # override; the 'items' sub-spec is deep-merged so a configured per-element
        # bound does not drop the schema's element type.
        merged_spec = spec
        if key in extra_constraints:
            merged_spec = dict(spec)
            for ck, cv in extra_constraints[key].items():
                if ck == "items" and isinstance(cv, dict) and isinstance(spec.get("items"), dict):
                    merged_spec["items"] = {**spec["items"], **cv}
                else:
                    merged_spec[ck] = cv
        err = _check_value(key, val, merged_spec)
        if err is not None:
            err.tool_name = tool.name
            return err
    return None


##########################
# Retry with exponential backoff
##########################


async def invoke_tool_with_retry(
    tool: Tool,
    input: BaseModel,
    context: ToolContext,
    *,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    sleeper: Optional[Callable[[float], Any]] = None,
) -> ToolResult[Any]:
    """Invoke ``Tool.call`` with exponential backoff on retryable errors.

    Backoff: ``delay = min(max_delay, base_delay * 2**attempt) + jitter``, where
    jitter is ``random.uniform(0, base_delay)``. Retries only for retryable
    errors (see :func:`classify_retryable_error`) and only while
    ``attempt < max_retries``.

    Raises :class:`ToolExecutionFailure` when retries are exhausted or the error
    is non-retryable -- the caller (entry point) renders the structured error.
    ``sleeper`` is injectable for tests to avoid real delays.
    """
    sleeper = sleeper or asyncio.sleep
    attempt = 0
    while True:
        try:
            return await tool.call(input, context)
        except Exception as exc:  # noqa: BLE001 -- classify then decide
            error_type, retryable = classify_retryable_error(exc)
            if not retryable or attempt >= max_retries:
                final_type = (
                    ToolErrorType.max_retries_exceeded
                    if retryable and attempt >= max_retries
                    else error_type
                )
                raise ToolExecutionFailure(final_type, attempt + 1, exc) from exc
            delay = min(max_delay, base_delay * (2 ** attempt)) + random.uniform(0, base_delay)
            logger.debug(
                "Tool %s failed with %s (retryable); retry %d/%d after %.2fs",
                tool.name, error_type.value, attempt + 1, max_retries, delay,
            )
            # Record this retry on the span opened by observe_tool_call (noop if
            # observability is disabled or no span is active).
            get_trace_recorder(context.config).active_span().record_retry(
                attempt=attempt + 1,
                error_type=error_type.value,
                http_status=_safe_status(exc),
                retryable=True,
                delay_s=delay,
                message=_safe_exc_str(exc),
            )
            await sleeper(delay)
            attempt += 1


##########################
# Permission check
##########################


def _origin_blocklist(role: AgentRole, config: RunnableConfig) -> set[ToolOrigin]:
    """Resolve the set of blocked origins for ``role`` from config."""
    configurable = Configuration.from_runnable_config(config)
    raw = (
        configurable.supervisor_blocked_origins
        if role is AgentRole.SUPERVISOR
        else configurable.researcher_blocked_origins
    )
    if not raw:
        return set()
    blocked: set[ToolOrigin] = set()
    for value in raw:
        try:
            blocked.add(ToolOrigin(value))
        except ValueError:
            logger.warning("Unknown tool origin %r in %s_blocked_origins; ignored", value, role.value)
    return blocked


def resolve_allowed_tools(
    role: AgentRole, config: RunnableConfig, assembled_names: set[str]
) -> Optional[set[str]]:
    """Resolve the per-role tool-name whitelist.

    Returns ``None`` when no whitelist is configured (backward compatible: all
    assembled tools allowed). Otherwise returns the intersection with
    ``assembled_names`` so a stale whitelist cannot reference tools that are not
    actually present.
    """
    configurable = Configuration.from_runnable_config(config)
    whitelist = (
        configurable.supervisor_tool_whitelist
        if role is AgentRole.SUPERVISOR
        else configurable.researcher_tool_whitelist
    )
    if whitelist is None:
        return None
    return {name for name in whitelist if name in assembled_names}


def _peek_mcp_tokens(config: RunnableConfig) -> Optional[Any]:
    """Check whether an MCP access token is present in config (no exchange work)."""
    configurable = config.get("configurable", {}) or {} if isinstance(config, dict) else {}
    tokens = configurable.get("mcp_tokens")
    return tokens if tokens else None


def get_user_permissions(config: RunnableConfig) -> list[str]:
    """Read the authenticated user's roles/permissions from the run config.

    The LangGraph server injects the authenticated user (as returned by
    ``auth.authenticate``) into ``config["configurable"]["langgraph_auth_user"]``,
    whose ``permissions`` field carries the JWT roles extracted in
    :mod:`open_deep_research.security.auth`. We also accept a plain
    ``configurable["user_permissions"]`` list as a fallback (e.g. for local
    tests or non-server invocations). Returns ``[]`` when no user/roles are
    present (anonymous -> only agent-scope policy applies).
    """
    configurable = config.get("configurable", {}) or {} if isinstance(config, dict) else {}
    auth_user = configurable.get("langgraph_auth_user")
    if auth_user is not None:
        # BaseUser instance (server path) or a plain dict (tests).
        perms = getattr(auth_user, "permissions", None)
        if perms is None and isinstance(auth_user, dict):
            perms = auth_user.get("permissions")
        if isinstance(perms, list | tuple):
            return [str(p) for p in perms]
    fallback = configurable.get("user_permissions")
    if isinstance(fallback, list | tuple):
        return [str(p) for p in fallback]
    return []


def check_permission(
    tool_name: str,
    tool: Tool,
    role: AgentRole,
    allowed: Optional[set[str]],
    config: RunnableConfig,
) -> Optional[ToolError]:
    """Run the permission gate: whitelist membership + origin policy + MCP auth.

    Returns ``None`` when permitted, or a ``permission_denied`` :class:`ToolError`.
    """
    # 1. Whitelist membership.
    if allowed is not None and tool_name not in allowed:
        return ToolError(
            error_type=ToolErrorType.permission_denied,
            tool_name=tool_name,
            message=f"Tool '{tool_name}' is not in the {role.value} tool whitelist.",
            detail={"role": role.value},
        )

    # 2. Origin policy.
    origin = get_tool_origin(tool)
    blocked = _origin_blocklist(role, config)
    if origin in blocked:
        return ToolError(
            error_type=ToolErrorType.permission_denied,
            tool_name=tool_name,
            message=f"Tool '{tool_name}' (origin={origin.value}) is not permitted for the {role.value}.",
            detail={"role": role.value, "origin": origin.value},
        )

    # 3. MCP auth-required: a tool is only permitted if it was loaded with a
    #    valid token. load_mcp_tools tags the tool's metadata with
    #    ``mcp_auth_satisfied=True`` when auth_required and fetch_tokens
    #    succeeded, so we trust that marker rather than probing config (tokens
    #    are never written back into the run config). Fall back to a config token
    #    peek only for tools that predate the marker.
    if origin is ToolOrigin.MCP:
        configurable = Configuration.from_runnable_config(config)
        if configurable.mcp_config and configurable.mcp_config.auth_required:
            auth_satisfied = bool(getattr(tool, "auth_satisfied", False))
            if not auth_satisfied and not _peek_mcp_tokens(config):
                return ToolError(
                    error_type=ToolErrorType.permission_denied,
                    tool_name=tool_name,
                    message="MCP tool requires authentication but was not loaded with a valid token.",
                    detail={"origin": ToolOrigin.MCP.value, "auth_required": True},
                )

    # 4. User-role blacklist (deny). Layered on top of the agent-scope policy:
    # for each role the authenticated user holds, blocklisted tool names and
    # origins are denied. Users with no roles (anonymous) skip this layer, so
    # existing deployments and local Studio runs are unaffected.
    user_roles = get_user_permissions(config)
    if user_roles:
        configurable = Configuration.from_runnable_config(config)
        role_tool_bl = configurable.role_tool_blacklist or {}
        role_origin_bl = configurable.role_blocked_origins or {}
        for user_role in user_roles:
            role_blocked_tools = role_tool_bl.get(user_role) or []
            if tool_name in role_blocked_tools:
                return ToolError(
                    error_type=ToolErrorType.permission_denied,
                    tool_name=tool_name,
                    message=f"Tool '{tool_name}' is blocked for user role '{user_role}'.",
                    detail={"user_role": user_role, "scope": "tool"},
                )
            role_blocked_origins = role_origin_bl.get(user_role) or []
            if origin.value in role_blocked_origins:
                return ToolError(
                    error_type=ToolErrorType.permission_denied,
                    tool_name=tool_name,
                    message=f"Tool '{tool_name}' (origin={origin.value}) is blocked for user role '{user_role}'.",
                    detail={"user_role": user_role, "scope": "origin", "origin": origin.value},
                )
    return None


def filter_tools_by_permission(
    tools: list[Tool],
    role: AgentRole,
    config: RunnableConfig,
) -> list[Any]:
    """Filter assembled ``tools`` down to those the ``role`` is permitted to bind.

    Applies the permission gate (whitelist membership + origin policy + user-role
    blacklist + MCP auth) to each tool and returns only the permitted ones. This
    runs *before* ``bind_tools`` so disallowed tool names and schemas are never
    exposed to the model. Parameter validation is intentionally not done here --
    it is an execution-time concern handled by :func:`execute_governed_tool_call`.

    Provider-native search ``dict`` tools are filtered by name like the rest.

    When no whitelist, no origin blocklist, and no user-role blacklist apply,
    every tool passes through unchanged (zero-cost backward compatibility).
    """
    names = {t.name for t in tools}
    allowed = resolve_allowed_tools(role, config, names)
    # Fast path: nothing to enforce -> return input as-is.
    if allowed is None and not _origin_blocklist(role, config) and not _user_role_blocklists_active(config):
        return tools

    filtered: list[Tool] = []
    for t in tools:
        name = t.name
        if check_permission(name, t, role, allowed, config) is None:
            filtered.append(t)
    return filtered


def _user_role_blocklists_active(config: RunnableConfig) -> bool:
    """Whether any user-role blacklist config is present (cheap fast-path check)."""
    configurable = Configuration.from_runnable_config(config)
    return bool(configurable.role_tool_blacklist or configurable.role_blocked_origins)


##########################
# Egress domain allowlist
##########################


def _egress_host_for_tool(
    tool: Tool, args: dict[str, Any], configurable: Configuration
) -> Optional[str]:
    """Return the egress host a URL-bearing tool targets, or ``None`` to skip.

    Only URL-bearing tools are intercepted (per the allowlist design): MCP tools
    (host from ``mcp_config.url``) and the ``fetch_webpage`` tool (host from
    ``args["url"]``). Search/model/think tools are skipped because their hosts are
    already statically derived in :func:`allowed_domains`.
    """
    origin = get_tool_origin(tool)
    if origin is ToolOrigin.MCP:
        if configurable.mcp_config and configurable.mcp_config.url:
            return egress_host_from_url(configurable.mcp_config.url)
        return None
    if tool.name == "fetch_webpage":
        url = args.get("url")
        if isinstance(url, str):
            return egress_host_from_url(url)
        return None
    return None


def _find_task_for_run(
    registry: Any, run_id: str, task_id: Optional[str] = None
) -> Optional[TaskRecord]:
    """Locate the active task for a run (used to pause it for a domain decision).

    Prefers an explicit ``task_id`` threaded via config metadata; otherwise falls
    back to any RUNNING/WAITING task for the run, preferring a WAITING one.
    """
    if task_id:
        rec = registry.get(task_id)
        if rec is not None and rec.run_id == run_id and rec.status in (
            TaskStatus.RUNNING,
            TaskStatus.WAITING_FOR_CONFIRMATION,
        ):
            return rec
    candidates = [
        r
        for r in registry.list(run_id=run_id)
        if r.status in (TaskStatus.RUNNING, TaskStatus.WAITING_FOR_CONFIRMATION)
    ]
    if not candidates:
        return None
    return next(
        (r for r in candidates if r.status == TaskStatus.WAITING_FOR_CONFIRMATION),
        candidates[0],
    )


async def check_egress_domain(
    tool_call: dict[str, Any],
    tool: Tool,
    args: dict[str, Any],
    config: RunnableConfig,
) -> Optional[ToolMessage]:
    """Enforce the egress domain allowlist for URL-bearing tools.

    Returns ``None`` to allow the call to proceed, or a ``ToolMessage`` carrying a
    structured :class:`ToolError` to block it. In the in-process path an undecided
    domain **blocks inline** on a per-run ``asyncio.Future`` (the researcher task
    pauses as ``WAITING_FOR_CONFIRMATION`` until the supervisor decides); in the
    Docker sandbox path the worker cannot share host asyncio state, so it returns
    a pending ``ToolError`` telling the researcher to request approval and retry.
    """
    tool_call_id = tool_call["id"]
    configurable = Configuration.from_runnable_config(config)
    if not is_enforced_mode(configurable):
        return None

    host = _egress_host_for_tool(tool, args, configurable)
    if host is None:
        return None
    if host in set(allowed_domains(configurable)):
        return None

    run_id = config.get("metadata", {}).get("run_id", "default")
    task_id = config.get("metadata", {}).get("task_id")
    approvals = get_domain_approval_registry()
    decision = approvals.is_allowed(run_id, host)
    if decision is None:
        from open_deep_research.tasks.coordination import FileDomainDecisionStore

        decision = await FileDomainDecisionStore(configurable, run_id).get(host)
        if decision is not None:
            approvals.record_decision(run_id, host, decision)
    if decision is True:
        return None
    if decision is False:
        return ToolError(
            error_type=ToolErrorType.egress_domain_denied,
            tool_name=getattr(tool, "name", "unknown"),
            message=(
                f"Domain '{host}' is not on the egress allowlist and was denied "
                f"for this run."
            ),
            detail={"domain": host, "run_id": run_id, "denied": True},
        ).to_tool_message(tool_call_id)

    # Undecided domain. The Docker sandbox worker sets SANDBOX_NETWORK_MODE; inside
    # the container it cannot block on a host-side Future, so it returns a pending
    # error and the researcher is told to ask the supervisor to approve and retry.
    if os.getenv("SANDBOX_NETWORK_MODE") is not None:
        return ToolError(
            error_type=ToolErrorType.egress_domain_pending,
            tool_name=getattr(tool, "name", "unknown"),
            message=(
                f"Domain '{host}' requires supervisor approval before it can be "
                f"fetched. Ask the supervisor to approve it with "
                f"ApproveResearchDomain(task_id, domain='{host}', allow=True), "
                f"then retry this fetch."
            ),
            detail={"domain": host, "run_id": run_id, "pending": True},
        ).to_tool_message(tool_call_id)

    # In-process path: pause inline (Option 1) until the supervisor decides.
    registry = get_task_registry()
    record = _find_task_for_run(registry, run_id, task_id)
    if record is None:
        return ToolError(
            error_type=ToolErrorType.egress_domain_denied,
            tool_name=getattr(tool, "name", "unknown"),
            message=(
                f"Domain '{host}' is not on the egress allowlist and no active "
                f"task context is available to request approval."
            ),
            detail={"domain": host, "run_id": run_id},
        ).to_tool_message(tool_call_id)

    req = approvals.request_decision(run_id, host, getattr(tool, "name", "unknown"))
    record.pending_domain = host
    record.pending_domain_tool = getattr(tool, "name", "unknown")
    registry.update_status(record.task_id, TaskStatus.WAITING_FOR_CONFIRMATION)
    await emit_task_state_change(
        record,
        config,
        event_type=EventType.TASK_DOMAIN_CONFIRMATION_REQUESTED,
        runs_dir=configurable.runs_dir,
        run_id=run_id,
        event_log_enabled=configurable.event_log_enabled,
        data={"domain": host, "tool": getattr(tool, "name", "unknown")},
    )
    try:
        allowed = await req.wait()  # blocks until ApproveResearchDomain resolves it
    except asyncio.CancelledError:
        record.pending_domain = None
        record.pending_domain_tool = None
        return ToolError(
            error_type=ToolErrorType.egress_domain_denied,
            tool_name=getattr(tool, "name", "unknown"),
            message=f"Domain '{host}' request cancelled (task ended).",
            detail={"domain": host, "run_id": run_id, "cancelled": True},
        ).to_tool_message(tool_call_id)
    finally:
        record.pending_domain = None
        record.pending_domain_tool = None

    if not allowed:
        return ToolError(
            error_type=ToolErrorType.egress_domain_denied,
            tool_name=getattr(tool, "name", "unknown"),
            message=f"Domain '{host}' was denied by the supervisor for this run.",
            detail={"domain": host, "run_id": run_id, "denied": True},
        ).to_tool_message(tool_call_id)

    # Allowed -> the domain is now cached for the rest of the run; proceed.
    registry.update_status(record.task_id, TaskStatus.RUNNING)
    await emit_task_state_change(
        record,
        config,
        event_type=EventType.TASK_DOMAIN_DECISION,
        runs_dir=configurable.runs_dir,
        run_id=run_id,
        event_log_enabled=configurable.event_log_enabled,
        data={"domain": host, "allow": True},
    )
    return None


async def execute_governed_tool_call(
    tool_call: dict[str, Any],
    tools_by_name: dict[str, Tool],
    role: AgentRole,
    config: RunnableConfig,
    *,
    allowed_tools: Optional[set[str]] = None,
    apply_retry: bool = True,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    sleeper: Optional[Callable[[float], Any]] = None,
) -> GovernedToolCallResult:
    """Execute a single tool call under full governance.

    Pipeline (each branch returns a ``GovernedToolCallResult``, never raises):
    1. ``tool_not_found`` if the named tool is not registered for this role.
    2. Permission gate (whitelist + origin + MCP auth).
    3. Configured constraints and Pydantic input validation.
    4. Egress policy.
    5. ``Tool.call`` with optional retry and stable result rendering.
    """
    name = tool_call["name"]
    tool_call_id = tool_call["id"]
    args = tool_call.get("args", {}) or {}

    def observed_error(error: ToolError) -> GovernedToolCallResult:
        get_trace_recorder(config).active_span().record_outcome(
            error_type=error.error_type.value,
            http_status=error.detail.get("status") if isinstance(error.detail, dict) else None,
        )
        return _error_result(error, tool_call_id)

    # tool_not_found
    tool = tools_by_name.get(name)
    if tool is None:
        return observed_error(ToolError(
            error_type=ToolErrorType.tool_not_found,
            tool_name=name,
            message=f"No tool named '{name}' is registered for the {role.value}.",
            detail={"role": role.value},
        ))

    # Permission gate.
    perm_err = check_permission(name, tool, role, allowed_tools, config)
    if perm_err is not None:
        return observed_error(perm_err)

    # Parameter validation (with per-tool configured constraints).
    val_err = validate_tool_args(tool, args, config)
    if val_err is not None:
        return observed_error(val_err)
    try:
        validated_input = tool.input_schema.model_validate(args)
    except ValidationError as exc:
        return observed_error(
            ToolError(
                error_type=ToolErrorType.validation_error,
                tool_name=name,
                message="Input failed Pydantic validation.",
                detail={"errors": exc.errors(include_url=False)},
            ),
        )

    # Egress domain allowlist for URL-bearing tools (MCP + fetch_webpage). May
    # block inline (in-process) until a supervisor decision arrives.
    egress_err = await check_egress_domain(tool_call, tool, args, config)
    if egress_err is not None:
        get_trace_recorder(config).active_span().record_outcome(
            error_type=ToolErrorType.egress_domain_denied.value,
        )
        return GovernedToolCallResult(message=egress_err)

    context = ToolContext(
        config=config,
        role=role.value,
        tool_call_id=tool_call_id,
    )

    # Execution. Retry is applied only when the caller enables it (apply_retry)
    # AND the tool is marked retryable (network-bound SEARCH/MCP tools). System
    # orchestration tools are tagged non-retryable and always execute once.
    effective_retry = apply_retry and get_tool_retryable(tool)
    if not effective_retry:
        try:
            result = await tool.call(validated_input, context)
            return GovernedToolCallResult(
                message=ToolMessage(
                    content=serialize_tool_output(result.output),
                    name=name,
                    tool_call_id=tool_call_id,
                ),
                result=result,
            )
        except Exception as exc:  # noqa: BLE001
            error_type, _ = classify_retryable_error(exc)
            get_trace_recorder(config).active_span().record_outcome(
                error_type=error_type.value,
                http_status=_safe_status(exc),
            )
            return _error_result(ToolError(
                error_type=error_type,
                tool_name=name,
                message=f"Tool execution failed: {_safe_exc_str(exc)}",
                detail={"status": _safe_status(exc)},
            ), tool_call_id)

    try:
        result = await invoke_tool_with_retry(
            tool, validated_input, context,
            max_retries=max_retries, base_delay=base_delay, max_delay=max_delay, sleeper=sleeper,
        )
        return GovernedToolCallResult(
            message=ToolMessage(
                content=serialize_tool_output(result.output),
                name=name,
                tool_call_id=tool_call_id,
            ),
            result=result,
        )
    except ToolExecutionFailure as failure:
        get_trace_recorder(config).active_span().record_outcome(
            error_type=failure.error_type.value,
            http_status=_safe_status(failure.inner),
            retry_count=failure.attempts - 1,
        )
        return _error_result(ToolError(
            error_type=failure.error_type,
            tool_name=name,
            message=f"Tool execution failed after {failure.attempts} attempt(s): {_safe_exc_str(failure.inner)}",
            attempts=failure.attempts,
            retryable=False,
            detail={"status": _safe_status(failure.inner)},
        ), tool_call_id)
    except Exception as exc:  # noqa: BLE001 -- non-retryable, surfaced directly
        error_type, _ = classify_retryable_error(exc)
        get_trace_recorder(config).active_span().record_outcome(
            error_type=error_type.value,
            http_status=_safe_status(exc),
        )
        return _error_result(ToolError(
            error_type=error_type,
            tool_name=name,
            message=f"Tool execution failed: {_safe_exc_str(exc)}",
            detail={"status": _safe_status(exc)},
        ), tool_call_id)
