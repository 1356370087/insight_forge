"""Tests for the tool governance layer.

Covers: tool_governance.py (origin labeling, whitelist/permission, parameter
validation including configured constraints, error classification, retry with
exponential backoff scoped by origin, the governed execution entry point, the
supervisor gate, pre-bind filtering, and user-role blacklists), the JWT role
extraction in security/auth.py, plus integration tests that exercise
researcher_tools and supervisor_tools end-to-end.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import os
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import aiohttp
import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool as lc_tool

from open_deep_research.configuration import Configuration
from open_deep_research.tasks.registry import TaskStatus
from open_deep_research.tools.adapters import adapt_langchain_tool
from open_deep_research.tools.base import Tool, ToolContext, ToolEffect, ToolOrigin
from open_deep_research.tools.governance import (
    AgentRole,
    ToolErrorType,
    ToolExecutionFailure,
    classify_retryable_error,
    filter_tools_by_permission,
    get_tool_origin,
    get_tool_retryable,
    get_user_permissions,
    resolve_allowed_tools,
    validate_tool_args,
)
from open_deep_research.tools.governance import (
    check_permission as _check_permission,
)
from open_deep_research.tools.governance import (
    execute_governed_tool_call as _execute_governed_tool_call,
)
from open_deep_research.tools.governance import (
    invoke_tool_with_retry as _invoke_tool_with_retry,
)
from open_deep_research.tools.research_complete import ResearchComplete
from open_deep_research.tools.supervisor.conduct_research import ConductResearch
from open_deep_research.tools.utils import tavily_search, think_tool

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _config(**configurable: Any) -> RunnableConfig:
    """Build a RunnableConfig with the given configurable overrides."""
    return RunnableConfig(configurable=configurable, metadata={"run_id": "test"})


def _client_response_error(status: int, message: str = "err") -> aiohttp.ClientResponseError:
    """Construct an aiohttp.ClientResponseError with a minimal but str-safe payload."""
    return aiohttp.ClientResponseError(
        request_info=None, history=(), status=status, message=message,
    )


def _make_tool(fn, *, origin: ToolOrigin, retryable: bool, name: str | None = None) -> Any:
    """Wrap a coroutine behind the project Tool Adapter.

    ``fn`` is a bare async function (with a docstring) -- we apply ``lc_tool``
    fresh so the resulting StructuredTool has a clean schema, then tag metadata.
    """
    t = lc_tool(fn)
    if name is not None:
        t.name = name
    return adapt_langchain_tool(t, origin=origin, retryable=retryable)


def _research_complete_tool(
    *,
    origin: ToolOrigin = ToolOrigin.SYSTEM,
    retryable: bool = False,
    auth_satisfied: bool = False,
):
    """Build a fresh ResearchComplete Tool Adapter."""
    return adapt_langchain_tool(
        lc_tool(ResearchComplete),
        origin=origin,
        retryable=retryable,
        auth_satisfied=auth_satisfied,
    )


@lc_tool
async def ok_tool() -> str:
    """A tool that always succeeds."""
    return "ok-result"


async def _flaky_503_fn() -> str:
    """A tool that always raises an HTTP 503."""
    raise _client_response_error(503, "busy")


async def _bad_request_404_fn() -> str:
    """A tool that always raises an HTTP 404 (non-retryable)."""
    raise _client_response_error(404, "nope")


# Module-level @tool wrappers (kept for the retry-unit tests that call
# invoke_tool_with_retry directly with a StructuredTool).
ok_tool = adapt_langchain_tool(ok_tool, origin=ToolOrigin.SYSTEM)
flaky_503 = adapt_langchain_tool(
    lc_tool(_flaky_503_fn), origin=ToolOrigin.SEARCH, retryable=True
)
bad_request_404 = adapt_langchain_tool(
    lc_tool(_bad_request_404_fn), origin=ToolOrigin.SEARCH, retryable=True
)


async def _probe_search_fn(query: str) -> str:
    """A probe search tool."""
    return query


def _is_denied(msg) -> bool:
    """True if a ToolMessage carries a permission_denied structured error."""
    try:
        return json.loads(msg.content).get("error_type") == "permission_denied"
    except Exception:
        return False  # non-JSON content means the tool executed (success)
    """True if a ToolMessage carries a permission_denied structured error."""
    try:
        return json.loads(msg.content).get("error_type") == "permission_denied"
    except Exception:
        return False  # non-JSON content means the tool executed (success)


def check_permission(
    tool_name,
    tool,
    role,
    allowed,
    origin_index=None,
    config=None,
):
    """Bridge the former test call shape to the Tool-owned origin contract."""
    del origin_index
    return _check_permission(tool_name, tool, role, allowed, config)


async def execute_governed_tool_call(*args, **kwargs):
    """Return the transport message for legacy scenario assertions."""
    kwargs.pop("origin_index", None)
    outcome = await _execute_governed_tool_call(*args, **kwargs)
    return outcome.message


async def invoke_tool_with_retry(
    tool,
    args,
    config,
    **kwargs,
):
    """Invoke the new typed retry seam while preserving scenario assertions."""
    config = config or _config()
    if not isinstance(tool, Tool):
        tool = adapt_langchain_tool(
            tool,
            origin=ToolOrigin.SEARCH,
            retryable=True,
        )
    input = tool.input_schema.model_validate(args)
    result = await _invoke_tool_with_retry(
        tool,
        input,
        ToolContext(config=config, role="researcher", tool_call_id="retry-test"),
        **kwargs,
    )
    return result.output


# ---------------------------------------------------------------------------
# Tool origin tagging (4-category model)
# ---------------------------------------------------------------------------


class TestToolOriginFields:
    def test_system_origin_is_declared_on_tool(self):
        tool = _make_tool(
            _probe_search_fn,
            origin=ToolOrigin.SYSTEM,
            retryable=False,
        )
        assert get_tool_origin(tool) is ToolOrigin.SYSTEM

    def test_search_origin_and_retry_policy_are_direct_fields(self):
        tool = _make_tool(
            _probe_search_fn,
            origin=ToolOrigin.SEARCH,
            retryable=True,
        )
        assert tool.origin is ToolOrigin.SEARCH
        assert get_tool_retryable(tool) is True

    def test_retryable_defaults_are_conservative(self):
        assert get_tool_retryable(ok_tool) is False

    def test_effectful_langchain_tool_is_serial_unless_explicitly_opted_in(self):
        effectful_tool = adapt_langchain_tool(
            lc_tool(_probe_search_fn),
            origin=ToolOrigin.MCP,
            effect=ToolEffect.EXTERNAL_WRITE,
        )

        assert effectful_tool.concurrency_safe is False


# ---------------------------------------------------------------------------
# Whitelist filtering + pre-bind filtering
# ---------------------------------------------------------------------------



class TestWhitelistFiltering:
    def test_resolve_allowed_none_when_whitelist_unset(self):
        # Arrange -- default config has no whitelist
        config = _config()
        # Act
        allowed = resolve_allowed_tools(AgentRole.RESEARCHER, config, {"tavily_search", "think_tool"})
        # Assert
        assert allowed is None  # backward compatible: all assembled tools allowed

    def test_resolve_allowed_intersects_with_assembled(self):
        # Arrange -- whitelist references a tool that is not assembled (stale)
        config = _config(researcher_tool_whitelist=["tavily_search", "ghost_tool"])
        # Act
        allowed = resolve_allowed_tools(AgentRole.RESEARCHER, config, {"tavily_search", "think_tool"})
        # Assert
        assert allowed == {"tavily_search"}  # stale name dropped

    def test_tool_whitelist_env_accepts_comma_separated_values(
        self,
        monkeypatch,
    ):
        monkeypatch.setenv(
            "RESEARCHER_TOOL_WHITELIST",
            "fetch_url, think_tool, ResearchComplete",
        )

        configurable = Configuration.from_runnable_config({"configurable": {}})

        assert configurable.researcher_tool_whitelist == [
            "fetch_url",
            "think_tool",
            "ResearchComplete",
        ]

    def test_resolve_allowed_supervisor_uses_supervisor_whitelist(self):
        # Arrange
        config = _config(supervisor_tool_whitelist=["think_tool"])
        # Act
        allowed = resolve_allowed_tools(AgentRole.SUPERVISOR, config, {"think_tool", "ConductResearch"})
        # Assert
        assert allowed == {"think_tool"}

    def test_permission_denied_when_tool_not_in_whitelist(self):
        # Arrange
        tool = _research_complete_tool()
        config = _config(researcher_tool_whitelist=["think_tool"])
        # Act
        err = check_permission(
            "ResearchComplete", tool, AgentRole.RESEARCHER,
            allowed={"think_tool"}, origin_index=None, config=config,
        )
        # Assert
        assert err is not None
        assert err.error_type == ToolErrorType.permission_denied
        assert err.detail["role"] == "researcher"

    def test_permission_passes_when_whitelist_is_none(self):
        # Arrange
        tool = _research_complete_tool()
        config = _config()
        # Act
        err = check_permission(
            "ResearchComplete", tool, AgentRole.RESEARCHER,
            allowed=None, origin_index=None, config=config,
        )
        # Assert
        assert err is None

    def test_permission_denied_by_origin_blocklist(self):
        # Arrange -- a tool tagged MCP, blocked for researchers
        mcp_tool = _research_complete_tool(origin=ToolOrigin.MCP)
        config = _config(researcher_blocked_origins=["mcp"])
        # Act
        err = check_permission(
            "ResearchComplete", mcp_tool, AgentRole.RESEARCHER,
            allowed=None, origin_index={"ResearchComplete": ToolOrigin.MCP}, config=config,
        )
        # Assert
        assert err is not None
        assert err.error_type == ToolErrorType.permission_denied
        assert err.detail["origin"] == "mcp"


class TestPreBindFiltering:
    def test_no_filter_config_returns_all(self):
        # Arrange
        tools = [_research_complete_tool(), think_tool, tavily_search]
        # Act
        out = filter_tools_by_permission(tools, AgentRole.RESEARCHER, _config())
        # Assert -- backward compatible: everything passes
        assert len(out) == len(tools)

    def test_whitelist_filters_before_bind(self):
        # Arrange -- whitelist allows only tavily_search
        tools = [_research_complete_tool(), think_tool, tavily_search]
        config = _config(researcher_tool_whitelist=["tavily_search"])
        # Act
        out = filter_tools_by_permission(tools, AgentRole.RESEARCHER, config)
        names = {t.name if hasattr(t, "name") else t.get("name") for t in out}
        # Assert -- only the whitelisted tool remains; others never reach bind_tools
        assert names == {"tavily_search"}

    def test_origin_blocklist_filters_system_tools(self):
        # Arrange -- block system origin -> only search/MCP remain
        tools = [_research_complete_tool(), think_tool, tavily_search]
        config = _config(researcher_blocked_origins=["system"])
        # Act
        out = filter_tools_by_permission(tools, AgentRole.RESEARCHER, config)
        names = {t.name if hasattr(t, "name") else t.get("name") for t in out}
        # Assert
        assert "ResearchComplete" not in names
        assert "think_tool" not in names
        assert "tavily_search" in names

    def test_skill_tool_filtered_by_name(self):
        # Arrange
        skill_tool = _make_tool(
            _probe_search_fn,
            origin=ToolOrigin.SKILL,
            retryable=True,
            name="skill_search",
        )
        tools = [_research_complete_tool(), skill_tool]
        config = _config(researcher_tool_whitelist=["ResearchComplete"])
        # Act
        out = filter_tools_by_permission(tools, AgentRole.RESEARCHER, config)
        names = {t.name for t in out}
        # Assert -- the non-whitelisted Tool is filtered out by name
        assert names == {"ResearchComplete"}


# ---------------------------------------------------------------------------
# Parameter validation (incl. configured constraints)
# ---------------------------------------------------------------------------


class TestParamValidation:
    def test_missing_required_arg(self):
        # Arrange / Act
        err = validate_tool_args(tavily_search, {})
        # Assert
        assert err is not None
        assert err.error_type == ToolErrorType.validation_error
        assert err.detail["missing"] == ["queries"]

    def test_wrong_type_arg(self):
        # Arrange / Act
        err = validate_tool_args(tavily_search, {"queries": "not-a-list"})
        # Assert
        assert err is not None
        assert err.error_type == ToolErrorType.validation_error
        assert err.detail["argument"] == "queries"
        assert err.detail["expected_type"] == "array"

    def test_array_element_wrong_type(self):
        # Arrange / Act
        err = validate_tool_args(tavily_search, {"queries": ["ok", 123]})
        # Assert
        assert err is not None
        assert err.error_type == ToolErrorType.validation_error
        assert err.detail["index"] == 1
        assert err.detail["expected_type"] == "string"

    def test_valid_args_pass(self):
        # Arrange / Act / Assert
        assert validate_tool_args(tavily_search, {"queries": ["a", "b"]}) is None

    def test_empty_schema_passes(self):
        # Arrange -- ResearchComplete has no fields
        tool = _research_complete_tool()
        # Act / Assert
        assert validate_tool_args(tool, {}) is None

    def test_injected_args_not_required(self):
        """InjectedToolArg params (max_results/topic/config) must NOT be required."""
        # Arrange / Act / Assert
        assert validate_tool_args(tavily_search, {"queries": ["x"]}) is None

    def test_unknown_arg_lenient_by_default(self):
        # Arrange / Act / Assert
        assert validate_tool_args(tavily_search, {"queries": ["x"], "bogus": 1}) is None

    def test_conduct_research_missing_topic(self):
        # Arrange
        tool = adapt_langchain_tool(
            lc_tool(ConductResearch),
            origin=ToolOrigin.SYSTEM,
        )
        # Act
        err = validate_tool_args(tool, {})
        # Assert
        assert err is not None
        assert err.detail["missing"] == ["research_topic"]

    def test_config_maxItems_on_queries(self):
        # Arrange -- configured constraint: tavily queries maxItems=3
        config = _config(tool_param_constraints={"tavily_search": {"queries": {"maxItems": 3}}})
        # Act / Assert -- too many queries rejected
        err = validate_tool_args(tavily_search, {"queries": ["a", "b", "c", "d"]}, config)
        assert err is not None and err.detail["constraint"] == "maxItems"
        # within limit passes
        assert validate_tool_args(tavily_search, {"queries": ["a", "b"]}, config) is None

    def test_config_minItems_on_queries(self):
        # Arrange
        config = _config(tool_param_constraints={"tavily_search": {"queries": {"minItems": 1}}})
        # Act / Assert
        err = validate_tool_args(tavily_search, {"queries": []}, config)
        assert err is not None and err.detail["constraint"] == "minItems"

    def test_config_per_element_maxLength_deep_merged(self):
        # Arrange -- per-query maxLength=5, deep-merged with schema items type=string
        config = _config(
            tool_param_constraints={"tavily_search": {"queries": {"items": {"maxLength": 5}}}},
        )
        # Act / Assert -- a too-long query is rejected
        err = validate_tool_args(tavily_search, {"queries": ["short", "toolongstring"]}, config)
        assert err is not None and "maxLength" in str(err.detail)
        # short queries pass
        assert validate_tool_args(tavily_search, {"queries": ["ok", "ok2"]}, config) is None
        # element type still validated (deep merge preserved type=string)
        err2 = validate_tool_args(tavily_search, {"queries": ["ok", 123]}, config)
        assert err2 is not None and err2.detail["expected_type"] == "string"

    def test_no_config_means_no_extra_constraints(self):
        # Arrange / Act / Assert -- without config, 4 queries pass (schema has no maxItems)
        assert validate_tool_args(tavily_search, {"queries": ["a", "b", "c", "d"]}) is None


# ---------------------------------------------------------------------------
# Error classification
# ---------------------------------------------------------------------------


class TestClassifyRetryableError:
    def test_timeout_retryable(self):
        assert classify_retryable_error(asyncio.TimeoutError()) == (ToolErrorType.timeout, True)

    def test_429_rate_limited(self):
        assert classify_retryable_error(_client_response_error(429)) == (ToolErrorType.rate_limited, True)

    def test_503_service_unavailable(self):
        assert classify_retryable_error(_client_response_error(503)) == (ToolErrorType.service_unavailable, True)

    def test_500_service_unavailable(self):
        assert classify_retryable_error(_client_response_error(500)) == (ToolErrorType.service_unavailable, True)

    def test_404_not_retryable(self):
        assert classify_retryable_error(_client_response_error(404)) == (ToolErrorType.unknown, False)

    def test_408_rate_limited(self):
        assert classify_retryable_error(_client_response_error(408)) == (ToolErrorType.rate_limited, True)

    def test_clienterror_network_retryable(self):
        assert classify_retryable_error(aiohttp.ClientError("boom")) == (ToolErrorType.network_error, True)

    def test_oserror_network_retryable(self):
        assert classify_retryable_error(OSError("net down")) == (ToolErrorType.network_error, True)

    def test_generic_runtime_not_retryable(self):
        assert classify_retryable_error(RuntimeError("whatever")) == (ToolErrorType.unknown, False)

    def test_cause_chain_recursion(self):
        # Arrange -- wrap a 503 inside a RuntimeError via raise...from
        try:
            try:
                raise _client_response_error(503, "up")
            except Exception as inner:
                raise RuntimeError("wrapped") from inner
        except RuntimeError as outer:
            # Act / Assert
            assert classify_retryable_error(outer) == (ToolErrorType.service_unavailable, True)

    def test_toolexception_with_429_string(self):
        from langchain_core.tools import ToolException
        # Act / Assert
        assert classify_retryable_error(ToolException("HTTP 429 Too Many Requests")) == (
            ToolErrorType.rate_limited, True,
        )


# ---------------------------------------------------------------------------
# Retry with exponential backoff
# ---------------------------------------------------------------------------


class TestInvokeToolWithRetry:
    @pytest.mark.asyncio
    async def test_succeeds_first_try(self):
        # Arrange
        sleeper = AsyncMock()
        # Act
        result = await invoke_tool_with_retry(ok_tool, {}, None, max_retries=3, sleeper=sleeper)
        # Assert
        assert result == "ok-result"
        sleeper.assert_not_called()

    @pytest.mark.asyncio
    async def test_retries_then_succeeds_with_growing_delays(self):
        # Arrange -- a tool that 503s twice then succeeds
        state = {"n": 0}

        @lc_tool
        async def transient() -> str:
            """Fails twice then succeeds."""
            state["n"] += 1
            if state["n"] < 3:
                raise _client_response_error(503, "busy")
            return "recovered"

        delays: list[float] = []

        async def recorder(d: float) -> None:
            delays.append(d)

        # Act
        result = await invoke_tool_with_retry(transient, {}, None, max_retries=3, base_delay=1.0, max_delay=30.0, sleeper=recorder)
        # Assert
        assert result == "recovered"
        assert len(delays) == 2  # two retries before success
        assert 1.0 <= delays[0] < 2.0   # attempt 0: 1 + [0,1)
        assert 2.0 <= delays[1] < 3.0   # attempt 1: 2 + [0,1)
        assert delays[1] > delays[0]    # backoff grows (base dominates jitter)

    @pytest.mark.asyncio
    async def test_max_retries_exceeded(self):
        # Arrange
        sleeper = AsyncMock()
        # Act
        with pytest.raises(ToolExecutionFailure) as exc_info:
            await invoke_tool_with_retry(flaky_503, {}, None, max_retries=2, base_delay=1.0, sleeper=sleeper)
        # Assert
        failure = exc_info.value
        assert failure.error_type == ToolErrorType.max_retries_exceeded
        assert failure.attempts == 3  # initial attempt + 2 retries
        assert sleeper.await_count == 2

    @pytest.mark.asyncio
    async def test_non_retryable_no_retry(self):
        # Arrange
        sleeper = AsyncMock()
        # Act
        with pytest.raises(ToolExecutionFailure) as exc_info:
            await invoke_tool_with_retry(bad_request_404, {}, None, max_retries=3, sleeper=sleeper)
        # Assert
        assert exc_info.value.error_type == ToolErrorType.unknown
        assert exc_info.value.attempts == 1
        sleeper.assert_not_called()


# ---------------------------------------------------------------------------
# Governed execution entry point (retry scoped by origin)
# ---------------------------------------------------------------------------


class TestExecuteGovernedToolCall:
    @pytest.mark.asyncio
    async def test_tool_not_found(self):
        # Arrange
        tc = {"name": "nope", "args": {}, "id": "tc1"}
        # Act
        msg = await execute_governed_tool_call(tc, {}, AgentRole.RESEARCHER, _config())
        # Assert
        parsed = json.loads(msg.content)
        assert parsed["error_type"] == "tool_not_found"
        assert msg.name == "nope"

    @pytest.mark.asyncio
    async def test_provider_native_dict_must_be_adapted_before_execution(self):
        from open_deep_research.tools.base import build_tool_registry

        provider_dict = {"type": "web_search_preview", "name": "web_search"}
        with pytest.raises(TypeError):
            build_tool_registry([provider_dict])

    @pytest.mark.asyncio
    async def test_validation_error_short_circuits_before_invoke(self):
        """Validation must run before the tool is invoked."""
        # Arrange -- a spy tool with tavily's schema; missing required `queries`.
        spy = AsyncMock(return_value="should-not-happen")
        spy.name = "tavily_search"
        spy.metadata = {"tool_retryable": True}
        spy.input_schema = tavily_search.input_schema
        tc = {"name": "tavily_search", "args": {}, "id": "tc3"}
        # Act
        msg = await execute_governed_tool_call(
            tc, {"tavily_search": spy}, AgentRole.RESEARCHER, _config(),
            apply_retry=True, max_retries=2,
        )
        # Assert
        parsed = json.loads(msg.content)
        assert parsed["error_type"] == "validation_error"
        assert parsed["detail"]["missing"] == ["queries"]
        spy.assert_not_called()

    @pytest.mark.asyncio
    async def test_success_returns_toolmessage_with_content(self):
        # Arrange -- ok_tool is untagged (system, non-retryable); apply_retry=True
        # but get_tool_retryable(ok_tool) is False -> single execution, succeeds.
        tc = {"name": "ok_tool", "args": {}, "id": "tc4"}
        # Act
        msg = await execute_governed_tool_call(
            tc, {"ok_tool": ok_tool}, AgentRole.RESEARCHER, _config(),
            apply_retry=True, max_retries=2,
        )
        # Assert
        assert msg.content == "ok-result"
        assert msg.name == "ok_tool"

    @pytest.mark.asyncio
    async def test_per_tool_output_budget_is_applied_during_serialization(self):
        async def long_output() -> str:
            """Return content longer than the declared output budget."""
            return "abcdefghij"

        declared = adapt_langchain_tool(
            lc_tool(long_output),
            origin=ToolOrigin.SYSTEM,
            max_output_chars=5,
        )
        result = await execute_governed_tool_call(
            {"name": declared.name, "args": {}, "id": "tc-budget"},
            {declared.name: declared},
            AgentRole.RESEARCHER,
            _config(),
        )

        assert result.content == "abcde\n[truncated 5 chars]"

    @pytest.mark.asyncio
    async def test_search_tool_retryable_failure_returns_structured_error(self):
        # Arrange -- flaky_503 tagged as SEARCH + retryable
        search_503 = _make_tool(_flaky_503_fn, origin=ToolOrigin.SEARCH, retryable=True)
        sleeper = AsyncMock()
        tc = {"name": "flaky_503", "args": {}, "id": "tc5"}
        # Act
        msg = await execute_governed_tool_call(
            tc, {"flaky_503": search_503}, AgentRole.RESEARCHER, _config(),
            apply_retry=True, max_retries=1, base_delay=1.0, sleeper=sleeper,
        )
        # Assert -- retried (max_retries_exceeded) because SEARCH tools are retryable
        parsed = json.loads(msg.content)
        assert parsed["error_type"] == "max_retries_exceeded"
        assert parsed["attempts"] == 2
        assert parsed["tool_name"] == "flaky_503"
        assert sleeper.await_count == 1

    @pytest.mark.asyncio
    async def test_system_tool_not_retried_even_when_apply_retry_true(self):
        # Arrange -- flaky_503 tagged as SYSTEM + non-retryable
        system_503 = _make_tool(_flaky_503_fn, origin=ToolOrigin.SYSTEM, retryable=False)
        sleeper = AsyncMock()
        tc = {"name": "flaky_503", "args": {}, "id": "tc5b"}
        # Act
        msg = await execute_governed_tool_call(
            tc, {"flaky_503": system_503}, AgentRole.RESEARCHER, _config(),
            apply_retry=True, max_retries=3, sleeper=sleeper,
        )
        # Assert -- NOT retried: single attempt, service_unavailable (not max_retries_exceeded)
        parsed = json.loads(msg.content)
        assert parsed["error_type"] == "service_unavailable"
        sleeper.assert_not_called()

    @pytest.mark.asyncio
    async def test_mcp_auth_satisfied_tool_is_permitted(self):
        """An auth-required MCP tool loaded with a token (mcp_auth_satisfied=True)
        is permitted to execute -- not wrongly denied (P1#3)."""
        # Arrange
        mcp_tool = _research_complete_tool(
            origin=ToolOrigin.MCP,
            auth_satisfied=True,
        )
        config = _config()
        config["configurable"]["mcp_config"] = {"url": "http://x", "tools": ["ResearchComplete"], "auth_required": True}
        tc = {"name": "ResearchComplete", "args": {}, "id": "tc6"}
        # Act
        msg = await execute_governed_tool_call(
            tc, {"ResearchComplete": mcp_tool}, AgentRole.RESEARCHER, config,
            origin_index={"ResearchComplete": ToolOrigin.MCP}, apply_retry=False,
        )
        # Assert -- permitted (executes; ResearchComplete returns empty content, not a denial)
        assert not _is_denied(msg)

    @pytest.mark.asyncio
    async def test_mcp_auth_required_without_marker_denied(self):
        # Arrange -- MCP tool without the auth_satisfied marker + auth_required
        mcp_tool = _research_complete_tool(origin=ToolOrigin.MCP)
        config = _config()
        config["configurable"]["mcp_config"] = {"url": "http://x", "tools": ["ResearchComplete"], "auth_required": True}
        tc = {"name": "ResearchComplete", "args": {}, "id": "tc7"}
        # Act
        msg = await execute_governed_tool_call(
            tc, {"ResearchComplete": mcp_tool}, AgentRole.RESEARCHER, config,
            origin_index={"ResearchComplete": ToolOrigin.MCP}, apply_retry=False,
        )
        # Assert
        parsed = json.loads(msg.content)
        assert parsed["error_type"] == "permission_denied"
        assert parsed["detail"]["auth_required"] is True

    @pytest.mark.asyncio
    async def test_user_role_tool_blacklist_denied(self):
        # Arrange -- user has role 'admin', which is blacklisted from tavily_search
        config = _config(
            langgraph_auth_user={"identity": "u1", "permissions": ["admin"]},
            role_tool_blacklist={"admin": ["tavily_search"]},
        )
        tc = {"name": "tavily_search", "args": {"queries": ["x"]}, "id": "tc8"}
        # Act
        msg = await execute_governed_tool_call(
            tc, {"tavily_search": tavily_search}, AgentRole.RESEARCHER, config, apply_retry=False,
        )
        # Assert
        parsed = json.loads(msg.content)
        assert parsed["error_type"] == "permission_denied"
        assert parsed["detail"]["user_role"] == "admin"

    @pytest.mark.asyncio
    async def test_user_role_origin_blacklist_denied(self):
        # Arrange -- admin blocked from 'search' origin
        config = _config(
            langgraph_auth_user={"identity": "u1", "permissions": ["admin"]},
            role_blocked_origins={"admin": ["search"]},
        )
        tc = {"name": "tavily_search", "args": {"queries": ["x"]}, "id": "tc9"}
        # Act
        msg = await execute_governed_tool_call(
            tc, {"tavily_search": tavily_search}, AgentRole.RESEARCHER, config, apply_retry=False,
        )
        # Assert
        parsed = json.loads(msg.content)
        assert parsed["error_type"] == "permission_denied"
        assert parsed["detail"]["scope"] == "origin"

    @pytest.mark.asyncio
    async def test_anonymous_user_skips_role_layer(self):
        # Arrange -- no user permissions; role blacklist configured but does not apply
        config = _config(role_tool_blacklist={"admin": ["think_tool"]})
        assert get_user_permissions(config) == []
        tc = {"name": "think_tool", "args": {"reflection": "p"}, "id": "tc10"}
        # Act
        msg = await execute_governed_tool_call(
            tc, {"think_tool": think_tool}, AgentRole.RESEARCHER, config, apply_retry=False,
        )
        # Assert -- allowed (anonymous skips role layer)
        assert "Reflection recorded" in msg.content


# ---------------------------------------------------------------------------
# Supervisor gate
# ---------------------------------------------------------------------------


class TestSupervisorGovernedExecution:
    @pytest.mark.asyncio
    async def test_executor_blocks_unknown_tool(self):
        message = await execute_governed_tool_call(
            {"name": "ghost", "args": {}, "id": "g1"},
            {"think_tool": think_tool},
            AgentRole.SUPERVISOR,
            _config(),
        )
        assert json.loads(message.content)["error_type"] == "tool_not_found"

    def test_registry_rejects_unadapted_langchain_tool(self):
        from open_deep_research.tools.base import build_tool_registry

        with pytest.raises(TypeError):
            build_tool_registry([lc_tool(ConductResearch)])

    @pytest.mark.asyncio
    async def test_executor_blocks_whitelist_violation(self):
        tool = _make_tool(
            _probe_search_fn,
            origin=ToolOrigin.SYSTEM,
            retryable=False,
            name="ConductResearch",
        )
        message = await execute_governed_tool_call(
            {
                "name": "ConductResearch",
                "args": {"query": "x"},
                "id": "g3",
            },
            {"ConductResearch": tool},
            AgentRole.SUPERVISOR,
            _config(supervisor_tool_whitelist=["think_tool"]),
            allowed_tools={"think_tool"},
        )
        assert json.loads(message.content)["error_type"] == "permission_denied"

    @pytest.mark.asyncio
    async def test_executor_blocks_invalid_args(self):
        tool = _make_tool(
            _probe_search_fn,
            origin=ToolOrigin.SYSTEM,
            retryable=False,
            name="ConductResearch",
        )
        message = await execute_governed_tool_call(
            {"name": "ConductResearch", "args": {}, "id": "g4"},
            {"ConductResearch": tool},
            AgentRole.SUPERVISOR,
            _config(),
        )
        payload = json.loads(message.content)
        assert payload["error_type"] == "validation_error"
        assert payload["detail"]["missing"] == ["query"]

    @pytest.mark.asyncio
    async def test_executor_passes_valid_call(self):
        tool = _make_tool(
            _probe_search_fn,
            origin=ToolOrigin.SYSTEM,
            retryable=False,
            name="ConductResearch",
        )
        message = await execute_governed_tool_call(
            {
                "name": "ConductResearch",
                "args": {"query": "ai safety"},
                "id": "g5",
            },
            {"ConductResearch": tool},
            AgentRole.SUPERVISOR,
            _config(),
        )
        assert message.content == "ai safety"


# ---------------------------------------------------------------------------
# JWT user role extraction (security/auth.py)
# ---------------------------------------------------------------------------



def _load_auth_module():
    """Load src/security/auth.py by path (it is not an installed package)."""
    src_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "src")
    spec = importlib.util.spec_from_file_location(
        "odr_security_auth", os.path.join(src_dir, "security", "auth.py"),
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestJwtRoleExtraction:
    def test_extract_single_role(self):
        # Arrange / Act / Assert
        auth = _load_auth_module()
        assert auth._extract_roles(SimpleNamespace(app_metadata={"role": "admin"})) == ["admin"]

    def test_extract_roles_list(self):
        auth = _load_auth_module()
        assert auth._extract_roles(SimpleNamespace(app_metadata={"roles": ["researcher", "viewer"]})) == ["researcher", "viewer"]

    def test_extract_dedupes(self):
        auth = _load_auth_module()
        # role + roles both contain 'admin' -> deduplicated, order preserved
        assert auth._extract_roles(SimpleNamespace(app_metadata={"role": "admin", "roles": ["admin", "researcher"]})) == ["admin", "researcher"]

    def test_extract_none_metadata(self):
        auth = _load_auth_module()
        assert auth._extract_roles(SimpleNamespace(app_metadata=None)) == []

    def test_extract_empty_metadata(self):
        auth = _load_auth_module()
        assert auth._extract_roles(SimpleNamespace(app_metadata={})) == []

    def test_get_user_permissions_reads_langgraph_auth_user(self):
        # Arrange -- server-injected user dict
        config = _config(langgraph_auth_user={"identity": "u1", "permissions": ["admin", "researcher"]})
        # Act / Assert
        assert get_user_permissions(config) == ["admin", "researcher"]

    def test_get_user_permissions_fallback_user_permissions(self):
        # Arrange -- plain fallback key (tests / non-server)
        config = _config(user_permissions=["viewer"])
        # Act / Assert
        assert get_user_permissions(config) == ["viewer"]

    def test_get_user_permissions_empty_when_absent(self):
        # Act / Assert
        assert get_user_permissions(_config()) == []


# ---------------------------------------------------------------------------
# Integration: researcher_tools with a retryable failure (retry scoped to SEARCH)
# ---------------------------------------------------------------------------


class TestResearcherToolsIntegration:
    @pytest.mark.asyncio
    async def test_retryable_search_failure_yields_structured_error(self):
        """A SEARCH researcher tool raising HTTP 503 returns max_retries_exceeded."""
        # Arrange -- a SEARCH+retryable tool that always 503s
        search_503 = _make_tool(
            _flaky_503_fn,
            origin=ToolOrigin.SEARCH,
            retryable=True,
            name="flaky_503",
        )
        from open_deep_research.agents.deep_researcher import researcher_tools
        ai_msg = AIMessage(content="", tool_calls=[{"name": "flaky_503", "args": {}, "id": "it1"}])
        state = {
            "researcher_messages": [ai_msg], "tool_call_iterations": 1,
            "research_topic": "test topic", "memory_context": None,
        }
        config = _config(max_tool_retries=1, tool_retry_base_delay=1.0, tool_retry_max_delay=30.0, max_react_tool_calls=10)

        import asyncio as _asyncio

        import open_deep_research.agents.deep_researcher as mod

        async def fake_get_all_tools(_config):
            return [search_503]

        fake_sleep_calls: list[float] = []

        async def patched_sleep(d: float) -> None:
            fake_sleep_calls.append(d)

        original_get_all_tools = mod.get_all_tools
        orig_sleep = _asyncio.sleep
        mod.get_all_tools = fake_get_all_tools
        _asyncio.sleep = patched_sleep
        try:
            # Act
            result = await researcher_tools(state, config)
        finally:
            mod.get_all_tools = original_get_all_tools
            _asyncio.sleep = orig_sleep

        # Assert
        messages = result.update["researcher_messages"]
        assert len(messages) == 1
        parsed = json.loads(messages[0].content)
        assert parsed["error_type"] == "max_retries_exceeded"
        assert parsed["tool_name"] == "flaky_503"
        assert parsed["attempts"] == 2  # initial + 1 retry
        assert len(fake_sleep_calls) == 1  # one backoff before the final attempt


# ---------------------------------------------------------------------------
# Integration: supervisor_tools -- ResearchComplete whitelist bypass + id filtering
# ---------------------------------------------------------------------------


class TestSupervisorToolsIntegration:
    @pytest.mark.asyncio
    async def test_research_complete_bypasses_whitelist_is_blocked(self):
        """A ResearchComplete call excluded by the supervisor whitelist must NOT
        end the research phase -- it is denied and the loop continues (P1#1)."""
        # Arrange
        from open_deep_research.agents.deep_researcher import supervisor_tools
        ai_msg = AIMessage(content="", tool_calls=[{"name": "ResearchComplete", "args": {}, "id": "rc-1"}])
        state = {
            "supervisor_messages": [ai_msg], "research_iterations": 1, "research_brief": "b",
            "enable_async_research": False, "memory_context": None, "notes": [], "raw_notes": [],
        }
        config = _config(
            max_researcher_iterations=100,
            supervisor_tool_whitelist=["think_tool", "ConductResearch"],  # ResearchComplete excluded
        )
        # Act
        result = await supervisor_tools(state, config)
        # Assert -- does NOT end; continues to supervisor with the denial
        assert result.goto != "__end__"
        msgs = result.update.get("supervisor_messages", [])
        assert any(m.name == "ResearchComplete" and json.loads(m.content)["error_type"] == "permission_denied" for m in msgs)

    @pytest.mark.asyncio
    async def test_research_complete_allowed_ends_research(self):
        # Arrange
        from open_deep_research.agents.deep_researcher import supervisor_tools
        ai_msg = AIMessage(content="", tool_calls=[{"name": "ResearchComplete", "args": {}, "id": "rc-2"}])
        state = {
            "supervisor_messages": [ai_msg], "research_iterations": 1, "research_brief": "b",
            "enable_async_research": False, "memory_context": None, "notes": [], "raw_notes": [],
        }
        config = _config(
            max_researcher_iterations=100,
            supervisor_tool_whitelist=["think_tool", "ConductResearch", "ResearchComplete"],
        )
        # Act
        result = await supervisor_tools(state, config)
        # Assert -- ends research
        assert result.goto == "__end__"

    @pytest.mark.asyncio
    async def test_id_filtering_preserves_same_name_valid_call(self):
        """An invalid same-name call must not drag a valid same-name call out of
        the active dispatch set (filtered by tool_call id, not name)."""
        # Arrange
        import open_deep_research.agents.deep_researcher as mod
        from open_deep_research.agents.deep_researcher import supervisor_tools

        async def fake_ainvoke(*a, **k):
            return {"compressed_research": "GOOD-RAN", "raw_notes": []}

        ai_msg = AIMessage(content="", tool_calls=[
            {"name": "ConductResearch", "args": {}, "id": "cr-bad"},  # missing research_topic
            {"name": "ConductResearch", "args": {"research_topic": "ai safety"}, "id": "cr-good"},
        ])
        state = {
            "supervisor_messages": [ai_msg], "research_iterations": 1, "research_brief": "b",
            "enable_async_research": False, "memory_context": None, "notes": [], "raw_notes": [],
        }
        config = _config(max_researcher_iterations=100, max_concurrent_research_units=5)
        orig_ainvoke = mod.researcher_runtime.ainvoke
        mod.researcher_runtime.ainvoke = fake_ainvoke
        try:
            # Act
            result = await supervisor_tools(state, config)
        finally:
            mod.researcher_runtime.ainvoke = orig_ainvoke
        # Assert
        msgs = result.update.get("supervisor_messages", [])
        bad = [m for m in msgs if m.tool_call_id == "cr-bad"]
        good = [m for m in msgs if m.tool_call_id == "cr-good"]
        assert len(bad) == 1 and json.loads(bad[0].content)["error_type"] == "validation_error"
        assert len(good) == 1
        assert json.loads(good[0].content)["compressed_research"] == "GOOD-RAN"

    def test_build_supervisor_tools_includes_approve_domain(self):
        from open_deep_research.agents.deep_researcher import build_supervisor_tools

        async_tools = build_supervisor_tools({"enable_async_research": True})
        names = [t.name for t in async_tools]
        assert "ApproveResearchDomain" in names

    @pytest.mark.asyncio
    async def test_supervisor_dispatches_approve_research_domain(self):
        """In async mode an ApproveResearchDomain tool call is routed to its handler."""
        from open_deep_research.agents.deep_researcher import supervisor_tools
        from open_deep_research.tools.supervisor.approve_research_domain import (
            definition as approve_definition,
        )

        called: dict[str, Any] = {}

        async def fake_approve(tool_call, config, registry, event_writer=None, state_store=None):
            called["tool_call_id"] = tool_call["id"]
            called["domain"] = tool_call["args"]["domain"]
            from langchain_core.messages import ToolMessage

            return ToolMessage(
                content="approved", name="ApproveResearchDomain", tool_call_id=tool_call["id"]
            )

        orig = approve_definition.handle_approve_research_domain
        approve_definition.handle_approve_research_domain = fake_approve
        try:
            ai_msg = AIMessage(content="", tool_calls=[{
                "name": "ApproveResearchDomain",
                "args": {"task_id": "t1", "domain": "x.example", "allow": True},
                "id": "ard-1",
            }])
            state = {
                "supervisor_messages": [ai_msg], "research_iterations": 1,
                "research_brief": "b", "enable_async_research": True,
                "memory_context": None, "notes": [], "raw_notes": [],
            }
            config = _config(max_researcher_iterations=100)
            result = await supervisor_tools(state, config)
        finally:
            approve_definition.handle_approve_research_domain = orig

        assert called.get("tool_call_id") == "ard-1"
        assert called.get("domain") == "x.example"
        msgs = result.update.get("supervisor_messages", [])
        assert any(m.name == "ApproveResearchDomain" and m.content == "approved" for m in msgs)


# ---------------------------------------------------------------------------
# Egress domain allowlist (check_egress_domain inside execute_governed_tool_call)
# ---------------------------------------------------------------------------


async def _ok_fetch_fn(url: str) -> str:
    """A fetch_webpage stand-in that returns fixed content (no real network)."""
    return f"fetched:{url}"


@pytest.fixture
def egress_env(monkeypatch):
    """Ensure in-process mode (no SANDBOX_NETWORK_MODE) and fresh registries."""
    monkeypatch.delenv("SANDBOX_NETWORK_MODE", raising=False)
    # Reset the registry singleton so prior tests' tasks don't leak in.
    import open_deep_research.tasks.registry as registry_mod
    from open_deep_research.tasks.domain_approvals import (
        get_domain_approval_registry,
        reset_domain_approval_registry,
    )
    from open_deep_research.tasks.registry import get_task_registry

    registry_mod._registry = None
    reset_domain_approval_registry()
    yield get_task_registry(), get_domain_approval_registry()
    reset_domain_approval_registry()
    registry_mod._registry = None


def _fetch_tool() -> Any:
    """Build a retryable fetch_webpage Tool used by egress scenarios."""
    t = lc_tool(_ok_fetch_fn)
    t.name = "fetch_webpage"
    return adapt_langchain_tool(
        t,
        origin=ToolOrigin.SYSTEM,
        retryable=True,
        egress_urls=lambda args: [args["url"]],
    )


def _egress_config(**configurable: Any) -> RunnableConfig:
    base = {
        "sandbox_network_mode": "allowlist-domain",
        "sandbox_allowed_domains": ["api.tavily.com"],
    }
    base.update(configurable)
    return RunnableConfig(
        configurable=base, metadata={"run_id": "egress-run"}
    )


def _seed_task(registry, run_id="egress-run") -> Any:
    """Create a RUNNING task in the registry and return it."""
    rec = registry.create("topic", run_id=run_id)
    registry.update_status(rec.task_id, TaskStatus.RUNNING)
    return rec


class TestEgressAllowlist:
    def test_allowed_domain_passes(self, egress_env):
        registry, _ = egress_env
        tool = _fetch_tool()
        cfg = _egress_config(sandbox_allowed_domains=["example.com"])
        rec = _seed_task(registry)
        cfg["metadata"]["task_id"] = rec.task_id
        msg = asyncio.run(
            execute_governed_tool_call(
                {"name": "fetch_webpage", "id": "tc1", "args": {"url": "https://example.com/x"}},
                {"fetch_webpage": tool},
                AgentRole.RESEARCHER,
                cfg,
                apply_retry=False,
            )
        )
        assert msg.content == "fetched:https://example.com/x"

    def test_no_network_mode_denies_egress(self, egress_env):
        registry, _ = egress_env
        tool = _fetch_tool()
        cfg = _egress_config(sandbox_network_mode="no-network")
        rec = _seed_task(registry)
        cfg["metadata"]["task_id"] = rec.task_id
        msg = asyncio.run(
            execute_governed_tool_call(
                {"name": "fetch_webpage", "id": "tc1", "args": {"url": "https://untrusted.example/x"}},
                {"fetch_webpage": tool},
                AgentRole.RESEARCHER,
                cfg,
                apply_retry=False,
            )
        )
        payload = json.loads(msg.content)
        assert payload["error_type"] == "egress_domain_denied"
        assert payload["detail"]["network_mode"] == "no-network"

    def test_open_network_mode_skips_egress(self, egress_env):
        registry, _ = egress_env
        tool = _fetch_tool()
        cfg = _egress_config(sandbox_network_mode="open-network")
        rec = _seed_task(registry)
        cfg["metadata"]["task_id"] = rec.task_id
        msg = asyncio.run(
            execute_governed_tool_call(
                {"name": "fetch_webpage", "id": "tc1", "args": {"url": "https://untrusted.example/x"}},
                {"fetch_webpage": tool},
                AgentRole.RESEARCHER,
                cfg,
                apply_retry=False,
            )
        )
        assert msg.content == "fetched:https://untrusted.example/x"

    @pytest.mark.asyncio
    async def test_unknown_domain_blocks_and_waits_then_resumes(self, egress_env):
        registry, approvals = egress_env
        tool = _fetch_tool()
        cfg = _egress_config()  # allowlist only api.tavily.com
        rec = _seed_task(registry)
        cfg["metadata"]["task_id"] = rec.task_id

        async def approve_after_pause():
            # Wait until the task flips to WAITING, then approve the domain.
            for _ in range(50):
                if rec.status == TaskStatus.WAITING_FOR_CONFIRMATION:
                    break
                await asyncio.sleep(0.005)
            approvals.record_decision("egress-run", "untrusted.example", True)

        msg, _ = await asyncio.gather(
            execute_governed_tool_call(
                {"name": "fetch_webpage", "id": "tc1", "args": {"url": "https://untrusted.example/x"}},
                {"fetch_webpage": tool},
                AgentRole.RESEARCHER,
                cfg,
                apply_retry=False,
            ),
            approve_after_pause(),
        )
        assert msg.content == "fetched:https://untrusted.example/x"
        assert rec.status == TaskStatus.RUNNING
        # decision cached for the run
        assert approvals.is_allowed("egress-run", "untrusted.example") is True

    @pytest.mark.asyncio
    async def test_unknown_domain_denied(self, egress_env):
        registry, approvals = egress_env
        tool = _fetch_tool()
        cfg = _egress_config()
        rec = _seed_task(registry)
        cfg["metadata"]["task_id"] = rec.task_id

        async def deny_after_pause():
            for _ in range(50):
                if rec.status == TaskStatus.WAITING_FOR_CONFIRMATION:
                    break
                await asyncio.sleep(0.005)
            approvals.record_decision("egress-run", "untrusted.example", False)

        msg, _ = await asyncio.gather(
            execute_governed_tool_call(
                {"name": "fetch_webpage", "id": "tc1", "args": {"url": "https://untrusted.example/x"}},
                {"fetch_webpage": tool},
                AgentRole.RESEARCHER,
                cfg,
                apply_retry=False,
            ),
            deny_after_pause(),
        )
        payload = json.loads(msg.content)
        assert payload["error_type"] == "egress_domain_denied"
        assert approvals.is_allowed("egress-run", "untrusted.example") is False

    @pytest.mark.asyncio
    async def test_decision_scoped_to_run(self, egress_env):
        registry, approvals = egress_env
        approvals.record_decision("run-A", "shared.example", True)
        assert approvals.is_allowed("run-A", "shared.example") is True
        assert approvals.is_allowed("run-B", "shared.example") is None

    def test_search_tool_skipped_by_egress(self, egress_env):
        # tavily_search is SEARCH origin with no url arg -> _egress_host_for_tool
        # returns None -> no egress check, tool would proceed (we only assert no
        # egress error is produced for a SEARCH tool targeting an unknown host).
        registry, _ = egress_env
        rec = _seed_task(registry)
        cfg = _egress_config()
        cfg["metadata"]["task_id"] = rec.task_id
        # Use check_egress_domain directly with a SEARCH-origin tool.
        from open_deep_research.tools.governance import check_egress_domain

        result = asyncio.run(
            check_egress_domain(
                {"id": "tc"},
                tavily_search,
                {"queries": ["x"]},
                cfg,
            )
        )
        assert result is None  # SEARCH tool -> no egress interception

    def test_pending_error_in_container_mode(self, egress_env, monkeypatch):
        # Simulate the Docker worker: SANDBOX_NETWORK_MODE is set -> undecided
        # domain returns egress_domain_pending instead of blocking.
        monkeypatch.setenv("SANDBOX_NETWORK_MODE", "allowlist-domain")
        tool = _fetch_tool()
        cfg = _egress_config()
        msg = asyncio.run(
            execute_governed_tool_call(
                {"name": "fetch_webpage", "id": "tc1", "args": {"url": "https://untrusted.example/x"}},
                {"fetch_webpage": tool},
                AgentRole.RESEARCHER,
                cfg,
                apply_retry=False,
            )
        )
        payload = json.loads(msg.content)
        assert payload["error_type"] == "egress_domain_pending"
        assert payload["detail"]["pending"] is True

    def test_allow_search_only_without_task_reports_mode_restriction(self, egress_env):
        # No seeded task -> synchronous researcher has no approval channel; the
        # denial must name the mode restriction instead of implying that a task
        # context could have provided approval.
        tool = _fetch_tool()
        cfg = _egress_config(sandbox_network_mode="allow-search-only")
        msg = asyncio.run(
            execute_governed_tool_call(
                {"name": "fetch_webpage", "id": "tc1", "args": {"url": "https://untrusted.example/x"}},
                {"fetch_webpage": tool},
                AgentRole.RESEARCHER,
                cfg,
                apply_retry=False,
            )
        )
        payload = json.loads(msg.content)
        assert payload["error_type"] == "egress_domain_denied"
        assert payload["detail"]["network_mode"] == "allow-search-only"
        assert "allow-search-only" in payload["message"]
        assert "no active task context" not in payload["message"]

    def test_allowlist_domain_without_task_keeps_approval_hint(self, egress_env):
        # In allowlist-domain mode supervisor approval is the intended path, so
        # the missing-task-context hint stays accurate there.
        tool = _fetch_tool()
        cfg = _egress_config()  # default sandbox_network_mode=allowlist-domain
        msg = asyncio.run(
            execute_governed_tool_call(
                {"name": "fetch_webpage", "id": "tc1", "args": {"url": "https://untrusted.example/x"}},
                {"fetch_webpage": tool},
                AgentRole.RESEARCHER,
                cfg,
                apply_retry=False,
            )
        )
        payload = json.loads(msg.content)
        assert payload["error_type"] == "egress_domain_denied"
        assert payload["detail"]["network_mode"] == "allowlist-domain"
        assert "no active task context" in payload["message"]


