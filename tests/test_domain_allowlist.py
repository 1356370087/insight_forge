"""Tests for the enforced domain-allowlist + new-domain confirmation flow."""

import asyncio

import pytest

from open_deep_research.tasks.domain_approvals import (
    DomainApprovalRegistry,
    get_domain_approval_registry,
    reset_domain_approval_registry,
)


@pytest.fixture(autouse=True)
def reset_approvals():
    reset_domain_approval_registry()
    yield
    reset_domain_approval_registry()


class TestDomainApprovalRegistry:
    def test_is_allowed_undecided_returns_none(self):
        reg = DomainApprovalRegistry()
        assert reg.is_allowed("run-1", "example.com") is None

    @pytest.mark.asyncio
    async def test_record_decision_allow_caches_and_resolves_future(self):
        reg = DomainApprovalRegistry()
        req = reg.request_decision("run-1", "Example.COM", "fetch_webpage")
        assert req.future is None  # not bound until awaited
        resolved = reg.record_decision("run-1", "example.com", True)
        assert resolved is req
        # future is created lazily inside wait(); pre-resolved -> already-done
        result = await req.wait()
        assert result is True
        assert reg.is_allowed("run-1", "example.com") is True
        # domain normalized to lowercase
        assert reg.is_allowed("run-1", "EXAMPLE.com") is True

    @pytest.mark.asyncio
    async def test_record_decision_denied(self):
        reg = DomainApprovalRegistry()
        req = reg.request_decision("run-1", "bad.example", "fetch_webpage")
        reg.record_decision("run-1", "bad.example", False)
        assert await req.wait() is False
        assert reg.is_allowed("run-1", "bad.example") is False

    @pytest.mark.asyncio
    async def test_clear_run_cancels_pending_future(self):
        reg = DomainApprovalRegistry()
        req = reg.request_decision("run-1", "pending.example", "fetch_webpage")
        # bind the future before clearing
        loop = asyncio.get_running_loop()
        req.future = loop.create_future()
        reg.clear_run("run-1")
        assert req.future.cancelled()
        assert reg.get_pending("run-1") == []

    def test_request_decision_is_idempotent(self):
        reg = DomainApprovalRegistry()
        r1 = reg.request_decision("run-1", "x.example", "fetch_webpage")
        r2 = reg.request_decision("run-1", "x.example", "fetch_webpage")
        assert r1 is r2

    def test_pre_approval_without_pending_request(self):
        reg = DomainApprovalRegistry()
        # record a decision with no pending request -> caches, returns None
        req = reg.record_decision("run-1", "pre.example", True)
        assert req is None
        assert reg.is_allowed("run-1", "pre.example") is True

    def test_decision_scoped_to_run(self):
        reg = DomainApprovalRegistry()
        reg.record_decision("run-A", "shared.example", True)
        assert reg.is_allowed("run-A", "shared.example") is True
        assert reg.is_allowed("run-B", "shared.example") is None

    def test_clear_run_removes_decisions(self):
        reg = DomainApprovalRegistry()
        reg.record_decision("run-1", "a.example", True)
        reg.record_decision("run-1", "b.example", False)
        reg.clear_run("run-1")
        assert reg.is_allowed("run-1", "a.example") is None
        assert reg.is_allowed("run-1", "b.example") is None

    def test_singleton(self):
        assert get_domain_approval_registry() is get_domain_approval_registry()

    @pytest.mark.asyncio
    async def test_block_until_decision_resolved(self):
        reg = get_domain_approval_registry()

        async def resolver():
            await asyncio.sleep(0)
            reg.record_decision("run-1", "block.example", True)

        async def blocked():
            req = reg.request_decision("run-1", "block.example", "fetch_webpage")
            return await req.wait()

        result, _ = await asyncio.gather(blocked(), resolver())
        assert result is True


# ---------------------------------------------------------------------------
# fetch_webpage tool (network mocked; egress enforced separately in governance tests)
# ---------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, status: int, text: str) -> None:
        self.status = status
        self._text = text

    async def text(self, errors: str = "strict") -> str:
        return self._text

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _FakeSession:
    def __init__(self, *args, **kwargs) -> None:
        self.status = 200
        self.text = "hello world page content"

    def get(self, url, **kwargs):
        return _FakeResponse(self.status, self.text)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class TestFetchWebpageTool:
    @pytest.mark.asyncio
    async def test_fetch_webpage_returns_content_no_summarize(self, monkeypatch):
        from open_deep_research.tools import utils
        from open_deep_research.tools.base import ToolContext

        monkeypatch.setattr(utils.aiohttp, "ClientSession", _FakeSession)
        tool = utils.fetch_webpage
        config = {"configurable": {"search_api": "none"}, "metadata": {"run_id": "test"}}
        result = (
            await tool.call(
                tool.input_schema.model_validate(
                    {"url": "https://example.com/page", "summarize": False}
                ),
                ToolContext(config=config, role="researcher", tool_call_id="fetch-1"),
            )
        ).output
        assert "hello world page content" in result
        assert "example.com/page" in result

    @pytest.mark.asyncio
    async def test_fetch_webpage_raises_on_http_error(self, monkeypatch):
        from langchain_core.tools import ToolException

        from open_deep_research.tools import utils
        from open_deep_research.tools.base import ToolContext

        class _ErrSession(_FakeSession):
            def __init__(self, *a, **k):
                super().__init__(*a, **k)
                self.status = 404

        monkeypatch.setattr(utils.aiohttp, "ClientSession", _ErrSession)
        with pytest.raises(ToolException):
            tool = utils.fetch_webpage
            config = {"configurable": {"search_api": "none"}, "metadata": {"run_id": "test"}}
            await tool.call(
                tool.input_schema.model_validate(
                    {"url": "https://example.com/missing", "summarize": False}
                ),
                ToolContext(config=config, role="researcher", tool_call_id="fetch-2"),
            )


# ---------------------------------------------------------------------------
# End-to-end: run_task_with_control pauses and resumes on a domain decision
# ---------------------------------------------------------------------------


class TestRunTaskPauseAndResume:
    @pytest.mark.asyncio
    async def test_clear_run_on_terminal(self, tmp_path):
        import open_deep_research.tasks.registry as registry_mod
        from open_deep_research.tasks.executor import run_task_with_control
        from open_deep_research.tasks.registry import (
            TaskStatus,
            get_task_registry,
        )

        registry_mod._registry = None
        registry = get_task_registry()
        approvals = get_domain_approval_registry()
        approvals.record_decision("run-end", "cached.example", True)
        assert approvals.is_allowed("run-end", "cached.example") is True

        record = registry.create("topic", run_id="run-end")
        cfg = {
            "configurable": {
                "runs_dir": str(tmp_path),
                "search_api": "none",
                "task_timeout_seconds": 5,
            },
            "metadata": {"run_id": "run-end"},
        }

        async def fake_execute(state, config):
            return {"compressed_research": "done", "raw_notes": [], "metrics": {"sources_read": 1}}

        await run_task_with_control(
            record, cfg, registry, fake_execute,
            runs_dir=str(tmp_path), run_id="run-end", event_log_enabled=False,
        )
        assert record.status == TaskStatus.COMPLETED
        # terminal branch cleared the run's approvals
        assert approvals.is_allowed("run-end", "cached.example") is None
        registry_mod._registry = None

    @pytest.mark.asyncio
    async def test_pause_and_resume_through_executor(self, tmp_path):
        """End-to-end in-process: a researcher tool hitting an unapproved domain
        pauses the task (WAITING_FOR_CONFIRMATION); ApproveResearchDomain resumes
        it and the task completes."""
        from langchain_core.tools import tool as lc_tool

        import open_deep_research.tasks.registry as registry_mod
        from open_deep_research.tasks.async_tools import handle_approve_research_domain
        from open_deep_research.tasks.executor import run_task_with_control
        from open_deep_research.tasks.registry import (
            TaskStatus,
            get_task_registry,
        )
        from open_deep_research.tools.adapters import adapt_langchain_tool
        from open_deep_research.tools.base import ToolOrigin
        from open_deep_research.tools.governance import (
            AgentRole,
            execute_governed_tool_call,
        )

        registry_mod._registry = None
        registry = get_task_registry()
        approvals = get_domain_approval_registry()

        async def _fetch_fn(url: str) -> str:
            """A fetch_webpage-named tool whose body only runs if egress allows."""
            return f"fetched:{url}"

        langchain_fetch = lc_tool(_fetch_fn)
        langchain_fetch.name = "fetch_webpage"
        fetch_tool = adapt_langchain_tool(
            langchain_fetch,
            origin=ToolOrigin.SYSTEM,
            retryable=True,
        )

        record = registry.create("topic", run_id="run-e2e")
        cfg = {
            "configurable": {
                "runs_dir": str(tmp_path),
                "search_api": "none",
                "task_timeout_seconds": 10,
                "sandbox_network_mode": "allowlist-domain",
                "sandbox_allowed_domains": ["api.tavily.com"],
            },
            "metadata": {"run_id": "run-e2e"},
        }

        async def fake_execute(state, config):
            # The researcher "calls" fetch_webpage on an unapproved domain.
            outcome = await execute_governed_tool_call(
                {"name": "fetch_webpage", "id": "fw-1",
                 "args": {"url": "https://untrusted.example/page"}},
                {"fetch_webpage": fetch_tool},
                AgentRole.RESEARCHER,
                config,
                apply_retry=False,
            )
            return {
                "compressed_research": outcome.message.content,
                "raw_notes": [],
                "metrics": {"sources_read": 1},
            }

        async def approver():
            # Wait for the task to enter WAITING, then approve.
            for _ in range(200):
                if record.status == TaskStatus.WAITING_FOR_CONFIRMATION:
                    break
                await asyncio.sleep(0.005)
            assert record.status == TaskStatus.WAITING_FOR_CONFIRMATION
            await handle_approve_research_domain(
                {"id": "ap", "name": "ApproveResearchDomain",
                 "args": {"task_id": record.task_id, "domain": "untrusted.example", "allow": True}},
                cfg, registry,
            )

        await asyncio.gather(
            run_task_with_control(
                record, cfg, registry, fake_execute,
                runs_dir=str(tmp_path), run_id="run-e2e", event_log_enabled=False,
            ),
            approver(),
        )

        assert record.status == TaskStatus.COMPLETED
        assert "fetched:https://untrusted.example/page" in record.result["compressed_research"]
        # The decision was cached for the run, then cleared on terminal status.
        assert approvals.is_allowed("run-e2e", "untrusted.example") is None
        registry_mod._registry = None
