"""Tests for V7 durable approvals and SSRF-safe legacy fetching."""

import time

import pytest

from open_deep_research.sandbox.approvals import SecurityApprovalStore


def test_allow_run_decision_is_scoped_to_one_run(tmp_path):
    first = SecurityApprovalStore("run-a", runs_dir=str(tmp_path))
    second = SecurityApprovalStore("run-b", runs_dir=str(tmp_path))
    approval = first.request(
        task_id="task-a",
        fence_token=1,
        kind="network",
        capability="tool.egress",
        target={"domain": "example.com", "port": 443},
        operation_id="op-a",
        expires_at=time.time() + 60,
    )
    first.resolve(
        approval.approval_id,
        decision="allow_run",
        actor="user-a",
        reason="needed",
        expected_fence_token=1,
    )
    assert first.list(status="resolved")[1][0].decision == "allow_run"
    assert second.list()[1] == []


class _FakeResponse:
    def __init__(self, status: int, text: str) -> None:
        self.status = status
        self._text = text

    async def text(self, errors: str = "strict") -> str:
        del errors
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
        del url, kwargs
        return _FakeResponse(self.status, self.text)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


@pytest.mark.asyncio
async def test_fetch_webpage_returns_content_no_summarize(monkeypatch):
    from open_deep_research.tools import utils
    from open_deep_research.tools.base import ToolContext
    from open_deep_research.tools.fetch_webpage import definition

    monkeypatch.setattr(definition.aiohttp, "ClientSession", _FakeSession)
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


@pytest.mark.asyncio
async def test_fetch_webpage_raises_on_http_error(monkeypatch):
    from langchain_core.tools import ToolException

    from open_deep_research.tools import utils
    from open_deep_research.tools.base import ToolContext
    from open_deep_research.tools.fetch_webpage import definition

    class _ErrSession(_FakeSession):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.status = 404

    monkeypatch.setattr(definition.aiohttp, "ClientSession", _ErrSession)
    tool = utils.fetch_webpage
    with pytest.raises(ToolException):
        await tool.call(
            tool.input_schema.model_validate(
                {"url": "https://example.com/missing", "summarize": False}
            ),
            ToolContext(
                config={"configurable": {"search_api": "none"}},
                role="researcher",
                tool_call_id="fetch-2",
            ),
        )
