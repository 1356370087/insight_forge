"""Tests for mem0 long-term memory integration.

Covers: memory.py (models, stores, factory), memory_policy.py (extraction,
filtering), graph-level behaviour (recall, extract-and-write nodes), and
event emission.
"""

import json
import os

import pytest

from open_deep_research.memory.store import (
    MemoryCandidate,
    MemoryCategory,
    MemoryStore,
    NoopMemoryStore,
    RetrievedMemory,
    create_memory_store,
)
from open_deep_research.tasks.events import EventType, ResearchEvent

# ---------------------------------------------------------------------------
# Data model tests
# ---------------------------------------------------------------------------


class TestMemoryCategory:
    def test_all_categories_exist(self):
        assert MemoryCategory.USER_RESEARCH_PREFERENCE.value == "user_research_preference"
        assert MemoryCategory.DOMAIN_PROFILE.value == "domain_profile"
        assert MemoryCategory.PROJECT_MEMORY.value == "project_memory"

    def test_category_from_string(self):
        assert MemoryCategory("user_research_preference") == MemoryCategory.USER_RESEARCH_PREFERENCE
        assert MemoryCategory("domain_profile") == MemoryCategory.DOMAIN_PROFILE

    def test_invalid_category_raises(self):
        with pytest.raises(ValueError):
            MemoryCategory("invalid_category")


class TestMemoryCandidate:
    def test_minimal_construction(self):
        c = MemoryCandidate(
            category=MemoryCategory.PROJECT_MEMORY,
            content="User uses PyTorch",
            confidence=0.9,
        )
        assert c.category == MemoryCategory.PROJECT_MEMORY
        assert c.content == "User uses PyTorch"
        assert c.confidence == 0.9
        assert c.source == "user_message"  # default

    def test_full_construction(self):
        c = MemoryCandidate(
            category=MemoryCategory.USER_RESEARCH_PREFERENCE,
            content="Prefers Chinese reports",
            confidence=0.85,
            reason="Mentioned in user message",
            source="user_message",
            metadata={"app_id": "test"},
        )
        assert c.reason == "Mentioned in user message"
        assert c.metadata == {"app_id": "test"}

    def test_content_truncation(self):
        long_content = "x" * 300
        # Pydantic max_length=240 is validated on construction when using model_validate
        with pytest.raises(Exception):
            MemoryCandidate.model_validate({
                "category": "domain_profile",
                "content": long_content,
                "confidence": 0.8,
            })

    def test_content_max_length_accepted(self):
        content_240 = "x" * 240
        c = MemoryCandidate(
            category=MemoryCategory.DOMAIN_PROFILE,
            content=content_240,
            confidence=0.8,
        )
        assert len(c.content) == 240

    def test_confidence_bounds(self):
        # Confidence below 0
        with pytest.raises(Exception):
            MemoryCandidate(category=MemoryCategory.DOMAIN_PROFILE, content="test", confidence=-0.1)
        # Confidence above 1
        with pytest.raises(Exception):
            MemoryCandidate(category=MemoryCategory.DOMAIN_PROFILE, content="test", confidence=1.1)


class TestRetrievedMemory:
    def test_minimal_construction(self):
        rm = RetrievedMemory(
            category=MemoryCategory.PROJECT_MEMORY,
            content="Uses FastAPI",
            score=0.92,
        )
        assert rm.memory_id == ""
        assert rm.metadata == {}

    def test_full_construction(self):
        rm = RetrievedMemory(
            category=MemoryCategory.USER_RESEARCH_PREFERENCE,
            content="Likes tables",
            score=0.78,
            memory_id="mem-abc",
            metadata={"source": "prior_session"},
        )
        assert rm.memory_id == "mem-abc"
        assert rm.metadata["source"] == "prior_session"


# ---------------------------------------------------------------------------
# Memory store tests
# ---------------------------------------------------------------------------


class TestNoopMemoryStore:
    def test_search_returns_empty(self):
        store = NoopMemoryStore()
        import asyncio
        result = asyncio.run(store.search("query", "user-1"))
        assert result == []

    def test_add_returns_noop(self):
        store = NoopMemoryStore()
        import asyncio
        result = asyncio.run(store.add("content", "user-1", MemoryCategory.PROJECT_MEMORY))
        assert result == "noop"

    def test_satisfies_protocol(self):
        store = NoopMemoryStore()
        assert isinstance(store, MemoryStore)


class TestCreateMemoryStore:
    def test_returns_noop_when_disabled(self):
        class FakeConfig:
            enable_memory = False
        store = create_memory_store(FakeConfig())
        assert isinstance(store, NoopMemoryStore)

    def test_returns_noop_when_no_api_key(self):
        class FakeConfig:
            enable_memory = True
            memory_provider = "platform"
            memory_project_id = None
        # Ensure MEM0_API_KEY is not set
        old_key = os.environ.pop("MEM0_API_KEY", None)
        try:
            store = create_memory_store(FakeConfig())
            assert isinstance(store, NoopMemoryStore)
        finally:
            if old_key:
                os.environ["MEM0_API_KEY"] = old_key


# ---------------------------------------------------------------------------
# Memory policy tests
# ---------------------------------------------------------------------------


class TestFilterCandidates:
    """Tests for filter_candidates in memory_policy.py."""

    def test_filters_below_confidence(self):
        from open_deep_research.memory.policy import (
            MemoryCandidateModel,
            filter_candidates,
        )

        raw = [
            MemoryCandidateModel(
                category="project_memory",
                content="Valid high-confidence candidate",
                confidence=0.9,
                reason="Useful",
            ),
            MemoryCandidateModel(
                category="project_memory",
                content="Low confidence candidate",
                confidence=0.3,
                reason="Weak",
            ),
        ]
        result = filter_candidates(raw, min_confidence=0.75)
        assert len(result) == 1
        assert result[0].content == "Valid high-confidence candidate"

    def test_filters_forbidden_keyword_api_key(self):
        from open_deep_research.memory.policy import (
            MemoryCandidateModel,
            filter_candidates,
        )

        raw = [
            MemoryCandidateModel(
                category="project_memory",
                content="My api_key is abc123",
                confidence=0.9,
                reason="Bad",
            ),
        ]
        result = filter_candidates(raw, min_confidence=0.5)
        assert len(result) == 0

    def test_filters_forbidden_keyword_url(self):
        from open_deep_research.memory.policy import (
            MemoryCandidateModel,
            filter_candidates,
        )

        raw = [
            MemoryCandidateModel(
                category="project_memory",
                content="Source: https://example.com/article",
                confidence=0.9,
                reason="Contains URL",
            ),
        ]
        result = filter_candidates(raw, min_confidence=0.5)
        assert len(result) == 0

    def test_credential_word_matching_does_not_reject_benign_compounds(self):
        from open_deep_research.memory.policy import (
            MemoryCandidateModel,
            filter_candidates,
        )

        raw = [
            MemoryCandidateModel(
                category="user_research_preference",
                content="Prefers tokenizer statistics in reports",
                confidence=0.9,
                reason="Benign technical preference",
            ),
            MemoryCandidateModel(
                category="project_memory",
                content="The deployment token is abc123",
                confidence=0.9,
                reason="Credential",
            ),
        ]
        result = filter_candidates(raw, min_confidence=0.5)
        assert [candidate.content for candidate in result] == [
            "Prefers tokenizer statistics in reports",
        ]

    def test_filters_markdown_link_pattern(self):
        from open_deep_research.memory.policy import (
            MemoryCandidateModel,
            filter_candidates,
        )

        raw = [
            MemoryCandidateModel(
                category="project_memory",
                content="[Some Link](https://example.com)",
                confidence=0.9,
                reason="Markdown link",
            ),
        ]
        result = filter_candidates(raw, min_confidence=0.5)
        assert len(result) == 0

    def test_filters_invalid_category(self):
        from open_deep_research.memory.policy import (
            MemoryCandidateModel,
            filter_candidates,
        )

        raw = [
            MemoryCandidateModel(
                category="not_a_real_category",
                content="Some content",
                confidence=0.9,
                reason="Bad category",
            ),
        ]
        result = filter_candidates(raw, min_confidence=0.5)
        assert len(result) == 0

    def test_truncates_long_content(self):
        from open_deep_research.memory.policy import (
            MemoryCandidateModel,
            filter_candidates,
        )

        long = "a" * 300
        raw = [
            MemoryCandidateModel(
                category="project_memory",
                content=long,
                confidence=0.9,
                reason="Long",
            ),
        ]
        result = filter_candidates(raw, min_confidence=0.5)
        assert len(result) == 1
        assert len(result[0].content) <= 240

    def test_empty_list_returns_empty(self):
        from open_deep_research.memory.policy import filter_candidates
        result = filter_candidates([], min_confidence=0.5)
        assert result == []

    def test_preserves_valid_candidates(self):
        from open_deep_research.memory.policy import (
            MemoryCandidateModel,
            filter_candidates,
        )

        raw = [
            MemoryCandidateModel(
                category="user_research_preference",
                content="Prefers reports in Chinese",
                confidence=0.92,
                reason="Mentioned in user message",
            ),
            MemoryCandidateModel(
                category="domain_profile",
                content="Frequently researches AI agent architectures",
                confidence=0.88,
                reason="Recurring topic across sessions",
            ),
            MemoryCandidateModel(
                category="project_memory",
                content="Uses Spring Boot for backend services",
                confidence=0.95,
                reason="Explicitly stated in project config",
            ),
        ]
        result = filter_candidates(raw, min_confidence=0.75)
        assert len(result) == 3
        categories = {c.category for c in result}
        assert categories == {
            MemoryCategory.USER_RESEARCH_PREFERENCE,
            MemoryCategory.DOMAIN_PROFILE,
            MemoryCategory.PROJECT_MEMORY,
        }


# ---------------------------------------------------------------------------
# Memory context formatting tests
# ---------------------------------------------------------------------------


class TestFormatMemoryContext:
    def test_basic_format(self):
        from open_deep_research.agents.deep_researcher import _format_memory_context

        results = [
            {
                "content": "User prefers Chinese reports",
                "memory": "User prefers Chinese reports",
                "metadata": {"category": "user_research_preference"},
            },
            {
                "content": "Uses Spring Boot",
                "memory": "Uses Spring Boot",
                "metadata": {"category": "project_memory"},
            },
        ]
        formatted = _format_memory_context(results)
        assert "<Memory Context>" in formatted
        assert "advisory only" in formatted
        assert "[user_research_preference] User prefers Chinese reports" in formatted
        assert "[project_memory] Uses Spring Boot" in formatted
        assert "</Memory Context>" in formatted

    def test_falls_back_to_memory_key(self):
        from open_deep_research.agents.deep_researcher import _format_memory_context

        results = [{"memory": "Legacy format memory", "metadata": {}}]
        formatted = _format_memory_context(results)
        assert "[general] Legacy format memory" in formatted

    def test_non_dict_metadata_handled(self):
        from open_deep_research.agents.deep_researcher import _format_memory_context

        results = [{"content": "Test", "memory": "Test", "metadata": "not-a-dict"}]
        formatted = _format_memory_context(results)
        assert "[general] Test" in formatted


# ---------------------------------------------------------------------------
# Event type tests
# ---------------------------------------------------------------------------


class TestMemoryEventTypes:
    def test_all_memory_events_constructable(self):
        for et_value in [
            "memory.recalled",
            "memory.candidate_extracted",
            "memory.written",
            "memory.skipped",
            "memory.failed",
        ]:
            et = EventType(et_value)
            # Events should serialize correctly
            event = ResearchEvent(
                event_type=et,
                task_id="lead_agent",
                run_id="test-run",
                data={"summary": "test"},
            )
            raw = event.model_dump_json()
            parsed = json.loads(raw)
            assert parsed["event_type"].startswith("memory.")

    def test_memory_events_have_expected_values(self):
        assert EventType.MEMORY_RECALLED == "memory.recalled"
        assert EventType.MEMORY_CANDIDATE_EXTRACTED == "memory.candidate_extracted"
        assert EventType.MEMORY_WRITTEN == "memory.written"
        assert EventType.MEMORY_SKIPPED == "memory.skipped"
        assert EventType.MEMORY_FAILED == "memory.failed"


# ---------------------------------------------------------------------------
# Configuration tests
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# SDK surface tests — verify real mem0 import paths and call signatures
# ---------------------------------------------------------------------------


class TestMem0SDKImports:
    """Verify that the mem0 SDK exports match what our stores expect."""

    def test_async_memory_client_import(self):
        from mem0 import AsyncMemoryClient
        assert AsyncMemoryClient is not None

    def test_memory_import(self):
        from mem0 import Memory
        assert Memory is not None

    def test_memory_client_import(self):
        from mem0 import MemoryClient
        assert MemoryClient is not None

    def test_add_memory_options_import(self):
        from mem0.client.types import AddMemoryOptions
        assert AddMemoryOptions is not None


class TestPlatformMem0StoreSignature:
    """Verify PlatformMem0Store passes args in the correct shape to the SDK."""

    def test_add_passes_messages_as_first_positional(self, monkeypatch):
        import asyncio

        from open_deep_research.memory.store import MemoryCategory, PlatformMem0Store

        store = PlatformMem0Store.__new__(PlatformMem0Store)

        # Fake client whose add() records its call args
        class FakeClient:
            def __init__(self, *a, **kw):
                pass

            async def search(self, *a, **kw):
                return []

            async def add(self, messages, options=None, **kwargs):
                self._last_messages = messages
                self._last_options = options
                return {"id": "mem-fake"}

        fake = FakeClient()
        store._client = fake

        result = asyncio.run(store.add(
            content="Test memory",
            user_id="u1",
            category=MemoryCategory.PROJECT_MEMORY,
            metadata={"key": "val"},
        ))
        # messages should be a list of message dicts
        assert isinstance(fake._last_messages, list)
        assert fake._last_messages[0]["role"] == "user"
        assert fake._last_messages[0]["content"] == "Test memory"
        # options should carry user_id in filters and metadata in metadata
        assert fake._last_options is not None
        assert fake._last_options.filters == {"user_id": "u1"}
        assert fake._last_options.metadata.get("category") == "project_memory"
        assert result == "mem-fake"

    def test_add_normalises_result(self, monkeypatch):
        import asyncio

        from open_deep_research.memory.store import MemoryCategory, PlatformMem0Store

        store = PlatformMem0Store.__new__(PlatformMem0Store)

        class FakeClient:
            async def add(self, *a, **kw):
                return {"id": "abc", "other": "x"}

        store._client = FakeClient()
        result = asyncio.run(store.add("x", "u", MemoryCategory.DOMAIN_PROFILE))
        assert result == "abc"


class TestOSSMem0StoreSignature:
    """Verify OSSMem0Store merges user_id into filters for search."""

    def test_search_merges_user_id_into_filters(self, monkeypatch):
        import asyncio

        from open_deep_research.memory.store import OSSMem0Store

        store = OSSMem0Store.__new__(OSSMem0Store)

        class FakeMemory:
            def search(self, query, top_k, filters, **kw):
                self._last_query = query
                self._last_filters = filters
                return [{"content": "found", "score": 0.9}]

        store._memory = FakeMemory()

        result = asyncio.run(store.search(query="test", user_id="u1", top_k=5))
        assert result[0]["content"] == "found"
        assert store._memory._last_filters == {"user_id": "u1"}

    def test_search_merges_extra_filters(self, monkeypatch):
        import asyncio

        from open_deep_research.memory.store import OSSMem0Store

        store = OSSMem0Store.__new__(OSSMem0Store)

        class FakeMemory:
            def search(self, query, top_k, filters, **kw):
                self._last_filters = filters
                return []

        store._memory = FakeMemory()
        asyncio.run(store.search(
            query="q", user_id="u1", top_k=8,
            filters={"project_id": "p1"},
        ))
        assert store._memory._last_filters == {"project_id": "p1", "user_id": "u1"}

    def test_add_passes_messages_and_user_id_correctly(self, monkeypatch):
        import asyncio

        from open_deep_research.memory.store import MemoryCategory, OSSMem0Store

        store = OSSMem0Store.__new__(OSSMem0Store)

        class FakeMemory:
            def add(self, messages, *, user_id=None, agent_id=None, metadata=None, **kw):
                self._last_messages = messages
                self._last_user_id = user_id
                self._last_metadata = metadata
                return "mem-oss-1"

        store._memory = FakeMemory()
        result = asyncio.run(store.add(
            content="OSS memory", user_id="u2",
            category=MemoryCategory.USER_RESEARCH_PREFERENCE,
            metadata={"a": "b"},
        ))
        assert isinstance(store._memory._last_messages, list)
        assert store._memory._last_messages[0]["content"] == "OSS memory"
        assert store._memory._last_user_id == "u2"
        assert store._memory._last_metadata["category"] == "user_research_preference"
        assert result == "mem-oss-1"


class TestMemoryConfiguration:
    def test_default_memory_disabled(self):
        from open_deep_research.configuration import Configuration
        c = Configuration()
        assert c.enable_memory is False
        assert c.memory_provider == "platform"
        assert c.memory_top_k == 8
        assert c.memory_min_confidence == 0.75
        assert c.memory_auto_write is True
        assert c.memory_write_after_report is True
        assert c.memory_fail_open is True

    def test_memory_config_from_env(self):
        from open_deep_research.configuration import Configuration

        os.environ["MEM0_PROVIDER"] = "oss"
        os.environ["MEM0_MEMORY_PROJECT_ID"] = "my-project"
        try:
            c = Configuration.from_runnable_config({})
            assert c.memory_provider == "oss"
            assert c.memory_project_id == "my-project"
        finally:
            del os.environ["MEM0_PROVIDER"]
            del os.environ["MEM0_MEMORY_PROJECT_ID"]

    def test_configurable_overrides_env(self):
        from open_deep_research.configuration import Configuration

        os.environ["MEM0_PROVIDER"] = "oss"
        try:
            c = Configuration.from_runnable_config({
                "configurable": {"memory_provider": "platform"},
            })
            # configurable takes precedence over env
            assert c.memory_provider == "platform"
        finally:
            del os.environ["MEM0_PROVIDER"]
