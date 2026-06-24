"""Unified mem0 long-term memory interface for Open Deep Research.

Provides a :class:`MemoryStore` Protocol that abstracts the difference between
the mem0 Platform (``AsyncMemoryClient``) and OSS (``Memory``) backends, plus a
``NoopMemoryStore`` fallback for when memory is disabled or unavailable.

All ``mem0`` imports are deferred to store constructors so the graph compiles
cleanly even when the package is not installed.
"""

from __future__ import annotations

import asyncio
import os
from enum import Enum
from typing import Any, Optional, Protocol, runtime_checkable

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class MemoryCategory(str, Enum):
    """Well-known memory categories used for filtering and display."""

    USER_RESEARCH_PREFERENCE = "user_research_preference"
    DOMAIN_PROFILE = "domain_profile"
    PROJECT_MEMORY = "project_memory"


class MemoryCandidate(BaseModel):
    """A potential memory extracted by the policy engine (before writing)."""

    category: MemoryCategory
    content: str = Field(
        description="Memory content, max 240 characters. Factual and derived from user input.",
        max_length=240,
    )
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = Field(default="")
    source: str = Field(default="user_message")
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievedMemory(BaseModel):
    """A memory retrieved from mem0 during recall."""

    category: MemoryCategory
    content: str
    score: float
    memory_id: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Store protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class MemoryStore(Protocol):
    """Protocol satisfied by all memory store backends."""

    async def search(
        self,
        query: str,
        user_id: str,
        top_k: int = 8,
        filters: Optional[dict[str, Any]] = None,
    ) -> list[dict[str, Any]]:
        """Search for relevant memories."""
        ...

    async def add(
        self,
        content: str,
        user_id: str,
        category: MemoryCategory,
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        """Persist a single memory and return its ID."""
        ...


# ---------------------------------------------------------------------------
# Noop store (safe default)
# ---------------------------------------------------------------------------


class NoopMemoryStore:
    """No-op store that silently discards all reads and writes."""

    async def search(
        self,
        query: str,
        user_id: str,
        top_k: int = 8,
        filters: Optional[dict[str, Any]] = None,
    ) -> list[dict[str, Any]]:
        return []

    async def add(
        self,
        content: str,
        user_id: str,
        category: MemoryCategory,
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        return "noop"


# ---------------------------------------------------------------------------
# Platform store (mem0 cloud) — uses ``mem0.AsyncMemoryClient``
# ---------------------------------------------------------------------------


class PlatformMem0Store:
    """Uses ``mem0.AsyncMemoryClient`` for the hosted platform.

    The SDK signature is::

        AsyncMemoryClient.add(messages, options=None, **kwargs)
        AsyncMemoryClient.search(query, user_id, top_k, filters)

    where ``messages`` is a plain string or ``[{"role":"user","content":...}]``
    and ``options`` is an ``AddMemoryOptions`` dataclass carrying
    ``user_id``, ``agent_id``, ``metadata``, etc.
    """

    def __init__(
        self,
        api_key: str,
        org_id: str = "default",
        project_id: str = "default",
    ) -> None:
        from mem0 import AsyncMemoryClient

        self._client = AsyncMemoryClient(
            api_key=api_key,
            org_id=org_id,
            project_id=project_id,
        )

    async def search(
        self,
        query: str,
        user_id: str,
        top_k: int = 8,
        filters: Optional[dict[str, Any]] = None,
    ) -> list[dict[str, Any]]:
        result = await self._client.search(
            query=query,
            user_id=user_id,
            top_k=top_k,
            filters=filters,
        )
        if isinstance(result, list):
            return result
        if isinstance(result, dict):
            return result.get("results", [])
        return []

    async def add(
        self,
        content: str,
        user_id: str,
        category: MemoryCategory,
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        from mem0.client.types import AddMemoryOptions  # deferred

        meta = metadata or {}
        meta["category"] = category.value

        options = AddMemoryOptions(
            filters={"user_id": user_id},
            metadata=meta,
        )
        # messages is the first positional argument (plain string or message list)
        result = await self._client.add(
            [{"role": "user", "content": content}],
            options=options,
        )
        if isinstance(result, dict):
            return result.get("id", str(result))
        return str(result)


# ---------------------------------------------------------------------------
# OSS store (self-hosted) — uses ``mem0.Memory``
# ---------------------------------------------------------------------------


class OSSMem0Store:
    """Uses ``mem0.Memory``, wrapping synchronous calls in ``asyncio.to_thread``.

    SDK signatures::

        Memory.add(messages, *, user_id=None, agent_id=None, metadata=None, ...)
        Memory.search(query, *, top_k=20, filters=None, ...)

    ``Memory.search`` does **not** accept a top-level ``user_id`` — it must
    be merged into the ``filters`` dict.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        from mem0 import Memory

        self._memory = Memory.from_config(config_dict=config)

    async def search(
        self,
        query: str,
        user_id: str,
        top_k: int = 8,
        filters: Optional[dict[str, Any]] = None,
    ) -> list[dict[str, Any]]:
        merged_filters = filters or {}
        merged_filters["user_id"] = user_id
        return await asyncio.to_thread(
            self._memory.search,
            query=query,
            top_k=top_k,
            filters=merged_filters,
        )

    async def add(
        self,
        content: str,
        user_id: str,
        category: MemoryCategory,
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        meta = metadata or {}
        meta["category"] = category.value
        return await asyncio.to_thread(
            self._memory.add,
            [{"role": "user", "content": content}],
            user_id=user_id,
            metadata=meta,
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_memory_store(config: Any) -> MemoryStore:  # 'Configuration' at runtime
    """Build the appropriate :class:`MemoryStore` for the current configuration.

    Deferred imports of ``mem0`` keep the graph importable when the
    package is not installed and ``enable_memory`` is ``False``.
    """
    if not getattr(config, "enable_memory", False):
        return NoopMemoryStore()

    api_key = os.environ.get("MEM0_API_KEY")
    if not api_key:
        return NoopMemoryStore()

    provider = getattr(config, "memory_provider", "platform")

    if provider == "platform":
        project_id = getattr(config, "memory_project_id", None) or "default"
        return PlatformMem0Store(api_key=api_key, project_id=project_id)

    if provider == "oss":
        oss_config_path = os.environ.get("MEM0_CONFIG_PATH")
        oss_config: dict[str, Any] = {}
        if oss_config_path:
            import json
            with open(oss_config_path, encoding="utf-8") as fh:
                oss_config = json.load(fh)
        return OSSMem0Store(config=oss_config)

    return NoopMemoryStore()
