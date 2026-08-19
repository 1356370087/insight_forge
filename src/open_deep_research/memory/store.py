"""Unified Mem0 long-term-memory stores and versioned memory contracts."""

# Adapter methods intentionally share their contract documentation via MemoryStore.
# ruff: noqa: D102, D107

from __future__ import annotations

import asyncio
import builtins
import os
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional, Protocol, runtime_checkable

from pydantic import BaseModel, Field, model_validator

MEMORY_SCHEMA_VERSION = 2
OSS_LIST_HARD_LIMIT = 10001


class MemoryListLimitExceeded(RuntimeError):
    """Raised when OSS Mem0 cannot prove that a full listing was returned."""


class MemoryCategory(str, Enum):
    """Well-known memory categories used for filtering and display."""

    USER_RESEARCH_PREFERENCE = "user_research_preference"
    DOMAIN_PROFILE = "domain_profile"
    PROJECT_MEMORY = "project_memory"
    VERIFIED_RESEARCH_INSIGHT = "verified_research_insight"


class MemoryKind(str, Enum):
    """Lifecycle role of one durable memory."""

    OBSERVATION = "observation"
    REFLECTION = "reflection"
    PROFILE = "profile"


class MemoryStatus(str, Enum):
    """Recall eligibility of one durable memory."""

    ACTIVE = "active"
    ARCHIVED = "archived"
    SUPERSEDED = "superseded"


class MemorySourceKind(str, Enum):
    """Trusted provenance boundary for memory creation."""

    USER_MESSAGE = "user_message"
    PROJECT_CONFIG = "project_config"
    VERIFIED_EVIDENCE = "verified_evidence"
    REFLECTION = "reflection"


def utc_now_iso() -> str:
    """Return an RFC 3339 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


class MemoryCandidate(BaseModel):
    """A potential observation extracted before conflict resolution."""

    category: MemoryCategory
    content: str = Field(max_length=240)
    confidence: float = Field(ge=0.0, le=1.0)
    importance: int = Field(default=5, ge=1, le=10)
    reason: str = ""
    source: str = MemorySourceKind.USER_MESSAGE.value
    source_refs: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class MemoryRecord(BaseModel):
    """Versioned application-level memory persisted through Mem0."""

    memory_id: str = ""
    kind: MemoryKind = MemoryKind.OBSERVATION
    category: MemoryCategory = MemoryCategory.PROJECT_MEMORY
    content: str = Field(min_length=1, max_length=4000)
    importance: int = Field(default=5, ge=1, le=10)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    status: MemoryStatus = MemoryStatus.ACTIVE
    observed_at: str = Field(default_factory=utc_now_iso)
    last_accessed_at: Optional[str] = None
    access_count: int = Field(default=0, ge=0)
    source_kind: MemorySourceKind = MemorySourceKind.USER_MESSAGE
    source_refs: list[str] = Field(default_factory=list)
    valid_from: Optional[str] = None
    valid_to: Optional[str] = None
    conflict_group: Optional[str] = None
    supersedes_ids: list[str] = Field(default_factory=list)
    app_id: str = ""
    project_id: str = ""
    agent_id: str = ""
    user_id: str = ""
    schema_version: int = MEMORY_SCHEMA_VERSION
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_content_limit(self) -> MemoryRecord:
        """Apply kind-specific content limits."""
        limits = {
            MemoryKind.OBSERVATION: 240,
            MemoryKind.REFLECTION: 2000,
            MemoryKind.PROFILE: 4000,
        }
        if len(self.content) > limits[self.kind]:
            raise ValueError(f"{self.kind.value} content exceeds {limits[self.kind]} characters")
        return self

    def mem0_metadata(self) -> dict[str, Any]:
        """Return the complete searchable metadata payload."""
        return {
            **self.metadata,
            "schema_version": self.schema_version,
            "kind": self.kind.value,
            "category": self.category.value,
            "importance": self.importance,
            "confidence": self.confidence,
            "status": self.status.value,
            "observed_at": self.observed_at,
            "last_accessed_at": self.last_accessed_at,
            "access_count": self.access_count,
            "source_kind": self.source_kind.value,
            "source_refs": self.source_refs,
            "valid_from": self.valid_from,
            "valid_to": self.valid_to,
            "conflict_group": self.conflict_group,
            "supersedes_ids": self.supersedes_ids,
            "app_id": self.app_id,
            "project_id": self.project_id,
            "agent_id": self.agent_id,
            "user_id": self.user_id,
        }

    @classmethod
    def from_mem0(cls, item: dict[str, Any]) -> MemoryRecord:
        """Normalize a Platform or OSS response into one record."""
        metadata = item.get("metadata") or item.get("payload") or {}
        if not isinstance(metadata, dict):
            metadata = {}
        content = str(item.get("content", item.get("memory", metadata.get("data", ""))))
        category = metadata.get("category", MemoryCategory.PROJECT_MEMORY.value)
        try:
            parsed_category = MemoryCategory(category)
        except ValueError:
            parsed_category = MemoryCategory.PROJECT_MEMORY
        kind = metadata.get("kind", MemoryKind.OBSERVATION.value)
        status = metadata.get("status", MemoryStatus.ACTIVE.value)
        source_kind = metadata.get("source_kind", MemorySourceKind.USER_MESSAGE.value)
        try:
            parsed_kind = MemoryKind(kind)
        except ValueError:
            parsed_kind = MemoryKind.OBSERVATION
        try:
            parsed_status = MemoryStatus(status)
        except ValueError:
            parsed_status = MemoryStatus.ACTIVE
        try:
            parsed_source = MemorySourceKind(source_kind)
        except ValueError:
            parsed_source = MemorySourceKind.USER_MESSAGE
        source_refs_value = metadata.get("source_refs", [])
        if isinstance(source_refs_value, str):
            source_refs_value = [source_refs_value]
        supersedes_value = metadata.get("supersedes_ids", [])
        if isinstance(supersedes_value, str):
            supersedes_value = [supersedes_value]
        return cls(
            memory_id=str(item.get("id", item.get("memory_id", ""))),
            kind=parsed_kind,
            category=parsed_category,
            content=content[: {MemoryKind.OBSERVATION: 240, MemoryKind.REFLECTION: 2000, MemoryKind.PROFILE: 4000}[parsed_kind]],
            importance=int(metadata.get("importance", 5) or 5),
            confidence=float(metadata.get("confidence", 1.0) or 1.0),
            status=parsed_status,
            observed_at=str(metadata.get("observed_at") or item.get("created_at") or utc_now_iso()),
            last_accessed_at=metadata.get("last_accessed_at"),
            access_count=max(0, int(metadata.get("access_count", 0) or 0)),
            source_kind=parsed_source,
            source_refs=[str(value) for value in source_refs_value],
            valid_from=metadata.get("valid_from"),
            valid_to=metadata.get("valid_to"),
            conflict_group=metadata.get("conflict_group"),
            supersedes_ids=[str(value) for value in supersedes_value],
            app_id=str(metadata.get("app_id", "")),
            project_id=str(metadata.get("project_id", "")),
            agent_id=str(metadata.get("agent_id", "")),
            user_id=str(metadata.get("user_id", "")),
            schema_version=int(metadata.get("schema_version", 1) or 1),
            metadata=metadata,
        )


class RetrievedMemory(BaseModel):
    """A memory plus explainable retrieval scores."""

    category: MemoryCategory
    content: str
    score: float
    memory_id: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    relevance_score: float = 0.0
    recency_score: float = 0.0
    importance_score: float = 0.5
    final_score: float = 0.0


@runtime_checkable
class MemoryStore(Protocol):
    """Common capability surface for managed and OSS Mem0 backends."""

    async def search(
        self,
        query: str,
        user_id: str,
        top_k: int = 8,
        filters: Optional[dict[str, Any]] = None,
        *,
        threshold: float = 0.1,
        rerank: bool = False,
        reference_date: Optional[str] = None,
    ) -> list[dict[str, Any]]: ...

    async def add(
        self,
        content: str,
        user_id: str,
        category: MemoryCategory,
        metadata: Optional[dict[str, Any]] = None,
        *,
        infer: bool = True,
        timestamp: Optional[int] = None,
    ) -> str: ...

    async def update(self, memory_id: str, *, text: Optional[str] = None, metadata: Optional[dict[str, Any]] = None) -> None: ...
    async def delete(self, memory_id: str) -> None: ...
    async def list(self, user_id: str, filters: Optional[dict[str, Any]] = None) -> list[dict[str, Any]]: ...
    async def list_users(self) -> builtins.list[str]: ...
    async def configure_project(self, *, decay: bool) -> None: ...


class NoopMemoryStore:
    """No-op store used when durable memory is unavailable."""

    metadata_updates_reembed = False

    async def search(self, query: str, user_id: str, top_k: int = 8, filters: Optional[dict[str, Any]] = None, **_: Any) -> list[dict[str, Any]]:
        return []

    async def add(self, content: str, user_id: str, category: MemoryCategory, metadata: Optional[dict[str, Any]] = None, **_: Any) -> str:
        return "noop"

    async def update(self, memory_id: str, **_: Any) -> None:
        return None

    async def delete(self, memory_id: str) -> None:
        return None

    async def list(self, user_id: str, filters: Optional[dict[str, Any]] = None) -> list[dict[str, Any]]:
        return []

    async def list_users(self) -> builtins.list[str]:
        return []

    async def configure_project(self, *, decay: bool) -> None:
        return None


class PlatformMem0Store:
    """Mem0 Platform v3 adapter using the mem0ai 2.x typed client surface."""

    metadata_updates_reembed = False

    def __init__(self, api_key: str, org_id: str = "default", project_id: str = "default") -> None:
        from mem0 import AsyncMemoryClient

        # In mem0ai 2.x project selection is associated with the API key/client
        # project manager. org_id/project_id are retained only for source compatibility.
        self.org_id = org_id
        self.project_id = project_id
        self._client = AsyncMemoryClient(api_key=api_key)

    async def search(self, query: str, user_id: str, top_k: int = 8, filters: Optional[dict[str, Any]] = None, *, threshold: float = 0.1, rerank: bool = False, reference_date: Optional[str] = None) -> list[dict[str, Any]]:
        from mem0.client.types import SearchMemoryOptions

        merged_filters = {**(filters or {}), "user_id": user_id}
        options = SearchMemoryOptions(
            filters=merged_filters,
            top_k=top_k,
            threshold=threshold,
            rerank=rerank,
        )
        kwargs = {"reference_date": reference_date} if reference_date else {}
        result = await self._client.search(query, options=options, **kwargs)
        return result.get("results", []) if isinstance(result, dict) else list(result or [])

    async def add(self, content: str, user_id: str, category: MemoryCategory, metadata: Optional[dict[str, Any]] = None, *, infer: bool = True, timestamp: Optional[int] = None) -> str:
        from mem0.client.types import AddMemoryOptions

        meta = {**(metadata or {}), "category": category.value}
        options = AddMemoryOptions(
            filters={"user_id": user_id},
            metadata=meta,
            infer=infer,
            timestamp=timestamp,
        )
        result = await self._client.add([{"role": "user", "content": content}], options=options)
        if isinstance(result, dict):
            results = result.get("results") or []
            if results and isinstance(results[0], dict):
                return str(results[0].get("id", ""))
            return str(result.get("id", ""))
        return str(result)

    async def update(self, memory_id: str, *, text: Optional[str] = None, metadata: Optional[dict[str, Any]] = None) -> None:
        from mem0.client.types import UpdateMemoryOptions

        await self._client.update(memory_id, options=UpdateMemoryOptions(text=text, metadata=metadata))

    async def delete(self, memory_id: str) -> None:
        await self._client.delete(memory_id)

    async def list(self, user_id: str, filters: Optional[dict[str, Any]] = None) -> list[dict[str, Any]]:
        from mem0.client.types import GetAllMemoryOptions

        page = 1
        page_size = 1000
        memories: list[dict[str, Any]] = []
        while True:
            result = await self._client.get_all(options=GetAllMemoryOptions(
                filters={**(filters or {}), "user_id": user_id},
                page=page,
                page_size=page_size,
            ))
            if not isinstance(result, dict):
                memories.extend(list(result or []))
                break
            batch = result.get("results", [])
            memories.extend(batch)
            count = result.get("count")
            if (
                not result.get("next")
                or not batch
                or (count is not None and len(memories) >= int(count))
            ):
                break
            page += 1
        return memories

    async def list_users(self) -> builtins.list[str]:
        result = await self._client.users()
        entities = result.get("results", result) if isinstance(result, dict) else result
        users: list[str] = []
        for item in entities or []:
            if isinstance(item, dict):
                if item.get("type") not in {None, "user", "user_id"}:
                    continue
                value = item.get("user_id") or item.get("id") or item.get("name")
                if value:
                    users.append(str(value))
        return users

    async def configure_project(self, *, decay: bool) -> None:
        await self._client.project.update(decay=decay)


class OSSMem0Store:
    """Self-hosted Mem0 adapter with explicit Platform capability degradation."""

    metadata_updates_reembed = True

    def __init__(self, config: dict[str, Any]) -> None:
        from mem0 import Memory

        self._memory = Memory.from_config(config_dict=config)

    async def search(self, query: str, user_id: str, top_k: int = 8, filters: Optional[dict[str, Any]] = None, *, threshold: float = 0.1, rerank: bool = False, reference_date: Optional[str] = None) -> list[dict[str, Any]]:
        merged_filters = {**(filters or {}), "user_id": user_id}
        kwargs = {"reference_date": reference_date} if reference_date else {}
        result = await asyncio.to_thread(
            self._memory.search,
            query=query,
            top_k=top_k,
            filters=merged_filters,
            threshold=threshold,
            rerank=rerank,
            **kwargs,
        )
        return result.get("results", []) if isinstance(result, dict) else list(result or [])

    async def add(self, content: str, user_id: str, category: MemoryCategory, metadata: Optional[dict[str, Any]] = None, *, infer: bool = True, timestamp: Optional[int] = None) -> str:
        meta = {**(metadata or {}), "category": category.value}
        result = await asyncio.to_thread(
            self._memory.add,
            [{"role": "user", "content": content}],
            user_id=user_id,
            metadata=meta,
            infer=infer,
            timestamp=timestamp,
        )
        if isinstance(result, dict):
            results = result.get("results") or []
            if results and isinstance(results[0], dict):
                return str(results[0].get("id", ""))
            return str(result.get("id", ""))
        return str(result)

    async def update(self, memory_id: str, *, text: Optional[str] = None, metadata: Optional[dict[str, Any]] = None) -> None:
        content = text
        if content is None:
            existing = await asyncio.to_thread(self._memory.get, memory_id)
            if not existing:
                raise KeyError(f"Memory not found: {memory_id}")
            content = str(existing.get("memory", existing.get("data", "")))
        await asyncio.to_thread(
            self._memory.update,
            memory_id,
            content,
            metadata=metadata,
        )

    async def delete(self, memory_id: str) -> None:
        await asyncio.to_thread(self._memory.delete, memory_id)

    async def list(self, user_id: str, filters: Optional[dict[str, Any]] = None) -> list[dict[str, Any]]:
        result = await asyncio.to_thread(
            self._memory.get_all,
            filters={**(filters or {}), "user_id": user_id},
            top_k=OSS_LIST_HARD_LIMIT,
        )
        memories = result.get("results", []) if isinstance(result, dict) else list(result or [])
        if len(memories) >= OSS_LIST_HARD_LIMIT:
            raise MemoryListLimitExceeded(
                "OSSMem0Store listing reached its hard limit; refusing to use a truncated lifecycle view"
            )
        return memories

    async def list_users(self) -> builtins.list[str]:
        return []

    async def configure_project(self, *, decay: bool) -> None:
        # Decay is a Platform v3 project feature. OSS recall still uses the
        # deterministic application-side importance/recency reranker.
        return None


def create_memory_store(config: Any) -> MemoryStore:
    """Build the configured backend, preserving the existing no-op fallback."""
    if not getattr(config, "enable_memory", False):
        return NoopMemoryStore()
    api_key = os.environ.get("MEM0_API_KEY")
    provider = getattr(config, "memory_provider", "platform")
    if provider == "platform":
        if not api_key:
            return NoopMemoryStore()
        return PlatformMem0Store(
            api_key=api_key,
            project_id=getattr(config, "memory_project_id", None) or "default",
        )
    if provider == "oss":
        import json

        path = os.environ.get("MEM0_CONFIG_PATH")
        oss_config: dict[str, Any] = {}
        if path:
            with open(path, encoding="utf-8") as handle:
                oss_config = json.load(handle)
        return OSSMem0Store(config=oss_config)
    return NoopMemoryStore()
