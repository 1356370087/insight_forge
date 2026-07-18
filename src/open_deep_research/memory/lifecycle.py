"""Application-side lifecycle for schema-v2 long-term memories."""

from __future__ import annotations

import hashlib
import math
from datetime import datetime, timedelta, timezone
from typing import Any, Awaitable, Callable, Optional

from pydantic import BaseModel, Field

from open_deep_research.memory.policy import (
    MemoryConflictDecisionModel,
    generate_reflections,
    generate_research_profile,
)
from open_deep_research.memory.store import (
    MEMORY_SCHEMA_VERSION,
    MemoryCandidate,
    MemoryCategory,
    MemoryKind,
    MemoryRecord,
    MemorySourceKind,
    MemoryStatus,
    MemoryStore,
    utc_now_iso,
)

_PROJECT_DECAY_STATE: dict[tuple[str, str], bool] = {}


class MaintenanceResult(BaseModel):
    """Summary returned by run-end and daily maintenance."""

    trigger_reasons: list[str] = Field(default_factory=list)
    reflections_generated: int = 0
    profile_updated: bool = False
    archived: int = 0
    would_write: int = 0
    status_counts: dict[str, int] = Field(default_factory=dict)
    category_counts: dict[str, int] = Field(default_factory=dict)
    profile_version: int = 0


def advanced_app_id(config: Any) -> str:
    """Return the isolated application namespace for schema-v2 memories."""
    return f"{config.memory_app_id}{config.memory_v2_app_suffix}"


def v2_filters(config: Any, **extra: Any) -> dict[str, Any]:
    """Build mandatory project/app/schema tenant filters."""
    return {
        "project_id": config.memory_project_id,
        "app_id": advanced_app_id(config),
        "schema_version": MEMORY_SCHEMA_VERSION,
        **extra,
    }


def _parse_datetime(value: Any, fallback: datetime) -> datetime:
    """Parse Mem0 timestamps into timezone-aware UTC datetimes."""
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, int | float):
        parsed = datetime.fromtimestamp(value, tz=timezone.utc)
    elif isinstance(value, str) and value:
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return fallback
    else:
        return fallback
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def memory_half_life_days(record: MemoryRecord, config: Any) -> int:
    """Resolve category or reflection half-life with safe defaults."""
    key = "reflection" if record.kind == MemoryKind.REFLECTION else record.category.value
    defaults = {
        MemoryCategory.USER_RESEARCH_PREFERENCE.value: 180,
        MemoryCategory.DOMAIN_PROFILE.value: 180,
        MemoryCategory.PROJECT_MEMORY.value: 90,
        MemoryCategory.VERIFIED_RESEARCH_INSIGHT.value: 30,
        "reflection": 90,
    }
    configured = getattr(config, "memory_half_life_days", {}) or {}
    return max(1, int(configured.get(key, defaults[key])))


def recency_score(record: MemoryRecord, config: Any, now: Optional[datetime] = None) -> float:
    """Return exponential freshness using the configured category half-life."""
    current = now or datetime.now(timezone.utc)
    observed = _parse_datetime(record.observed_at, current)
    age_days = max(0.0, (current - observed).total_seconds() / 86400)
    return math.exp(-math.log(2) * age_days / memory_half_life_days(record, config))


def rank_v2_memories(
    results: list[dict[str, Any]],
    config: Any,
    *,
    now: Optional[datetime] = None,
    top_k: Optional[int] = None,
) -> list[dict[str, Any]]:
    """Rerank Mem0's candidate set with explainable application scores."""
    ranked: list[dict[str, Any]] = []
    for item in results:
        record = MemoryRecord.from_mem0(item)
        if record.schema_version != MEMORY_SCHEMA_VERSION or record.status != MemoryStatus.ACTIVE:
            continue
        relevance = max(0.0, min(1.0, float(item.get("score", 0.0) or 0.0)))
        importance = record.importance / 10
        freshness = recency_score(record, config, now)
        final_score = (
            config.memory_relevance_weight * relevance
            + config.memory_importance_weight * importance
            + config.memory_recency_weight * freshness
        )
        ranked.append({
            **item,
            "id": record.memory_id,
            "memory": record.content,
            "content": record.content,
            "metadata": record.mem0_metadata(),
            "score": final_score,
            "retrieval_scores": {
                "relevance": relevance,
                "importance": importance,
                "recency": freshness,
                "final": final_score,
            },
        })
    ranked.sort(key=lambda item: float(item["score"]), reverse=True)
    return ranked[: top_k or config.memory_top_k]


def rank_legacy_memories(
    results: list[dict[str, Any]],
    config: Any,
    *,
    top_k: Optional[int] = None,
) -> list[dict[str, Any]]:
    """Apply neutral importance/freshness to immutable legacy memories."""
    ranked: list[dict[str, Any]] = []
    for item in results:
        relevance = max(0.0, min(1.0, float(item.get("score", 0.0) or 0.0)))
        final_score = (
            config.memory_relevance_weight * relevance
            + config.memory_importance_weight * 0.5
            + config.memory_recency_weight * 0.5
        )
        ranked.append({
            **item,
            "score": final_score,
            "retrieval_scores": {
                "relevance": relevance,
                "importance": 0.5,
                "recency": 0.5,
                "final": final_score,
                "legacy": True,
            },
        })
    ranked.sort(key=lambda item: float(item["score"]), reverse=True)
    return ranked[: top_k or config.memory_top_k]


async def configure_advanced_store(store: MemoryStore, config: Any) -> None:
    """Synchronize Platform Decay with the advanced-memory feature state."""
    if getattr(config, "memory_provider", "platform") != "platform":
        return
    key = (config.memory_provider, config.memory_project_id)
    desired = bool(config.memory_advanced_enabled and config.memory_decay_enabled)
    if _PROJECT_DECAY_STATE.get(key) == desired:
        return
    await store.configure_project(decay=desired)
    _PROJECT_DECAY_STATE[key] = desired


async def reinforce_access(
    store: MemoryStore,
    results: list[dict[str, Any]],
    *,
    accessed_at: Optional[str] = None,
) -> int:
    """Persist access timestamps and counts for memories actually injected."""
    updated = 0
    timestamp = accessed_at or utc_now_iso()
    for item in results:
        record = MemoryRecord.from_mem0(item)
        if not record.memory_id or record.schema_version != MEMORY_SCHEMA_VERSION:
            continue
        record.last_accessed_at = timestamp
        record.access_count += 1
        try:
            await store.update(record.memory_id, metadata=record.mem0_metadata())
            updated += 1
        except Exception:
            # Access reinforcement is advisory and must never break research.
            continue
    return updated


async def list_v2_records(store: MemoryStore, user_id: str, config: Any) -> list[MemoryRecord]:
    """List and normalize only records inside the exact v2 tenant boundary."""
    raw = await store.list(user_id, filters=v2_filters(config))
    records: list[MemoryRecord] = []
    for item in raw:
        try:
            record = MemoryRecord.from_mem0(item)
        except (TypeError, ValueError):
            continue
        if (
            record.schema_version == MEMORY_SCHEMA_VERSION
            and record.app_id == advanced_app_id(config)
            and record.project_id == config.memory_project_id
        ):
            records.append(record)
    return records


async def _persist_record(store: MemoryStore, record: MemoryRecord) -> str:
    """Write an already extracted record without Mem0 inference."""
    memory_id = await store.add(
        content=record.content,
        user_id=record.user_id,
        category=record.category,
        metadata=record.mem0_metadata(),
        infer=False,
    )
    record.memory_id = memory_id
    return memory_id


def _conflict_group(candidate: MemoryCandidate, target_ids: list[str]) -> str:
    payload = "|".join(sorted(target_ids) + [candidate.category.value, candidate.content.casefold()])
    return f"conflict:{hashlib.sha256(payload.encode()).hexdigest()[:16]}"


async def write_observation(
    store: MemoryStore,
    candidate: MemoryCandidate,
    *,
    user_id: str,
    config: Any,
    run_id: str,
    decide: Optional[
        Callable[[MemoryCandidate, list[MemoryRecord]], Awaitable[MemoryConflictDecisionModel]]
    ] = None,
) -> tuple[str, str]:
    """Resolve one candidate against active same-category memories and persist it."""
    records = await list_v2_records(store, user_id, config)
    existing = [
        record for record in records
        if record.kind == MemoryKind.OBSERVATION
        and record.category == candidate.category
        and record.status == MemoryStatus.ACTIVE
    ]
    normalized = " ".join(candidate.content.casefold().split())
    duplicate = next(
        (
            record for record in existing
            if " ".join(record.content.casefold().split()) == normalized
        ),
        None,
    )
    if duplicate:
        return "NOOP", duplicate.memory_id

    decision = await decide(candidate, existing) if decide and existing else MemoryConflictDecisionModel(action="ADD")
    by_id = {record.memory_id: record for record in existing}
    targets = [by_id[memory_id] for memory_id in decision.target_memory_ids if memory_id in by_id]
    now = utc_now_iso()
    conflict_group: Optional[str] = None
    supersedes: list[str] = []

    if decision.action == "NOOP" and targets:
        return "NOOP", targets[0].memory_id
    if decision.action in {"SUPERSEDE", "TEMPORAL_CHANGE"}:
        supersedes = [record.memory_id for record in targets]
        for record in targets:
            record.status = MemoryStatus.SUPERSEDED
            if decision.action == "TEMPORAL_CHANGE":
                record.valid_to = now
            await store.update(record.memory_id, metadata=record.mem0_metadata())
    elif decision.action == "CONFLICT":
        conflict_group = _conflict_group(candidate, decision.target_memory_ids)
        for record in targets:
            record.conflict_group = conflict_group
            record.metadata["conflict_status"] = "open"
            await store.update(record.memory_id, metadata=record.mem0_metadata())

    try:
        source_kind = MemorySourceKind(candidate.source)
    except ValueError:
        source_kind = MemorySourceKind.USER_MESSAGE
    record = MemoryRecord(
        category=candidate.category,
        content=candidate.content,
        importance=candidate.importance,
        confidence=candidate.confidence,
        source_kind=source_kind,
        source_refs=list(dict.fromkeys([*candidate.source_refs, f"run:{run_id}"])),
        valid_from=now,
        conflict_group=conflict_group,
        supersedes_ids=supersedes,
        app_id=advanced_app_id(config),
        project_id=config.memory_project_id,
        agent_id=config.memory_agent_id or "lead_researcher",
        user_id=user_id,
        metadata={
            "conflict_decision": decision.action,
            "conflict_reason": decision.reason,
            "conflict_status": "open" if conflict_group else None,
        },
    )
    return decision.action, await _persist_record(store, record)


def _source_fingerprint(records: list[MemoryRecord]) -> str:
    values = sorted(
        f"{record.memory_id}:{record.status.value}:{record.content}"
        for record in records
        if record.kind != MemoryKind.PROFILE
    )
    return hashlib.sha256("\n".join(values).encode()).hexdigest()


async def maintain_user_memories(
    store: MemoryStore,
    *,
    user_id: str,
    config: Any,
    model: Any,
    model_name: str,
    model_max_tokens: int,
    runnable_config: Any,
    daily: bool = False,
    dry_run: bool = False,
    now: Optional[datetime] = None,
) -> MaintenanceResult:
    """Run reflection, canonical profile update, and soft forgetting."""
    current = now or datetime.now(timezone.utc)
    records = await list_v2_records(store, user_id, config)
    result = MaintenanceResult()
    reflected_ids = {
        source_id
        for record in records
        if record.kind == MemoryKind.REFLECTION
        for source_id in record.source_refs
    }
    unreflected = [
        record for record in records
        if record.kind == MemoryKind.OBSERVATION
        and record.status == MemoryStatus.ACTIVE
        and record.memory_id not in reflected_ids
        and not (
            daily
            and record.metadata.get("reflection_attempt_window") == current.date().isoformat()
        )
    ]
    if len(unreflected) >= config.memory_reflection_observation_threshold:
        result.trigger_reasons.append("observation_count")
    if sum(record.importance for record in unreflected) >= config.memory_reflection_importance_threshold:
        result.trigger_reasons.append("importance_sum")
    stale_before = current - timedelta(hours=config.memory_reflection_max_age_hours)
    if daily and any(_parse_datetime(record.observed_at, current) <= stale_before for record in unreflected):
        result.trigger_reasons.append("daily_stale")

    if config.memory_reflection_enabled and result.trigger_reasons and unreflected:
        async def retrieve_reflection_observations(question: str) -> list[MemoryRecord]:
            raw = await store.search(
                question,
                user_id,
                top_k=config.memory_top_k,
                filters=v2_filters(
                    config,
                    status=MemoryStatus.ACTIVE.value,
                    kind=MemoryKind.OBSERVATION.value,
                ),
                threshold=config.memory_search_threshold,
                rerank=config.memory_search_rerank,
            )
            relevant: list[MemoryRecord] = []
            for item in raw:
                try:
                    record = MemoryRecord.from_mem0(item)
                except (TypeError, ValueError):
                    continue
                if (
                    record.kind == MemoryKind.OBSERVATION
                    and record.status == MemoryStatus.ACTIVE
                    and record.app_id == advanced_app_id(config)
                    and record.project_id == config.memory_project_id
                    and record.user_id == user_id
                ):
                    relevant.append(record)
            return relevant

        reflections = await generate_reflections(
            unreflected,
            model=model,
            model_name=model_name,
            model_max_tokens=model_max_tokens,
            config=runnable_config,
            retrieve=retrieve_reflection_observations,
        )
        if daily and not dry_run:
            for record in unreflected:
                record.metadata["reflection_attempt_window"] = current.date().isoformat()
                await store.update(record.memory_id, metadata=record.mem0_metadata())
        existing_fingerprints = {
            str(record.metadata.get("reflection_fingerprint", ""))
            for record in records
            if record.kind == MemoryKind.REFLECTION
        }
        for reflection in reflections:
            source_ids = sorted(set(reflection.source_memory_ids))
            fingerprint = hashlib.sha256(
                f"{reflection.question}|{'|'.join(source_ids)}".encode()
            ).hexdigest()
            if fingerprint in existing_fingerprints:
                continue
            result.would_write += 1
            if not dry_run:
                record = MemoryRecord(
                    kind=MemoryKind.REFLECTION,
                    category=MemoryCategory.PROJECT_MEMORY,
                    content=reflection.content,
                    importance=reflection.importance,
                    confidence=reflection.confidence,
                    source_kind=MemorySourceKind.REFLECTION,
                    source_refs=source_ids,
                    app_id=advanced_app_id(config),
                    project_id=config.memory_project_id,
                    agent_id=config.memory_agent_id or "lead_researcher",
                    user_id=user_id,
                    metadata={
                        "question": reflection.question,
                        "scope": reflection.scope,
                        "reflection_fingerprint": fingerprint,
                    },
                )
                await _persist_record(store, record)
                records.append(record)
                existing_fingerprints.add(fingerprint)
                result.reflections_generated += 1

    if config.memory_profile_enabled:
        source_records = [record for record in records if record.status == MemoryStatus.ACTIVE]
        fingerprint = _source_fingerprint(source_records)
        profiles = [
            record for record in records
            if record.kind == MemoryKind.PROFILE and record.status == MemoryStatus.ACTIVE
        ]
        canonical = profiles[0] if profiles else None
        current_fingerprint = str(canonical.metadata.get("profile_source_fingerprint", "")) if canonical else ""
        if source_records and fingerprint != current_fingerprint:
            profile = await generate_research_profile(
                source_records,
                model=model,
                model_name=model_name,
                model_max_tokens=model_max_tokens,
                config=runnable_config,
            )
            content = profile.render()[: config.memory_profile_max_chars]
            result.would_write += 1
            if not dry_run:
                metadata = {
                    "canonical": True,
                    "profile_version": int(canonical.metadata.get("profile_version", 0) or 0) + 1 if canonical else 1,
                    "profile_source_fingerprint": fingerprint,
                    "profile_updated_date": current.date().isoformat(),
                    "source_refs": profile.source_memory_ids,
                }
                if canonical:
                    canonical.content = content
                    canonical.source_refs = profile.source_memory_ids
                    canonical.observed_at = current.isoformat()
                    canonical.metadata.update(metadata)
                    await store.update(
                        canonical.memory_id,
                        text=canonical.content,
                        metadata=canonical.mem0_metadata(),
                    )
                else:
                    canonical = MemoryRecord(
                        kind=MemoryKind.PROFILE,
                        category=MemoryCategory.DOMAIN_PROFILE,
                        content=content,
                        importance=8,
                        confidence=1.0,
                        source_kind=MemorySourceKind.REFLECTION,
                        source_refs=profile.source_memory_ids,
                        app_id=advanced_app_id(config),
                        project_id=config.memory_project_id,
                        agent_id=config.memory_agent_id or "lead_researcher",
                        user_id=user_id,
                        metadata=metadata,
                    )
                    await _persist_record(store, canonical)
                    records.append(canonical)
                result.profile_updated = True
        for duplicate_profile in profiles[1:]:
            result.would_write += 1
            if not dry_run:
                duplicate_profile.status = MemoryStatus.ARCHIVED
                await store.update(
                    duplicate_profile.memory_id,
                    metadata=duplicate_profile.mem0_metadata(),
                )

    if config.memory_soft_forgetting_enabled:
        for record in records:
            if record.status != MemoryStatus.ACTIVE or record.kind == MemoryKind.PROFILE:
                continue
            if record.conflict_group or record.importance > 3 or record.access_count > 1:
                continue
            last_touch = _parse_datetime(record.last_accessed_at or record.observed_at, current)
            if current - last_touch <= timedelta(days=4 * memory_half_life_days(record, config)):
                continue
            result.would_write += 1
            if not dry_run:
                record.status = MemoryStatus.ARCHIVED
                await store.update(record.memory_id, metadata=record.mem0_metadata())
                result.archived += 1

    for record in records:
        result.status_counts[record.status.value] = result.status_counts.get(record.status.value, 0) + 1
        result.category_counts[record.category.value] = result.category_counts.get(record.category.value, 0) + 1
        if record.kind == MemoryKind.PROFILE and record.status == MemoryStatus.ACTIVE:
            result.profile_version = max(
                result.profile_version,
                int(record.metadata.get("profile_version", 0) or 0),
            )
    return result
