"""Shared, stateless helpers for Supervisor tool definitions."""

from __future__ import annotations

from typing import Any

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field, create_model

from open_deep_research.configuration import Configuration
from open_deep_research.quality.contract import ResearchCoverageContract
from open_deep_research.run_context import RunContextStore
from open_deep_research.tasks.events import JSONLEventWriter
from open_deep_research.tools.base import ToolContext


def bind_run_context_fence(
    store: RunContextStore,
    config: RunnableConfig,
) -> RunContextStore:
    """Bind a context store to propagated Lead ownership when available."""
    metadata = config.get("metadata", {})
    token = metadata.get("run_fence_token")
    owner_id = metadata.get("run_lease_owner_id")
    if token is not None and owner_id:
        store.bind_fence_token(
            int(token),
            str(owner_id),
            advance_manifest=False,
        )
    return store


def coverage_bound_input_schema(
    base_schema: type[BaseModel],
    contract: ResearchCoverageContract | None,
) -> type[BaseModel]:
    """Expose legal coverage IDs as an enum while retaining call validation."""
    if contract is None or not contract.requirements:
        return base_schema
    requirement_field = base_schema.model_fields["requirement_ids"]
    return create_model(
        base_schema.__name__,
        __base__=base_schema,
        requirement_ids=(
            list[str],
            Field(
                default_factory=list,
                description=requirement_field.description,
                json_schema_extra={
                    "items": {
                        "type": "string",
                        "enum": list(contract.requirement_ids()),
                    }
                },
            ),
        ),
    )


def validate_requirement_ids(
    requirement_ids: list[str],
    contract: ResearchCoverageContract | None,
    *,
    required: bool,
) -> list[str]:
    """Deduplicate and validate delegated coverage requirement IDs."""
    normalized = list(dict.fromkeys(str(item) for item in requirement_ids))
    if contract is None:
        return normalized
    if required and not normalized:
        raise ValueError("coverage_requirement_ids_required")
    known_ids = set(contract.requirement_ids())
    unknown = [item for item in normalized if item not in known_ids]
    if unknown:
        raise ValueError("unknown_coverage_requirement_ids:" + ",".join(unknown))
    return normalized


def tool_call_payload(name: str, input: BaseModel, context: ToolContext) -> dict[str, Any]:
    """Translate validated input for the persistent task handlers."""
    return {
        "name": name,
        "args": input.model_dump(),
        "id": context.tool_call_id,
    }


def event_writer(
    configurable: Configuration,
    run_id: str,
) -> JSONLEventWriter | None:
    """Create the optional per-invocation task event writer."""
    if not configurable.event_log_enabled:
        return None
    return JSONLEventWriter(run_id=run_id, runs_dir=configurable.runs_dir)
