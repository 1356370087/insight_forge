"""Explicit dependency seam for Supervisor tool assembly."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Coroutine, Mapping
from dataclasses import dataclass, field
from typing import Any

from langchain_core.runnables import RunnableConfig

from open_deep_research.quality.contract import (
    ResearchCoverageContract,
    ResearchRiskProfile,
)

ResearcherInvoker = Callable[
    [dict[str, Any], RunnableConfig], Coroutine[Any, Any, dict[str, Any]]
]
HandoffEvaluator = Callable[..., Awaitable[Any]]


async def _unconfigured_dependency(*args: Any, **kwargs: Any) -> Any:
    del args, kwargs
    raise RuntimeError("Supervisor runtime dependency was not configured")


@dataclass(frozen=True, slots=True)
class SupervisorToolDeps:
    """Narrow, immutable inputs used by module-level Supervisor tool calls.

    Async task tools intentionally resolve the task registry and teammate pool via
    their module-level infrastructure locators. Those lifecycle-managed singletons
    are not per-invocation behavior, so they are deliberately excluded from this
    dependency object.
    """

    enable_async_research: bool = False
    sandbox_enabled: bool = False
    coverage_contract: ResearchCoverageContract | None = None
    risk_profile: ResearchRiskProfile = field(
        default_factory=lambda: ResearchRiskProfile(level="standard")
    )
    memory_context: Any = None
    research_artifact_refs: Mapping[str, Mapping[str, Any]] = field(
        default_factory=dict
    )
    handoff_assessments: tuple[Mapping[str, Any], ...] = ()
    researcher_ainvoke: ResearcherInvoker = _unconfigured_dependency
    evaluate_handoff: HandoffEvaluator = _unconfigured_dependency
