"""Checkpoint sink adapters for recoverable Query loop state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from open_deep_research.agents.query_state import QueryLoopState


@dataclass(frozen=True, slots=True)
class RunContextQueryCheckpointSink:
    """Persist Query state in the authoritative run journal."""

    store: Any
    channel: Literal["lead", "supervisor"] = "supervisor"

    async def save(self, state: QueryLoopState) -> None:
        """Append and fsync one Query state transition."""
        await self.store.save_query_state(state, channel=self.channel)


@dataclass(frozen=True, slots=True)
class CallbackQueryCheckpointSink:
    """Adapt a checkpoint callback owned by the async task runtime."""

    callback: Any

    async def save(self, state: QueryLoopState) -> None:
        """Pass the serializable snapshot to the task checkpoint owner."""
        await self.callback(state)
