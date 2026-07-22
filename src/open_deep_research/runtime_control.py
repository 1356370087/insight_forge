"""Shared cancellation and terminal-outcome contracts for one research run."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, TypeVar


class TerminationReason(str, Enum):
    """Stable reason codes propagated across runtime layers."""

    completed = "completed"
    explicit_completion = "explicit_completion"
    completion_policy_satisfied = "completion_policy_satisfied"
    no_tool_calls = "no_tool_calls"
    max_turns = "max_turns"
    budget_exhausted = "budget_exhausted"
    deadline_exceeded = "deadline_exceeded"
    cancel_requested = "cancel_requested"
    tool_protocol_violation = "tool_protocol_violation"
    context_recovery_exhausted = "context_recovery_exhausted"
    permission_denied = "permission_denied"
    model_error = "model_error"
    tool_error = "tool_error"
    report_failed = "report_failed"
    persistence_failed = "persistence_failed"
    lease_lost = "lease_lost"
    internal_error = "internal_error"


@dataclass(frozen=True, slots=True)
class PermissionDenial:
    """One sanitized governance denial admitted to run-level state."""

    tool_call_id: str
    tool_name: str
    role: str
    reason_code: str
    detail: dict[str, Any] = field(default_factory=dict)
    turn: int | None = None
    task_id: str | None = None


@dataclass(frozen=True, slots=True)
class CompletionOutcome:
    """Immutable terminal lifecycle and result-quality decision."""

    terminal_status: str
    result_status: str
    reason: TerminationReason
    stage: str
    message: str = ""
    gaps: tuple[str, ...] = ()
    budget: dict[str, Any] = field(default_factory=dict)
    permission_denials: tuple[PermissionDenial, ...] = ()


class RunCancelled(asyncio.CancelledError):
    """Raised after the active operation is cancelled and drained."""

    def __init__(self, reason: str, stage: str) -> None:
        """Store the stable request reason and observed stage."""
        super().__init__(reason)
        self.reason = reason
        self.stage = stage


T = TypeVar("T")


class CancellationScope:
    """Run-owned cooperative cancellation that drains raced awaitables."""

    def __init__(self) -> None:
        """Create an uncancelled scope."""
        self._event = asyncio.Event()
        self._reason = "cancel_requested"
        self._completion_claimed = False

    @property
    def is_cancelled(self) -> bool:
        """Return whether cancellation has been requested."""
        return self._event.is_set()

    @property
    def is_completed(self) -> bool:
        """Return whether successful completion won the terminal race."""
        return self._completion_claimed

    @property
    def reason(self) -> str:
        """Return the first cancellation reason."""
        return self._reason

    def request(self, reason: str = "cancel_requested") -> bool:
        """Request cancellation once and return whether this call won."""
        if self._completion_claimed or self._event.is_set():
            return False
        self._reason = reason
        self._event.set()
        return True

    def claim_completion(self, stage: str) -> bool:
        """Atomically claim successful termination unless cancellation won first."""
        self.checkpoint(stage)
        if self._completion_claimed:
            return False
        self._completion_claimed = True
        return True

    async def wait(self) -> None:
        """Wait until cancellation is requested."""
        await self._event.wait()

    def checkpoint(self, stage: str) -> None:
        """Raise immediately when cancellation is already requested."""
        if self._event.is_set():
            raise RunCancelled(self._reason, stage)

    async def run(
        self,
        awaitable: Awaitable[T],
        *,
        stage: str,
        timeout_seconds: float | None = None,
    ) -> T:
        """Race work against cancellation/timeout and drain every owned task."""
        self.checkpoint(stage)
        work = asyncio.ensure_future(awaitable)
        cancellation = asyncio.create_task(self._event.wait())
        timeout: asyncio.Task[None] | None = None
        if timeout_seconds is not None:
            timeout = asyncio.create_task(asyncio.sleep(timeout_seconds))
        owned = {work, cancellation}
        if timeout is not None:
            owned.add(timeout)
        try:
            done, pending = await asyncio.wait(
                owned,
                return_when=asyncio.FIRST_COMPLETED,
            )
        except BaseException:
            for task in owned:
                task.cancel()
            await asyncio.gather(*owned, return_exceptions=True)
            raise
        if work in done:
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            return await work

        work.cancel()
        await asyncio.gather(work, return_exceptions=True)
        for task in pending:
            if task is not work:
                task.cancel()
        await asyncio.gather(
            *(task for task in pending if task is not work),
            return_exceptions=True,
        )
        if cancellation in done:
            raise RunCancelled(self._reason, stage)
        raise TimeoutError(f"{stage} exceeded {timeout_seconds}s")
