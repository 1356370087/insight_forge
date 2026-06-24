"""Per-run domain-approval registry for the enforced egress allowlist.

Holds, per ``run_id``, the set of domains the supervisor has approved or denied
for the current run, plus the pending :class:`asyncio.Future` objects that block
in-process researcher tool calls until a decision arrives.

This registry is a module-level singleton (like :mod:`tasks.registry`) so every
part of the process — the tool-governance layer that *requests* a decision and
the supervisor tool handler that *records* one — shares the same state. It is
**not persisted and not cross-run**: decisions live only for the current run and
are cleared when a task reaches a terminal status or is cancelled.
"""

from __future__ import annotations

import asyncio
from typing import Optional


class DomainRequest:
    """A pending domain-confirmation request (live, non-serializable)."""

    def __init__(self, run_id: str, domain: str, tool_name: str) -> None:
        """Store the run, domain, and tool that triggered the confirmation request."""
        self.run_id = run_id
        self.domain = domain
        self.tool_name = tool_name
        # Created lazily on the first await so it binds to the running loop.
        self.future: Optional[asyncio.Future[bool]] = None
        # Set by record_decision when no future exists yet (decision arrived
        # before anyone awaited). Lets wait() return without creating a future.
        self._pre_result: Optional[bool] = None

    async def wait(self) -> bool:
        """Block until a decision is recorded. Lazily binds the future to the loop."""
        if self._pre_result is not None:
            return self._pre_result
        if self.future is None:
            self.future = asyncio.get_running_loop().create_future()
        return await self.future


class DomainApprovalRegistry:
    """Per-run approval set plus pending-request futures.

    Not persisted, not cross-run. The live ``asyncio.Future`` lives here (never
    on a :class:`TaskRecord` / :class:`TaskSnapshot`, which must stay serializable).
    """

    def __init__(self) -> None:
        """Initialize empty per-run allowed/denied sets and pending-request maps."""
        self._allowed: dict[str, set[str]] = {}
        self._denied: dict[str, set[str]] = {}
        self._pending: dict[tuple[str, str], DomainRequest] = {}

    def is_allowed(self, run_id: str, domain: str) -> Optional[bool]:
        """Return ``True`` if approved, ``False`` if denied, ``None`` if undecided."""
        d = domain.lower()
        if d in self._denied.get(run_id, set()):
            return False
        if d in self._allowed.get(run_id, set()):
            return True
        return None

    def request_decision(self, run_id: str, domain: str, tool_name: str) -> DomainRequest:
        """Get-or-create the pending ``DomainRequest`` for ``(run_id, domain)``.

        The ``Future`` is created lazily inside :meth:`DomainRequest.wait` on the
        running loop, so this method is safe to call from a non-async context.
        """
        key = (run_id, domain.lower())
        if key not in self._pending:
            self._pending[key] = DomainRequest(run_id, domain.lower(), tool_name)
        return self._pending[key]

    def record_decision(
        self, run_id: str, domain: str, allowed: bool
    ) -> Optional[DomainRequest]:
        """Record the decision, cache it per-run, and resolve the pending future.

        Returns the resolved :class:`DomainRequest` (or ``None`` if there was no
        pending request for this domain — e.g. a pre-approval).
        """
        d = domain.lower()
        bucket = self._allowed if allowed else self._denied
        bucket.setdefault(run_id, set()).add(d)
        key = (run_id, d)
        req = self._pending.pop(key, None)
        if req is not None:
            if req.future is not None and not req.future.done():
                req.future.set_result(allowed)
            elif req.future is None:
                # Decision arrived before anyone awaited: stash the result so
                # wait() returns it without creating a (loop-less) future.
                req._pre_result = allowed
        return req

    def get_pending(self, run_id: str) -> list[DomainRequest]:
        """Return all pending (not-yet-decided) requests for a run."""
        return [r for (rid, _), r in self._pending.items() if rid == run_id]

    def clear_run(self, run_id: str) -> None:
        """Drop all decisions and cancel pending requests for a run (terminal/cancel)."""
        self._allowed.pop(run_id, None)
        self._denied.pop(run_id, None)
        for key in list(self._pending):
            if key[0] == run_id:
                req = self._pending.pop(key, None)
                if req is not None and req.future is not None and not req.future.done():
                    req.future.cancel()

    # Useful for tests and inspection ----------------------------------------

    def allowed_for_run(self, run_id: str) -> set[str]:
        """Return a copy of the approved domains for a run."""
        return set(self._allowed.get(run_id, set()))

    def denied_for_run(self, run_id: str) -> set[str]:
        """Return a copy of the denied domains for a run."""
        return set(self._denied.get(run_id, set()))


_approvals: Optional[DomainApprovalRegistry] = None


def get_domain_approval_registry() -> DomainApprovalRegistry:
    """Return the module-level :class:`DomainApprovalRegistry` singleton."""
    global _approvals
    if _approvals is None:
        _approvals = DomainApprovalRegistry()
    return _approvals


def reset_domain_approval_registry() -> None:
    """Reset the singleton (tests only)."""
    global _approvals
    _approvals = None
