"""Process-local API admission controls for the single-worker deployment."""

from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import dataclass


@dataclass
class _Window:
    started_at: float
    count: int


class FixedWindowRateLimiter:
    """Thread-safe fixed-window counter with bounded stale-state cleanup."""

    def __init__(self) -> None:
        """Create an empty process-local counter map."""
        self._windows: dict[str, _Window] = {}
        self._lock = threading.Lock()

    def allow(
        self,
        key: str,
        limit: int,
        *,
        window_seconds: float = 60,
        now: float | None = None,
    ) -> tuple[bool, int]:
        """Return allowed and whole-second Retry-After values."""
        if limit <= 0:
            return True, 0
        current = time.monotonic() if now is None else now
        with self._lock:
            window = self._windows.get(key)
            if window is None or current - window.started_at >= window_seconds:
                self._windows[key] = _Window(current, 1)
                self._cleanup(current, window_seconds)
                return True, 0
            if window.count >= limit:
                retry_after = max(1, int(window_seconds - (current - window.started_at) + 0.999))
                return False, retry_after
            window.count += 1
            return True, 0

    def _cleanup(self, now: float, window_seconds: float) -> None:
        if len(self._windows) < 1000:
            return
        cutoff = now - window_seconds * 2
        self._windows = {
            key: value
            for key, value in self._windows.items()
            if value.started_at >= cutoff
        }

    def clear(self) -> None:
        """Clear process-local counters for tests and controlled restarts."""
        with self._lock:
            self._windows.clear()


class ConnectionLimiter:
    """Async-safe process-wide concurrent connection counter."""

    def __init__(self) -> None:
        """Create a zeroed connection counter."""
        self._active = 0
        self._lock = asyncio.Lock()

    async def acquire(self, limit: int) -> bool:
        """Reserve a slot, returning false when the configured cap is full."""
        if limit <= 0:
            return True
        async with self._lock:
            if self._active >= limit:
                return False
            self._active += 1
            return True

    async def release(self, limit: int) -> None:
        """Release a previously reserved slot."""
        if limit <= 0:
            return
        async with self._lock:
            self._active = max(0, self._active - 1)

    @property
    def active(self) -> int:
        """Return the current process-local connection count."""
        return self._active


__all__ = ["ConnectionLimiter", "FixedWindowRateLimiter"]
