"""Process-local circuit breaking for model availability and first-token latency."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import threading
import time
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class ModelCircuitState(str, Enum):
    """Availability states for one model circuit."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitFailureKind(str, Enum):
    """Terminal failure kinds that contribute to a model circuit."""

    MODEL_UNAVAILABLE = "model_unavailable"
    RATE_LIMITED = "rate_limited"
    TRANSIENT = "transient"


@dataclass(frozen=True, slots=True)
class ModelCircuitPolicy:
    """Frozen decision thresholds shared by one process-local breaker."""

    failure_threshold: int = 5
    failure_window_seconds: float = 300.0
    open_cooldown_seconds: float = 60.0
    max_cooldown_seconds: float = 600.0
    slow_ratio_threshold: float = 0.5
    slow_min_samples: int = 4
    first_packet_probe: str = "shadow"
    slow_first_packet_threshold_seconds: float = 8.0

    @property
    def fingerprint(self) -> str:
        """Return a stable, non-secret fingerprint for policy compatibility."""
        payload = {
            "failure_threshold": int(self.failure_threshold),
            "failure_window_seconds": float(self.failure_window_seconds),
            "open_cooldown_seconds": float(self.open_cooldown_seconds),
            "max_cooldown_seconds": float(self.max_cooldown_seconds),
            "slow_ratio_threshold": float(self.slow_ratio_threshold),
            "slow_min_samples": int(self.slow_min_samples),
            "first_packet_probe": str(self.first_packet_probe),
            "slow_first_packet_threshold_seconds": float(
                self.slow_first_packet_threshold_seconds
            ),
        }
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


def model_circuit_policy_from_configuration(
    configuration: Any,
) -> ModelCircuitPolicy:
    """Build a circuit policy from a Configuration-like object."""
    return ModelCircuitPolicy(
        failure_threshold=int(configuration.model_circuit_failure_threshold),
        failure_window_seconds=float(
            configuration.model_circuit_failure_window_seconds
        ),
        open_cooldown_seconds=float(
            configuration.model_circuit_open_cooldown_seconds
        ),
        max_cooldown_seconds=float(
            configuration.model_circuit_max_cooldown_seconds
        ),
        slow_ratio_threshold=float(
            configuration.model_circuit_slow_ratio_threshold
        ),
        slow_min_samples=int(configuration.model_circuit_slow_min_samples),
        first_packet_probe=str(configuration.model_first_packet_probe),
        slow_first_packet_threshold_seconds=float(
            configuration.model_slow_first_packet_threshold_seconds
        ),
    )


class CircuitOpenError(RuntimeError):
    """Signal that a provider request was rejected by the local circuit."""

    def __init__(
        self,
        model_id: str,
        state: ModelCircuitState,
        *,
        retry_after_seconds: float | None,
        reason: str,
    ) -> None:
        """Initialize a bounded circuit rejection without provider details."""
        super().__init__(f"model circuit {state.value}: {model_id} ({reason})")
        self.model_id = model_id
        self.state = state
        self.retry_after_seconds = retry_after_seconds
        self.reason = reason


@dataclass(frozen=True, slots=True)
class CircuitPermit:
    """Authorize one high-level model call against a circuit generation."""

    model_id: str
    generation: int
    is_probe: bool = False


@dataclass(frozen=True, slots=True)
class CircuitTransition:
    """Describe one state transition for best-effort observability."""

    model_id: str
    from_state: ModelCircuitState
    to_state: ModelCircuitState
    reason: str
    timestamp: float
    failure_count: int
    slow_count: int
    sample_count: int
    slow_ratio: float
    cooldown_seconds: float
    forced_probe: bool = False


@dataclass(frozen=True, slots=True)
class ModelCircuitSnapshot:
    """Expose one consistent breaker snapshot without mutable internals."""

    model_id: str
    state: ModelCircuitState
    policy_fingerprint: str
    failure_count: int
    slow_count: int
    sample_count: int
    slow_ratio: float
    opened_at: float | None
    cooldown_seconds: float
    retry_after_seconds: float | None
    probe_in_flight: bool

    @property
    def rejecting(self) -> bool:
        """Return whether a new call would be rejected by this snapshot."""
        if self.state is ModelCircuitState.HALF_OPEN:
            return self.probe_in_flight
        if self.state is ModelCircuitState.OPEN:
            return bool(
                self.retry_after_seconds is None
                or self.retry_after_seconds > 0
            )
        return False


class ModelCircuitBreaker:
    """Maintain one model's sliding windows and three-state circuit."""

    def __init__(
        self,
        model_id: str,
        policy: ModelCircuitPolicy,
        *,
        now: Callable[[], float] = time.monotonic,
    ) -> None:
        """Initialize a closed model circuit."""
        self.model_id = model_id
        self.policy = policy
        self._now = now
        self._state = ModelCircuitState.CLOSED
        self._failures: deque[float] = deque()
        self._slow_samples: deque[tuple[float, bool]] = deque()
        self._opened_at: float | None = None
        self._current_cooldown_seconds = policy.open_cooldown_seconds
        self._consecutive_reopens = 0
        self._generation = 0
        self._probe_in_flight = False
        self._lock = asyncio.Lock()

    @property
    def policy_fingerprint(self) -> str:
        """Return the immutable policy fingerprint bound to this breaker."""
        return self.policy.fingerprint

    def policy_matches(self, policy: ModelCircuitPolicy) -> bool:
        """Return whether a caller can safely share this breaker."""
        return self.policy_fingerprint == policy.fingerprint

    def _prune(self, now: float) -> None:
        cutoff = now - self.policy.failure_window_seconds
        while self._failures and self._failures[0] < cutoff:
            self._failures.popleft()
        while self._slow_samples and self._slow_samples[0][0] < cutoff:
            self._slow_samples.popleft()

    def _counts(self) -> tuple[int, int, int, float]:
        failure_count = len(self._failures)
        sample_count = len(self._slow_samples)
        slow_count = sum(1 for _timestamp, slow in self._slow_samples if slow)
        slow_ratio = slow_count / sample_count if sample_count else 0.0
        return failure_count, slow_count, sample_count, slow_ratio

    def _transition(
        self,
        from_state: ModelCircuitState,
        to_state: ModelCircuitState,
        reason: str,
        now: float,
        *,
        forced_probe: bool = False,
    ) -> CircuitTransition:
        failure_count, slow_count, sample_count, slow_ratio = self._counts()
        return CircuitTransition(
            model_id=self.model_id,
            from_state=from_state,
            to_state=to_state,
            reason=reason,
            timestamp=now,
            failure_count=failure_count,
            slow_count=slow_count,
            sample_count=sample_count,
            slow_ratio=slow_ratio,
            cooldown_seconds=self._current_cooldown_seconds,
            forced_probe=forced_probe,
        )

    def _open(
        self,
        reason: str,
        now: float,
        *,
        from_state: ModelCircuitState,
        probe_failure: bool = False,
    ) -> CircuitTransition:
        if probe_failure:
            self._consecutive_reopens += 1
            self._current_cooldown_seconds = min(
                self.policy.max_cooldown_seconds,
                self.policy.open_cooldown_seconds
                * (2**self._consecutive_reopens),
            )
        else:
            self._current_cooldown_seconds = self.policy.open_cooldown_seconds
        self._state = ModelCircuitState.OPEN
        self._opened_at = now
        self._probe_in_flight = False
        self._generation += 1
        return self._transition(from_state, self._state, reason, now)

    async def before_call(self) -> tuple[CircuitPermit, CircuitTransition | None]:
        """Authorize one high-level call or raise a fast circuit rejection."""
        async with self._lock:
            now = self._now()
            self._prune(now)
            transition: CircuitTransition | None = None
            if self._state is ModelCircuitState.OPEN:
                opened_at = self._opened_at if self._opened_at is not None else now
                retry_after = max(
                    0.0,
                    self._current_cooldown_seconds - (now - opened_at),
                )
                if retry_after > 0:
                    raise CircuitOpenError(
                        self.model_id,
                        self._state,
                        retry_after_seconds=retry_after,
                        reason="cooldown_active",
                    )
                previous = self._state
                self._state = ModelCircuitState.HALF_OPEN
                self._probe_in_flight = True
                self._generation += 1
                transition = self._transition(
                    previous,
                    self._state,
                    "cooldown_elapsed",
                    now,
                )
                return (
                    CircuitPermit(self.model_id, self._generation, is_probe=True),
                    transition,
                )
            if self._state is ModelCircuitState.HALF_OPEN:
                if self._probe_in_flight:
                    raise CircuitOpenError(
                        self.model_id,
                        self._state,
                        retry_after_seconds=None,
                        reason="half_open_probe_in_flight",
                    )
                self._probe_in_flight = True
                return (
                    CircuitPermit(self.model_id, self._generation, is_probe=True),
                    None,
                )
            return CircuitPermit(self.model_id, self._generation), None

    async def force_half_open(self) -> CircuitTransition | None:
        """Make an OPEN circuit probeable for the all-open fallback policy."""
        async with self._lock:
            now = self._now()
            self._prune(now)
            if self._state is not ModelCircuitState.OPEN:
                return None
            previous = self._state
            self._state = ModelCircuitState.HALF_OPEN
            self._probe_in_flight = False
            self._generation += 1
            return self._transition(
                previous,
                self._state,
                "all_candidates_open",
                now,
                forced_probe=True,
            )

    async def record_success(
        self,
        permit: CircuitPermit,
        *,
        ttft_seconds: float | None = None,
    ) -> CircuitTransition | None:
        """Record one successful high-level call and possibly close/open."""
        async with self._lock:
            now = self._now()
            self._prune(now)
            if permit.model_id != self.model_id or permit.generation != self._generation:
                return None
            if permit.is_probe and self._state is ModelCircuitState.HALF_OPEN:
                previous = self._state
                self._failures.clear()
                self._slow_samples.clear()
                self._state = ModelCircuitState.CLOSED
                self._opened_at = None
                self._probe_in_flight = False
                self._consecutive_reopens = 0
                self._current_cooldown_seconds = self.policy.open_cooldown_seconds
                self._generation += 1
                return self._transition(previous, self._state, "probe_succeeded", now)
            if self._state is not ModelCircuitState.CLOSED:
                return None
            if ttft_seconds is not None:
                is_slow = (
                    ttft_seconds
                    > self.policy.slow_first_packet_threshold_seconds
                )
                self._slow_samples.append((now, is_slow))
            if self.policy.first_packet_probe != "enforced":
                return None
            _failures, slow_count, sample_count, slow_ratio = self._counts()
            if (
                sample_count >= self.policy.slow_min_samples
                and slow_ratio >= self.policy.slow_ratio_threshold
                and slow_count > 0
            ):
                return self._open(
                    "slow_first_packet_ratio",
                    now,
                    from_state=ModelCircuitState.CLOSED,
                )
            return None

    async def record_failure(
        self,
        permit: CircuitPermit,
        *,
        failure_kind: CircuitFailureKind,
    ) -> CircuitTransition | None:
        """Record one exhausted availability failure and possibly open."""
        async with self._lock:
            now = self._now()
            self._prune(now)
            if permit.model_id != self.model_id or permit.generation != self._generation:
                return None
            self._failures.append(now)
            if permit.is_probe and self._state is ModelCircuitState.HALF_OPEN:
                return self._open(
                    f"probe_failed:{failure_kind.value}",
                    now,
                    from_state=ModelCircuitState.HALF_OPEN,
                    probe_failure=True,
                )
            if (
                self._state is ModelCircuitState.CLOSED
                and len(self._failures) >= self.policy.failure_threshold
            ):
                return self._open(
                    f"failure_threshold:{failure_kind.value}",
                    now,
                    from_state=ModelCircuitState.CLOSED,
                )
            return None

    async def record_inconclusive(
        self,
        permit: CircuitPermit,
    ) -> CircuitTransition | None:
        """Release a probe after a terminal error that must not be counted."""
        async with self._lock:
            now = self._now()
            self._prune(now)
            if permit.model_id != self.model_id or permit.generation != self._generation:
                return None
            if not permit.is_probe or self._state is not ModelCircuitState.HALF_OPEN:
                return None
            previous = self._state
            self._state = ModelCircuitState.OPEN
            self._opened_at = now
            self._probe_in_flight = False
            self._generation += 1
            return self._transition(
                previous,
                self._state,
                "probe_inconclusive",
                now,
            )

    async def snapshot(self) -> ModelCircuitSnapshot:
        """Return one window-pruned immutable snapshot."""
        async with self._lock:
            now = self._now()
            self._prune(now)
            failure_count, slow_count, sample_count, slow_ratio = self._counts()
            retry_after: float | None = None
            if self._state is ModelCircuitState.OPEN:
                opened_at = self._opened_at if self._opened_at is not None else now
                retry_after = max(
                    0.0,
                    self._current_cooldown_seconds - (now - opened_at),
                )
            return ModelCircuitSnapshot(
                model_id=self.model_id,
                state=self._state,
                policy_fingerprint=self.policy_fingerprint,
                failure_count=failure_count,
                slow_count=slow_count,
                sample_count=sample_count,
                slow_ratio=slow_ratio,
                opened_at=self._opened_at,
                cooldown_seconds=self._current_cooldown_seconds,
                retry_after_seconds=retry_after,
                probe_in_flight=self._probe_in_flight,
            )


class ModelCircuitRegistry:
    """Own process-local breakers keyed by normalized model identifier."""

    def __init__(self) -> None:
        """Initialize an empty registry."""
        self._breakers: dict[str, ModelCircuitBreaker] = {}
        self._lock = threading.Lock()
        self._policy_mismatch_warned: set[tuple[str, str, str]] = set()

    def get_or_create(
        self,
        model_id: str,
        policy: ModelCircuitPolicy,
        *,
        now: Callable[[], float] = time.monotonic,
    ) -> ModelCircuitBreaker | None:
        """Return a matching breaker, failing open on a policy mismatch."""
        normalized = str(model_id).strip()
        if not normalized:
            return None
        with self._lock:
            breaker = self._breakers.get(normalized)
            if breaker is None:
                breaker = ModelCircuitBreaker(normalized, policy, now=now)
                self._breakers[normalized] = breaker
            elif not breaker.policy_matches(policy):
                # Fail open, but warn once per model/policy pair so a long-lived
                # process does not flood logs on every call from the mismatching
                # configuration.
                mismatch_key = (
                    normalized,
                    breaker.policy_fingerprint,
                    policy.fingerprint,
                )
                if mismatch_key not in self._policy_mismatch_warned:
                    self._policy_mismatch_warned.add(mismatch_key)
                    logger.warning(
                        "Model circuit policy mismatch; failing open for model %s "
                        "(existing=%s requested=%s)",
                        normalized,
                        breaker.policy_fingerprint[:12],
                        policy.fingerprint[:12],
                    )
                return None
            return breaker

    def get(
        self,
        model_id: str,
        policy: ModelCircuitPolicy | None = None,
    ) -> ModelCircuitBreaker | None:
        """Return an existing compatible breaker without creating it."""
        with self._lock:
            breaker = self._breakers.get(str(model_id).strip())
        if breaker is not None and policy is not None and not breaker.policy_matches(policy):
            return None
        return breaker

    async def snapshots(self) -> list[ModelCircuitSnapshot]:
        """Return consistent snapshots for all known breakers."""
        with self._lock:
            breakers = tuple(self._breakers.values())
        return [await breaker.snapshot() for breaker in breakers]

    async def select_candidate_index(
        self,
        model_ids: Sequence[str],
        policy: ModelCircuitPolicy,
        *,
        start_index: int = 0,
        force_if_all_open: bool = True,
    ) -> tuple[int, CircuitTransition | None]:
        """Select the first callable candidate or force the oldest OPEN probe."""
        if not model_ids:
            raise ValueError("model circuit candidate chain is empty")
        start = min(max(0, int(start_index)), len(model_ids) - 1)
        snapshots: list[tuple[int, ModelCircuitBreaker, ModelCircuitSnapshot]] = []
        for index in range(start, len(model_ids)):
            breaker = self.get(model_ids[index], policy)
            if breaker is None:
                return index, None
            snapshot = await breaker.snapshot()
            if not snapshot.rejecting:
                return index, None
            snapshots.append((index, breaker, snapshot))
        if not force_if_all_open:
            return start, None
        forceable = [
            item
            for item in snapshots
            if item[2].state is ModelCircuitState.OPEN
        ]
        if not forceable:
            return start, None
        index, breaker, _snapshot = min(
            forceable,
            key=lambda item: (
                item[2].opened_at
                if item[2].opened_at is not None
                else float("inf")
            ),
        )
        return index, await breaker.force_half_open()


_model_circuit_registry = ModelCircuitRegistry()


def get_model_circuit_registry() -> ModelCircuitRegistry:
    """Return the process-local model circuit registry."""
    return _model_circuit_registry


def _reset_model_circuit_registry() -> None:
    """Reset the process-local registry for deterministic tests."""
    global _model_circuit_registry
    _model_circuit_registry = ModelCircuitRegistry()
