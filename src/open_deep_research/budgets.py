"""Durable file-backed run budget reservations."""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import portalocker
from pydantic import BaseModel, Field


class ModelPriceNotConfigured(RuntimeError):
    """Raised when a cost cap is configured but a model has no price entry."""


class DeadlineExceeded(RuntimeError):
    """Raised when the run's wall-clock deadline has elapsed."""


class BudgetDimension(str, Enum):
    """Resource counters enforceable by the run ledger."""

    MODEL_CALLS = "model_calls"
    TOOL_CALLS = "tool_calls"
    INPUT_TOKENS = "input_tokens"
    OUTPUT_TOKENS = "output_tokens"
    COST_MICRO_USD = "cost_micro_usd"
    FETCH_CALLS = "fetch_calls"


@dataclass(frozen=True, slots=True)
class RunBudgetPolicy:
    """Optional maximums for each durable run resource."""

    max_model_calls: int | None = None
    max_tool_calls: int | None = None
    max_input_tokens: int | None = None
    max_output_tokens: int | None = None
    max_cost_micro_usd: int | None = None
    max_fetch_calls: int | None = None

    def limit_for(self, dimension: BudgetDimension) -> int | None:
        """Return the configured limit for one dimension."""
        return {
            BudgetDimension.MODEL_CALLS: self.max_model_calls,
            BudgetDimension.TOOL_CALLS: self.max_tool_calls,
            BudgetDimension.INPUT_TOKENS: self.max_input_tokens,
            BudgetDimension.OUTPUT_TOKENS: self.max_output_tokens,
            BudgetDimension.COST_MICRO_USD: self.max_cost_micro_usd,
            BudgetDimension.FETCH_CALLS: self.max_fetch_calls,
        }[dimension]


class BudgetReservation(BaseModel):
    """One idempotent pre-operation resource reservation."""

    reservation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    operation_key: str
    dimension: BudgetDimension
    reserved: int
    actual: int | None = None
    status: str = "reserved"
    created_at: float = Field(default_factory=time.time)


class RunBudgetSnapshot(BaseModel):
    """Current settled plus outstanding usage."""

    model_calls: int = 0
    tool_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_micro_usd: int = 0
    fetch_calls: int = 0
    exhausted: bool = False
    exhausted_dimension: BudgetDimension | None = None


class BudgetExhausted(RuntimeError):
    """Raised before an operation would exceed a configured limit."""

    def __init__(self, dimension: BudgetDimension) -> None:
        """Store the exhausted resource dimension."""
        super().__init__(f"budget_exhausted:{dimension.value}")
        self.dimension = dimension


class RunBudgetLedger:
    """Persist idempotent budget reservations under an inter-process lock."""

    def __init__(
        self,
        run_id: str,
        *,
        runs_dir: str = ".runs",
        policy: RunBudgetPolicy | None = None,
    ) -> None:
        """Initialize durable paths and the immutable policy."""
        if not run_id or ".." in run_id or any(char in run_id for char in "/\\"):
            raise ValueError("Invalid run_id")
        self.run_id = run_id
        self.policy = policy or RunBudgetPolicy()
        root = Path(runs_dir).resolve() / run_id / "context"
        self.path = root / "budget_ledger.json"
        self.lock_path = root / "budget_ledger.lock"

    def _load_unlocked(self) -> dict:
        if not self.path.exists():
            return {"schema_version": 1, "reservations": {}, "exhausted": None}
        return json.loads(self.path.read_text(encoding="utf-8"))

    def _write_unlocked(self, payload: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.path.with_name(f".{self.path.name}.{uuid.uuid4().hex}.tmp")
        try:
            with temp.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp, self.path)
        finally:
            if temp.exists():
                temp.unlink()

    @staticmethod
    def _used(data: dict, dimension: BudgetDimension) -> int:
        return sum(
            int(item.get("actual") if item.get("actual") is not None else item["reserved"])
            for item in data["reservations"].values()
            if item["dimension"] == dimension.value
            and item.get("status") in {"reserved", "settled", "uncertain"}
        )

    def reserve(
        self,
        operation_key: str,
        dimension: BudgetDimension,
        amount: int,
    ) -> BudgetReservation:
        """Atomically reserve a positive amount before external work."""
        if amount <= 0:
            raise ValueError("reservation amount must be positive")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with portalocker.Lock(str(self.lock_path), mode="a+b", timeout=30):
            data = self._load_unlocked()
            existing = data["reservations"].get(operation_key)
            if existing is not None:
                reservation = BudgetReservation.model_validate(existing)
                if reservation.dimension != dimension or reservation.reserved != amount:
                    raise ValueError("operation reservation mismatch")
                return reservation
            limit = self.policy.limit_for(dimension)
            if limit is not None and self._used(data, dimension) + amount > limit:
                data["exhausted"] = dimension.value
                self._write_unlocked(data)
                raise BudgetExhausted(dimension)
            reservation = BudgetReservation(
                operation_key=operation_key,
                dimension=dimension,
                reserved=amount,
            )
            data["reservations"][operation_key] = reservation.model_dump(mode="json")
            self._write_unlocked(data)
            return reservation

    def settle(self, operation_key: str, *, actual: int) -> BudgetReservation:
        """Settle an existing reservation to provider-reported usage once."""
        if actual < 0:
            raise ValueError("actual usage cannot be negative")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with portalocker.Lock(str(self.lock_path), mode="a+b", timeout=30):
            data = self._load_unlocked()
            reservation = BudgetReservation.model_validate(
                data["reservations"][operation_key]
            )
            if reservation.status == "settled":
                if reservation.actual != actual:
                    raise ValueError(
                        "operation already settled with different usage"
                    )
                return reservation
            reservation.actual = actual
            reservation.status = "settled"
            data["reservations"][operation_key] = reservation.model_dump(mode="json")
            dimension = reservation.dimension
            limit = self.policy.limit_for(dimension)
            if limit is not None and self._used(data, dimension) > limit:
                data["exhausted"] = dimension.value
            self._write_unlocked(data)
            return reservation

    def release(self, operation_key: str) -> BudgetReservation | None:
        """Release a reservation when an operation is known not to consume it."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with portalocker.Lock(str(self.lock_path), mode="a+b", timeout=30):
            data = self._load_unlocked()
            raw = data["reservations"].get(operation_key)
            if raw is None:
                return None
            reservation = BudgetReservation.model_validate(raw)
            if reservation.status == "settled":
                return reservation
            reservation.actual = 0
            reservation.status = "released"
            data["reservations"][operation_key] = reservation.model_dump(mode="json")
            self._write_unlocked(data)
            return reservation

    def mark_uncertain(self, operation_key: str) -> BudgetReservation | None:
        """Keep a conservative reservation for an indeterminate remote attempt."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with portalocker.Lock(str(self.lock_path), mode="a+b", timeout=30):
            data = self._load_unlocked()
            raw = data["reservations"].get(operation_key)
            if raw is None:
                return None
            reservation = BudgetReservation.model_validate(raw)
            if reservation.status not in {"settled", "released"}:
                reservation.status = "uncertain"
                data["reservations"][operation_key] = reservation.model_dump(
                    mode="json"
                )
                self._write_unlocked(data)
            return reservation

    def outstanding_by_dimension(self) -> dict[str, int]:
        """Return unresolved conservative reservations grouped by dimension."""
        if not self.path.exists():
            return {dimension.value: 0 for dimension in BudgetDimension}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with portalocker.Lock(str(self.lock_path), mode="a+b", timeout=30):
            data = self._load_unlocked()
        return {
            dimension.value: sum(
                int(item.get("actual") if item.get("actual") is not None else item["reserved"])
                for item in data["reservations"].values()
                if item["dimension"] == dimension.value
                and item.get("status") in {"reserved", "uncertain"}
            )
            for dimension in BudgetDimension
        }

    def snapshot(self) -> RunBudgetSnapshot:
        """Return current conservative usage across all dimensions."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with portalocker.Lock(str(self.lock_path), mode="a+b", timeout=30):
            data = self._load_unlocked()
        values = {
            dimension.value: self._used(data, dimension)
            for dimension in BudgetDimension
        }
        exhausted = data.get("exhausted")
        return RunBudgetSnapshot(
            **values,
            exhausted=exhausted is not None,
            exhausted_dimension=(
                BudgetDimension(exhausted) if exhausted is not None else None
            ),
        )


def budget_policy_from_config(configurable) -> RunBudgetPolicy | None:
    """Build a run budget policy, or ``None`` when no resource cap is set."""
    policy = RunBudgetPolicy(
        max_model_calls=getattr(configurable, "max_run_model_calls", None),
        max_tool_calls=getattr(configurable, "max_run_tool_calls", None),
        max_input_tokens=getattr(configurable, "max_run_input_tokens", None),
        max_output_tokens=getattr(configurable, "max_run_output_tokens", None),
        max_cost_micro_usd=getattr(configurable, "max_run_cost_micro_usd", None),
        max_fetch_calls=getattr(configurable, "max_fetches_per_run", None),
    )
    if all((
        policy.max_model_calls is None,
        policy.max_tool_calls is None,
        policy.max_input_tokens is None,
        policy.max_output_tokens is None,
        policy.max_cost_micro_usd is None,
        policy.max_fetch_calls is None,
    )):
        return None
    return policy


def budget_ledger_for(
    configurable,
    run_id: str,
    *,
    created_at: float | None = None,
) -> RunBudgetLedger | None:
    """Return a durable ledger for the run, or ``None`` when uncapped."""
    policy = budget_policy_from_config(configurable)
    if policy is None:
        return None
    return RunBudgetLedger(run_id, runs_dir=configurable.runs_dir, policy=policy)


def estimate_cost_micro_usd(
    model_name: str,
    input_tokens: int,
    output_tokens: int,
    costs: dict[str, dict[str, float]],
) -> int:
    """Convert token usage to integer micro-USD using per-model rates.

    Matching uses the exact model id first, then the provider-stripped id, then
    the longest key that is a prefix of either form.
    """
    if not costs:
        raise ModelPriceNotConfigured(
            f"Run cost cap is configured but no price table was provided for {model_name!r}."
        )
    stripped = model_name.split(":", 1)[-1]
    candidates = {model_name, stripped}
    matched: dict[str, float] | None = None
    for key in sorted(costs, key=len, reverse=True):
        if any(candidate.startswith(key) for candidate in candidates):
            matched = costs[key]
            break
    if matched is None:
        raise ModelPriceNotConfigured(
            f"Run cost cap is configured but model {model_name!r} has no price entry."
        )
    input_rate = float(matched.get("input", 0.0))
    output_rate = float(matched.get("output", 0.0))
    usd = (input_tokens * input_rate + output_tokens * output_rate) / 1_000_000
    return int(round(usd * 1_000_000))


@dataclass(frozen=True, slots=True)
class BudgetGate:
    """Unified deadline + durable-reservation seam for model/tool boundaries."""

    ledger: RunBudgetLedger | None
    deadline_at: float | None = None
    cost_pricing: dict[str, dict[str, float]] = None  # type: ignore[assignment]

    @classmethod
    def from_config(
        cls,
        configurable,
        run_id: str,
        *,
        started_at: float | None = None,
    ) -> BudgetGate:
        """Resolve the gate from configuration; absent limits disable it."""
        ledger = budget_ledger_for(configurable, run_id)
        deadline_seconds = getattr(configurable, "run_deadline_seconds", None)
        deadline_at = (
            (started_at if started_at is not None else time.time()) + deadline_seconds
            if deadline_seconds
            else None
        )
        return cls(
            ledger=ledger,
            deadline_at=deadline_at,
            cost_pricing=getattr(configurable, "model_costs_per_million", {}) or {},
        )

    @property
    def enabled(self) -> bool:
        """Return whether any budget or deadline constraint is active."""
        return self.ledger is not None or self.deadline_at is not None

    def check_deadline(self, stage: str) -> None:
        """Raise :class:`DeadlineExceeded` when the run deadline elapsed."""
        if self.deadline_at is not None and time.time() >= self.deadline_at:
            raise DeadlineExceeded(f"{stage}: run deadline exceeded")

    def _reserve(self, key: str, dimension: BudgetDimension, amount: int) -> None:
        if self.ledger is not None:
            self.ledger.reserve(key, dimension, amount)

    def _settle(self, key: str, actual: int) -> None:
        if self.ledger is not None:
            try:
                self.ledger.settle(key, actual=actual)
            except KeyError:
                # The dimension was not pre-reserved (limit not configured); skip.
                pass

    def reserve_model_call(
        self,
        op_key: str,
        *,
        estimated_input_tokens: int,
        estimated_output_tokens: int,
        model_name: str,
    ) -> None:
        """Pre-reserve one model call plus estimated token/cost usage."""
        self.check_deadline("model")
        reserved_keys: list[str] = []
        try:
            self._reserve(op_key, BudgetDimension.MODEL_CALLS, 1)
            reserved_keys.append(op_key)
            policy = self.ledger.policy if self.ledger is not None else None
            if policy is not None and policy.max_input_tokens is not None:
                self._reserve(
                    f"{op_key}:input",
                    BudgetDimension.INPUT_TOKENS,
                    max(1, estimated_input_tokens),
                )
                reserved_keys.append(f"{op_key}:input")
            if policy is not None and policy.max_output_tokens is not None:
                self._reserve(
                    f"{op_key}:output",
                    BudgetDimension.OUTPUT_TOKENS,
                    max(1, estimated_output_tokens),
                )
                reserved_keys.append(f"{op_key}:output")
            if policy is not None and policy.max_cost_micro_usd is not None:
                estimated_cost = estimate_cost_micro_usd(
                    model_name,
                    max(1, estimated_input_tokens),
                    max(1, estimated_output_tokens),
                    self.cost_pricing,
                )
                if estimated_cost > 0:
                    self._reserve(
                        f"{op_key}:cost",
                        BudgetDimension.COST_MICRO_USD,
                        estimated_cost,
                    )
                    reserved_keys.append(f"{op_key}:cost")
        except Exception:
            if self.ledger is not None:
                for key in reserved_keys:
                    self.ledger.release(key)
            raise

    def settle_model_call(
        self,
        op_key: str,
        *,
        input_tokens: int,
        output_tokens: int,
        model_name: str,
    ) -> None:
        """Settle provider-reported usage for one model call."""
        policy = self.ledger.policy if self.ledger is not None else None
        self._settle(op_key, 1)
        if policy is not None and policy.max_input_tokens is not None:
            self._settle(f"{op_key}:input", input_tokens)
        if policy is not None and policy.max_output_tokens is not None:
            self._settle(f"{op_key}:output", output_tokens)
        if policy is not None and policy.max_cost_micro_usd is not None:
            actual_cost = estimate_cost_micro_usd(
                model_name,
                input_tokens,
                output_tokens,
                self.cost_pricing,
            )
            self._settle(f"{op_key}:cost", actual_cost)

    def fail_model_call(self, op_key: str, *, uncertain: bool) -> None:
        """Finalize reservations for a failed physical model attempt."""
        if self.ledger is None:
            return
        self._settle(op_key, 1)
        for suffix in ("input", "output", "cost"):
            key = f"{op_key}:{suffix}"
            if uncertain:
                self.ledger.mark_uncertain(key)
            else:
                self.ledger.release(key)

    def outstanding_reservations(self) -> dict[str, int]:
        """Expose unresolved reservations for the business usage projection."""
        if self.ledger is None:
            return {dimension.value: 0 for dimension in BudgetDimension}
        return self.ledger.outstanding_by_dimension()

    def reserve_tool_call(self, op_key: str) -> None:
        """Pre-reserve one tool execution."""
        self.check_deadline("tool")
        self._reserve(op_key, BudgetDimension.TOOL_CALLS, 1)

    def reserve_fetch_call(self, op_key: str) -> None:
        """Pre-reserve one webpage fetch against the run fetch budget."""
        self._reserve(op_key, BudgetDimension.FETCH_CALLS, 1)
