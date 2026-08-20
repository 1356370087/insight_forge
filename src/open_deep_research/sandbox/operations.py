"""Durable, idempotent model-operation journal owned by the API."""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any, Literal

import portalocker
from pydantic import BaseModel, ConfigDict, Field

from open_deep_research.tasks.mailbox import (
    atomic_write_json,
    read_json_file,
    validate_component,
)

_RETRYABLE_FAILED_ERROR_TYPES = frozenset(
    {"rate_limited", "transient", "model_unavailable"}
)


class ModelOperationRecord(BaseModel):
    """One logical operation across one or more deterministic physical attempts."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    run_id: str
    task_id: str
    stage: str
    logical_operation_id: str
    physical_attempt_id: str
    physical_attempts: list[str] = Field(default_factory=list)
    status: Literal["reserved", "dispatched", "completed", "failed", "uncertain"]
    outcome: dict[str, Any] | None = None
    error_type: str | None = None
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)
    version: int = 1


class ModelOperationStore:
    """File-locked API authority for model operation recovery."""

    def __init__(self, run_id: str, *, runs_dir: str = ".runs") -> None:
        """Bind operation records to one validated run directory."""
        self.run_id = validate_component(run_id, "run_id")
        self.root = Path(runs_dir).resolve() / self.run_id / "sandbox" / "model_operations"

    @staticmethod
    def _key(logical_operation_id: str) -> str:
        return hashlib.sha256(logical_operation_id.encode()).hexdigest()

    def _paths(self, logical_operation_id: str) -> tuple[Path, Path]:
        key = self._key(logical_operation_id)
        return self.root / f"{key}.json", self.root / f"{key}.lock"

    def get(self, logical_operation_id: str) -> ModelOperationRecord | None:
        """Return the current logical operation record, if present."""
        path, lock = self._paths(logical_operation_id)
        self.root.mkdir(parents=True, exist_ok=True)
        with portalocker.Lock(str(lock), mode="a+b", timeout=30):
            if not path.exists():
                return None
            return ModelOperationRecord.model_validate(read_json_file(path))

    def reserve(
        self,
        *,
        task_id: str,
        stage: str,
        logical_operation_id: str,
        physical_attempt_id: str,
    ) -> ModelOperationRecord:
        """Reserve an idempotent physical attempt under one logical operation."""
        path, lock = self._paths(logical_operation_id)
        self.root.mkdir(parents=True, exist_ok=True)
        with portalocker.Lock(str(lock), mode="a+b", timeout=30):
            if path.exists():
                current = ModelOperationRecord.model_validate(read_json_file(path))
                if physical_attempt_id not in current.physical_attempts:
                    retryable_failure = (
                        current.status == "failed"
                        and current.error_type in _RETRYABLE_FAILED_ERROR_TYPES
                    )
                    if current.status in {"completed", "uncertain"} or (
                        current.status == "failed" and not retryable_failure
                    ):
                        raise ValueError("model_operation_already_terminal")
                    current.physical_attempts.append(physical_attempt_id)
                    current.physical_attempt_id = physical_attempt_id
                    if retryable_failure:
                        current.status = "reserved"
                        current.outcome = None
                        current.error_type = None
                    current.updated_at = time.time()
                    current.version += 1
                    atomic_write_json(path, current.model_dump(mode="json"))
                return current
            record = ModelOperationRecord(
                run_id=self.run_id,
                task_id=validate_component(task_id, "task_id"),
                stage=stage,
                logical_operation_id=logical_operation_id,
                physical_attempt_id=physical_attempt_id,
                physical_attempts=[physical_attempt_id],
                status="reserved",
            )
            atomic_write_json(path, record.model_dump(mode="json"))
            return record

    def transition(
        self,
        logical_operation_id: str,
        *,
        expected: set[str],
        status: Literal["dispatched", "completed", "failed", "uncertain"],
        outcome: dict[str, Any] | None = None,
        error_type: str | None = None,
    ) -> ModelOperationRecord:
        """Apply a checked journal state transition."""
        path, lock = self._paths(logical_operation_id)
        self.root.mkdir(parents=True, exist_ok=True)
        with portalocker.Lock(str(lock), mode="a+b", timeout=30):
            if not path.exists():
                raise KeyError(logical_operation_id)
            record = ModelOperationRecord.model_validate(read_json_file(path))
            if record.status == status:
                if record.outcome != outcome or record.error_type != error_type:
                    raise ValueError("model_operation_terminal_mismatch")
                return record
            if record.status not in expected:
                raise ValueError(
                    f"model_operation_invalid_transition:{record.status}->{status}"
                )
            record.status = status
            record.outcome = outcome
            record.error_type = error_type
            record.updated_at = time.time()
            record.version += 1
            atomic_write_json(path, record.model_dump(mode="json"))
            return record
