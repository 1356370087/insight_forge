"""Durable run-scoped control inbox for multi-worker HTTP deployments."""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

_COMMAND_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class RunControlCommand(BaseModel):
    """One idempotent command addressed to an active QueryEngine."""

    command_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    run_id: str
    type: Literal["cancel", "human_action", "feedback"]
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: float = Field(default_factory=time.time)


class RunControlStore:
    """File inbox that can be written by any worker on the same host."""

    def __init__(self, run_id: str, *, runs_dir: str = ".runs") -> None:
        """Initialize the pending and processed command directories."""
        if not run_id or ".." in run_id or any(char in run_id for char in "/\\"):
            raise ValueError("Invalid run_id")
        root = Path(runs_dir).resolve() / run_id / "coordination" / "run_control"
        self.run_id = run_id
        self.pending_dir = root / "pending"
        self.processed_dir = root / "processed"

    @staticmethod
    def _write_atomic(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            with temp.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp, path)
        finally:
            if temp.exists():
                temp.unlink()

    async def enqueue(
        self,
        command_type: Literal["cancel", "human_action", "feedback"],
        payload: dict[str, Any],
        *,
        command_id: str | None = None,
    ) -> RunControlCommand:
        """Durably enqueue a command and return its stable identifier."""
        command = RunControlCommand(
            command_id=command_id or str(uuid.uuid4()),
            run_id=self.run_id,
            type=command_type,
            payload=payload,
        )
        if not _COMMAND_ID_RE.fullmatch(command.command_id) or ".." in command.command_id:
            raise ValueError("Invalid command_id")
        path = self.pending_dir / f"{command.command_id}.json"
        processed = self.processed_dir / path.name
        if path.exists() or processed.exists():
            existing_path = path if path.exists() else processed
            return RunControlCommand.model_validate_json(
                await asyncio.to_thread(existing_path.read_text, encoding="utf-8")
            )
        await asyncio.to_thread(
            self._write_atomic,
            path,
            command.model_dump(mode="json"),
        )
        return command

    async def pending(self) -> list[RunControlCommand]:
        """Return queued commands in creation order."""
        if not self.pending_dir.exists():
            return []
        commands = [
            RunControlCommand.model_validate_json(
                await asyncio.to_thread(path.read_text, encoding="utf-8")
            )
            for path in self.pending_dir.glob("*.json")
        ]
        return sorted(commands, key=lambda command: (command.created_at, command.command_id))

    async def ack(self, command: RunControlCommand) -> None:
        """Move a successfully consumed command to the processed directory."""
        source = self.pending_dir / f"{command.command_id}.json"
        target = self.processed_dir / source.name
        if not source.exists():
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(os.replace, source, target)
