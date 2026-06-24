"""Redis Pub/Sub notifications for async task-state changes."""

from __future__ import annotations

import asyncio
import json
import os
import time
from typing import Any, Optional

from pydantic import BaseModel, Field

from open_deep_research.tasks.events import EventType, ResearchEvent
from open_deep_research.tasks.state import TaskSnapshot


class TaskNotification(BaseModel):
    """Lightweight notification that tells the orchestrator state changed."""

    run_id: str
    task_id: str
    event_type: str
    status: str
    phase: str
    version: int
    updated_at: float = Field(default_factory=time.time)


def task_updates_channel(run_id: str) -> str:
    """Return the Redis channel for task updates in a research run."""
    return f"odr:run:{run_id}:task_updates"


def notification_from_snapshot(
    snapshot: TaskSnapshot, event_type: EventType | str
) -> TaskNotification:
    """Create the lightweight notification payload for a snapshot."""
    event_value = event_type.value if isinstance(event_type, EventType) else event_type
    return TaskNotification(
        run_id=snapshot.run_id,
        task_id=snapshot.task_id,
        event_type=event_value,
        status=snapshot.status.value,
        phase=snapshot.phase.value,
        version=snapshot.version,
        updated_at=snapshot.updated_at,
    )


def _redis_url(configurable: Any) -> Optional[str]:
    return getattr(configurable, "redis_url", None) or os.getenv("REDIS_URL")


async def publish_task_notification(
    configurable: Any,
    snapshot: TaskSnapshot,
    event_type: EventType | str,
) -> None:
    """Publish a lightweight Redis notification.

    Missing Redis configuration is treated as disabled so local memory-state
    runs keep working without a Redis service.
    """
    if not getattr(configurable, "task_notification_enabled", True):
        return
    redis_url = _redis_url(configurable)
    if not redis_url:
        return
    try:
        from redis import asyncio as redis_async
    except ImportError as exc:
        raise RuntimeError("Install redis>=5 to use task notifications.") from exc

    client = redis_async.from_url(redis_url, decode_responses=True)
    try:
        notification = notification_from_snapshot(snapshot, event_type)
        await client.publish(
            task_updates_channel(snapshot.run_id),
            notification.model_dump_json(),
        )
    finally:
        await client.aclose()


async def wait_for_task_notifications(
    configurable: Any,
    *,
    run_id: str,
    timeout_seconds: Optional[float] = None,
) -> list[TaskNotification]:
    """Collect task notifications from Redis during a short wait window."""
    if not getattr(configurable, "task_notification_enabled", True):
        return []
    redis_url = _redis_url(configurable)
    if not redis_url:
        return []
    wait_seconds = (
        timeout_seconds
        if timeout_seconds is not None
        else getattr(configurable, "task_notification_wait_seconds", 5)
    )
    if wait_seconds <= 0:
        return []

    try:
        from redis import asyncio as redis_async
    except ImportError as exc:
        raise RuntimeError("Install redis>=5 to use task notifications.") from exc

    client = redis_async.from_url(redis_url, decode_responses=True)
    pubsub = client.pubsub()
    notifications: list[TaskNotification] = []
    deadline = time.monotonic() + wait_seconds

    try:
        await pubsub.subscribe(task_updates_channel(run_id))
        while time.monotonic() < deadline:
            remaining = max(0.0, deadline - time.monotonic())
            message = await pubsub.get_message(
                ignore_subscribe_messages=True,
                timeout=min(0.25, remaining),
            )
            if not message:
                await asyncio.sleep(0)
                continue
            data = message.get("data")
            if not data:
                continue
            try:
                payload = json.loads(data)
                notifications.append(TaskNotification.model_validate(payload))
            except Exception:
                continue
    finally:
        await pubsub.unsubscribe(task_updates_channel(run_id))
        await pubsub.aclose()
        await client.aclose()

    return notifications


def notification_failure_event(
    *,
    task_id: str,
    run_id: str,
    phase: Optional[str],
    error: Exception,
) -> ResearchEvent:
    """Build a JSONL event for non-fatal notification publish failures."""
    return ResearchEvent(
        event_type=EventType.TASK_NOTIFICATION_FAILED,
        task_id=task_id,
        run_id=run_id,
        phase=phase,
        data={"error": str(error)},
    )
