"""Internal CLI for daily advanced-memory maintenance."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import portalocker
from dotenv import load_dotenv

from open_deep_research.configuration import Configuration
from open_deep_research.memory.lifecycle import (
    configure_advanced_store,
    maintain_user_memories,
)
from open_deep_research.memory.store import NoopMemoryStore, create_memory_store
from open_deep_research.models.resolution import get_configurable_model_template

# Backward-compatible patch point for tests and integrations.  The implementation
# now delegates to the shared lazy template instead of constructing a second one.
init_chat_model = get_configurable_model_template


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Maintain Mem0 schema-v2 long-term memories")
    subparsers = parser.add_subparsers(dest="command", required=True)
    daily = subparsers.add_parser("daily", help="Run reflection, profile, and forgetting maintenance")
    daily.add_argument("--user-id", help="Limit maintenance to one trusted user ID")
    daily.add_argument("--dry-run", action="store_true", help="Evaluate without writing to Mem0")
    daily.add_argument(
        "--loop",
        action="store_true",
        help="Keep the lock and repeat daily maintenance until terminated",
    )
    daily.add_argument(
        "--interval-hours",
        type=float,
        default=24,
        help="Hours between loop iterations (default: 24)",
    )
    decay = subparsers.add_parser(
        "configure-decay",
        help="Explicitly apply the project-wide Mem0 Platform decay setting",
    )
    decay_group = decay.add_mutually_exclusive_group(required=True)
    decay_group.add_argument("--enabled", dest="decay", action="store_true")
    decay_group.add_argument("--disabled", dest="decay", action="store_false")
    return parser


async def _run_daily(args: argparse.Namespace, config: Configuration) -> dict[str, Any]:
    if not config.enable_memory or not config.memory_advanced_enabled:
        raise RuntimeError("ENABLE_MEMORY and MEMORY_ADVANCED_ENABLED must both be true")
    if not config.memory_project_id or not config.memory_app_id:
        raise RuntimeError("MEMORY_PROJECT_ID and MEMORY_APP_ID are required tenant boundaries")

    store = create_memory_store(config)
    if isinstance(store, NoopMemoryStore):
        raise RuntimeError("Configured memory backend is unavailable")
    user_ids = [args.user_id] if args.user_id else await store.list_users()
    if not user_ids:
        raise RuntimeError("No users returned by Mem0; OSS maintenance requires --user-id")

    model = get_configurable_model_template()
    runnable_config: dict[str, Any] = {
        "configurable": {},
        "metadata": {"run_id": "memory-maintenance-daily"},
    }
    summaries: dict[str, Any] = {}
    for user_id in user_ids:
        result = await maintain_user_memories(
            store,
            user_id=user_id,
            config=config,
            model=model,
            model_name=config.research_model,
            model_max_tokens=config.research_model_max_tokens,
            runnable_config=runnable_config,
            daily=True,
            dry_run=args.dry_run,
        )
        summaries[user_id] = result.model_dump()
    return {"dry_run": args.dry_run, "users": summaries}


async def _run_configure_decay(args: argparse.Namespace, config: Configuration) -> dict[str, Any]:
    """Apply one explicit deployment-level project setting change."""
    if config.memory_provider != "platform":
        raise RuntimeError("configure-decay is only available for the Mem0 Platform provider")
    store = create_memory_store(config)
    if isinstance(store, NoopMemoryStore):
        raise RuntimeError("Configured memory backend is unavailable")
    await configure_advanced_store(store, config, decay=bool(args.decay))
    return {"provider": "platform", "decay": bool(args.decay)}


async def _run_daily_loop(
    args: argparse.Namespace,
    config: Configuration,
    *,
    sleep: Any = asyncio.sleep,
    max_iterations: int | None = None,
) -> list[dict[str, Any]]:
    """Run daily maintenance repeatedly while the caller holds the file lock."""
    if args.interval_hours <= 0:
        raise ValueError("--interval-hours must be greater than zero")
    results: list[dict[str, Any]] = []
    while max_iterations is None or len(results) < max_iterations:
        result = await _run_daily(args, config)
        results.append(result)
        sys.stdout.write(json.dumps(result, ensure_ascii=False) + "\n")
        sys.stdout.flush()
        if max_iterations is not None and len(results) >= max_iterations:
            break
        await sleep(args.interval_hours * 3600)
    return results


def _command_lock_path(config: Configuration, command: str) -> Path:
    """Return a lock scoped to the side effect performed by one CLI command."""
    filename = (
        "memory-configure-decay.lock"
        if command == "configure-decay"
        else "memory-maintenance.lock"
    )
    return Path(config.runs_dir) / filename


def main() -> None:
    """Run each maintenance command under its own process-wide file lock."""
    load_dotenv()
    args = _parser().parse_args()
    config = Configuration.from_runnable_config()
    lock_path = _command_lock_path(config, args.command)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with portalocker.Lock(str(lock_path), mode="a+b", timeout=0):
            if args.command == "daily":
                if args.loop:
                    asyncio.run(_run_daily_loop(args, config))
                    result = None
                else:
                    result = asyncio.run(_run_daily(args, config))
            elif args.command == "configure-decay":
                result = asyncio.run(_run_configure_decay(args, config))
            else:  # pragma: no cover - argparse enforces known subcommands
                raise RuntimeError(f"Unknown command: {args.command}")
    except portalocker.exceptions.LockException as exc:
        raise SystemExit("Another memory maintenance process is already running") from exc
    if result is not None:
        sys.stdout.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
