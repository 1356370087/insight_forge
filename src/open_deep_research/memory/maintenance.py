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
from open_deep_research.memory.store import create_memory_store
from open_deep_research.model_resolution import get_configurable_model_template

# Backward-compatible patch point for tests and integrations.  The implementation
# now delegates to the shared lazy template instead of constructing a second one.
init_chat_model = get_configurable_model_template


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Maintain Mem0 schema-v2 long-term memories")
    subparsers = parser.add_subparsers(dest="command", required=True)
    daily = subparsers.add_parser("daily", help="Run reflection, profile, and forgetting maintenance")
    daily.add_argument("--user-id", help="Limit maintenance to one trusted user ID")
    daily.add_argument("--dry-run", action="store_true", help="Evaluate without writing to Mem0")
    return parser


async def _run_daily(args: argparse.Namespace, config: Configuration) -> dict[str, Any]:
    if not config.enable_memory or not config.memory_advanced_enabled:
        raise RuntimeError("ENABLE_MEMORY and MEMORY_ADVANCED_ENABLED must both be true")
    if not config.memory_project_id or not config.memory_app_id:
        raise RuntimeError("MEMORY_PROJECT_ID and MEMORY_APP_ID are required tenant boundaries")

    store = create_memory_store(config)
    await configure_advanced_store(store, config)
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


def main() -> None:
    """Run the maintenance command under a process-wide file lock."""
    load_dotenv()
    args = _parser().parse_args()
    config = Configuration.from_runnable_config()
    lock_path = Path(config.runs_dir) / "memory-maintenance.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with portalocker.Lock(str(lock_path), mode="a+b", timeout=0):
            if args.command == "daily":
                result = asyncio.run(_run_daily(args, config))
            else:  # pragma: no cover - argparse enforces known subcommands
                raise RuntimeError(f"Unknown command: {args.command}")
    except portalocker.exceptions.LockException as exc:
        raise SystemExit("Another memory maintenance process is already running") from exc
    sys.stdout.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
