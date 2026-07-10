"""Container entrypoint for Docker-isolated Researcher SubAgents."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import traceback
from pathlib import Path
from typing import Any

from langchain_core.messages import messages_from_dict

DEFAULT_PAYLOAD_PATH = "/workspace/input/task_payload.json"
DEFAULT_RESULT_PATH = "/workspace/output/result.json"
DEFAULT_LOG_PATH = "/workspace/logs/worker.log"


def _setup_logging() -> None:
    log_path = Path(os.getenv("SANDBOX_LOG_PATH", DEFAULT_LOG_PATH))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=str(log_path),
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _safe_error(exc: BaseException) -> str:
    return str(exc)[:1000]


async def _run() -> int:
    _setup_logging()
    payload_path = Path(os.getenv("SANDBOX_TASK_PAYLOAD_PATH", DEFAULT_PAYLOAD_PATH))
    result_path = Path(os.getenv("SANDBOX_RESULT_PATH", DEFAULT_RESULT_PATH))

    try:
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
        task_id = payload["task_id"]
        state = payload["researcher_state"]
        state["researcher_messages"] = messages_from_dict(
            state.get("researcher_messages", [])
        )

        configurable = dict(payload.get("configurable", {}))
        configurable["enable_docker_sandbox"] = False
        configurable["enable_memory"] = False
        configurable["memory_auto_write"] = False

        config = {
            "configurable": configurable,
            "metadata": payload.get("metadata", {}),
        }

        logging.info("Starting sandbox worker for task_id=%s", task_id)
        from open_deep_research.agents.deep_researcher import researcher_runtime
        from open_deep_research.observability import bind_span_context

        with bind_span_context(
            str(config["metadata"].get("run_id") or "default"),
            config["metadata"].get("trace_parent_span_id"),
            config["metadata"].get("langfuse_parent_span_id"),
        ):
            result = await researcher_runtime.ainvoke(state, config)
        raw_notes = result.get("raw_notes", [])
        compressed_research = result.get("compressed_research", "")
        result_metrics = dict(result.get("metrics", {}))
        result_metrics.setdefault("sources_read", 0)
        result_metrics.setdefault(
            "tool_calls", int(result.get("tool_call_iterations", 0) or 0)
        )
        result_metrics["failures"] = 0

        _write_json(
            result_path,
            {
                "task_id": task_id,
                "status": "completed",
                "compressed_research": compressed_research,
                "raw_notes": raw_notes,
                "metrics": result_metrics,
            },
        )
        logging.info("Sandbox worker completed task_id=%s", task_id)
        return 0
    except Exception as exc:
        logging.exception("Sandbox worker failed")
        _write_json(
            result_path,
            {
                "task_id": locals().get("task_id", "unknown"),
                "status": "failed",
                "error": _safe_error(exc),
                "traceback": traceback.format_exc(limit=8),
                "metrics": {
                    "sources_read": 0,
                    "tool_calls": 0,
                    "failures": 1,
                },
            },
        )
        return 1


def main() -> None:
    """Run the sandbox worker entrypoint."""
    raise SystemExit(asyncio.run(_run()))


if __name__ == "__main__":
    main()

