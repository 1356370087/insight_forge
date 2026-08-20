"""Container entrypoint for Docker-isolated Researcher SubAgents."""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import traceback
from pathlib import Path

from langchain_core.messages import messages_from_dict

from open_deep_research.sandbox.wire import (
    WORKER_EXIT_CODE_PATH,
    SandboxTaskPayloadV1,
    SandboxTaskResultV1,
)

DEFAULT_PAYLOAD_PATH = "/workspace/input/task_payload.json"
DEFAULT_RESULT_PATH = "/workspace/output/result.json"
DEFAULT_LOG_PATH = "/workspace/logs/worker.log"


def _setup_logging() -> None:
    log_path = Path(os.getenv("SANDBOX_LOG_PATH", DEFAULT_LOG_PATH))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(),
        ],
        force=True,
    )


def _safe_error(exc: BaseException) -> str:
    return str(exc)[:1000]


async def _run() -> int:
    _setup_logging()
    payload_path = Path(os.getenv("SANDBOX_TASK_PAYLOAD_PATH", DEFAULT_PAYLOAD_PATH))
    result_path = Path(os.getenv("SANDBOX_RESULT_PATH", DEFAULT_RESULT_PATH))

    try:
        payload = SandboxTaskPayloadV1.model_validate_json(
            payload_path.read_text(encoding="utf-8")
        )
        task_id = payload.task_id
        state = dict(payload.researcher_state)
        state["researcher_messages"] = messages_from_dict(
            state.get("researcher_messages", [])
        )

        configurable = dict(payload.runtime_config)
        configurable["sandbox_enabled"] = False
        configurable["enable_memory"] = False
        configurable["memory_auto_write"] = False

        config = {
            "configurable": configurable,
            "metadata": {
                "run_id": payload.run_id,
                "task_id": payload.task_id,
                "run_fence_token": payload.fence_token,
                "sandbox_profile_id": payload.profile_id,
                "sandbox_policy_digest": payload.policy_digest,
            },
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

        completed = SandboxTaskResultV1(
            task_id=task_id,
            status="completed",
            compressed_research=compressed_research,
            raw_notes=raw_notes,
            metrics=result_metrics,
            candidate_registry=list(result.get("candidate_registry", [])),
            document_registry=list(result.get("document_registry", [])),
            evidence_registry=list(result.get("evidence_registry", [])),
            web_research_iterations=list(result.get("web_research_iterations", [])),
            permission_denials=list(result.get("permission_denials", [])),
            coverage_ledger=dict(result.get("coverage_ledger", {})),
            completion_decision=dict(result.get("completion_decision", {})),
        )
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(completed.model_dump_json(), encoding="utf-8")
        logging.info("Sandbox worker completed task_id=%s", task_id)
        return 0
    except Exception as exc:
        logging.exception("Sandbox worker failed")
        logging.error("Sandbox traceback: %s", traceback.format_exc(limit=8))
        failed = SandboxTaskResultV1(
            task_id=locals().get("task_id", "unknown"),
            status="failed",
            error=_safe_error(exc),
            metrics={"sources_read": 0, "tool_calls": 0, "failures": 1},
        )
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(failed.model_dump_json(), encoding="utf-8")
        return 1


def main() -> None:
    """Run, publish a tmpfs-ready sentinel, then wait for safe collection."""
    terminal_code: dict[str, int | None] = {"value": None}

    def _terminate(_signum: int, _frame: object) -> None:
        code = terminal_code["value"]
        raise SystemExit(code if code is not None else 143)

    signal.signal(signal.SIGTERM, _terminate)
    signal.signal(signal.SIGINT, _terminate)
    code = asyncio.run(_run())
    terminal_code["value"] = code
    sentinel = Path(WORKER_EXIT_CODE_PATH)
    sentinel.write_text(str(code), encoding="ascii")
    logging.info("Sandbox worker result is ready for Controller collection")
    while True:
        signal.pause()


if __name__ == "__main__":
    main()
