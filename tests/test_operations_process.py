"""Process-boundary acceptance tests for operational lifecycle governance."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _run_child(source: str, tmp_path: Path) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["SPEC_PROCESS_TMP"] = str(tmp_path)
    return subprocess.run(
        [sys.executable, "-c", source],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


def test_ws2_memory_bound_holds_across_a_real_process(tmp_path) -> None:
    """A worker process evicts terminal runs and bounds each event buffer."""
    completed = _run_child(
        r"""
import asyncio
import json
from types import SimpleNamespace

from open_deep_research import server

async def main():
    config = {"configurable": {
        "inflight_event_buffer_size": 7,
        "inflight_run_memory_retention_seconds": 0,
        "max_inflight_runs_in_memory": 10,
    }}
    tasks = []
    max_events = 0
    for index in range(50):
        record = server._new_run_record(
            run_id=f"run-{index}",
            engine=SimpleNamespace(config=config),
            status="completed",
            config=config,
        )
        for sequence in range(100):
            record.events.append({"sequence": sequence})
        max_events = max(max_events, len(record.events))
        server._remember_run(record, config)
        server._schedule_run_eviction(record, config)
        task = server._run_eviction_tasks.get(record.run_id)
        if task is not None:
            tasks.append(task)
    await asyncio.gather(*tasks, return_exceptions=True)
    print(json.dumps({"runs": len(server._runs), "max_events": max_events}))

asyncio.run(main())
""",
        tmp_path,
    )

    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout.strip().splitlines()[-1])
    assert result == {"runs": 0, "max_events": 7}


@pytest.mark.skipif(os.name == "nt", reason="Windows terminate() bypasses SIGTERM handlers")
def test_ws4_sigterm_drains_manifest_and_trace_in_child_process(tmp_path) -> None:
    """A real SIGTERM drives an in-flight run to durable interrupted state."""
    env = os.environ.copy()
    env["SPEC_PROCESS_TMP"] = str(tmp_path)
    child = subprocess.Popen(
        [
            sys.executable,
            "-c",
            r"""
import asyncio
import os
import signal
from pathlib import Path
from types import SimpleNamespace

from open_deep_research import server
from open_deep_research.observability.core import SQLiteTraceStore
from open_deep_research.run_context import RunContextStore

async def main():
    root = Path(os.environ["SPEC_PROCESS_TMP"])
    runs_dir = root / "runs"
    trace_path = root / "traces.sqlite3"
    run_id = "sigterm-run"
    config = {
        "configurable": {
            "runs_dir": str(runs_dir),
            "trace_store_path": str(trace_path),
            "observability_enabled": True,
            "sqlite_observability_enabled": True,
            "prometheus_enabled": False,
        },
        "metadata": {"run_id": run_id},
    }
    store = RunContextStore(run_id, runs_dir=str(runs_dir))
    store.initialize("owner", config)
    store._update_manifest(status="running")
    SQLiteTraceStore(str(trace_path)).start_run(run_id, "owner", {})
    engine = SimpleNamespace(config=config, context_store=store)
    record = server._new_run_record(run_id=run_id, engine=engine, config=config)
    record.task = asyncio.create_task(asyncio.sleep(3600))
    server._remember_run(record, config)
    stopping = asyncio.Event()
    asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, stopping.set)
    (root / "ready").write_text("ready", encoding="utf-8")
    await stopping.wait()
    await server._drain_inflight_runs(5)

asyncio.run(main())
""",
        ],
        cwd=PROJECT_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    ready = tmp_path / "ready"
    deadline = time.monotonic() + 15
    while not ready.exists() and child.poll() is None and time.monotonic() < deadline:
        time.sleep(0.05)
    assert ready.exists(), child.communicate(timeout=5)[1]

    child.terminate()
    stdout, stderr = child.communicate(timeout=15)
    assert child.returncode == 0, f"{stdout}\n{stderr}"

    manifest = json.loads(
        (tmp_path / "runs" / "sigterm-run" / "context" / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["status"] == "interrupted"
    verifier = _run_child(
        r"""
import json
import os
from pathlib import Path
from open_deep_research.observability.core import SQLiteTraceStore
root = Path(os.environ["SPEC_PROCESS_TMP"])
print(json.dumps(SQLiteTraceStore(str(root / "traces.sqlite3")).get_run("sigterm-run")))
""",
        tmp_path,
    )
    assert verifier.returncode == 0, verifier.stderr
    assert json.loads(verifier.stdout.strip().splitlines()[-1])["status"] == "interrupted"


def test_ws5_retention_reclaims_files_in_a_real_process(tmp_path) -> None:
    """The retention worker removes expired durable artifacts process-locally."""
    completed = _run_child(
        r"""
import asyncio
import json
import os
import time
from pathlib import Path

from open_deep_research import server
from open_deep_research.configuration import Configuration
from open_deep_research.observability.core import SQLiteTraceStore
from open_deep_research.run_context import RunContextStore

async def main():
    root = Path(os.environ["SPEC_PROCESS_TMP"])
    runs_dir = root / "runs"
    trace_path = root / "traces.sqlite3"
    run_id = "expired-process-run"
    store = RunContextStore(run_id, runs_dir=str(runs_dir))
    store.initialize("owner", {"configurable": {"runs_dir": str(runs_dir)}})
    store._update_manifest(status="completed", ended_at=time.time() - 3 * 86400)
    SQLiteTraceStore(str(trace_path)).start_run(run_id, "owner", {})
    config = Configuration(
        runs_dir=str(runs_dir),
        trace_store_path=str(trace_path),
        run_retention_days=1,
        trace_retention_days=0,
        prometheus_enabled=False,
    )
    result = await server._run_retention_sweep(config)
    trace = SQLiteTraceStore(str(trace_path)).get_run(run_id)
    print(json.dumps({
        "deleted": result["deleted_by_age"],
        "directory_exists": (runs_dir / run_id).exists(),
        "trace_exists": trace is not None,
    }))

asyncio.run(main())
""",
        tmp_path,
    )

    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout.strip().splitlines()[-1])
    assert result == {
        "deleted": 1,
        "directory_exists": False,
        "trace_exists": False,
    }
