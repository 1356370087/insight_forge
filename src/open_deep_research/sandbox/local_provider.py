"""Linux/WSL2 local command sandbox used by optional developer tools."""

from __future__ import annotations

import asyncio
import os
import shutil
import sys
from contextlib import suppress
from pathlib import Path
from typing import Any

from langchain_core.runnables import RunnableConfig

from open_deep_research.configuration import Configuration

MAX_COMMAND_OUTPUT_BYTES = 1_000_000


def task_workspace(config: RunnableConfig) -> Path:
    """Resolve the task-owned workspace without accepting a caller path root."""
    if os.getenv("SANDBOX_TASK_TOKEN"):
        return Path(
            os.getenv("INSIGHTFORGE_TASK_WORKSPACE_ROOT", "/workspace/work")
        ).resolve()
    configurable = Configuration.from_runnable_config(config)
    metadata = config.get("metadata", {})
    run_id = str(metadata.get("run_id") or "default")
    task_id = str(metadata.get("task_id") or "developer")
    root = (Path(configurable.runs_dir).resolve() / run_id / "workspaces" / task_id).resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def safe_workspace_path(config: RunnableConfig, value: str) -> Path:
    """Resolve a path below the task workspace and reject existing symlinks."""
    root = task_workspace(config)
    supplied = Path(value)
    if supplied.is_absolute() or ".." in supplied.parts:
        raise ValueError("sandbox_path_outside_workspace")
    lexical = root.joinpath(*[part for part in supplied.parts if part not in {"", "."}])
    cursor = root
    for part in lexical.relative_to(root).parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ValueError("sandbox_path_symlink_denied")
    candidate = lexical.resolve(strict=False)
    if root not in candidate.parents and candidate != root:
        raise ValueError("sandbox_path_outside_workspace")
    return candidate


def developer_tools_enabled(config: RunnableConfig, permission: str) -> bool:
    """Require both administrator Profile selection and an RBAC capability."""
    profile_id = os.getenv("SANDBOX_PROFILE_ID") or str(
        config.get("configurable", {}).get("sandbox_profile_id") or ""
    )
    if profile_id != "developer-workspace":
        return False
    auth_user = config.get("configurable", {}).get("langgraph_auth_user") or {}
    permissions = set(auth_user.get("permissions") or []) if isinstance(auth_user, dict) else set()
    return permission in permissions


class BubblewrapSandboxProvider:
    """Execute one bounded shell command with inherited OS-level restrictions."""

    @staticmethod
    def _build_bubblewrap_argv(
        *,
        bwrap: str,
        workspace: Path,
        working_dir: Path,
        command: str,
    ) -> list[str]:
        """Build a minimal runtime view without exposing the host root tree."""
        relative_cwd = working_dir.relative_to(workspace)
        sandbox_cwd = Path("/workspace").joinpath(relative_cwd).as_posix()
        return [
            bwrap,
            "--die-with-parent",
            "--new-session",
            "--unshare-user",
            "--unshare-pid",
            "--unshare-ipc",
            "--unshare-uts",
            "--unshare-net",
            "--ro-bind",
            "/usr",
            "/usr",
            "--ro-bind-try",
            "/bin",
            "/bin",
            "--ro-bind-try",
            "/sbin",
            "/sbin",
            "--ro-bind-try",
            "/lib",
            "/lib",
            "--ro-bind-try",
            "/lib64",
            "/lib64",
            "--dir",
            "/workspace",
            "--bind",
            workspace.as_posix(),
            "/workspace",
            "--tmpfs",
            "/tmp",
            "--tmpfs",
            "/run",
            "--tmpfs",
            "/home",
            "--tmpfs",
            "/root",
            "--tmpfs",
            "/mnt",
            "--dir",
            "/etc",
            "--dir",
            "/etc/ssl",
            "--ro-bind-try",
            "/etc/ssl/certs",
            "/etc/ssl/certs",
            "--proc",
            "/proc",
            "--dev",
            "/dev",
            "--chdir",
            sandbox_cwd,
            "/bin/sh",
            "-lc",
            command,
        ]

    async def run(
        self,
        command: str,
        *,
        config: RunnableConfig,
        cwd: str | None = None,
        timeout_seconds: int = 120,
    ) -> dict[str, Any]:
        """Execute a command inside Docker or a host bubblewrap boundary."""
        workspace = task_workspace(config)
        working_dir = safe_workspace_path(config, cwd or ".")
        if not working_dir.is_dir():
            raise ValueError("sandbox_command_cwd_not_directory")
        shim = None
        environment = {
            "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
            "LANG": os.environ.get("LANG", "C.UTF-8"),
            "HOME": "/tmp/sandbox-home",
            "TMPDIR": "/tmp",
            "SANDBOX_TASK_TOKEN": os.environ.get("SANDBOX_TASK_TOKEN", ""),
        }
        if os.getenv("SANDBOX_TASK_TOKEN"):
            shim = await asyncio.create_subprocess_exec(
                sys.executable,
                "-m",
                "open_deep_research.sandbox.proxy_shim",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
                env=environment,
            )
            for _ in range(50):
                try:
                    probe_reader, probe_writer = await asyncio.open_connection(
                        "127.0.0.1", 3128
                    )
                    probe_writer.close()
                    await probe_writer.wait_closed()
                    del probe_reader
                    break
                except OSError:
                    await asyncio.sleep(0.02)
            else:
                shim.terminate()
                raise RuntimeError("sandbox_unavailable:proxy_shim")
            environment.update(
                {
                    "HTTP_PROXY": "http://127.0.0.1:3128",
                    "HTTPS_PROXY": "http://127.0.0.1:3128",
                    "http_proxy": "http://127.0.0.1:3128",
                    "https_proxy": "http://127.0.0.1:3128",
                }
            )
            argv = ["/bin/sh", "-lc", command]
        else:
            bwrap = shutil.which("bwrap")
            if bwrap is None:
                raise RuntimeError("sandbox_unavailable:bubblewrap")
            argv = self._build_bubblewrap_argv(
                bwrap=bwrap,
                workspace=workspace,
                working_dir=working_dir,
                command=command,
            )
        process = await asyncio.create_subprocess_exec(
            *argv,
            cwd=str(working_dir),
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=environment,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=timeout_seconds
            )
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            if shim is not None:
                shim.terminate()
                with suppress(ProcessLookupError):
                    await shim.wait()
            raise TimeoutError("sandbox command timed out") from None
        if shim is not None:
            shim.terminate()
            with suppress(ProcessLookupError):
                await shim.wait()
        return {
            "exit_code": int(process.returncode or 0),
            "stdout": stdout[:MAX_COMMAND_OUTPUT_BYTES].decode("utf-8", errors="replace"),
            "stderr": stderr[:MAX_COMMAND_OUTPUT_BYTES].decode("utf-8", errors="replace"),
            "truncated": len(stdout) > MAX_COMMAND_OUTPUT_BYTES or len(stderr) > MAX_COMMAND_OUTPUT_BYTES,
        }
