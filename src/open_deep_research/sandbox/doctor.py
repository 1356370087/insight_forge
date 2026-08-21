"""Fail-closed sandbox deployment diagnostics."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import httpx

from open_deep_research.configuration import Configuration
from open_deep_research.sandbox.crypto import SandboxDerivedKeys
from open_deep_research.sandbox.schema import load_policy_bundle, policy_digest


def _developer_profile_warnings(bundle: object | None) -> list[str]:
    """Return non-blocking release warnings for mapped Developer profiles."""
    profile_by_role = getattr(bundle, "profile_by_role", {}) if bundle is not None else {}
    developer_mapped = "developer-workspace" in set(profile_by_role.values())
    if developer_mapped and not sys.platform.startswith("linux"):
        return [
            "developer-workspace is mapped but the current platform is not Linux; "
            "this profile is not release-qualified"
        ]
    return []


def diagnose() -> dict:
    """Return a content-free readiness report for administrator use."""
    failures: list[str] = []
    try:
        configurable = Configuration.from_runnable_config(None)
    except Exception as exc:
        return {"ready": False, "failures": [str(exc)], "warnings": []}
    if not configurable.sandbox_enabled:
        failures.append("SANDBOX_ENABLED is false")
    if not configurable.enable_async_research:
        failures.append("ENABLE_ASYNC_RESEARCH is false")
    try:
        bundle = load_policy_bundle(configurable.sandbox_policy_path)
    except Exception as exc:
        failures.append(str(exc))
        bundle = None
    try:
        SandboxDerivedKeys.from_root(configurable.sandbox_root_signing_key or "")
    except Exception as exc:
        failures.append(str(exc))
    socket_path = Path(configurable.sandbox_controller_socket)
    controller = None
    if not socket_path.exists():
        failures.append(f"controller socket missing: {socket_path}")
    else:
        try:
            transport = httpx.HTTPTransport(uds=str(socket_path))
            with httpx.Client(
                transport=transport,
                base_url="http://sandbox-controller",
                timeout=10,
            ) as client:
                response = client.get("/healthz")
                response.raise_for_status()
                controller = response.json()
            if controller.get("status") != "ok":
                failures.append("controller reports not_ready")
        except Exception as exc:
            failures.append(f"controller health failed: {exc}")
    gateway = None
    try:
        with httpx.Client(base_url=configurable.sandbox_gateway_url, timeout=10) as client:
            response = client.get("/healthz")
            response.raise_for_status()
            gateway = response.json()
    except Exception as exc:
        failures.append(f"gateway health failed: {exc}")
    developer_needed = bool(
        bundle is not None
        and "developer-workspace" in set(bundle.profile_by_role.values())
    )
    warnings = _developer_profile_warnings(bundle)
    inside_container = Path("/.dockerenv").exists()
    if developer_needed and not inside_container and shutil.which("bwrap") is None:
        failures.append("bubblewrap unavailable for developer-workspace")
    if developer_needed and not inside_container and shutil.which("socat") is None:
        failures.append("socat unavailable for developer-workspace")
    return {
        "ready": not failures,
        "failures": failures,
        "warnings": warnings,
        "policy_digest": policy_digest(bundle) if bundle is not None else None,
        "controller": controller,
        "gateway": gateway,
    }


def main() -> None:
    """Print readiness JSON and exit nonzero when any invariant fails."""
    report = diagnose()
    sys.stdout.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    raise SystemExit(0 if report["ready"] else 1)


if __name__ == "__main__":
    main()
