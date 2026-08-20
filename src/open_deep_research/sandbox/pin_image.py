"""Pin a locally built Worker image ID into the administrator policy bundle."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import uuid
from pathlib import Path

from open_deep_research.sandbox.schema import load_policy_bundle

_IMAGE_ID = re.compile(r"^sha256:[0-9a-f]{64}$")
_WORKER_LINE = re.compile(
    r'(?m)^(worker_image_digest\s*=\s*)"[^"]+"\s*$'
)


def inspect_image_id(image: str) -> str:
    """Resolve an immutable local Docker image ID without using Docker Socket SDK."""
    process = subprocess.run(
        ["docker", "image", "inspect", image, "--format", "{{json .Id}}"],
        check=True,
        capture_output=True,
        text=True,
    )
    image_id = str(json.loads(process.stdout.strip()))
    if not _IMAGE_ID.fullmatch(image_id):
        raise ValueError("sandbox_worker_image_id_invalid")
    return image_id


def pin_image(policy_path: str | Path, image_id: str) -> None:
    """Atomically replace every Profile Worker reference and revalidate TOML."""
    if not _IMAGE_ID.fullmatch(image_id):
        raise ValueError("sandbox_worker_image_id_invalid")
    path = Path(policy_path).resolve()
    original = path.read_text(encoding="utf-8")
    updated, count = _WORKER_LINE.subn(
        lambda match: f'{match.group(1)}"{image_id}"',
        original,
    )
    if count == 0:
        raise ValueError("sandbox_policy_has_no_worker_image")
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(updated, encoding="utf-8", newline="\n")
        load_policy_bundle(temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> None:
    """Inspect a local tag and pin its content ID into the selected policy."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default="insightforge-sandbox-worker:local")
    parser.add_argument("--policy", default="config/sandbox-policy.toml")
    args = parser.parse_args()
    image_id = inspect_image_id(args.image)
    pin_image(args.policy, image_id)
    sys.stdout.write(image_id + "\n")


if __name__ == "__main__":
    main()
