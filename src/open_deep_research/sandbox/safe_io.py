"""Host-side collectors that never follow container-controlled links."""

from __future__ import annotations

import io
import os
import shutil
import tarfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath


class UnsafeSandboxArchive(ValueError):
    """Raised when a container archive violates the extraction contract."""


@dataclass(frozen=True, slots=True)
class ArchiveLimits:
    """Bounded archive extraction limits."""

    max_bytes: int
    max_files: int = 10_000
    max_path_chars: int = 1024


def _safe_member_path(name: str) -> PurePosixPath:
    normalized = PurePosixPath(name)
    if normalized.is_absolute() or ".." in normalized.parts:
        raise UnsafeSandboxArchive(f"unsafe archive path: {name}")
    if not normalized.parts or str(normalized) in {"", "."}:
        raise UnsafeSandboxArchive("empty archive path")
    return normalized


def extract_safe_tar(
    archive: bytes,
    destination: str | Path,
    *,
    limits: ArchiveLimits,
) -> list[Path]:
    """Extract regular files/directories from tar bytes without link traversal."""
    target_root = Path(destination).resolve()
    target_root.mkdir(parents=True, exist_ok=True)
    extracted: list[Path] = []
    total_bytes = 0
    file_count = 0

    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:*") as bundle:
        for member in bundle:
            relative = _safe_member_path(member.name)
            if len(str(relative)) > limits.max_path_chars:
                raise UnsafeSandboxArchive("archive path exceeds limit")
            if member.issym() or member.islnk() or member.isdev() or member.isfifo():
                raise UnsafeSandboxArchive(f"archive contains special entry: {member.name}")
            if not member.isdir() and not member.isfile():
                raise UnsafeSandboxArchive(f"archive entry type is unsupported: {member.name}")

            output = (target_root / Path(*relative.parts)).resolve()
            if target_root not in output.parents and output != target_root:
                raise UnsafeSandboxArchive(f"archive escaped destination: {member.name}")
            if member.isdir():
                output.mkdir(parents=True, exist_ok=True)
                continue

            file_count += 1
            total_bytes += int(member.size)
            if file_count > limits.max_files:
                raise UnsafeSandboxArchive("archive file count exceeds limit")
            if total_bytes > limits.max_bytes:
                raise UnsafeSandboxArchive("archive bytes exceed limit")
            source = bundle.extractfile(member)
            if source is None:
                raise UnsafeSandboxArchive(f"archive file has no data: {member.name}")
            output.parent.mkdir(parents=True, exist_ok=True)
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            descriptor = os.open(output, flags, 0o600)
            try:
                with os.fdopen(descriptor, "wb", closefd=False) as handle:
                    shutil.copyfileobj(source, handle, length=64 * 1024)
            finally:
                os.close(descriptor)
            extracted.append(output)
    return extracted


def read_regular_file_from_tar(
    archive: bytes,
    expected_name: str,
    *,
    max_bytes: int,
) -> bytes:
    """Read one exact regular file from a Docker tar response without extracting."""
    expected = _safe_member_path(expected_name)
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:*") as bundle:
        matches = []
        for member in bundle:
            relative = _safe_member_path(member.name)
            if member.issym() or member.islnk() or member.isdev() or member.isfifo():
                raise UnsafeSandboxArchive(f"archive contains special entry: {member.name}")
            if relative == expected or relative.name == expected.name:
                matches.append(member)
        if len(matches) != 1:
            raise UnsafeSandboxArchive("archive does not contain exactly one expected file")
        member = matches[0]
        if not member.isfile() or member.size > max_bytes:
            raise UnsafeSandboxArchive("expected archive file is invalid or too large")
        source = bundle.extractfile(member)
        if source is None:
            raise UnsafeSandboxArchive("expected archive file has no data")
        data = source.read(max_bytes + 1)
        if len(data) > max_bytes:
            raise UnsafeSandboxArchive("expected archive file exceeds limit")
        return data


def repack_docker_archives(
    archives: list[tuple[str, bytes]],
    *,
    limits: ArchiveLimits,
) -> bytes:
    """Validate Docker tar responses and return a canonical regular-file tar.

    Each source is assigned an administrator-chosen top-level prefix. Container
    member ownership, modes, links and timestamps are discarded.
    """
    output = io.BytesIO()
    total_bytes = 0
    file_count = 0
    seen: set[str] = set()
    with tarfile.open(fileobj=output, mode="w") as destination:
        for prefix, archive in archives:
            safe_prefix = _safe_member_path(prefix)
            if len(safe_prefix.parts) != 1:
                raise UnsafeSandboxArchive("archive prefix must be one component")
            with tarfile.open(fileobj=io.BytesIO(archive), mode="r:*") as source_tar:
                for member in source_tar:
                    relative = _safe_member_path(member.name)
                    if member.issym() or member.islnk() or member.isdev() or member.isfifo():
                        raise UnsafeSandboxArchive(
                            f"archive contains special entry: {member.name}"
                        )
                    if not member.isdir() and not member.isfile():
                        raise UnsafeSandboxArchive(
                            f"archive entry type is unsupported: {member.name}"
                        )
                    parts = relative.parts
                    if parts and parts[0] == safe_prefix.name:
                        parts = parts[1:]
                    if not parts or member.isdir():
                        continue
                    canonical = safe_prefix / PurePosixPath(*parts)
                    canonical_name = canonical.as_posix()
                    if len(canonical_name) > limits.max_path_chars:
                        raise UnsafeSandboxArchive("archive path exceeds limit")
                    if canonical_name in seen:
                        raise UnsafeSandboxArchive(
                            f"archive contains duplicate path: {canonical_name}"
                        )
                    seen.add(canonical_name)
                    file_count += 1
                    total_bytes += int(member.size)
                    if file_count > limits.max_files:
                        raise UnsafeSandboxArchive("archive file count exceeds limit")
                    if total_bytes > limits.max_bytes:
                        raise UnsafeSandboxArchive("archive bytes exceed limit")
                    source = source_tar.extractfile(member)
                    if source is None:
                        raise UnsafeSandboxArchive(
                            f"archive file has no data: {member.name}"
                        )
                    data = source.read(int(member.size) + 1)
                    if len(data) != member.size:
                        raise UnsafeSandboxArchive(
                            f"archive member size mismatch: {member.name}"
                        )
                    clean = tarfile.TarInfo(canonical_name)
                    clean.size = len(data)
                    clean.mode = 0o600
                    clean.uid = 0
                    clean.gid = 0
                    clean.mtime = 0
                    destination.addfile(clean, io.BytesIO(data))
    return output.getvalue()
