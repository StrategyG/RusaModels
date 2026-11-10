"""Pinned checksum verification for CPU GGUF model artifacts."""

from __future__ import annotations

import hashlib
import hmac
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class VerifiedGGUF:
    path: Path
    size_bytes: int
    sha256: str


def verify_gguf_file(model_path: str | Path, expected_sha256: str) -> VerifiedGGUF:
    """Verify an absolute, regular, non-symlink GGUF file against a pinned SHA-256."""
    path = Path(model_path)
    if not path.is_absolute():
        raise ValueError("GGUF model path must be absolute")
    if path.suffix.lower() != ".gguf":
        raise ValueError("Model artifact must use the .gguf extension")
    if not isinstance(expected_sha256, str) or not re.fullmatch(
        r"[0-9a-fA-F]{64}", expected_sha256
    ):
        raise ValueError("Expected GGUF SHA-256 must contain exactly 64 hexadecimal characters")

    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(path, flags)
    with os.fdopen(descriptor, "rb") as model_file:
        before = os.fstat(model_file.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("GGUF model must be a regular file")
        if before.st_size <= 0:
            raise ValueError("GGUF model file must not be empty")

        digest = hashlib.sha256()
        for chunk in iter(lambda: model_file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
        after = os.fstat(model_file.fileno())

    stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, field) != getattr(after, field) for field in stable_fields):
        raise ValueError("GGUF model changed while its checksum was being verified")

    actual_sha256 = digest.hexdigest()
    if not hmac.compare_digest(actual_sha256, expected_sha256.lower()):
        raise ValueError("GGUF model SHA-256 does not match the approved checksum")
    return VerifiedGGUF(path=path, size_bytes=after.st_size, sha256=actual_sha256)
