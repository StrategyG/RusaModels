#!/usr/bin/env python3
"""Download all immutable artifacts for a registered RusaModels model version."""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import tempfile
from pathlib import Path

import httpx


def safe_filename(value: str) -> str:
    name = Path(value).name
    if not name or name in {".", ".."} or name != value or "/" in value or "\\" in value:
        raise ValueError(f"Unsafe artifact filename: {value!r}")
    return name


def download_version(base_url: str, token: str, model_id: str, version_id: str, output_dir: Path) -> list[Path]:
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    headers = {"Authorization": f"Bearer {token}"}
    timeout = httpx.Timeout(300.0, connect=15.0)
    written: list[Path] = []

    with httpx.Client(timeout=timeout, follow_redirects=False) as client:
        response = client.get(
            f"{base_url.rstrip('/')}/v1/models/{model_id}/versions/{version_id}/artifacts",
            headers=headers,
        )
        response.raise_for_status()
        artifacts = response.json()
        if not isinstance(artifacts, list) or not artifacts:
            raise RuntimeError("The selected model version has no registered artifacts")

        names: set[str] = set()
        for artifact in artifacts:
            filename = safe_filename(artifact.get("original_filename") or "")
            if filename in names:
                raise RuntimeError(f"Artifact filename collision: {filename}")
            names.add(filename)
            destination = output_dir / filename
            if destination.exists() or destination.is_symlink():
                raise FileExistsError(f"Refusing to overwrite existing path: {destination}")

            expected_sha = str(artifact.get("sha256") or "").lower()
            if not re.fullmatch(r"[0-9a-f]{64}", expected_sha):
                raise RuntimeError(f"Artifact {artifact.get('id')} has no valid SHA-256")
            temporary_path = None
            digest = hashlib.sha256()
            size = 0
            try:
                with client.stream(
                    "GET",
                    f"{base_url.rstrip('/')}/v1/artifacts/{artifact['id']}/download",
                    headers=headers,
                ) as download:
                    download.raise_for_status()
                    header_sha = download.headers.get("X-Checksum-SHA256", "").lower()
                    if header_sha != expected_sha:
                        raise RuntimeError("Artifact checksum header does not match registry metadata")
                    with tempfile.NamedTemporaryFile(
                        mode="wb", dir=output_dir, prefix=f".{filename}.", suffix=".part", delete=False
                    ) as target:
                        temporary_path = Path(target.name)
                        for chunk in download.iter_bytes():
                            digest.update(chunk)
                            size += len(chunk)
                            target.write(chunk)
                if digest.hexdigest() != expected_sha:
                    raise RuntimeError(f"SHA-256 verification failed for {filename}")
                if size != int(artifact.get("size_bytes", -1)):
                    raise RuntimeError(f"Size verification failed for {filename}")
                os.replace(temporary_path, destination)
                written.append(destination)
            except Exception:
                if temporary_path is not None:
                    temporary_path.unlink(missing_ok=True)
                raise
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--version-id", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--base-url", default=os.getenv("RUSAMODELS_API_BASE_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--token", default=os.getenv("RUSAMODELS_API_TOKEN"))
    args = parser.parse_args()
    if not args.token:
        parser.error("Set RUSAMODELS_API_TOKEN or pass --token")
    files = download_version(args.base_url, args.token, args.model_id, args.version_id, args.output_dir)
    for path in files:
        print(path)
    print(f"Downloaded and SHA-256 verified {len(files)} artifacts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
