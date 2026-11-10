#!/usr/bin/env python3
"""Create a checksummed RusaModels model manifest for a local model directory."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


MANIFEST_NAME = "rusa-model-manifest.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--license", required=True, help="Exact license identifier or documented license reference")
    parser.add_argument("--tokenizer-id", required=True)
    parser.add_argument("--tokenizer-revision", required=True)
    args = parser.parse_args()

    root = args.model_dir.expanduser().resolve(strict=True)
    if not root.is_dir():
        parser.error("--model-dir must point to a directory")
    files: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.is_symlink() or path.name == MANIFEST_NAME:
            continue
        relative = path.relative_to(root).as_posix()
        if relative.startswith(".cache/") or "/.cache/" in relative:
            continue
        files[relative] = sha256_file(path)

    required = {"config.json"}
    if not required.issubset(files):
        parser.error("Model directory must contain config.json")
    if not any(name in files for name in ("tokenizer.json", "tokenizer.model", "tokenizer_config.json", "spiece.model", "vocab.json")):
        parser.error("Model directory must contain tokenizer files")
    if not any(name.lower().endswith(".safetensors") for name in files):
        parser.error("Model directory must contain safetensors weights; pickle-based checkpoints are not accepted")

    manifest = {
        "schema_version": 1,
        "model_id": args.model_id,
        "revision": args.revision,
        "license": args.license,
        "tokenizer_id": args.tokenizer_id,
        "tokenizer_revision": args.tokenizer_revision,
        "files": files,
    }
    output = root / MANIFEST_NAME
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Manifest: {output}")
    print(f"Files: {len(files)}")
    print(f"Manifest SHA-256: {sha256_file(output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
