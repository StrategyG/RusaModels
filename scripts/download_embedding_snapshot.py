#!/usr/bin/env python3
"""Download a pinned Hugging Face embedding snapshot for local CPU inference."""
from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

from huggingface_hub import snapshot_download


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default="intfloat/multilingual-e5-small")
    parser.add_argument("--revision", required=True, help="Immutable 40-character Hugging Face commit SHA")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--license-accepted", action="store_true", help="Confirm the model license was reviewed")
    args = parser.parse_args()
    if not args.license_accepted:
        parser.error("Review the model card/license and pass --license-accepted")
    if not re.fullmatch(r"[0-9a-fA-F]{40}", args.revision):
        parser.error("--revision must be an immutable 40-character commit SHA")
    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        parser.error("--output-dir must be empty or not exist")
    output.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=args.repo_id,
        revision=args.revision.lower(),
        token=os.getenv("HF_TOKEN"),
        local_dir=str(output),
    )
    (output / "rusa-source-revision.txt").write_text(
        f"repo_id={args.repo_id}\nrevision={args.revision.lower()}\n",
        encoding="utf-8",
    )
    print(f"Embedding snapshot downloaded to {output}")
    print("Keep this directory outside Git and mount it read-only into the embedding service.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
