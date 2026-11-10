#!/usr/bin/env python3
"""Verify the pinned GGUF model artifact before starting the RusaModels CPU inference stack."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rusamodels.gguf import verify_gguf_file  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--env-file",
        type=Path,
        default=ROOT / ".env",
        help="dotenv file to read (default: repository .env)",
    )
    parser.add_argument("--model", help="absolute path to the GGUF file")
    parser.add_argument("--sha256", help="approved 64-character SHA-256 digest")
    args = parser.parse_args(argv)

    load_dotenv(dotenv_path=args.env_file, override=False)
    model_path = args.model or os.getenv("RUSAMODELS_GGUF_MODEL_PATH")
    expected_sha256 = args.sha256 or os.getenv("RUSAMODELS_GGUF_MODEL_SHA256")
    if not model_path:
        parser.error("provide --model or set RUSAMODELS_GGUF_MODEL_PATH in the env file")
    if not expected_sha256:
        parser.error("provide --sha256 or set RUSAMODELS_GGUF_MODEL_SHA256 in the env file")

    try:
        verified = verify_gguf_file(model_path, expected_sha256)
    except (OSError, ValueError) as exc:
        print(f"GGUF verification failed: {exc}", file=sys.stderr)
        return 1

    print(
        "GGUF verified: "
        f"path={verified.path} size_bytes={verified.size_bytes} sha256={verified.sha256}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
