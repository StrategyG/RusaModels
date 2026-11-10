#!/usr/bin/env python3
"""Validate RusaModels agent SFT JSONL files and check for split leakage."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rusamodels.training.dataset import load_jsonl, validate_dataset_splits  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-file", required=True, type=Path)
    parser.add_argument("--eval-file", required=True, type=Path)
    args = parser.parse_args()
    train = load_jsonl(args.train_file)
    evaluation = load_jsonl(args.eval_file)
    validate_dataset_splits(train, evaluation)
    print(f"PASS: train={len(train)} eval={len(evaluation)}; no duplicate IDs/content or split leakage")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
