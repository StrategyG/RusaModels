#!/usr/bin/env python3
"""Train an immutable RusaAi BPE tokenizer from a verified Dataset Engine train export."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rusamodels.tokenizers.training import TokenizerTrainingError, train_tokenizer  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--corpus-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--min-frequency", type=int, default=2)
    parser.add_argument("--tokenizer-id", default="rusaai-research-100m-bpe-32k")
    parser.add_argument("--tokenizer-version", default="0.1.0")
    args = parser.parse_args(argv)
    try:
        result = train_tokenizer(
            args.corpus,
            args.corpus_manifest,
            args.output_dir,
            vocab_size=args.vocab_size,
            min_frequency=args.min_frequency,
            tokenizer_id=args.tokenizer_id,
            tokenizer_version=args.tokenizer_version,
        )
    except (TokenizerTrainingError, OSError, UnicodeError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
