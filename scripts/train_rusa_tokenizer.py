#!/usr/bin/env python3
"""Train an immutable RusaAi BPE tokenizer from a verified Dataset Engine train export."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rusamodels.tokenizers.training import TokenizerTrainingError, train_tokenizer  # noqa: E402
from rusamodels.tokenizers.vocabulary import SPECIAL_TOKENS  # noqa: E402


_CONFIG_KEYS = {
    "schema_version",
    "tokenizer_id",
    "tokenizer_version",
    "algorithm",
    "vocab_size",
    "min_frequency",
    "special_tokens",
    "training_split",
    "pre_tokenizer",
    "normalizer",
    "notes",
}


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant is prohibited: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate tokenizer config key: {key}")
        result[key] = value
    return result


def _load_config(path: Path) -> dict[str, Any]:
    if path.is_symlink():
        raise TokenizerTrainingError("tokenizer config must not be a symlink")
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise TokenizerTrainingError(f"cannot load tokenizer config: {exc}") from exc
    if not isinstance(payload, dict) or set(payload) != _CONFIG_KEYS:
        raise TokenizerTrainingError("tokenizer config has unsupported or missing fields")
    if payload["schema_version"] != "1.0.0" or payload["algorithm"] != "byte_level_bpe":
        raise TokenizerTrainingError("unsupported tokenizer config version or algorithm")
    if payload["special_tokens"] != SPECIAL_TOKENS:
        raise TokenizerTrainingError("tokenizer config special tokens differ from the versioned contract")
    if payload["training_split"] != "train":
        raise TokenizerTrainingError("tokenizer config must use training_split=train")
    if payload["pre_tokenizer"] != "bytelevel" or payload["normalizer"] != "none":
        raise TokenizerTrainingError("unsupported tokenizer pre-tokenizer or normalizer")
    if type(payload["vocab_size"]) is not int or payload["vocab_size"] < 260:
        raise TokenizerTrainingError("tokenizer config vocab_size must be at least 260")
    if type(payload["min_frequency"]) is not int or payload["min_frequency"] < 1:
        raise TokenizerTrainingError("tokenizer config min_frequency must be positive")
    if not isinstance(payload["tokenizer_id"], str) or not isinstance(payload["tokenizer_version"], str):
        raise TokenizerTrainingError("tokenizer config identity fields are invalid")
    if not isinstance(payload["notes"], list) or any(not isinstance(item, str) for item in payload["notes"]):
        raise TokenizerTrainingError("tokenizer config notes must be strings")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/tokenizers/rusaai-research-100m-bpe.json",
    )
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--corpus-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--vocab-size", type=int)
    parser.add_argument("--min-frequency", type=int)
    parser.add_argument("--tokenizer-id")
    parser.add_argument("--tokenizer-version")
    args = parser.parse_args(argv)
    try:
        config = _load_config(args.config)
        result = train_tokenizer(
            args.corpus,
            args.corpus_manifest,
            args.output_dir,
            vocab_size=args.vocab_size if args.vocab_size is not None else config["vocab_size"],
            min_frequency=(
                args.min_frequency
                if args.min_frequency is not None
                else config["min_frequency"]
            ),
            tokenizer_id=args.tokenizer_id or config["tokenizer_id"],
            tokenizer_version=args.tokenizer_version or config["tokenizer_version"],
        )
    except (TokenizerTrainingError, OSError, UnicodeError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
