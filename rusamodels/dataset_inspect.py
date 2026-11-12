"""Inspect dataset distributions and produce deterministic, bounded content samples."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from rusamodels.dataset_text_export import (
    DatasetTextExportError,
    record_to_text,
)
from rusamodels.dataset_verification import (
    DatasetVerificationError,
    verify_dataset_release,
)


class DatasetInspectionError(ValueError):
    """Raised when a dataset release cannot be inspected safely."""


_WORD = re.compile(r"\w+", re.UNICODE)


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_dataset_release(
    release_path: str | Path,
    *,
    sample_per_split: int = 3,
    seed: int = 42,
    preview_characters: int = 300,
) -> dict[str, Any]:
    """Return split statistics and deterministic reservoir samples.

    Samples contain only pseudonymized record IDs and truncated text previews.
    Protect the output because previews still contain dataset content.
    """
    if type(sample_per_split) is not int or not 0 <= sample_per_split <= 100:
        raise DatasetInspectionError("sample_per_split must be an integer from 0 to 100")
    if type(seed) is not int:
        raise DatasetInspectionError("seed must be an integer")
    if type(preview_characters) is not int or not 1 <= preview_characters <= 2000:
        raise DatasetInspectionError("preview_characters must be between 1 and 2000")

    root = Path(release_path).expanduser().resolve()
    try:
        verification = verify_dataset_release(root)
    except (DatasetVerificationError, OSError, UnicodeError) as exc:
        raise DatasetInspectionError(f"dataset release verification failed: {exc}") from exc
    manifest_bytes = (root / "manifest.json").read_bytes()
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    manifest = json.loads(manifest_bytes.decode("utf-8"))
    random_source = random.Random(seed)

    split_summaries: dict[str, Any] = {}
    for split in ("train", "validation", "test"):
        split_path = root / f"{split}.jsonl"
        records = 0
        total_characters = 0
        total_words = 0
        minimum_characters: int | None = None
        maximum_characters = 0
        role_counts: Counter[str] = Counter()
        task_counts: Counter[str] = Counter()
        language_counts: Counter[str] = Counter()
        risk_counts: Counter[str] = Counter()
        samples: list[dict[str, Any]] = []
        with split_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise DatasetInspectionError(
                        f"{split_path.name}:{line_number}: blank lines are not allowed"
                    )
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise DatasetInspectionError(
                        f"{split_path.name}:{line_number}: invalid JSON"
                    ) from exc
                if not isinstance(record, dict) or not isinstance(record.get("id"), str):
                    raise DatasetInspectionError(
                        f"{split_path.name}:{line_number}: record ID is missing"
                    )
                try:
                    text = record_to_text(record)
                except DatasetTextExportError as exc:
                    raise DatasetInspectionError(
                        f"{split_path.name}:{line_number}: {exc}"
                    ) from exc
                character_count = len(text)
                word_count = len(_WORD.findall(text))
                records += 1
                total_characters += character_count
                total_words += word_count
                minimum_characters = (
                    character_count
                    if minimum_characters is None
                    else min(minimum_characters, character_count)
                )
                maximum_characters = max(maximum_characters, character_count)
                for field, counter in (
                    ("agent_role", role_counts),
                    ("task_type", task_counts),
                    ("language", language_counts),
                    ("risk_class", risk_counts),
                ):
                    value = record.get(field)
                    if isinstance(value, str) and value:
                        counter[value] += 1
                if sample_per_split:
                    sample = {
                        "record_id": record["id"],
                        "preview": text[:preview_characters],
                        "truncated": character_count > preview_characters,
                        "text_characters": character_count,
                        "word_count": word_count,
                    }
                    if len(samples) < sample_per_split:
                        samples.append(sample)
                    else:
                        replacement_index = random_source.randrange(records)
                        if replacement_index < sample_per_split:
                            samples[replacement_index] = sample
        actual_digest = _hash_file(split_path)
        expected_digest = manifest["files"][split_path.name]["sha256"]
        if actual_digest != expected_digest:
            raise DatasetInspectionError(f"{split_path.name} changed during inspection")
        split_summaries[split] = {
            "records": records,
            "total_characters": total_characters,
            "total_words": total_words,
            "average_characters": round(total_characters / records, 2) if records else 0.0,
            "average_words": round(total_words / records, 2) if records else 0.0,
            "minimum_characters": minimum_characters or 0,
            "maximum_characters": maximum_characters,
            "agent_role_counts": dict(sorted(role_counts.items())),
            "task_type_counts": dict(sorted(task_counts.items())),
            "language_counts": dict(sorted(language_counts.items())),
            "risk_class_counts": dict(sorted(risk_counts.items())),
            "samples": samples,
        }

    final_manifest_sha256 = _hash_file(root / "manifest.json")
    if final_manifest_sha256 != manifest_sha256 or final_manifest_sha256 != verification["manifest_sha256"]:
        raise DatasetInspectionError("release manifest changed during inspection")

    return {
        "dataset_id": verification["dataset_id"],
        "dataset_version": verification["dataset_version"],
        "manifest_sha256": verification["manifest_sha256"],
        "seed": seed,
        "sample_per_split": sample_per_split,
        "preview_characters": preview_characters,
        "splits": split_summaries,
        "note": "Sample previews contain dataset text. Treat inspection output with the same access controls as the source release.",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", required=True, type=Path)
    parser.add_argument("--sample-per-split", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--preview-characters", type=int, default=300)
    args = parser.parse_args(argv)
    try:
        result = inspect_dataset_release(
            args.release,
            sample_per_split=args.sample_per_split,
            seed=args.seed,
            preview_characters=args.preview_characters,
        )
    except (DatasetInspectionError, OSError, UnicodeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
