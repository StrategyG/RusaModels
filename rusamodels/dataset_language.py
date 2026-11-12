"""Offline language identification and deterministic dataset quality reporting."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
import unicodedata
from collections import Counter
from dataclasses import dataclass
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


class DatasetLanguageError(ValueError):
    """Raised when language analysis or quality policy evaluation fails."""


_WORD = re.compile(r"\w+", re.UNICODE)
_REPEATED = re.compile(r"(.)\1{4,}", re.UNICODE)
_LANGUAGE_CODE = re.compile(r"^[a-z]{2,3}$")


@dataclass(frozen=True)
class DatasetLanguagePolicy:
    detector_languages: tuple[str, ...] = ("id", "en")
    allowed_languages: tuple[str, ...] = ("id", "en", "mixed")
    minimum_confidence: float = 0.65
    mixed_margin: float = 0.15
    minimum_text_characters: int = 40
    minimum_word_count: int = 6
    minimum_unique_word_ratio: float = 0.20
    maximum_repeated_character_ratio: float = 0.25


def _validate_policy(policy: DatasetLanguagePolicy) -> None:
    if not isinstance(policy, DatasetLanguagePolicy):
        raise DatasetLanguageError("policy must be a DatasetLanguagePolicy")
    if not policy.detector_languages or len(set(policy.detector_languages)) != len(policy.detector_languages):
        raise DatasetLanguageError("detector_languages must be a non-empty list of unique language codes")
    if any(
        not isinstance(code, str) or not _LANGUAGE_CODE.fullmatch(code)
        for code in policy.detector_languages
    ):
        raise DatasetLanguageError("detector_languages must use lowercase ISO 639-1/3 codes")
    if not policy.allowed_languages:
        raise DatasetLanguageError("allowed_languages must not be empty")
    supported = set(policy.detector_languages) | {"mixed", "und"}
    if any(
        not isinstance(code, str) or code not in supported
        for code in policy.allowed_languages
    ):
        raise DatasetLanguageError(
            "allowed_languages must be selected detector languages plus mixed and/or und"
        )
    for name, value in (
        ("minimum_confidence", policy.minimum_confidence),
        ("mixed_margin", policy.mixed_margin),
        ("minimum_unique_word_ratio", policy.minimum_unique_word_ratio),
        ("maximum_repeated_character_ratio", policy.maximum_repeated_character_ratio),
    ):
        if type(value) not in {int, float} or not 0 <= value <= 1:
            raise DatasetLanguageError(f"{name} must be between 0 and 1")
    if type(policy.minimum_text_characters) is not int or policy.minimum_text_characters < 1:
        raise DatasetLanguageError("minimum_text_characters must be a positive integer")
    if type(policy.minimum_word_count) is not int or policy.minimum_word_count < 1:
        raise DatasetLanguageError("minimum_word_count must be a positive integer")


def _build_detector(language_codes: tuple[str, ...]) -> Any:
    try:
        from lingua import Language, LanguageDetectorBuilder
    except ImportError as exc:
        raise DatasetLanguageError(
            "language identification requires the optional dependency; install with "
            'python -m pip install -e ".[dataset]"'
        ) from exc
    available: dict[str, Any] = {}
    for language in Language:
        iso_code = language.iso_code_639_1
        if iso_code is not None:
            available[iso_code.name.lower()] = language
    missing = sorted(set(language_codes) - set(available))
    if missing:
        raise DatasetLanguageError(f"unsupported Lingua language codes: {missing}")
    selected = [available[code] for code in language_codes]
    return LanguageDetectorBuilder.from_languages(*selected).with_preloaded_language_models().build()


def _normalize_text(value: str) -> str:
    return " ".join(unicodedata.normalize(
        "NFC", value.replace("\r\n", "\n").replace("\r", "\n")
    ).split())


def _language_code(language: Any) -> str:
    iso_code = language.iso_code_639_1
    return iso_code.name.lower() if iso_code is not None else "und"


def _detect_language(
    text: str,
    detector: Any,
    policy: DatasetLanguagePolicy,
) -> tuple[str, float, str | None, float]:
    normalized = _normalize_text(text)
    words = _WORD.findall(normalized)
    if len(normalized) < policy.minimum_text_characters or len(words) < policy.minimum_word_count:
        return "und", 0.0, None, 0.0
    values = detector.compute_language_confidence_values(normalized)
    if not values:
        return "und", 0.0, None, 0.0
    top = values[0]
    top_code = _language_code(top.language)
    top_confidence = float(top.value)
    second_code: str | None = None
    second_confidence = 0.0
    if len(values) > 1:
        second_code = _language_code(values[1].language)
        second_confidence = float(values[1].value)
    if top_confidence < policy.minimum_confidence:
        return "und", top_confidence, second_code, second_confidence
    if (
        second_code is not None
        and second_confidence >= policy.minimum_confidence
        and top_confidence - second_confidence <= policy.mixed_margin
    ):
        return "mixed", top_confidence, second_code, second_confidence
    return top_code, top_confidence, second_code, second_confidence


def _quality_metrics(text: str) -> dict[str, int | float]:
    normalized = _normalize_text(text)
    words = _WORD.findall(normalized)
    repeated_characters = sum(
        len(match.group(0)) for match in _REPEATED.finditer(normalized)
    )
    return {
        "text_characters": len(normalized),
        "word_count": len(words),
        "unique_word_ratio": round(len(set(word.casefold() for word in words)) / len(words), 6)
        if words else 0.0,
        "repeated_character_ratio": round(repeated_characters / len(normalized), 6)
        if normalized else 0.0,
    }


def _analyze_record(
    record: dict[str, Any],
    detector: Any,
    policy: DatasetLanguagePolicy,
) -> dict[str, Any]:
    try:
        text = record_to_text(record)
    except DatasetTextExportError as exc:
        raise DatasetLanguageError(f"record has no exportable text: {exc}") from exc
    language, confidence, runner_up, runner_up_confidence = _detect_language(
        text, detector, policy
    )
    metrics = _quality_metrics(text)
    reasons: list[str] = []
    if metrics["text_characters"] < policy.minimum_text_characters:
        reasons.append("text_too_short")
    if metrics["word_count"] < policy.minimum_word_count:
        reasons.append("too_few_words")
    if metrics["unique_word_ratio"] < policy.minimum_unique_word_ratio:
        reasons.append("low_unique_word_ratio")
    if metrics["repeated_character_ratio"] > policy.maximum_repeated_character_ratio:
        reasons.append("excessive_repeated_characters")
    if confidence < policy.minimum_confidence:
        reasons.append("language_confidence_below_threshold")
    if language not in policy.allowed_languages:
        reasons.append("language_not_allowed")
    return {
        "language": language,
        "confidence": round(confidence, 6),
        "runner_up_language": runner_up,
        "runner_up_confidence": round(runner_up_confidence, 6),
        "metrics": metrics,
        "eligible": not reasons,
        "reasons": reasons,
    }


def _write_new_file(path: Path, payload: bytes) -> str:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return hashlib.sha256(payload).hexdigest()


def analyze_dataset_languages(
    release_path: str | Path,
    report_path: str | Path,
    *,
    policy: DatasetLanguagePolicy | None = None,
) -> dict[str, Any]:
    """Write language, lexical quality, and eligibility evidence for every record."""
    selected_policy = policy or DatasetLanguagePolicy()
    _validate_policy(selected_policy)
    root = Path(release_path).expanduser().resolve()
    output_input = Path(report_path).expanduser()
    if output_input.is_symlink():
        raise DatasetLanguageError("symlink paths are not allowed for quality reports")
    output = output_input.parent.resolve() / output_input.name
    summary_path = output.with_suffix(output.suffix + ".summary.json")
    if root == output or root in output.parents:
        raise DatasetLanguageError("quality report must be outside the immutable dataset release")
    if output.exists() or summary_path.exists():
        raise DatasetLanguageError("quality report or summary already exists")
    output.parent.mkdir(parents=True, exist_ok=True)

    try:
        verification = verify_dataset_release(root)
    except (DatasetVerificationError, OSError, UnicodeError) as exc:
        raise DatasetLanguageError(f"dataset release verification failed: {exc}") from exc

    detector = _build_detector(selected_policy.detector_languages)
    release_manifest_bytes = (root / "manifest.json").read_bytes()
    release_manifest_sha256 = hashlib.sha256(release_manifest_bytes).hexdigest()
    manifest = json.loads(release_manifest_bytes.decode("utf-8"))
    report_descriptor, report_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    report_temporary = Path(report_name)
    report_published = False
    summary_published = False
    language_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    split_counts = {"train": 0, "validation": 0, "test": 0}
    eligible_records = 0
    rejected_records = 0
    report_digest = hashlib.sha256()
    try:
        with os.fdopen(report_descriptor, "wb") as report_handle:
            for split in ("train", "validation", "test"):
                split_path = root / f"{split}.jsonl"
                with split_path.open("r", encoding="utf-8") as handle:
                    for line_number, line in enumerate(handle, start=1):
                        if not line.strip():
                            raise DatasetLanguageError(
                                f"{split_path.name}:{line_number}: blank lines are not allowed"
                            )
                        try:
                            record = json.loads(line)
                        except json.JSONDecodeError as exc:
                            raise DatasetLanguageError(
                                f"{split_path.name}:{line_number}: invalid JSON"
                            ) from exc
                        if not isinstance(record, dict) or not isinstance(record.get("id"), str):
                            raise DatasetLanguageError(
                                f"{split_path.name}:{line_number}: record ID is missing"
                            )
                        analysis = _analyze_record(record, detector, selected_policy)
                        language_counts[analysis["language"]] += 1
                        if analysis["eligible"]:
                            eligible_records += 1
                        else:
                            rejected_records += 1
                        for reason in analysis["reasons"]:
                            reason_counts[reason] += 1
                        item = {
                            "record_id": record["id"],
                            "split": split,
                            **analysis,
                        }
                        encoded = (json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n").encode()
                        report_handle.write(encoded)
                        report_digest.update(encoded)
                        split_counts[split] += 1
            report_handle.flush()
            os.fsync(report_handle.fileno())
        for split in ("train", "validation", "test"):
            actual_digest = hashlib.sha256((root / f"{split}.jsonl").read_bytes()).hexdigest()
            expected_digest = manifest["files"][f"{split}.jsonl"]["sha256"]
            if actual_digest != expected_digest:
                raise DatasetLanguageError(f"{split}.jsonl changed during quality analysis")
        actual_manifest_digest = hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest()
        if actual_manifest_digest != release_manifest_sha256 or actual_manifest_digest != verification["manifest_sha256"]:
            raise DatasetLanguageError("release manifest changed during quality analysis")
        os.link(report_temporary, output)
        report_published = True
        report_temporary.unlink(missing_ok=True)
        report_temporary = None
        summary = {
            "schema_version": "1.0.0",
            "report_type": "dataset_language_and_quality",
            "dataset_id": verification["dataset_id"],
            "dataset_version": verification["dataset_version"],
            "release_manifest_sha256": verification["manifest_sha256"],
            "report_file": output.name,
            "report_sha256": report_digest.hexdigest(),
            "records": sum(split_counts.values()),
            "eligible_records": eligible_records,
            "rejected_records": rejected_records,
            "split_counts": split_counts,
            "language_counts": dict(sorted(language_counts.items())),
            "reason_counts": dict(sorted(reason_counts.items())),
            "policy": {
                "detector_languages": list(selected_policy.detector_languages),
                "allowed_languages": list(selected_policy.allowed_languages),
                "minimum_confidence": selected_policy.minimum_confidence,
                "mixed_margin": selected_policy.mixed_margin,
                "minimum_text_characters": selected_policy.minimum_text_characters,
                "minimum_word_count": selected_policy.minimum_word_count,
                "minimum_unique_word_ratio": selected_policy.minimum_unique_word_ratio,
                "maximum_repeated_character_ratio": selected_policy.maximum_repeated_character_ratio,
            },
            "note": "Language and lexical-quality heuristics are screening signals, not semantic, toxicity, or safety certification.",
        }
        summary_bytes = (
            json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
        ).encode()
        _write_new_file(summary_path, summary_bytes)
        summary_published = True
    except (OSError, UnicodeError) as exc:
        raise DatasetLanguageError(f"dataset language analysis failed: {exc}") from exc
    finally:
        if report_temporary is not None:
            report_temporary.unlink(missing_ok=True)
        if report_published and not summary_published:
            output.unlink(missing_ok=True)

    return {
        "dataset_id": verification["dataset_id"],
        "dataset_version": verification["dataset_version"],
        "records": sum(split_counts.values()),
        "eligible_records": eligible_records,
        "rejected_records": rejected_records,
        "language_counts": dict(sorted(language_counts.items())),
        "report_path": str(output),
        "summary_path": str(summary_path),
        "report_sha256": report_digest.hexdigest(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--languages", default="id,en")
    parser.add_argument("--allowed-languages", default="id,en,mixed")
    parser.add_argument("--minimum-confidence", type=float, default=0.65)
    parser.add_argument("--mixed-margin", type=float, default=0.15)
    parser.add_argument("--minimum-text-characters", type=int, default=40)
    parser.add_argument("--minimum-word-count", type=int, default=6)
    parser.add_argument("--minimum-unique-word-ratio", type=float, default=0.20)
    parser.add_argument("--maximum-repeated-character-ratio", type=float, default=0.25)
    parser.add_argument("--fail-on-rejected", action="store_true")
    args = parser.parse_args(argv)
    try:
        policy = DatasetLanguagePolicy(
            detector_languages=tuple(item.strip() for item in args.languages.split(",") if item.strip()),
            allowed_languages=tuple(item.strip() for item in args.allowed_languages.split(",") if item.strip()),
            minimum_confidence=args.minimum_confidence,
            mixed_margin=args.mixed_margin,
            minimum_text_characters=args.minimum_text_characters,
            minimum_word_count=args.minimum_word_count,
            minimum_unique_word_ratio=args.minimum_unique_word_ratio,
            maximum_repeated_character_ratio=args.maximum_repeated_character_ratio,
        )
        result = analyze_dataset_languages(args.release, args.report, policy=policy)
    except (DatasetLanguageError, OSError, UnicodeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 1 if args.fail_on_rejected and result["rejected_records"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
