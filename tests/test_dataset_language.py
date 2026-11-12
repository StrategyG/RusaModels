import hashlib
import json
from pathlib import Path

import pytest

from rusamodels.data_pipeline import (
    DatasetReleaseConfig,
    DatasetSource,
    build_dataset_release,
)
from rusamodels.dataset_language import (
    DatasetLanguageError,
    DatasetLanguagePolicy,
    _quality_metrics,
    analyze_dataset_languages,
)


def _write_jsonl(path: Path, records: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    return path


def _release(tmp_path: Path) -> Path:
    records = [
        {
            "id": "indonesian",
            "group_id": "document-id",
            "text": (
                "Dokumentasi teknis menjelaskan langkah konfigurasi server, validasi keamanan, "
                "dan proses pemulihan sistem secara terperinci untuk pengelola aplikasi."
            ),
        },
        {
            "id": "english",
            "group_id": "document-en",
            "text": (
                "The engineering team verifies dataset provenance, model artifacts, deployment "
                "configuration, security controls, and operational recovery before production release."
            ),
        },
        {
            "id": "short",
            "group_id": "document-short",
            "text": "OK",
        },
    ]
    source = _write_jsonl(tmp_path / "source.jsonl", records)
    return build_dataset_release(
        DatasetReleaseConfig(
            dataset_id="language-test",
            version="1.0.0",
            sources=(DatasetSource("approved-source", source, "CC0-1.0"),),
            output_root=tmp_path / "releases",
            seed=43,
            validation_fraction=0.2,
            test_fraction=0.2,
        )
    )


def test_offline_language_and_quality_report_covers_every_split(tmp_path: Path) -> None:
    release = _release(tmp_path)
    report_path = tmp_path / "reports" / "language-quality.jsonl"
    policy = DatasetLanguagePolicy(
        detector_languages=("id", "en"),
        allowed_languages=("id", "en", "mixed"),
        minimum_confidence=0.50,
        minimum_text_characters=40,
        minimum_word_count=6,
        minimum_unique_word_ratio=0.10,
        maximum_repeated_character_ratio=0.50,
    )

    result = analyze_dataset_languages(release, report_path, policy=policy)
    rows = [
        json.loads(line)
        for line in report_path.read_text(encoding="utf-8").splitlines()
    ]
    summary = json.loads(
        report_path.with_suffix(report_path.suffix + ".summary.json").read_text(encoding="utf-8")
    )
    assert result["records"] == 3
    assert result["rejected_records"] >= 1
    assert any(row["language"] == "id" for row in rows)
    assert any(row["language"] == "en" for row in rows)
    assert any(row["language"] == "und" for row in rows)
    assert any("text_too_short" in row["reasons"] for row in rows)
    assert all("text" not in row for row in rows)
    assert summary["report_sha256"] == hashlib.sha256(report_path.read_bytes()).hexdigest()


def test_quality_metrics_detect_repeated_characters_and_low_diversity() -> None:
    metrics = _quality_metrics("AAAAAA AAAAAA AAAAAA AAAAAA")
    assert metrics["repeated_character_ratio"] > 0.5
    assert metrics["unique_word_ratio"] == 0.25


def test_language_policy_rejects_unknown_or_duplicate_codes() -> None:
    with pytest.raises(DatasetLanguageError, match="ISO 639"):
        analyze_dataset_languages(
            Path("unused"),
            Path("unused-report.jsonl"),
            policy=DatasetLanguagePolicy(detector_languages=("id", "invalid-code")),
        )
    with pytest.raises(DatasetLanguageError, match="unique"):
        analyze_dataset_languages(
            Path("unused"),
            Path("unused-report.jsonl"),
            policy=DatasetLanguagePolicy(detector_languages=("id", "id")),
        )


def test_quality_report_cannot_be_written_inside_immutable_release(tmp_path: Path) -> None:
    release = _release(tmp_path)
    with pytest.raises(DatasetLanguageError, match="outside the immutable dataset release"):
        analyze_dataset_languages(release, release / "quality.jsonl")


def test_quality_report_refuses_to_overwrite_existing_outputs(tmp_path: Path) -> None:
    release = _release(tmp_path)
    report = tmp_path / "quality.jsonl"
    report.write_text("existing", encoding="utf-8")
    with pytest.raises(DatasetLanguageError, match="already exists"):
        analyze_dataset_languages(release, report)
    assert report.read_text(encoding="utf-8") == "existing"
