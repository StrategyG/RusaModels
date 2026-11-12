import hashlib
import json
from pathlib import Path

import pytest

from rusamodels.data_pipeline import (
    DatasetReleaseConfig,
    DatasetSource,
    build_dataset_release,
)
from rusamodels.dataset_contamination import (
    DatasetContaminationError,
    check_frozen_evaluation_contamination,
)
from rusamodels.dataset_quality import simhash64_text, simhash_bands
from rusamodels.dataset_text_export import record_to_text


def _write_jsonl(path: Path, records: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    return path


def _release(tmp_path: Path, records: list[dict] | None = None) -> Path:
    if records is None:
        records = [
            {
                "id": f"example-{index}",
                "group_id": f"document-{index}",
                "text": f"Independent dataset example number {index} with distinct subject matter",
            }
            for index in range(1, 4)
        ]
    source = _write_jsonl(tmp_path / "source.jsonl", records)
    return build_dataset_release(
        DatasetReleaseConfig(
            dataset_id="contamination-test",
            version="1.0.0",
            sources=(DatasetSource("approved-source", source, "CC0-1.0"),),
            output_root=tmp_path / "releases",
            seed=31,
            validation_fraction=0.2,
            test_fraction=0.2,
        )
    )


def _frozen_set(tmp_path: Path, records: list[dict]) -> tuple[Path, Path]:
    evaluation = _write_jsonl(tmp_path / "frozen-evaluation.jsonl", records)
    manifest = {
        "schema_version": "1.0.0",
        "purpose": "frozen_evaluation",
        "dataset_id": "rusa-frozen-eval",
        "dataset_version": "1.0.0",
        "locked": True,
        "file": evaluation.name,
        "sha256": hashlib.sha256(evaluation.read_bytes()).hexdigest(),
        "size_bytes": evaluation.stat().st_size,
        "records": len(records),
        "created_at": "2026-01-01T00:00:00Z",
    }
    manifest_path = tmp_path / "frozen-evaluation.manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return evaluation, manifest_path


def test_exact_frozen_evaluation_match_fails_contamination_gate(tmp_path: Path) -> None:
    release = _release(tmp_path)
    target = json.loads((release / "train.jsonl").read_text(encoding="utf-8").splitlines()[0])
    evaluation, manifest = _frozen_set(
        tmp_path,
        [{"id": "frozen-001", "text": target["text"]}],
    )

    result = check_frozen_evaluation_contamination(
        release,
        evaluation,
        manifest,
        tmp_path / "reports" / "contamination.jsonl",
    )

    assert result["contamination_free"] is False
    assert result["exact_matches"] == 1
    report_line = (tmp_path / "reports" / "contamination.jsonl").read_text(encoding="utf-8")
    report = json.loads(report_line)
    assert report["match_type"] == "exact"
    assert target["text"] not in report_line
    assert (tmp_path / "reports" / "contamination.jsonl.summary.json").is_file()


def test_simhash_near_duplicate_against_frozen_evaluation_is_detected(tmp_path: Path) -> None:
    source_text = "The quick brown fox jumps over the lazy dog and jumps again"
    frozen_text = "The quick brown fox jumps over the lazy dog and jumps once"
    assert (simhash64_text(source_text) ^ simhash64_text(frozen_text)).bit_count() <= 7
    release = _release(
        tmp_path,
        [
            {"id": "a", "group_id": "a", "text": source_text},
            {"id": "b", "group_id": "b", "text": "A database migration should be reversible and tested"},
            {"id": "c", "group_id": "c", "text": "A tokenizer maps Unicode text into model-specific token IDs"},
        ],
    )
    evaluation, manifest = _frozen_set(
        tmp_path,
        [{"id": "frozen-near", "text": frozen_text}],
    )

    result = check_frozen_evaluation_contamination(
        release,
        evaluation,
        manifest,
        tmp_path / "near-duplicates.jsonl",
    )

    assert result["contamination_free"] is False
    assert result["near_duplicate_matches"] == 1


def test_unrelated_frozen_evaluation_set_passes(tmp_path: Path) -> None:
    release = _release(tmp_path)
    evaluation, manifest = _frozen_set(
        tmp_path,
        [{"id": "frozen-001", "text": "Oceanographic sensors measure salinity and temperature in deep water"}],
    )

    result = check_frozen_evaluation_contamination(
        release,
        evaluation,
        manifest,
        tmp_path / "clean-report.jsonl",
    )

    assert result["contamination_free"] is True
    assert result["total_matches"] == 0
    assert (tmp_path / "clean-report.jsonl").read_text(encoding="utf-8") == ""


def test_frozen_evaluation_file_tampering_is_rejected(tmp_path: Path) -> None:
    release = _release(tmp_path)
    evaluation, manifest = _frozen_set(
        tmp_path,
        [{"id": "frozen-001", "text": "A separately frozen evaluation prompt"}],
    )
    evaluation.write_text('{"id":"frozen-001","text":"modified"}\n', encoding="utf-8")

    with pytest.raises(DatasetContaminationError, match="SHA-256 does not match"):
        check_frozen_evaluation_contamination(
            release,
            evaluation,
            manifest,
            tmp_path / "report.jsonl",
        )


def test_frozen_evaluation_duplicate_content_is_rejected(tmp_path: Path) -> None:
    release = _release(tmp_path)
    evaluation, manifest = _frozen_set(
        tmp_path,
        [
            {"id": "frozen-001", "text": "Repeated frozen evaluation prompt"},
            {"id": "frozen-002", "text": "Repeated frozen evaluation prompt"},
        ],
    )

    with pytest.raises(DatasetContaminationError, match="duplicate frozen record ID or exact content"):
        check_frozen_evaluation_contamination(
            release,
            evaluation,
            manifest,
            tmp_path / "report.jsonl",
        )


def test_contamination_report_cannot_be_written_inside_immutable_release(tmp_path: Path) -> None:
    release = _release(tmp_path)
    evaluation, manifest = _frozen_set(
        tmp_path,
        [{"id": "frozen-001", "text": "Independent evaluation prompt"}],
    )

    with pytest.raises(DatasetContaminationError, match="outside the immutable dataset release"):
        check_frozen_evaluation_contamination(
            release,
            evaluation,
            manifest,
            release / "contamination.jsonl",
        )


def test_frozen_evaluation_manifest_must_be_locked(tmp_path: Path) -> None:
    release = _release(tmp_path)
    evaluation, manifest_path = _frozen_set(
        tmp_path,
        [{"id": "frozen-001", "text": "Independent evaluation prompt"}],
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["locked"] = False
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(DatasetContaminationError, match="locked=true"):
        check_frozen_evaluation_contamination(
            release,
            evaluation,
            manifest_path,
            tmp_path / "report.jsonl",
        )


def test_lsh_candidate_cap_is_applied_before_loading_all_candidates(tmp_path: Path) -> None:
    release = _release(tmp_path)
    target = json.loads((release / "train.jsonl").read_text(encoding="utf-8").splitlines()[0])
    base_text = record_to_text(target)
    base_bands = set(simhash_bands(simhash64_text(base_text)))
    variants = []
    for suffix in (
        "alpha", "bravo", "charlie", "delta", "echo", "foxtrot",
        "golf", "hotel", "india", "juliet", "kilo", "lima",
        "mike", "november", "oscar", "papa", "quebec", "romeo",
        "sierra", "tango", "uniform", "victor", "whiskey", "xray",
        "yankee", "zulu",
    ):
        text = f"{base_text} {suffix}"
        bands = set(simhash_bands(simhash64_text(text)))
        if base_bands.intersection(bands):
            variants.append({"id": f"frozen-{suffix}", "text": text})
        if len(variants) == 2:
            break
    assert len(variants) == 2
    evaluation, manifest = _frozen_set(tmp_path, variants)

    with pytest.raises(DatasetContaminationError, match="candidate set exceeds"):
        check_frozen_evaluation_contamination(
            release,
            evaluation,
            manifest,
            tmp_path / "candidate-cap.jsonl",
            max_candidates_per_record=1,
        )
