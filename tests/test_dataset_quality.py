import json
from pathlib import Path

import pytest

from rusamodels.data_pipeline import (
    DatasetReleaseConfig,
    DatasetSource,
    build_dataset_release,
)
from rusamodels.dataset_quality import (
    DatasetQualityError,
    find_near_duplicates,
)


def _write_jsonl(path: Path, records: list[dict]) -> Path:
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    return path


def _build_release(tmp_path: Path, records: list[dict]) -> Path:
    source = _write_jsonl(tmp_path / "source.jsonl", records)
    config = DatasetReleaseConfig(
        dataset_id="quality-test",
        version="1.0.0",
        sources=(DatasetSource("quality-source", source, "LicenseRef-Quality-Test"),),
        output_root=tmp_path / "releases",
        seed=19,
        validation_fraction=0.2,
        test_fraction=0.2,
    )
    return build_dataset_release(config)


def test_near_duplicate_across_splits_is_reported(tmp_path: Path) -> None:
    release = _build_release(
        tmp_path,
        [
            {
                "id": "record-a",
                "group_id": "conversation-a",
                "text": "The quick brown fox jumps over the lazy dog",
            },
            {
                "id": "record-b",
                "group_id": "conversation-b",
                "text": "The quick brown fox jumps over the lazy dog today",
            },
            {
                "id": "record-c",
                "group_id": "conversation-c",
                "text": "Quantum mechanics studies particles and fields in theoretical physics",
            },
        ],
    )
    report_path = tmp_path / "reports" / "near-duplicates.jsonl"
    summary = find_near_duplicates(release, report_path)
    pairs = [
        json.loads(line)
        for line in report_path.read_text(encoding="utf-8").splitlines()
    ]
    assert summary["records_scanned"] == 3
    assert summary["near_duplicate_pairs"] == 1
    assert summary["cross_split_pairs"] == 1
    assert len(pairs) == 1
    assert pairs[0]["split_a"] != pairs[0]["split_b"]
    assert pairs[0]["hamming_distance"] <= 7


def test_near_duplicates_within_same_group_are_not_flagged(tmp_path: Path) -> None:
    release = _build_release(
        tmp_path,
        [
            {
                "id": "record-a",
                "group_id": "same-conversation",
                "text": "The quick brown fox jumps over the lazy dog",
            },
            {
                "id": "record-b",
                "group_id": "same-conversation",
                "text": "The quick brown fox jumps over the lazy dog today",
            },
            {
                "id": "record-c",
                "group_id": "other-conversation",
                "text": "Quantum mechanics studies particles and fields in theoretical physics",
            },
            {
                "id": "record-d",
                "group_id": "third-conversation",
                "text": "A botanical field guide describes tropical plants and forest ecosystems",
            },
        ],
    )
    report_path = tmp_path / "reports" / "same-group.jsonl"
    summary = find_near_duplicates(release, report_path)
    assert summary["near_duplicate_pairs"] == 0
    assert summary["cross_split_pairs"] == 0
    assert report_path.read_text(encoding="utf-8") == ""


def test_near_duplicate_report_cannot_be_written_inside_immutable_release(
    tmp_path: Path,
) -> None:
    release = _build_release(
        tmp_path,
        [
            {"id": "a", "group_id": "a", "text": "first independent record"},
            {"id": "b", "group_id": "b", "text": "second unrelated record"},
            {"id": "c", "group_id": "c", "text": "third distinct example"},
        ],
    )
    with pytest.raises(DatasetQualityError, match="outside the immutable dataset release"):
        find_near_duplicates(release, release / "near-duplicates.jsonl")


def test_near_duplicate_threshold_is_limited_to_lsh_recall_guarantee(
    tmp_path: Path,
) -> None:
    release = _build_release(
        tmp_path,
        [
            {"id": "a", "group_id": "a", "text": "first independent record"},
            {"id": "b", "group_id": "b", "text": "second unrelated record"},
            {"id": "c", "group_id": "c", "text": "third distinct example"},
        ],
    )
    with pytest.raises(DatasetQualityError, match="between 0 and 7"):
        find_near_duplicates(
            release,
            tmp_path / "report.jsonl",
            max_hamming_distance=8,
        )
