import hashlib
import json
from pathlib import Path

import pytest

from rusamodels.data_pipeline import (
    DatasetPipelineError,
    DatasetReleaseConfig,
    DatasetSource,
    build_dataset_release,
)
from rusamodels.dataset_verification import (
    DatasetVerificationError,
    verify_dataset_release,
)


def _write_jsonl(path: Path, records: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    return path


def _record(index: int, *, group_id: str | None = None, text: str | None = None) -> dict:
    return {
        "id": f"record-{index}",
        "group_id": group_id or f"group-{index}",
        "text": text or f"Example training content {index}",
        "metadata": {"category": "unit-test"},
    }


def _config(tmp_path: Path, source: Path, *, version: str = "1.0.0") -> DatasetReleaseConfig:
    return DatasetReleaseConfig(
        dataset_id="sample-dataset",
        version=version,
        sources=(
            DatasetSource(
                source_id="curated-source",
                path=source,
                license_id="LicenseRef-Authorized-Test-Data",
            ),
        ),
        output_root=tmp_path / "releases",
        seed=17,
        validation_fraction=0.20,
        test_fraction=0.20,
    )


def test_build_release_records_provenance_redacts_and_verifies(tmp_path: Path) -> None:
    source_a = _write_jsonl(
        tmp_path / "source-a.jsonl",
        [
            _record(1, text="Contact alice@example.test; key sk-" + "A" * 24),
            _record(2),
            _record(3),
        ],
    )
    source_b = _write_jsonl(tmp_path / "source-b.jsonl", [_record(1, text="Contact alice@example.test; key sk-" + "A" * 24)])
    config = DatasetReleaseConfig(
        dataset_id="sample-dataset",
        version="1.0.0",
        sources=(
            DatasetSource("source-a", source_a, "LicenseRef-Authorized-Test-Data"),
            DatasetSource("source-b", source_b, "LicenseRef-Authorized-Test-Data"),
        ),
        output_root=tmp_path / "releases",
        seed=17,
        validation_fraction=0.20,
        test_fraction=0.20,
    )

    release = build_dataset_release(config)
    manifest = json.loads((release / "manifest.json").read_text(encoding="utf-8"))
    all_records = []
    for split in ("train", "validation", "test"):
        all_records.extend(
            json.loads(line)
            for line in (release / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()
        )
    redacted = next(item for item in all_records if "[REDACTED_EMAIL]" in item["text"])
    assert "alice@example.test" not in redacted["text"]
    assert "[REDACTED_EMAIL]" in redacted["text"]
    assert "[REDACTED_SECRET]" in redacted["text"]
    assert redacted["id"].startswith("record_")
    assert redacted["id"] != "record-1"
    assert manifest["identifier_policy"]["record_ids"] == "sha256_pseudonymized_or_content_derived"
    assert manifest["records"]["raw_records"] == 4
    assert manifest["records"]["unique_records"] == 3
    assert manifest["records"]["duplicates_removed"] == 1
    assert manifest["redaction"]["counts"]["emails"] == 2
    assert manifest["redaction"]["counts"]["secrets"] == 2
    assert verify_dataset_release(release)["unique_records"] == 3


def test_grouped_records_never_cross_splits(tmp_path: Path) -> None:
    records = [
        _record(1, group_id="conversation-a"),
        _record(2, group_id="conversation-a"),
        _record(3, group_id="conversation-b"),
        _record(4, group_id="conversation-b"),
        _record(5, group_id="conversation-c"),
        _record(6, group_id="conversation-c"),
    ]
    source = _write_jsonl(tmp_path / "grouped.jsonl", records)
    release = build_dataset_release(_config(tmp_path, source))
    provenance = [
        json.loads(line)
        for line in (release / "provenance.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    group_splits: dict[str, set[str]] = {}
    for item in provenance:
        group_splits.setdefault(item["group_id"], set()).add(item["split"])
    assert len(group_splits) == 3
    assert all(group_id.startswith("group_") for group_id in group_splits)
    assert all(len(splits) == 1 for splits in group_splits.values())
    assert verify_dataset_release(release)["raw_records"] == 6


def test_release_data_files_are_deterministic_for_same_inputs_and_seed(tmp_path: Path) -> None:
    source = _write_jsonl(tmp_path / "records.jsonl", [_record(i) for i in range(1, 9)])
    first = build_dataset_release(_config(tmp_path / "one", source))
    second_config = DatasetReleaseConfig(
        dataset_id="sample-dataset",
        version="1.0.0",
        sources=(DatasetSource("curated-source", source, "LicenseRef-Authorized-Test-Data"),),
        output_root=tmp_path / "two" / "releases",
        seed=17,
        validation_fraction=0.20,
        test_fraction=0.20,
    )
    second = build_dataset_release(second_config)
    for filename in ("train.jsonl", "validation.jsonl", "test.jsonl", "provenance.jsonl"):
        assert (first / filename).read_bytes() == (second / filename).read_bytes()


def test_invalid_jsonl_does_not_publish_partial_release(tmp_path: Path) -> None:
    source = tmp_path / "invalid.jsonl"
    source.write_text('{"id":"one","text":"valid"}\n{"id": broken}\n', encoding="utf-8")
    config = _config(tmp_path, source)
    with pytest.raises(DatasetPipelineError, match=r"curated-source:2"):
        build_dataset_release(config)
    release_parent = tmp_path / "releases" / "sample-dataset"
    assert not (release_parent / "1.0.0").exists()
    assert not (release_parent / ".1.0.0.lock").exists()
    assert not list(release_parent.glob(".dataset-release-*"))


def test_same_record_id_with_different_content_is_rejected(tmp_path: Path) -> None:
    source = _write_jsonl(
        tmp_path / "collision.jsonl",
        [{"id": "reused", "text": "first content"}, {"id": "reused", "text": "different content"}],
    )
    with pytest.raises(DatasetPipelineError, match="reused for different content"):
        build_dataset_release(_config(tmp_path, source))


def test_existing_release_is_immutable(tmp_path: Path) -> None:
    source = _write_jsonl(tmp_path / "records.jsonl", [_record(i) for i in range(1, 5)])
    config = _config(tmp_path, source)
    release = build_dataset_release(config)
    original_manifest = (release / "manifest.json").read_bytes()
    with pytest.raises(DatasetPipelineError, match="immutable and already exists"):
        build_dataset_release(config)
    assert (release / "manifest.json").read_bytes() == original_manifest


def test_verifier_detects_modified_split_artifact(tmp_path: Path) -> None:
    source = _write_jsonl(tmp_path / "records.jsonl", [_record(i) for i in range(1, 5)])
    release = build_dataset_release(_config(tmp_path, source))
    with (release / "train.jsonl").open("a", encoding="utf-8") as handle:
        handle.write('{"id":"tampered","text":"untracked"}\n')
    with pytest.raises(DatasetVerificationError, match="SHA-256 verification failed"):
        verify_dataset_release(release)


def test_invalid_split_fraction_is_rejected_before_writing(tmp_path: Path) -> None:
    source = _write_jsonl(tmp_path / "records.jsonl", [_record(i) for i in range(1, 5)])
    config = DatasetReleaseConfig(
        dataset_id="sample-dataset",
        version="1.0.0",
        sources=(DatasetSource("curated-source", source, "LicenseRef-Authorized-Test-Data"),),
        output_root=tmp_path / "releases",
        validation_fraction=0.7,
        test_fraction=0.4,
    )
    with pytest.raises(DatasetPipelineError, match="sum must be less than 1"):
        build_dataset_release(config)


def test_source_uri_cannot_embed_credentials_or_query_tokens(tmp_path: Path) -> None:
    source = _write_jsonl(tmp_path / "records.jsonl", [_record(i) for i in range(1, 4)])
    config = DatasetReleaseConfig(
        dataset_id="sample-dataset",
        version="1.0.0",
        sources=(
            DatasetSource(
                "curated-source",
                source,
                "LicenseRef-Authorized-Test-Data",
                "https://user:password@example.test/data?token=secret",
            ),
        ),
        output_root=tmp_path / "releases",
    )
    with pytest.raises(DatasetPipelineError, match="must not contain credentials"):
        build_dataset_release(config)


def test_exact_duplicate_content_with_conflicting_group_ids_is_rejected(tmp_path: Path) -> None:
    source = _write_jsonl(
        tmp_path / "conflicting-groups.jsonl",
        [
            {"id": "first", "group_id": "conversation-a", "text": "same content"},
            {"id": "second", "group_id": "conversation-b", "text": "same content"},
            _record(3),
            _record(4),
        ],
    )
    with pytest.raises(DatasetPipelineError, match="different group_id values"):
        build_dataset_release(_config(tmp_path, source))


def test_verifier_detects_group_leakage_even_when_manifest_checksums_are_recomputed(
    tmp_path: Path,
) -> None:
    source = _write_jsonl(
        tmp_path / "records.jsonl",
        [
            _record(1, group_id="group-a"),
            _record(2, group_id="group-b"),
            _record(3, group_id="group-c"),
        ],
    )
    release = build_dataset_release(_config(tmp_path, source))
    provenance_path = release / "provenance.jsonl"
    provenance = [
        json.loads(line)
        for line in provenance_path.read_text(encoding="utf-8").splitlines()
    ]
    assert provenance[0]["split"] != provenance[1]["split"]
    provenance[1]["group_id"] = provenance[0]["group_id"]
    provenance_bytes = (
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in provenance)
    ).encode("utf-8")
    provenance_path.write_bytes(provenance_bytes)

    manifest_path = release / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"]["provenance.jsonl"]["sha256"] = hashlib.sha256(
        provenance_bytes
    ).hexdigest()
    manifest["files"]["provenance.jsonl"]["size_bytes"] = len(provenance_bytes)
    manifest_bytes = (
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")
    manifest_path.write_bytes(manifest_bytes)
    (release / "manifest.sha256").write_text(
        hashlib.sha256(manifest_bytes).hexdigest() + "\n",
        encoding="ascii",
    )

    with pytest.raises(DatasetVerificationError, match="group crosses"):
        verify_dataset_release(release)


def test_disk_backed_verifier_handles_a_larger_dataset_release(tmp_path: Path) -> None:
    source = _write_jsonl(
        tmp_path / "larger.jsonl",
        [
            {
                "id": f"large-{index}",
                "group_id": f"conversation-{index}",
                "text": f"Distinct dataset example number {index} with unique topic {index * 17}",
            }
            for index in range(1, 1001)
        ],
    )
    release = build_dataset_release(_config(tmp_path, source))
    summary = verify_dataset_release(release)
    assert summary["unique_records"] == 1000
    assert summary["raw_records"] == 1000
    assert sum(summary["split_counts"].values()) == 1000
