import json
from pathlib import Path

import pytest

from rusamodels.data_pipeline import (
    DatasetReleaseConfig,
    DatasetSource,
    build_dataset_release,
)
from rusamodels.dataset_inspect import (
    DatasetInspectionError,
    inspect_dataset_release,
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
            "id": f"example-{index}",
            "group_id": f"document-{index}",
            "agent_role": "analyst" if index % 2 else "developer",
            "task_type": "technical_instruction",
            "language": "id" if index % 2 else "en",
            "text": f"Dataset inspection example {index} contains a distinct technical topic and supporting details.",
        }
        for index in range(1, 13)
    ]
    source = _write_jsonl(tmp_path / "source.jsonl", records)
    return build_dataset_release(
        DatasetReleaseConfig(
            dataset_id="inspection-test",
            version="1.0.0",
            sources=(DatasetSource("approved-source", source, "CC0-1.0"),),
            output_root=tmp_path / "releases",
            seed=47,
            validation_fraction=0.2,
            test_fraction=0.2,
        )
    )


def test_inspection_returns_deterministic_samples_and_split_metrics(tmp_path: Path) -> None:
    release = _release(tmp_path)

    first = inspect_dataset_release(
        release,
        sample_per_split=2,
        seed=101,
        preview_characters=35,
    )
    second = inspect_dataset_release(
        release,
        sample_per_split=2,
        seed=101,
        preview_characters=35,
    )

    assert first["dataset_id"] == "inspection-test"
    assert first["splits"] == second["splits"]
    for split in ("train", "validation", "test"):
        summary = first["splits"][split]
        assert summary["records"] > 0
        assert len(summary["samples"]) == 2
        assert all(len(sample["preview"]) <= 35 for sample in summary["samples"])
        assert summary["agent_role_counts"]
        assert summary["task_type_counts"] == {"technical_instruction": summary["records"]}


def test_inspection_can_disable_samples(tmp_path: Path) -> None:
    release = _release(tmp_path)
    result = inspect_dataset_release(release, sample_per_split=0)
    assert all(summary["samples"] == [] for summary in result["splits"].values())


def test_inspection_rejects_invalid_sample_limits(tmp_path: Path) -> None:
    release = _release(tmp_path)
    with pytest.raises(DatasetInspectionError, match="sample_per_split"):
        inspect_dataset_release(release, sample_per_split=101)
    with pytest.raises(DatasetInspectionError, match="preview_characters"):
        inspect_dataset_release(release, preview_characters=0)
