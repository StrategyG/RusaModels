import hashlib
import json
from pathlib import Path

import pytest

from rusamodels.data_pipeline import (
    DatasetReleaseConfig,
    DatasetSource,
    build_dataset_release,
)
from rusamodels.dataset_text_export import (
    DatasetTextExportError,
    export_dataset_split,
    record_to_text,
)


def _write_jsonl(path: Path, records: list[dict]) -> Path:
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    return path


def _release(tmp_path: Path) -> Path:
    source = _write_jsonl(
        tmp_path / "source.jsonl",
        [
            {
                "id": f"source-{index}",
                "group_id": f"conversation-{index}",
                "agent_role": "researcher",
                "messages": [{"role": "user", "content": f"Question number {index}"}],
                "response": f"Answer number {index}",
            }
            for index in range(1, 4)
        ],
    )
    config = DatasetReleaseConfig(
        dataset_id="text-export-test",
        version="1.0.0",
        sources=(DatasetSource("approved-source", source, "CC0-1.0"),),
        output_root=tmp_path / "releases",
        seed=23,
        validation_fraction=0.2,
        test_fraction=0.2,
    )
    return build_dataset_release(config)


def test_export_verified_split_and_write_source_manifest(tmp_path: Path) -> None:
    release = _release(tmp_path)
    output = tmp_path / "corpora" / "train.txt"
    result = export_dataset_split(release, "train", output)
    exported = output.read_text(encoding="utf-8")
    manifest = json.loads(
        output.with_suffix(output.suffix + ".manifest.json").read_text(encoding="utf-8")
    )
    assert "user: Question number" in exported
    assert "assistant: Answer number" in exported
    assert "source-1" not in exported
    assert "conversation-1" not in exported
    assert manifest["dataset_id"] == "text-export-test"
    assert manifest["dataset_version"] == "1.0.0"
    assert manifest["split"] == "train"
    assert manifest["license_ids"] == ["CC0-1.0"]
    assert manifest["output_sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    assert result["records"] == 1


def test_record_text_export_excludes_identifiers_and_metadata() -> None:
    rendered = record_to_text(
        {
            "id": "private-record-id",
            "group_id": "private-customer-id",
            "text": "Public training content",
            "metadata": {"owner": "private@example.test"},
            "source_uri": "https://example.invalid/private",
        }
    )
    assert rendered == "Public training content"
    assert "private-record-id" not in rendered
    assert "private-customer-id" not in rendered
    assert "private@example.test" not in rendered


def test_export_refuses_to_modify_immutable_release(tmp_path: Path) -> None:
    release = _release(tmp_path)
    with pytest.raises(DatasetTextExportError, match="outside the immutable dataset release"):
        export_dataset_split(release, "train", release / "train.txt")


def test_export_refuses_to_overwrite_existing_output(tmp_path: Path) -> None:
    release = _release(tmp_path)
    output = tmp_path / "train.txt"
    output.write_text("existing", encoding="utf-8")
    with pytest.raises(DatasetTextExportError, match="already exists"):
        export_dataset_split(release, "train", output)
    assert output.read_text(encoding="utf-8") == "existing"


def test_export_rejects_unsupported_split(tmp_path: Path) -> None:
    release = _release(tmp_path)
    with pytest.raises(DatasetTextExportError, match="split must be"):
        export_dataset_split(release, "eval", tmp_path / "eval.txt")
