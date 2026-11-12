from pathlib import Path

import pytest

from rusamodels.training.train_sft import (
    _assert_input_file_unchanged,
    _create_new_output_directory,
    _provenance_path,
    _snapshot_input_file,
    _write_new_file,
)


def test_training_input_snapshot_detects_dataset_mutation(tmp_path: Path) -> None:
    dataset = tmp_path / "train.jsonl"
    dataset.write_text('{"id":"one"}\n', encoding="utf-8")
    resolved, snapshot = _snapshot_input_file(dataset, "training dataset")

    _assert_input_file_unchanged(snapshot, "training dataset")
    assert resolved == dataset.resolve()

    dataset.write_text('{"id":"changed"}\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match="changed during training"):
        _assert_input_file_unchanged(snapshot, "training dataset")


def test_training_input_symlink_is_rejected(tmp_path: Path) -> None:
    target = tmp_path / "train.jsonl"
    target.write_text('{"id":"one"}\n', encoding="utf-8")
    link = tmp_path / "train-link.jsonl"
    try:
        link.symlink_to(target)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation is not supported in this environment")

    with pytest.raises(ValueError, match="must not be a symlink"):
        _snapshot_input_file(link, "training dataset")


def test_training_output_directory_must_be_new(tmp_path: Path) -> None:
    output = _create_new_output_directory("run-001", root=tmp_path)
    assert output.is_dir()

    with pytest.raises(ValueError, match="already exists"):
        _create_new_output_directory("run-001", root=tmp_path)


def test_training_provenance_file_write_does_not_overwrite(tmp_path: Path) -> None:
    destination = tmp_path / "provenance.json"
    destination.write_text("original", encoding="utf-8")

    with pytest.raises(FileExistsError):
        _write_new_file(destination, b"replacement")

    assert destination.read_text(encoding="utf-8") == "original"


def test_training_provenance_paths_do_not_expose_external_absolute_paths(tmp_path: Path) -> None:
    external = tmp_path / "private-home" / "customer-training-data.jsonl"
    external.parent.mkdir()
    external.write_text("private", encoding="utf-8")
    _, snapshot = _snapshot_input_file(external, "training dataset")

    value = _provenance_path(snapshot, root=tmp_path / "repository")

    assert value == f"external/sha256/{snapshot['sha256']}"
    assert "private-home" not in value
    assert "customer-training-data" not in value


def test_training_provenance_paths_keep_repository_relative_locator(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    dataset = repository / "data" / "train.jsonl"
    dataset.parent.mkdir(parents=True)
    dataset.write_text("data", encoding="utf-8")
    _, snapshot = _snapshot_input_file(dataset, "training dataset")

    assert _provenance_path(snapshot, root=repository) == "data/train.jsonl"
