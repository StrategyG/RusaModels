import hashlib
from pathlib import Path

import pytest

from rusamodels.evaluation.evaluate_agent_sft import (
    _hash_adapter_directory,
    _pseudonymize_record_id,
    _safe_report_path,
    _validate_provenance,
    _write_new_file,
)


def _provenance() -> dict:
    return {
        "schema_version": 1,
        "model": {
            "id": "organization/model",
            "configured_revision": "a" * 40,
            "resolved_revision": "a" * 40,
            "license": "apache-2.0",
        },
        "tokenizer": {
            "id": "organization/model",
            "configured_revision": "a" * 40,
            "resolved_revision": "a" * 40,
        },
        "dataset": {
            "train_sha256": "b" * 64,
            "eval_sha256": "c" * 64,
            "eval_examples": 36,
            "training_config_sha256": "d" * 64,
        },
    }


def test_evaluation_requires_exact_training_eval_digest_and_revision() -> None:
    model_id, revision = _validate_provenance(
        _provenance(),
        {"id": "organization/model", "revision": "a" * 40},
        "c" * 64,
        36,
        "d" * 64,
    )

    assert model_id == "organization/model"
    assert revision == "a" * 40


def test_evaluation_rejects_different_eval_file_or_model_revision() -> None:
    with pytest.raises(ValueError, match="SHA-256"):
        _validate_provenance(
            _provenance(),
            {"id": "organization/model", "revision": "a" * 40},
            "d" * 64,
            36,
            "d" * 64,
        )

    with pytest.raises(ValueError, match="does not match the configured model"):
        _validate_provenance(
            _provenance(),
            {"id": "organization/model", "revision": "e" * 40},
            "c" * 64,
            36,
            "d" * 64,
        )


def test_adapter_directory_digest_is_stable_and_detects_mutation(tmp_path: Path) -> None:
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text('{"r":8}', encoding="utf-8")
    first = _hash_adapter_directory(adapter)
    assert first == _hash_adapter_directory(adapter)

    (adapter / "adapter_config.json").write_text('{"r":16}', encoding="utf-8")
    assert first != _hash_adapter_directory(adapter)


def test_adapter_directory_symlinks_are_rejected(tmp_path: Path) -> None:
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    target = tmp_path / "outside.txt"
    target.write_text("outside", encoding="utf-8")
    try:
        (adapter / "linked.txt").symlink_to(target)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation is not supported in this environment")

    with pytest.raises(ValueError, match="contains a symlink"):
        _hash_adapter_directory(adapter)


def test_evaluation_report_must_be_new_and_outside_adapter(tmp_path: Path) -> None:
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    evaluation = tmp_path / "eval.jsonl"
    evaluation.write_text("{}", encoding="utf-8")
    config = tmp_path / "config.yaml"
    config.write_text("model: {}", encoding="utf-8")

    with pytest.raises(ValueError, match="outside adapter"):
        _safe_report_path(
            adapter / "report.json",
            adapter_dir=adapter,
            eval_file=evaluation,
            config_file=config,
        )

    report = _safe_report_path(
        tmp_path / "reports" / "evaluation.json",
        adapter_dir=adapter,
        eval_file=evaluation,
        config_file=config,
    )
    assert report.parent.is_dir()


def test_evaluation_report_write_is_no_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "report.json"
    output.write_text("original", encoding="utf-8")

    with pytest.raises(FileExistsError):
        _write_new_file(output, b"replacement")

    assert output.read_text(encoding="utf-8") == "original"


def test_report_record_ids_are_pseudonymized() -> None:
    value = _pseudonymize_record_id("customer@example.com")

    assert value.startswith("record_")
    assert len(value) == 31
    assert "customer" not in value
    assert hashlib.sha256(b"customer@example.com").hexdigest()[:24] in value
