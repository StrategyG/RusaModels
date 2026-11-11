from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import pytest


torch = pytest.importorskip("torch")

from scripts.train_carbon_cpu import (  # noqa: E402
    build_token_cache,
    capture_rng_state,
    main,
    restore_rng_state,
    sample_batch,
    save_checkpoint,
    training_contract,
)


def _write_corpus(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _write_config(tmp_path: Path, *, same_eval: bool = False) -> Path:
    train = _write_corpus(tmp_path / "train.txt", "training corpus with enough unique bytes. " * 80)
    evaluation = train if same_eval else _write_corpus(
        tmp_path / "eval.txt", "held out validation corpus with different content. " * 80
    )
    config = {
        "project": "Carbon-CPU",
        "run_name": "carbon-test",
        "data": {
            "train_file": str(train),
            "eval_file": str(evaluation),
            "validation_split": 0.1,
            "encoding": "utf-8",
            "tokenizer": "byte-level-v1",
            "dataset_id": "carbon-test",
            "dataset_version": "0.1.0",
            "license_id": "CC0-1.0",
            "source_uri": None,
        },
        "model": {
            "vocab_size": 258,
            "n_layers": 1,
            "d_model": 32,
            "n_heads": 4,
            "n_kv_heads": 2,
            "intermediate_size": 64,
            "max_seq_len": 32,
            "dropout": 0.0,
            "rope_theta": 10000.0,
        },
        "training": {
            "output_dir": str(tmp_path / "artifacts"),
            "steps": 2,
            "batch_size": 1,
            "block_size": 16,
            "gradient_accumulation_steps": 1,
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "warmup_steps": 1,
            "max_grad_norm": 1.0,
            "eval_interval": 1,
            "eval_batches": 1,
            "threads": 1,
            "seed": 42,
        },
    }
    config_path = tmp_path / "training.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return config_path


def test_build_token_cache_is_byte_exact_and_appends_eos(tmp_path: Path) -> None:
    source = _write_corpus(tmp_path / "corpus.txt", "RusaAi 🇮🇩")
    destination = tmp_path / "cache.u16"
    count = build_token_cache(source, destination)
    tokens = np.memmap(destination, dtype=np.uint16, mode="r", shape=(count,))
    assert count == len("RusaAi 🇮🇩".encode("utf-8")) + 1
    assert list(tokens[:-1]) == list("RusaAi 🇮🇩".encode("utf-8"))
    assert int(tokens[-1]) == 256


def test_sample_batch_returns_next_byte_targets() -> None:
    tokens = np.arange(100, dtype=np.uint16)
    inputs, targets = sample_batch(
        tokens, batch_size=4, block_size=12, rng=np.random.default_rng(5), torch=torch
    )
    assert inputs.shape == (4, 12)
    assert targets.shape == (4, 12)
    assert torch.equal(inputs[:, 1:], targets[:, :-1])


def test_training_contract_allows_more_steps_but_not_changed_optimizer() -> None:
    effective = {"batch_size": 2, "block_size": 32, "threads": 1}
    first = training_contract(
        {"steps": 10, "output_dir": "one", "learning_rate": 0.001}, effective
    )
    resumed = training_contract(
        {"steps": 100, "output_dir": "two", "learning_rate": 0.001}, effective
    )
    changed = training_contract(
        {"steps": 100, "output_dir": "two", "learning_rate": 0.002}, effective
    )
    assert first == resumed
    assert first != changed


def test_rng_state_restores_python_numpy_and_torch_streams() -> None:
    random.seed(41)
    numpy_rng = np.random.default_rng(42)
    eval_rng = np.random.default_rng(43)
    torch.manual_seed(44)
    state = capture_rng_state(numpy_rng, eval_rng, torch)
    expected = (
        random.random(),
        numpy_rng.integers(0, 100000),
        eval_rng.integers(0, 100000),
        torch.rand(3),
    )
    restore_rng_state(state, numpy_rng, eval_rng, torch)
    actual = (
        random.random(),
        numpy_rng.integers(0, 100000),
        eval_rng.integers(0, 100000),
        torch.rand(3),
    )
    assert actual[:3] == expected[:3]
    assert torch.equal(actual[3], expected[3])


def test_checkpoint_write_is_loadable_with_weights_only(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, {"step": 2, "weights": torch.ones(2)}, torch)
    loaded = torch.load(path, map_location="cpu", weights_only=True)
    assert loaded["step"] == 2
    assert torch.equal(loaded["weights"], torch.ones(2))


def test_cpu_training_run_can_resume_with_same_dataset_and_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = _write_config(tmp_path)
    monkeypatch.setattr(sys, "argv", ["train_carbon_cpu.py", "--config", str(config_path)])
    assert main() == 0
    output_dir = tmp_path / "artifacts"
    checkpoint = output_dir / "checkpoint.pt"
    assert checkpoint.is_file()
    assert (output_dir / "final.pt").is_file()
    first_metadata = json.loads((output_dir / "run_metadata.json").read_text(encoding="utf-8"))
    assert first_metadata["training"]["total_steps"] == 2
    assert len(first_metadata["artifacts"]["final_checkpoint"]["sha256"]) == 64

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_carbon_cpu.py",
            "--config",
            str(config_path),
            "--resume",
            str(checkpoint),
            "--steps",
            "3",
        ],
    )
    assert main() == 0
    second_metadata = json.loads((output_dir / "run_metadata.json").read_text(encoding="utf-8"))
    assert second_metadata["training"]["total_steps"] == 3
    metric_rows = (output_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(metric_rows) == 3


def test_cpu_training_rejects_identical_train_and_eval_corpora(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = _write_config(tmp_path, same_eval=True)
    monkeypatch.setattr(sys, "argv", ["train_carbon_cpu.py", "--config", str(config_path)])
    with pytest.raises(ValueError, match="different files"):
        main()


def test_cpu_training_requires_reviewed_license_declaration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = _write_config(tmp_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["data"]["license_id"] = "LicenseRef-REPLACE-WITH-APPROVED-LICENSE"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["train_carbon_cpu.py", "--config", str(config_path)])
    with pytest.raises(ValueError, match="non-placeholder declaration"):
        main()
