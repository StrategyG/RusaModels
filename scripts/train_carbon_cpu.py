#!/usr/bin/env python3
"""Train Carbon Tiny from random initialization using CPU-only PyTorch."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import UTC, datetime
from pathlib import Path
import platform
import random
import sys
import tempfile
import time
from urllib.parse import urlsplit

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rusamodels.carbon_cpu.model import (  # noqa: E402
    CarbonConfig,
    CarbonTiny,
    EOS_TOKEN_ID,
    count_parameters,
)


def resolve_path(value: str | Path, root: Path = ROOT) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()




def load_dataset_export_manifest(
    corpus_path: Path,
    expected_splits: set[str],
) -> dict | None:
    sidecar = corpus_path.with_suffix(corpus_path.suffix + ".manifest.json")
    if not sidecar.exists():
        return None
    if sidecar.is_symlink() or sidecar.resolve().parent != corpus_path.resolve().parent:
        raise ValueError("dataset export manifest is unsafe")
    try:
        manifest = json.loads(sidecar.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise ValueError("dataset export manifest is invalid JSON") from exc
    if not isinstance(manifest, dict) or manifest.get("schema_version") != "1.0.0":
        raise ValueError("unsupported dataset export manifest schema")
    required_strings = (
        "dataset_id",
        "dataset_version",
        "split",
        "source_split_file",
        "exporter",
        "output_file",
        "output_sha256",
        "release_manifest_sha256",
        "source_split_sha256",
        "tokenizer",
        "encoding",
    )
    if any(not isinstance(manifest.get(key), str) or not manifest[key] for key in required_strings):
        raise ValueError("dataset export manifest is missing required fields")
    if manifest["split"] not in expected_splits:
        raise ValueError(
            f"dataset export split {manifest['split']!r} is not valid for this training input"
        )
    if manifest["output_file"] != corpus_path.name:
        raise ValueError("dataset export manifest names a different corpus file")
    if manifest["exporter"] != "rusamodels.dataset_text_export":
        raise ValueError("dataset export manifest has an unsupported exporter")
    if manifest["source_split_file"] != f"{manifest['split']}.jsonl":
        raise ValueError("dataset export manifest names a different source split")
    if manifest["tokenizer"] != "byte-level-v1" or manifest["encoding"].lower() != "utf-8":
        raise ValueError("dataset export does not match the Carbon byte-level tokenizer")
    license_ids = manifest.get("license_ids")
    if (
        not isinstance(license_ids, list)
        or not license_ids
        or any(not isinstance(item, str) or not item.strip() for item in license_ids)
        or license_ids != sorted(set(license_ids))
    ):
        raise ValueError("dataset export manifest has invalid license_ids")
    if type(manifest.get("records")) is not int or manifest["records"] < 1:
        raise ValueError("dataset export manifest has invalid record count")
    if type(manifest.get("output_size_bytes")) is not int:
        raise ValueError("dataset export manifest has invalid output size")
    for key in ("output_sha256", "release_manifest_sha256", "source_split_sha256"):
        value = manifest[key]
        if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            raise ValueError(f"dataset export manifest has invalid {key}")
    if (
        sha256_file(corpus_path) != manifest["output_sha256"]
        or corpus_path.stat().st_size != manifest["output_size_bytes"]
    ):
        raise ValueError("dataset export corpus checksum or size does not match its manifest")
    manifest["sidecar_sha256"] = sha256_file(sidecar)
    return manifest


def canonical_digest(value: dict) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def training_contract(training: dict, effective: dict) -> dict:
    immutable = {
        key: value
        for key, value in training.items()
        if key not in {"steps", "output_dir"}
    }
    return {"training": immutable, "effective": effective}


def capture_rng_state(np_rng, eval_rng, torch) -> dict:
    return {
        "python": random.getstate(),
        "numpy": np_rng.bit_generator.state,
        "eval_numpy": eval_rng.bit_generator.state,
        "torch": torch.get_rng_state(),
    }


def restore_rng_state(state: dict, np_rng, eval_rng, torch) -> None:
    required = {"python", "numpy", "eval_numpy", "torch"}
    if not isinstance(state, dict) or set(state) != required:
        raise ValueError("Checkpoint is missing complete RNG state; deterministic resume is unavailable")
    random.setstate(state["python"])
    np_rng.bit_generator.state = state["numpy"]
    eval_rng.bit_generator.state = state["eval_numpy"]
    torch.set_rng_state(state["torch"])


def atomic_write_text(path: Path, content: str, encoding: str = "utf-8") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(file_descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(content, encoding=encoding)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def build_token_cache(source: Path, destination: Path) -> int:
    """Map UTF-8 bytes to token IDs 0..255 and append EOS=256, without loading all data."""
    if not source.is_file():
        raise FileNotFoundError(f"Corpus not found: {source}")
    byte_count = source.stat().st_size
    if byte_count == 0:
        raise ValueError(f"Corpus is empty: {source}")
    token_count = byte_count + 1
    destination.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(file_descriptor)
    temporary = Path(temporary_name)
    try:
        mapped = np.memmap(temporary, dtype=np.uint16, mode="w+", shape=(token_count,))
        offset = 0
        with source.open("rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                values = np.frombuffer(chunk, dtype=np.uint8).astype(np.uint16)
                mapped[offset : offset + len(values)] = values
                offset += len(values)
        mapped[offset] = EOS_TOKEN_ID
        mapped.flush()
        del mapped
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return token_count


def sample_batch(
    tokens: np.ndarray,
    batch_size: int,
    block_size: int,
    rng: np.random.Generator,
    torch,
) -> tuple:
    max_start = len(tokens) - block_size - 1
    if max_start < 1:
        raise ValueError(
            f"Not enough tokens ({len(tokens)}) for block_size={block_size}; "
            "provide a larger corpus or reduce block_size."
        )
    starts = rng.integers(0, max_start + 1, size=batch_size)
    inputs = np.stack(
        [np.asarray(tokens[start : start + block_size], dtype=np.int64) for start in starts]
    )
    targets = np.stack(
        [np.asarray(tokens[start + 1 : start + block_size + 1], dtype=np.int64) for start in starts]
    )
    return torch.from_numpy(inputs), torch.from_numpy(targets)


def evaluate(model, tokens, batch_size, block_size, batches, rng, torch) -> float:
    was_training = model.training
    model.eval()
    losses = []
    try:
        with torch.no_grad():
            for _ in range(batches):
                inputs, targets = sample_batch(tokens, batch_size, block_size, rng, torch)
                _, loss = model(inputs, targets)
                losses.append(float(loss.item()))
    finally:
        model.train(was_training)
    return sum(losses) / max(len(losses), 1)


def save_checkpoint(path: Path, payload: dict, torch) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(file_descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(payload, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/training/carbon-cpu-tiny.json")
    parser.add_argument("--data", type=Path, help="UTF-8 plain-text training corpus")
    parser.add_argument("--eval-data", type=Path, help="Optional separate UTF-8 evaluation corpus")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--threads", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--block-size", type=int)
    parser.add_argument("--resume", type=Path, help="Resume from a Carbon CPU checkpoint")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a tiny synthetic pipeline test; not useful training",
    )
    args = parser.parse_args()

    try:
        import torch
    except ImportError as exc:
        raise SystemExit(
            "PyTorch is required. Install a build compatible with your Python/OS/CPU. "
            "Standard Linux wheels may not work in Android Termux."
        ) from exc

    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not {"model", "training", "data"}.issubset(config):
        raise ValueError("training config must contain model, training, and data objects")
    model_values = config["model"]
    training = config["training"]
    data_config = config["data"]
    if not all(isinstance(value, dict) for value in (model_values, training, data_config)):
        raise ValueError("model, training, and data configuration values must be objects")
    output_dir = (
        args.output_dir if args.output_dir is not None
        else resolve_path(training["output_dir"])
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.smoke_test:
        dataset_id = data_config.get("dataset_id")
        dataset_version = data_config.get("dataset_version")
        raw_license_ids = data_config.get("license_ids", data_config.get("license_id"))
        if isinstance(raw_license_ids, str):
            license_ids = [raw_license_ids]
        elif isinstance(raw_license_ids, list):
            license_ids = raw_license_ids
        else:
            license_ids = []
        if not isinstance(dataset_id, str) or not dataset_id.strip():
            raise ValueError("data.dataset_id is required for non-smoke training")
        if not isinstance(dataset_version, str) or not dataset_version.strip():
            raise ValueError("data.dataset_version is required for non-smoke training")
        if (
            not license_ids
            or any(
                not isinstance(item, str) or not item.strip() or "REPLACE" in item.upper()
                for item in license_ids
            )
        ):
            raise ValueError("data.license_ids must contain reviewed, non-placeholder declarations")
        license_ids = sorted(set(license_ids))
        raw_eval_license_ids = data_config.get(
            "eval_license_ids", data_config.get("eval_license_id", license_ids)
        )
        if isinstance(raw_eval_license_ids, str):
            eval_license_ids = [raw_eval_license_ids]
        elif isinstance(raw_eval_license_ids, list):
            eval_license_ids = raw_eval_license_ids
        else:
            eval_license_ids = []
        if (
            not eval_license_ids
            or any(
                not isinstance(item, str) or not item.strip() or "REPLACE" in item.upper()
                for item in eval_license_ids
            )
        ):
            raise ValueError("data.eval_license_ids must contain reviewed declarations")
        eval_license_ids = sorted(set(eval_license_ids))
        source_uri = data_config.get("source_uri")
        if source_uri is not None:
            if not isinstance(source_uri, str) or len(source_uri) > 2048:
                raise ValueError("data.source_uri must be a string of at most 2048 characters")
            try:
                parsed_uri = urlsplit(source_uri)
            except ValueError as exc:
                raise ValueError("data.source_uri is malformed") from exc
            if parsed_uri.username or parsed_uri.password or parsed_uri.query or parsed_uri.fragment:
                raise ValueError("data.source_uri must not contain credentials, query, or fragment")
    else:
        dataset_id = "carbon-smoke-test"
        dataset_version = "synthetic"
        license_ids = ["LicenseRef-Synthetic-Smoke-Test"]
        eval_license_ids = license_ids
        source_uri = None

    if args.smoke_test:
        data_path = output_dir / "smoke-test-corpus.txt"
        sample = (
            "Carbon CPU is a small language model trained from random initialization. "
            "This synthetic text only checks the training pipeline.\n"
        )
        data_path.write_text(sample * 300, encoding="utf-8")
        eval_path = None
    else:
        data_path = (args.data or resolve_path(data_config["train_file"])).resolve()
        eval_value = args.eval_data or data_config.get("eval_file")
        eval_path = resolve_path(eval_value) if eval_value else None

    steps = args.steps if args.steps is not None else int(training["steps"])
    batch_size = args.batch_size if args.batch_size is not None else int(training["batch_size"])
    block_size = args.block_size if args.block_size is not None else int(training["block_size"])
    threads = args.threads if args.threads is not None else min(
        int(training.get("threads", 4)), os.cpu_count() or 1
    )
    grad_accum = int(training["gradient_accumulation_steps"])
    eval_interval = int(training["eval_interval"])
    eval_batches = int(training["eval_batches"])
    if min(steps, batch_size, block_size, threads, grad_accum, eval_interval, eval_batches) < 1:
        raise ValueError("steps, batch, block, threads and training intervals must be positive")
    seed = training.get("seed", 42)
    if type(seed) is not int:
        raise ValueError("training.seed must be an integer")
    warmup_steps = training.get("warmup_steps", 20)
    if type(warmup_steps) is not int or warmup_steps < 0:
        raise ValueError("warmup_steps must be a non-negative integer")
    if args.smoke_test and warmup_steps >= steps:
        warmup_steps = max(0, steps // 3)
    elif warmup_steps >= steps:
        raise ValueError("warmup_steps must be less than total steps")
    learning_rate_value = float(training["learning_rate"])
    weight_decay_value = float(training["weight_decay"])
    max_grad_norm = float(training["max_grad_norm"])
    if not math.isfinite(learning_rate_value) or learning_rate_value <= 0:
        raise ValueError("learning_rate must be finite and positive")
    if not math.isfinite(weight_decay_value) or weight_decay_value < 0:
        raise ValueError("weight_decay must be finite and non-negative")
    if not math.isfinite(max_grad_norm) or max_grad_norm <= 0:
        raise ValueError("max_grad_norm must be finite and positive")

    model_config = CarbonConfig.from_dict(model_values)
    if block_size > model_config.max_seq_len:
        raise ValueError("block_size cannot exceed model.max_seq_len")

    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    if data_config.get("encoding", "utf-8").lower() != "utf-8":
        raise ValueError("Carbon byte-level training requires UTF-8 source data")
    if not args.smoke_test and eval_path is not None:
        if not eval_path.is_file():
            raise FileNotFoundError(f"Evaluation corpus not found: {eval_path}")
        if eval_path.resolve() == data_path.resolve():
            raise ValueError("training and evaluation paths must be different files")
        if not eval_license_ids:
            raise ValueError("a reviewed evaluation license declaration is required")
    else:
        eval_license_ids = license_ids

    random.seed(seed)
    np_rng = np.random.default_rng(seed)
    eval_rng = np.random.default_rng(seed + 1)
    torch.manual_seed(seed)
    device = torch.device("cpu")

    source_hash_before = sha256_file(data_path)
    train_cache = output_dir / "train_tokens.u16"
    train_count = build_token_cache(data_path, train_cache)
    train_memmap = np.memmap(train_cache, dtype=np.uint16, mode="r", shape=(train_count,))
    source_hash = sha256_file(data_path)
    if source_hash != source_hash_before:
        raise ValueError("training corpus changed while its token cache was being built")

    validation_split = None
    if eval_path is not None and not args.smoke_test:
        eval_hash_before = sha256_file(eval_path)
        eval_cache = output_dir / "eval_tokens.u16"
        eval_count = build_token_cache(eval_path, eval_cache)
        eval_memmap = np.memmap(eval_cache, dtype=np.uint16, mode="r", shape=(eval_count,))
        eval_hash = sha256_file(eval_path)
        if eval_hash != eval_hash_before:
            raise ValueError("evaluation corpus changed while its token cache was being built")
        if eval_hash == source_hash:
            raise ValueError("training and evaluation corpora have identical SHA-256; held-out evaluation is invalid")
    else:
        validation_split = float(data_config.get("validation_split", 0.05))
        if not math.isfinite(validation_split) or not 0.0 < validation_split < 0.5:
            raise ValueError("validation_split must be finite and between 0 and 0.5")
        split = int(train_count * (1.0 - validation_split))
        if split <= block_size + 1 or train_count - split <= block_size + 1:
            raise ValueError("Corpus is too small for the configured block_size and validation split")
        eval_memmap = train_memmap[split:]
        train_memmap = train_memmap[:split]
        eval_hash = None

    tokenizer_name = data_config.get("tokenizer", "byte-level-v1")
    if tokenizer_name != "byte-level-v1":
        raise ValueError("Carbon CPU trainer currently supports only byte-level-v1")
    require_export_manifest = data_config.get("require_export_manifest", False)
    if type(require_export_manifest) is not bool:
        raise ValueError("data.require_export_manifest must be a boolean")
    train_export = load_dataset_export_manifest(data_path, {"train"})
    eval_export = (
        load_dataset_export_manifest(eval_path, {"validation", "test"})
        if eval_path is not None and eval_path.is_file() and not args.smoke_test
        else None
    )
    if require_export_manifest and train_export is None:
        raise ValueError("a verified Dataset Engine train export manifest is required")
    if (train_export is None) != (eval_export is None) and eval_path is not None:
        raise ValueError("train and evaluation corpora must both have Dataset Engine export manifests")
    if train_export is not None:
        if eval_export is None:
            raise ValueError("a separate verified validation/test export is required with a train export")
        for export in (train_export, eval_export):
            if export["dataset_id"] != dataset_id or export["dataset_version"] != dataset_version:
                raise ValueError("dataset export identity does not match the training configuration")
        if train_export["license_ids"] != license_ids:
            raise ValueError("training license declarations do not match the Dataset Engine export")
        if eval_export["license_ids"] != eval_license_ids:
            raise ValueError("evaluation license declarations do not match the Dataset Engine export")
        if (
            train_export["release_manifest_sha256"]
            != eval_export["release_manifest_sha256"]
        ):
            raise ValueError("train and evaluation exports come from different dataset releases")
    elif require_export_manifest:
        raise ValueError("Dataset Engine export manifests are required for this training run")

    dataset_metadata = {
        "dataset_id": dataset_id,
        "dataset_version": dataset_version,
        "license_ids": license_ids,
        "eval_license_ids": eval_license_ids,
        "source_uri": source_uri,
        "train_file_name": data_path.name,
        "train_sha256": source_hash,
        "train_export_manifest_sha256": (
            train_export["sidecar_sha256"] if train_export else None
        ),
        "eval_file_name": eval_path.name if eval_path and eval_path.is_file() else None,
        "eval_sha256": eval_hash,
        "eval_export_manifest_sha256": (
            eval_export["sidecar_sha256"] if eval_export else None
        ),
        "dataset_release_manifest_sha256": (
            train_export["release_manifest_sha256"] if train_export else None
        ),
        "validation_split": validation_split,
        "tokenizer": tokenizer_name,
    }
    dataset_fingerprint = canonical_digest(dataset_metadata)
    effective_training = {
        "batch_size": batch_size,
        "block_size": block_size,
        "threads": threads,
        "gradient_accumulation_steps": grad_accum,
        "eval_interval": eval_interval,
        "eval_batches": eval_batches,
        "seed": seed,
    }
    training_contract_digest = canonical_digest(
        training_contract(training, effective_training)
    )

    if len(train_memmap) <= block_size + 1 or len(eval_memmap) <= block_size + 1:
        raise ValueError("Training and evaluation splits must each contain more than block_size+1 tokens")
    if not args.smoke_test and data_path.stat().st_size < 100_000:
        print("WARNING: corpus is under 100 KB; this run is only a pipeline/prototype experiment.")

    model = CarbonTiny(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate_value,
        weight_decay=weight_decay_value,
        betas=(0.9, 0.95),
    )
    start_step = 0
    if args.resume:
        resume_path = args.resume.resolve()
        checkpoint = torch.load(resume_path, map_location="cpu", weights_only=True)
        if checkpoint.get("model_config") != model_config.to_dict():
            raise ValueError("Checkpoint architecture does not match the selected model config")
        if checkpoint.get("dataset_fingerprint") != dataset_fingerprint:
            raise ValueError("Checkpoint dataset fingerprint does not match the selected corpus/split")
        if checkpoint.get("training_contract_digest") != training_contract_digest:
            raise ValueError("Checkpoint training contract does not match the current training settings")
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_step = int(checkpoint.get("step", 0))
        if start_step >= steps:
            raise ValueError(f"Checkpoint is already at step {start_step}; requested total steps={steps}")
        restore_rng_state(checkpoint.get("rng_state"), np_rng, eval_rng, torch)
        print(f"Resuming from step {start_step}: {resume_path}")

    parameter_count = count_parameters(model)
    print(
        json.dumps(
            {
                "event": "training_started",
                "model": "Carbon-Tiny",
                "parameters": parameter_count,
                "device": str(device),
                "threads": threads,
                "steps": steps,
                "batch_size": batch_size,
                "block_size": block_size,
                "gradient_accumulation_steps": grad_accum,
                "corpus_bytes": data_path.stat().st_size,
                "corpus_sha256": source_hash,
                "smoke_test": bool(args.smoke_test),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    metrics_path = output_dir / "metrics.jsonl"
    if start_step == 0:
        atomic_write_text(metrics_path, "")
    started_at = datetime.now(UTC).isoformat()
    run_started = time.monotonic()

    def learning_rate(step: int) -> float:
        warmup = warmup_steps
        base = learning_rate_value
        if step <= warmup:
            return base * step / max(warmup, 1)
        progress = min(1.0, (step - warmup) / max(steps - warmup, 1))
        return base * 0.5 * (1.0 + math.cos(math.pi * progress))

    for step in range(start_step + 1, steps + 1):
        model.train()
        current_lr = learning_rate(step)
        for group in optimizer.param_groups:
            group["lr"] = current_lr
        optimizer.zero_grad(set_to_none=True)
        accumulated_loss = 0.0
        for _ in range(grad_accum):
            inputs, targets = sample_batch(train_memmap, batch_size, block_size, np_rng, torch)
            _, loss = model(inputs, targets)
            (loss / grad_accum).backward()
            accumulated_loss += float(loss.detach().item()) / grad_accum
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        if step == 1 or step % eval_interval == 0 or step == steps:
            validation_loss = evaluate(
                model, eval_memmap, batch_size, block_size, eval_batches, eval_rng, torch
            )
            elapsed = time.monotonic() - run_started
            record = {
                "step": step,
                "train_loss": accumulated_loss,
                "eval_loss": validation_loss,
                "learning_rate": current_lr,
                "elapsed_seconds": round(elapsed, 3),
            }
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            print(json.dumps(record), flush=True)

            payload = {
                "step": step,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "model_config": model_config.to_dict(),
                "training_config": training,
                "dataset_fingerprint": dataset_fingerprint,
                "training_contract_digest": training_contract_digest,
                "rng_state": capture_rng_state(np_rng, eval_rng, torch),
                "smoke_test": bool(args.smoke_test),
            }
            save_checkpoint(output_dir / "checkpoint.pt", payload, torch)

    final_path = output_dir / "final.pt"
    save_checkpoint(
        final_path,
        {
            "step": steps,
            "model_state": model.state_dict(),
            "model_config": model_config.to_dict(),
            "dataset_fingerprint": dataset_fingerprint,
            "training_contract_digest": training_contract_digest,
            "smoke_test": bool(args.smoke_test),
        },
        torch,
    )
    metadata = {
        "model": "Carbon-Tiny",
        "training_method": "from_scratch_next_token_prediction",
        "initialization": "random",
        "pretrained_base_model": None,
        "parameters": parameter_count,
        "model_config": model_config.to_dict(),
        "started_at": started_at,
        "finished_at": datetime.now(UTC).isoformat(),
        "hardware": {
            "device": "cpu",
            "processor": platform.processor(),
            "machine": platform.machine(),
            "logical_cpu_count": os.cpu_count(),
            "torch_threads": threads,
        },
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
        "dataset": dataset_metadata,
        "dataset_fingerprint": dataset_fingerprint,
        "training_contract_digest": training_contract_digest,
        "training": {
            "total_steps": steps,
            "effective_settings": effective_training,
        },
        "artifacts": {
            "final_checkpoint": {
                "file": final_path.name,
                "sha256": sha256_file(final_path),
                "size_bytes": final_path.stat().st_size,
            },
            "resumable_checkpoint": {
                "file": (output_dir / "checkpoint.pt").name,
                "sha256": sha256_file(output_dir / "checkpoint.pt"),
                "size_bytes": (output_dir / "checkpoint.pt").stat().st_size,
            },
        },
        "smoke_test_only": bool(args.smoke_test),
        "weights_status": (
            "smoke_test_only"
            if args.smoke_test
            else "experimental_randomly_initialized_cpu_trained_checkpoint"
        ),
    }
    atomic_write_text(
        output_dir / "run_metadata.json",
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
    )
    atomic_write_text(
        output_dir / "model_config.json",
        json.dumps(model_config.to_dict(), indent=2) + "\n",
    )
    print(f"Training complete. Final checkpoint: {final_path}", flush=True)
    if args.smoke_test:
        print("NOTE: This was a synthetic smoke test, not a useful language-model training run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
