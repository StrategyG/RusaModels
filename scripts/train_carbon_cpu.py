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

from rusamodels.carbon_cpu.model import CarbonConfig, CarbonTiny, count_parameters  # noqa: E402
from rusamodels.carbon_cpu.model import EOS_TOKEN_ID


def resolve_path(value: str | Path, root: Path = ROOT) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()




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
    model_values = config["model"]
    training = config["training"]
    data_config = config["data"]
    output_dir = (args.output_dir or resolve_path(training["output_dir"])).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

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

    steps = args.steps or int(training["steps"])
    batch_size = args.batch_size or int(training["batch_size"])
    block_size = args.block_size or int(training["block_size"])
    threads = args.threads or min(int(training.get("threads", 4)), os.cpu_count() or 1)
    grad_accum = int(training["gradient_accumulation_steps"])
    eval_interval = int(training["eval_interval"])
    eval_batches = int(training["eval_batches"])
    if min(steps, batch_size, block_size, threads, grad_accum, eval_interval, eval_batches) < 1:
        raise ValueError("steps, batch, block, threads and training intervals must be positive")

    model_config = CarbonConfig.from_dict(model_values)
    if block_size > model_config.max_seq_len:
        raise ValueError("block_size cannot exceed model.max_seq_len")

    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    random.seed(int(training["seed"]))
    np_rng = np.random.default_rng(int(training["seed"]))
    eval_rng = np.random.default_rng(int(training["seed"]) + 1)
    torch.manual_seed(int(training["seed"]))
    device = torch.device("cpu")

    train_cache = output_dir / "train_tokens.u16"
    train_count = build_token_cache(data_path, train_cache)
    train_memmap = np.memmap(train_cache, dtype=np.uint16, mode="r", shape=(train_count,))
    source_hash = sha256_file(data_path)

    if eval_path is not None and not eval_path.is_file() and not args.smoke_test:
        raise FileNotFoundError(f"Evaluation corpus not found: {eval_path}")

    if eval_path is not None and eval_path.is_file() and not args.smoke_test:
        eval_cache = output_dir / "eval_tokens.u16"
        eval_count = build_token_cache(eval_path, eval_cache)
        eval_memmap = np.memmap(eval_cache, dtype=np.uint16, mode="r", shape=(eval_count,))
        eval_hash = sha256_file(eval_path)
    else:
        split = int(train_count * (1.0 - float(data_config.get("validation_split", 0.05))))
        if split <= block_size + 1 or train_count - split <= block_size + 1:
            raise ValueError("Corpus is too small for the configured block_size and validation split")
        eval_memmap = train_memmap[split:]
        train_memmap = train_memmap[:split]
        eval_hash = None

    if len(train_memmap) <= block_size + 1 or len(eval_memmap) <= block_size + 1:
        raise ValueError("Training and evaluation splits must each contain more than block_size+1 tokens")
    if not args.smoke_test and data_path.stat().st_size < 100_000:
        print("WARNING: corpus is under 100 KB; this run is only a pipeline/prototype experiment.")

    model = CarbonTiny(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        betas=(0.9, 0.95),
    )
    start_step = 0
    if args.resume:
        resume_path = args.resume.resolve()
        checkpoint = torch.load(resume_path, map_location="cpu")
        if checkpoint.get("model_config") != model_config.to_dict():
            raise ValueError("Checkpoint architecture does not match the selected model config")
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_step = int(checkpoint.get("step", 0))
        if start_step >= steps:
            raise ValueError(f"Checkpoint is already at step {start_step}; requested total steps={steps}")
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
        metrics_path.write_text("", encoding="utf-8")
    started_at = datetime.now(UTC).isoformat()
    run_started = time.monotonic()

    def learning_rate(step: int) -> float:
        warmup = int(training.get("warmup_steps", 20))
        base = float(training["learning_rate"])
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(training["max_grad_norm"]))
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
            print(json.dumps(record), flush=True)

            payload = {
                "step": step,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "model_config": model_config.to_dict(),
                "training_config": training,
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
        "dataset": {
            "train_path": str(data_path),
            "train_sha256": source_hash,
            "train_bytes": data_path.stat().st_size,
            "eval_path": str(eval_path) if eval_path and eval_path.is_file() else None,
            "eval_sha256": eval_hash,
        },
        "smoke_test_only": bool(args.smoke_test),
        "weights_status": "experimental_randomly_initialized_cpu_trained_checkpoint",
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "model_config.json").write_text(
        json.dumps(model_config.to_dict(), indent=2) + "\n", encoding="utf-8"
    )
    print(f"Training complete. Final checkpoint: {final_path}", flush=True)
    if args.smoke_test:
        print("NOTE: This was a synthetic smoke test, not a useful language-model training run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
