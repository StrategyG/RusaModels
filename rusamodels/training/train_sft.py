#!/usr/bin/env python3
"""Train a RusaModels agent adapter with QLoRA and a completion-only loss."""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import tempfile
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rusamodels.training.dataset import (  # noqa: E402
    load_jsonl,
    sha256_file,
    validate_dataset_splits,
    validate_eval_distribution,
)

LOGGER = logging.getLogger("rusamodels.training")


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _resolve_path(value: str, root: Path = ROOT) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _snapshot_input_file(value: str | Path, label: str, *, root: Path = ROOT) -> tuple[Path, dict[str, Any]]:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = root / path
    if path.is_symlink():
        raise ValueError(f"{label} must not be a symlink")
    path = path.resolve(strict=True)
    if not path.is_file():
        raise ValueError(f"{label} must be a regular file")
    snapshot = {
        "path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }
    return path, snapshot


def _assert_input_file_unchanged(snapshot: dict[str, Any], label: str) -> None:
    path = Path(snapshot["path"])
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{label} disappeared or became unsafe during training")
    if sha256_file(path) != snapshot["sha256"] or path.stat().st_size != snapshot["size_bytes"]:
        raise RuntimeError(f"{label} changed during training")


def _create_new_output_directory(value: str | Path, *, root: Path = ROOT) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = root / path
    if path.is_symlink():
        raise ValueError("training output directory must not be a symlink")
    path = path.resolve()
    if path.exists():
        raise ValueError(f"training output directory already exists; use a new run path: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.mkdir(exist_ok=False)
    return path


def _write_new_file(path: Path, payload: bytes) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _build_torch_dataset(records: list[dict[str, Any]], tokenizer: Any, max_seq_length: int, torch: Any):
    class AgentSFTDataset(torch.utils.data.Dataset):
        def __init__(self) -> None:
            self.examples: list[dict[str, list[int]]] = []
            for record in records:
                prompt_ids = tokenizer.apply_chat_template(
                    record["messages"],
                    tokenize=True,
                    add_generation_prompt=True,
                )
                if hasattr(prompt_ids, "tolist"):
                    prompt_ids = prompt_ids.tolist()
                if prompt_ids and isinstance(prompt_ids[0], list):
                    prompt_ids = prompt_ids[0]
                completion_text = json.dumps(
                    record["response"], ensure_ascii=False, sort_keys=True, separators=(",", ":")
                )
                completion_ids = tokenizer.encode(completion_text, add_special_tokens=False)
                if tokenizer.eos_token_id is not None:
                    completion_ids = list(completion_ids) + [tokenizer.eos_token_id]
                else:
                    raise ValueError("Tokenizer must define eos_token_id for turn termination")
                prompt_ids = list(prompt_ids)
                if not prompt_ids or not completion_ids:
                    raise ValueError(f"Example {record['id']} tokenized to an empty sequence")
                if len(prompt_ids) + len(completion_ids) > max_seq_length:
                    raise ValueError(
                        f"Example {record['id']} exceeds max_seq_length={max_seq_length}; "
                        "shorten/redact the example instead of silently truncating policy context"
                    )
                input_ids = prompt_ids + completion_ids
                labels = ([-100] * len(prompt_ids)) + completion_ids
                self.examples.append({
                    "input_ids": input_ids,
                    "attention_mask": [1] * len(input_ids),
                    "labels": labels,
                })

        def __len__(self) -> int:
            return len(self.examples)

        def __getitem__(self, index: int) -> dict[str, Any]:
            return self.examples[index]

    return AgentSFTDataset()


def _collator(tokenizer: Any, torch: Any):
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        raise ValueError("Tokenizer must define pad_token_id")

    def collate(features: list[dict[str, list[int]]]) -> dict[str, Any]:
        width = max(len(item["input_ids"]) for item in features)
        input_ids = []
        attention_masks = []
        labels = []
        for item in features:
            pad = width - len(item["input_ids"])
            input_ids.append(item["input_ids"] + [pad_token_id] * pad)
            attention_masks.append(item["attention_mask"] + [0] * pad)
            labels.append(item["labels"] + [-100] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    return collate


def _load_config(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("Install requirements-train.txt to use the training pipeline") from exc
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("training config must be a YAML object")
    return config


def run_training(config_path: Path, *, train_path: Path | None = None, eval_path: Path | None = None,
                 output_path: Path | None = None, allow_small_dataset: bool = False) -> Path:
    config_path, config_snapshot = _snapshot_input_file(
        config_path, "training config", root=Path.cwd()
    )
    config_bytes = config_path.read_bytes()
    config = _load_config(config_path)
    _assert_input_file_unchanged(config_snapshot, "training config")
    model_cfg = config["model"]
    dataset_cfg = config["dataset"]
    training_cfg = config["training"]
    qlora_cfg = config["qlora"]
    lora_cfg = config["lora"]

    model_id = model_cfg.get("id")
    model_revision = model_cfg.get("revision")
    if not isinstance(model_id, str) or not model_id.strip() or len(model_id) > 256:
        raise ValueError("model.id must be a non-empty model identifier")
    if not isinstance(model_revision, str) or not re.fullmatch(r"[0-9a-f]{40}", model_revision):
        raise ValueError("model.revision must be a pinned 40-character commit SHA")
    if not isinstance(model_cfg.get("license"), str) or not model_cfg["license"].strip():
        raise ValueError("model.license must contain a reviewed license identifier")

    train_value = train_path if train_path is not None else dataset_cfg["train_file"]
    eval_value = eval_path if eval_path is not None else dataset_cfg["eval_file"]
    input_root = Path.cwd() if train_path is not None else ROOT
    train_file, train_snapshot = _snapshot_input_file(
        train_value, "training dataset", root=input_root
    )
    eval_root = Path.cwd() if eval_path is not None else ROOT
    eval_file, eval_snapshot = _snapshot_input_file(
        eval_value, "evaluation dataset", root=eval_root
    )
    train_records = load_jsonl(train_file)
    eval_records = load_jsonl(eval_file)
    _assert_input_file_unchanged(train_snapshot, "training dataset")
    _assert_input_file_unchanged(eval_snapshot, "evaluation dataset")
    validate_dataset_splits(train_records, eval_records)
    output_value = output_path if output_path is not None else training_cfg["output_dir"]
    output_root = Path.cwd() if output_path is not None else ROOT
    output_dir = output_value
    minimum = int(dataset_cfg.get("min_train_examples", 100))
    minimum_eval = int(dataset_cfg.get("min_eval_examples", 30))
    if not allow_small_dataset and len(train_records) < minimum:
        raise ValueError(
            f"Only {len(train_records)} training examples are available; minimum is {minimum}. "
            "Use a larger reviewed dataset or pass --allow-small-dataset for a non-production experiment."
        )
    if not allow_small_dataset and len(eval_records) < minimum_eval:
        raise ValueError(f"Only {len(eval_records)} evaluation examples are available; minimum is {minimum_eval}")
    required_roles = set(dataset_cfg.get("required_roles", []))
    if required_roles:
        train_roles = {item["agent_role"] for item in train_records}
        eval_roles = {item["agent_role"] for item in eval_records}
        if required_roles - train_roles or required_roles - eval_roles:
            raise ValueError(
                "train and eval splits must each cover all configured roles; "
                f"missing train={sorted(required_roles - train_roles)}, "
                f"missing eval={sorted(required_roles - eval_roles)}"
            )
    if not allow_small_dataset:
        try:
            validate_eval_distribution(
                eval_records,
                min_blocked_fraction=float(dataset_cfg.get("min_blocked_eval_fraction", 0.10)),
                min_unblocked_fraction=float(dataset_cfg.get("min_unblocked_eval_fraction", 0.50)),
            )
        except ValueError as exc:
            raise ValueError(str(exc)) from exc
    if training_cfg.get("method", "qlora") != "qlora":
        raise ValueError("This implementation currently supports method: qlora only")

    try:
        import torch
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        raise RuntimeError("Install requirements-train.txt in the dedicated training environment") from exc

    if not torch.cuda.is_available():
        raise RuntimeError("QLoRA training requires a CUDA GPU; CPU training is intentionally disabled")
    output_dir = _create_new_output_directory(output_dir, root=output_root)

    requested_dtype = str(qlora_cfg.get("compute_dtype", "bfloat16")).lower()
    if requested_dtype not in {"bfloat16", "float16"}:
        raise ValueError("qlora.compute_dtype must be bfloat16 or float16")
    bf16_supported = bool(torch.cuda.is_bf16_supported())
    use_bf16 = requested_dtype == "bfloat16" and bf16_supported
    if requested_dtype == "bfloat16" and not bf16_supported:
        LOGGER.warning("GPU does not support BF16; falling back to FP16 compute")
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16

    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["id"], revision=model_revision, trust_remote_code=False
    )
    tokenizer_kwargs = getattr(tokenizer, "init_kwargs", {})
    resolved_tokenizer_revision = (
        tokenizer_kwargs.get("_commit_hash", model_revision)
        if isinstance(tokenizer_kwargs, dict)
        else model_revision
    )
    if resolved_tokenizer_revision != model_revision:
        raise RuntimeError("Resolved tokenizer revision does not match the pinned model revision")
    if not getattr(tokenizer, "chat_template", None):
        raise RuntimeError("Base tokenizer must provide a chat_template")
    if tokenizer.eos_token_id is None:
        raise RuntimeError("Base tokenizer must define eos_token_id")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=str(qlora_cfg.get("quant_type", "nf4")),
        bnb_4bit_use_double_quant=bool(qlora_cfg.get("double_quant", True)),
        bnb_4bit_compute_dtype=compute_dtype,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["id"],
        revision=model_revision,
        quantization_config=quantization,
        torch_dtype=compute_dtype,
        device_map={"": torch.cuda.current_device()},
        low_cpu_mem_usage=True,
        trust_remote_code=False,
        use_safetensors=True,
    )
    resolved_model_revision = getattr(model.config, "_commit_hash", None) or model_revision
    if resolved_model_revision != model_revision:
        raise RuntimeError("Resolved base-model revision does not match the pinned model revision")
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=bool(training_cfg.get("gradient_checkpointing", True)),
    )
    adapter_config = LoraConfig(
        r=int(lora_cfg.get("r", 16)),
        lora_alpha=int(lora_cfg.get("alpha", 32)),
        lora_dropout=float(lora_cfg.get("dropout", 0.05)),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=lora_cfg.get("target_modules", "all-linear"),
    )
    model = get_peft_model(model, adapter_config)
    model.print_trainable_parameters()

    max_seq_length = int(dataset_cfg.get("max_seq_length", 4096))
    train_dataset = _build_torch_dataset(train_records, tokenizer, max_seq_length, torch)
    eval_dataset = _build_torch_dataset(eval_records, tokenizer, max_seq_length, torch)
    checkpoint_dir = output_dir / "checkpoints"
    adapter_dir = output_dir / "adapter"

    args = TrainingArguments(
        output_dir=str(checkpoint_dir),
        per_device_train_batch_size=int(training_cfg.get("per_device_train_batch_size", 1)),
        per_device_eval_batch_size=int(training_cfg.get("per_device_eval_batch_size", 1)),
        gradient_accumulation_steps=int(training_cfg.get("gradient_accumulation_steps", 16)),
        num_train_epochs=float(training_cfg.get("num_train_epochs", 3)),
        learning_rate=float(training_cfg.get("learning_rate", 1e-4)),
        weight_decay=float(training_cfg.get("weight_decay", 0.01)),
        warmup_ratio=float(training_cfg.get("warmup_ratio", 0.03)),
        lr_scheduler_type=str(training_cfg.get("lr_scheduler_type", "cosine")),
        logging_strategy="steps",
        logging_steps=int(training_cfg.get("logging_steps", 5)),
        eval_strategy=str(training_cfg.get("eval_strategy", "epoch")),
        save_strategy=str(training_cfg.get("save_strategy", "epoch")),
        save_total_limit=int(training_cfg.get("save_total_limit", 3)),
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        gradient_checkpointing=bool(training_cfg.get("gradient_checkpointing", True)),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=use_bf16,
        fp16=not use_bf16,
        optim="paged_adamw_8bit",
        max_grad_norm=float(training_cfg.get("max_grad_norm", 0.3)),
        report_to=[],
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        seed=int(training_cfg.get("seed", 42)),
        data_seed=int(training_cfg.get("seed", 42)),
        dataloader_num_workers=int(training_cfg.get("dataloader_num_workers", 2)),
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=_collator(tokenizer, torch),
    )
    train_result = trainer.train()
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(adapter_dir)
    trainer.save_state()
    trainer.save_metrics("train", train_result.metrics)
    eval_metrics = trainer.evaluate()
    trainer.save_metrics("eval", eval_metrics)

    _assert_input_file_unchanged(train_snapshot, "training dataset")
    _assert_input_file_unchanged(eval_snapshot, "evaluation dataset")
    _assert_input_file_unchanged(config_snapshot, "training config")
    gpu = torch.cuda.get_device_properties(torch.cuda.current_device())
    provenance = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "model": {
            "id": model_cfg["id"],
            "configured_revision": model_revision,
            "resolved_revision": resolved_model_revision,
            "license": model_cfg["license"],
        },
        "tokenizer": {
            "id": model_id,
            "configured_revision": model_revision,
            "resolved_revision": resolved_tokenizer_revision,
        },
        "method": "QLoRA",
        "dataset": {
            "train_path": train_snapshot["path"],
            "train_sha256": train_snapshot["sha256"],
            "train_size_bytes": train_snapshot["size_bytes"],
            "train_examples": len(train_records),
            "eval_path": eval_snapshot["path"],
            "eval_sha256": eval_snapshot["sha256"],
            "eval_size_bytes": eval_snapshot["size_bytes"],
            "eval_examples": len(eval_records),
            "training_config_sha256": config_snapshot["sha256"],
        },
        "training": training_cfg,
        "qlora": qlora_cfg,
        "lora": lora_cfg,
        "max_seq_length": max_seq_length,
        "hardware": {
            "gpu_name": torch.cuda.get_device_name(torch.cuda.current_device()),
            "gpu_vram_gib": round(gpu.total_memory / (1024 ** 3), 2),
            "cuda_version": torch.version.cuda,
        },
        "software": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "transformers": _package_version("transformers"),
            "peft": _package_version("peft"),
            "bitsandbytes": _package_version("bitsandbytes"),
            "accelerate": _package_version("accelerate"),
        },
        "metrics": {"train": train_result.metrics, "eval": eval_metrics},
        "weights_status": "adapter_trained_not_production_certified",
    }
    _write_new_file(
        adapter_dir / "rusa-finetune-provenance.json",
        (json.dumps(provenance, ensure_ascii=False, indent=2, default=str) + "\n").encode("utf-8"),
    )
    _write_new_file(output_dir / "training-config.yaml", config_bytes)
    LOGGER.info("QLoRA adapter saved to %s", adapter_dir)
    return adapter_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/training/agent-sft.yaml")
    parser.add_argument("--train-file", type=Path)
    parser.add_argument("--eval-file", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--allow-small-dataset", action="store_true",
                        help="Allow a dataset below the configured minimum for a non-production experiment")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_training(
        args.config,
        train_path=args.train_file,
        eval_path=args.eval_file,
        output_path=args.output_dir,
        allow_small_dataset=args.allow_small_dataset,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())