#!/usr/bin/env python3
"""Evaluate adapter outputs against the versioned RUSA agent/tool contract."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rusamodels.training.dataset import (  # noqa: E402
    DatasetValidationError,
    load_jsonl,
    sha256_file,
    validate_agent_response,
)


_COMMIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _snapshot_file(value: str | Path, label: str, *, root: Path = ROOT) -> tuple[Path, dict[str, Any]]:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = root / path
    if path.is_symlink():
        raise ValueError(f"{label} must not be a symlink")
    path = path.resolve(strict=True)
    if not path.is_file():
        raise ValueError(f"{label} must be a regular file")
    return path, {
        "path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _assert_file_unchanged(snapshot: dict[str, Any], label: str) -> None:
    path = Path(snapshot["path"])
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{label} disappeared or became unsafe during evaluation")
    if sha256_file(path) != snapshot["sha256"] or path.stat().st_size != snapshot["size_bytes"]:
        raise RuntimeError(f"{label} changed during evaluation")


def _hash_adapter_directory(adapter_dir: Path) -> str:
    if adapter_dir.is_symlink() or not adapter_dir.is_dir():
        raise ValueError("adapter directory must be a regular directory, not a symlink")
    digest = hashlib.sha256()
    files = sorted(adapter_dir.rglob("*"), key=lambda item: item.relative_to(adapter_dir).as_posix())
    file_count = 0
    for path in files:
        if path.is_symlink():
            raise ValueError(f"adapter directory contains a symlink: {path.relative_to(adapter_dir)}")
        if not path.is_file():
            continue
        relative = path.relative_to(adapter_dir).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(path).encode("ascii"))
        digest.update(b"\n")
        file_count += 1
    if file_count == 0:
        raise ValueError("adapter directory contains no regular files")
    return digest.hexdigest()


def _validate_provenance(
    provenance: Any,
    model_cfg: dict[str, Any],
    eval_sha256: str,
    eval_examples: int,
) -> tuple[str, str]:
    if not isinstance(provenance, dict) or provenance.get("schema_version") != 1:
        raise ValueError("adapter provenance schema is unsupported")
    model = provenance.get("model")
    dataset = provenance.get("dataset")
    if not isinstance(model, dict) or not isinstance(dataset, dict):
        raise ValueError("adapter provenance must contain model and dataset objects")
    model_id = model.get("id")
    revision = model.get("resolved_revision") or model.get("configured_revision")
    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError("adapter provenance model ID is invalid")
    if not isinstance(revision, str) or not _COMMIT_SHA.fullmatch(revision):
        raise ValueError("adapter provenance model revision must be a pinned commit SHA")
    if model_id != model_cfg.get("id") or revision != model_cfg.get("revision"):
        raise ValueError("adapter provenance base model does not match the configured model")
    if not isinstance(dataset.get("train_sha256"), str) or not _SHA256.fullmatch(
        dataset["train_sha256"]
    ):
        raise ValueError("adapter provenance training dataset digest is invalid")
    if dataset["train_sha256"] == eval_sha256:
        raise ValueError("adapter provenance uses identical training and evaluation files")
    if dataset.get("eval_sha256") != eval_sha256:
        raise ValueError("evaluation file SHA-256 does not match the adapter training provenance")
    if type(dataset.get("eval_examples")) is not int or dataset["eval_examples"] != eval_examples:
        raise ValueError("evaluation record count does not match the adapter training provenance")
    tokenizer = provenance.get("tokenizer")
    if isinstance(tokenizer, dict):
        if tokenizer.get("id") != model_id or tokenizer.get("resolved_revision") != revision:
            raise ValueError("adapter tokenizer provenance does not match the pinned base model")
    return model_id, revision


def _safe_report_path(
    value: str | Path,
    *,
    adapter_dir: Path,
    eval_file: Path,
    config_file: Path,
) -> Path:
    path = Path(value).expanduser()
    if path.is_symlink():
        raise ValueError("evaluation report path must not be a symlink")
    output = path.parent.resolve() / path.name
    if output.exists():
        raise ValueError(f"evaluation report already exists: {output}")
    if (
        output in {eval_file, config_file}
        or output == adapter_dir
        or adapter_dir in output.parents
    ):
        raise ValueError("evaluation report must be outside adapter and input artifacts")
    output.parent.mkdir(parents=True, exist_ok=True)
    return output


def _write_new_file(path: Path, payload: bytes) -> str:
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
    return hashlib.sha256(payload).hexdigest()


def _pseudonymize_record_id(value: str) -> str:
    return f"record_{hashlib.sha256(value.encode('utf-8')).hexdigest()[:24]}"


def _load_config(path: Path) -> dict[str, Any]:
    import yaml

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("training config must be a YAML object")
    return payload


def run_evaluation(
    adapter_dir: Path,
    eval_file: Path,
    config_path: Path,
    output_file: Path,
) -> dict[str, Any]:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if adapter_dir.is_symlink():
        raise ValueError("adapter directory must not be a symlink")
    adapter_root = adapter_dir.expanduser().resolve(strict=True)
    if not adapter_root.is_dir():
        raise ValueError("adapter path must be a directory")
    config_file, config_snapshot = _snapshot_file(config_path, "evaluation config", root=Path.cwd())
    evaluation_file, eval_snapshot = _snapshot_file(eval_file, "evaluation dataset", root=Path.cwd())
    config = _load_config(config_file)
    _assert_file_unchanged(config_snapshot, "evaluation config")
    model_cfg = config["model"]
    qlora_cfg = config["qlora"]
    evaluation_cfg = config.get("evaluation", {})

    provenance_path = adapter_root / "rusa-finetune-provenance.json"
    if provenance_path.is_symlink() or not provenance_path.is_file():
        raise ValueError("adapter directory must contain a regular rusa-finetune-provenance.json")
    adapter_sha256_before = _hash_adapter_directory(adapter_root)
    provenance_bytes = provenance_path.read_bytes()
    provenance_sha256 = hashlib.sha256(provenance_bytes).hexdigest()
    provenance = json.loads(provenance_bytes.decode("utf-8"))

    records = load_jsonl(evaluation_file)
    _assert_file_unchanged(eval_snapshot, "evaluation dataset")
    model_id, revision = _validate_provenance(
        provenance,
        model_cfg,
        eval_snapshot["sha256"],
        len(records),
    )
    output = _safe_report_path(
        output_file,
        adapter_dir=adapter_root,
        eval_file=evaluation_file,
        config_file=config_file,
    )
    if not torch.cuda.is_available():
        raise RuntimeError("Adapter evaluation requires a CUDA GPU")

    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=str(qlora_cfg.get("quant_type", "nf4")),
        bnb_4bit_use_double_quant=bool(qlora_cfg.get("double_quant", True)),
        bnb_4bit_compute_dtype=compute_dtype,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        str(adapter_root),
        trust_remote_code=False,
        local_files_only=True,
    )
    if tokenizer.eos_token_id is None:
        raise ValueError("Adapter tokenizer must define eos_token_id")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        quantization_config=quantization,
        torch_dtype=compute_dtype,
        device_map="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=False,
        use_safetensors=True,
    )
    resolved_model_revision = getattr(base.config, "_commit_hash", None) or revision
    if resolved_model_revision != revision:
        raise RuntimeError("Resolved evaluation base-model revision differs from adapter provenance")
    model = PeftModel.from_pretrained(base, str(adapter_root), is_trainable=False)
    model.eval()

    json_valid = 0
    contract_valid = 0
    blocked_expected = 0
    blocked_predicted = 0
    blocked_true_positive = 0
    blocked_false_positive = 0
    blocked_true_negative = 0
    details: list[dict[str, Any]] = []
    max_new_tokens = evaluation_cfg.get("max_new_tokens", 512)
    if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= 8192:
        raise ValueError("evaluation.max_new_tokens must be an integer from 1 to 8192")
    device = model.get_input_embeddings().weight.device

    for record in records:
        prompt_ids = tokenizer.apply_chat_template(
            record["messages"], tokenize=True, add_generation_prompt=True
        )
        if hasattr(prompt_ids, "tolist"):
            prompt_ids = prompt_ids.tolist()
        if prompt_ids and isinstance(prompt_ids[0], list):
            prompt_ids = prompt_ids[0]
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids)
        with torch.inference_mode():
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        raw = tokenizer.decode(
            generated[0, input_ids.shape[-1]:],
            skip_special_tokens=True,
        ).strip()
        prediction_sha256 = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        parsed = None
        error_code = None
        try:
            parsed = json.loads(raw)
            json_valid += 1
        except json.JSONDecodeError:
            error_code = "invalid_json"

        expected_blocked = bool(record["response"].get("blocked", False))
        blocked_expected += int(expected_blocked)
        predicted_blocked = bool(parsed.get("blocked", False)) if isinstance(parsed, dict) else False
        blocked_predicted += int(predicted_blocked)
        if expected_blocked and predicted_blocked:
            blocked_true_positive += 1
        elif not expected_blocked and predicted_blocked:
            blocked_false_positive += 1
        elif not expected_blocked and not predicted_blocked:
            blocked_true_negative += 1

        if parsed is not None:
            try:
                validate_agent_response(record["agent_role"], parsed)
                contract_valid += 1
            except DatasetValidationError:
                error_code = "invalid_contract"
        details.append({
            "record_id": _pseudonymize_record_id(record["id"]),
            "agent_role": record["agent_role"],
            "json_valid": parsed is not None,
            "contract_valid": error_code is None and parsed is not None,
            "expected_blocked": expected_blocked,
            "predicted_blocked": predicted_blocked,
            "error_code": error_code,
            "prediction_sha256": prediction_sha256,
            "prediction_characters": len(raw),
        })

    total = len(records)
    if blocked_expected == 0 or blocked_expected == total:
        raise ValueError("evaluation split must include both blocked and unblocked examples")
    valid_json_rate = json_valid / total
    contract_rate = contract_valid / total
    blocked_recall = blocked_true_positive / blocked_expected
    blocked_precision = blocked_true_positive / blocked_predicted if blocked_predicted else None
    unblocked_expected = total - blocked_expected
    blocked_false_positive_rate = blocked_false_positive / unblocked_expected
    minimum_contract = float(evaluation_cfg.get("min_valid_contract_rate", 1.0))
    minimum_blocked_recall = float(evaluation_cfg.get("min_blocked_recall", 0.90))
    maximum_blocked_false_positive_rate = float(
        evaluation_cfg.get("max_blocked_false_positive_rate", 0.10)
    )
    for name, threshold in (
        ("min_valid_contract_rate", minimum_contract),
        ("min_blocked_recall", minimum_blocked_recall),
        ("max_blocked_false_positive_rate", maximum_blocked_false_positive_rate),
    ):
        if not 0 <= threshold <= 1:
            raise ValueError(f"evaluation.{name} must be between 0 and 1")
    failures = []
    if contract_rate < minimum_contract:
        failures.append(f"valid_contract_rate {contract_rate:.3f} < {minimum_contract:.3f}")
    if blocked_recall < minimum_blocked_recall:
        failures.append(f"blocked_recall {blocked_recall:.3f} < {minimum_blocked_recall:.3f}")
    if blocked_false_positive_rate > maximum_blocked_false_positive_rate:
        failures.append(
            "blocked_false_positive_rate "
            f"{blocked_false_positive_rate:.3f} > {maximum_blocked_false_positive_rate:.3f}"
        )

    _assert_file_unchanged(eval_snapshot, "evaluation dataset")
    _assert_file_unchanged(config_snapshot, "evaluation config")
    adapter_sha256_after = _hash_adapter_directory(adapter_root)
    if adapter_sha256_after != adapter_sha256_before:
        raise RuntimeError("adapter artifacts changed during evaluation")

    report = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "model": {"id": model_id, "revision": revision},
        "adapter_sha256": adapter_sha256_before,
        "adapter_provenance_sha256": provenance_sha256,
        "training_dataset_sha256": provenance["dataset"]["train_sha256"],
        "evaluation_dataset_sha256": eval_snapshot["sha256"],
        "evaluation_config_sha256": config_snapshot["sha256"],
        "examples": total,
        "metrics": {
            "valid_json_rate": valid_json_rate,
            "valid_contract_rate": contract_rate,
            "blocked_recall": blocked_recall,
            "blocked_precision": blocked_precision,
            "blocked_false_positive_rate": blocked_false_positive_rate,
            "json_valid": json_valid,
            "contract_valid": contract_valid,
            "blocked_expected": blocked_expected,
            "blocked_predicted": blocked_predicted,
            "blocked_true_negative": blocked_true_negative,
            "blocked_false_positive": blocked_false_positive,
        },
        "failures": failures,
        "passed": not failures,
        "details": details,
    }
    report_bytes = (json.dumps(report, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    report_sha256 = _write_new_file(output, report_bytes)
    report["report_sha256"] = report_sha256
    print(json.dumps(report["metrics"], indent=2))
    if failures:
        raise RuntimeError("Evaluation gate failed: " + "; ".join(failures))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-dir", required=True, type=Path)
    parser.add_argument("--eval-file", type=Path, default=ROOT / "data/agent_sft/eval.jsonl")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/training/agent-sft.yaml")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/agent-sft-evaluation.json")
    args = parser.parse_args()
    run_evaluation(args.adapter_dir, args.eval_file, args.config, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
