#!/usr/bin/env python3
"""Evaluate adapter outputs against the versioned RUSA agent/tool contract."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rusamodels.training.dataset import load_jsonl, validate_agent_response  # noqa: E402


def _load_config(path: Path) -> dict[str, Any]:
    import yaml
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("training config must be a YAML object")
    return payload


def run_evaluation(adapter_dir: Path, eval_file: Path, config_path: Path, output_file: Path) -> dict[str, Any]:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    config = _load_config(config_path)
    model_cfg = config["model"]
    qlora_cfg = config["qlora"]
    evaluation_cfg = config.get("evaluation", {})
    provenance_path = adapter_dir / "rusa-finetune-provenance.json"
    if not provenance_path.is_file():
        raise ValueError("adapter directory must contain rusa-finetune-provenance.json")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    model_metadata = provenance["model"]
    model_id = model_metadata["id"]
    revision = model_metadata.get("resolved_revision") or model_metadata["configured_revision"]
    if model_id != model_cfg["id"]:
        raise ValueError("adapter provenance base model does not match the configured model")
    records = load_jsonl(eval_file)
    if not torch.cuda.is_available():
        raise RuntimeError("Adapter evaluation requires a CUDA GPU")

    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=str(qlora_cfg.get("quant_type", "nf4")),
        bnb_4bit_use_double_quant=bool(qlora_cfg.get("double_quant", True)),
        bnb_4bit_compute_dtype=compute_dtype,
    )
    tokenizer = AutoTokenizer.from_pretrained(str(adapter_dir), trust_remote_code=False)
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
    model = PeftModel.from_pretrained(base, str(adapter_dir), is_trainable=False)
    model.eval()

    json_valid = 0
    contract_valid = 0
    blocked_expected = 0
    blocked_predicted = 0
    blocked_true_positive = 0
    blocked_false_positive = 0
    blocked_true_negative = 0
    details: list[dict[str, Any]] = []
    max_new_tokens = int(evaluation_cfg.get("max_new_tokens", 512))
    device = model.get_input_embeddings().weight.device

    for record in records:
        prompt_ids = tokenizer.apply_chat_template(
            record["messages"], tokenize=True, add_generation_prompt=True
        )
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
        raw = tokenizer.decode(generated[0, input_ids.shape[-1]:], skip_special_tokens=True).strip()
        parsed = None
        error = None
        try:
            parsed = json.loads(raw)
            json_valid += 1
        except json.JSONDecodeError as exc:
            error = f"invalid_json: {exc.msg}"

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
            except ValueError as exc:
                error = f"invalid_contract: {exc}"
        details.append({
            "id": record["id"],
            "agent_role": record["agent_role"],
            "json_valid": parsed is not None,
            "contract_valid": error is None and parsed is not None,
            "expected_blocked": expected_blocked,
            "predicted_blocked": predicted_blocked,
            "error": error,
            "prediction": parsed if parsed is not None else raw[:2000],
        })

    total = len(records)
    if blocked_expected == 0 or blocked_expected == total:
        raise ValueError("evaluation split must include both blocked and unblocked examples")
    valid_json_rate = json_valid / total
    contract_rate = contract_valid / total
    blocked_recall = (blocked_true_positive / blocked_expected) if blocked_expected else None
    blocked_precision = (blocked_true_positive / blocked_predicted) if blocked_predicted else None
    unblocked_expected = total - blocked_expected
    blocked_false_positive_rate = (
        blocked_false_positive / unblocked_expected if unblocked_expected else None
    )
    report = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "adapter_dir": str(adapter_dir),
        "eval_file": str(eval_file),
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
        "details": details,
    }
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    minimum_contract = float(evaluation_cfg.get("min_valid_contract_rate", 1.0))
    minimum_blocked_recall = float(evaluation_cfg.get("min_blocked_recall", 0.90))
    maximum_blocked_false_positive_rate = float(
        evaluation_cfg.get("max_blocked_false_positive_rate", 0.10)
    )
    failures = []
    if contract_rate < minimum_contract:
        failures.append(f"valid_contract_rate {contract_rate:.3f} < {minimum_contract:.3f}")
    if blocked_recall is not None and blocked_recall < minimum_blocked_recall:
        failures.append(f"blocked_recall {blocked_recall:.3f} < {minimum_blocked_recall:.3f}")
    if (
        blocked_false_positive_rate is not None
        and blocked_false_positive_rate > maximum_blocked_false_positive_rate
    ):
        failures.append(
            "blocked_false_positive_rate "
            f"{blocked_false_positive_rate:.3f} > {maximum_blocked_false_positive_rate:.3f}"
        )
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
    run_evaluation(args.adapter_dir.resolve(strict=True), args.eval_file.resolve(),
                   args.config.resolve(), args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
