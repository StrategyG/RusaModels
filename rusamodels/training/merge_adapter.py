#!/usr/bin/env python3
"""Merge a trained PEFT adapter into a standalone RusaModels safetensors artifact."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    args = parser.parse_args()
    adapter_dir = args.adapter_dir.expanduser().resolve(strict=True)
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir == adapter_dir or adapter_dir in output_dir.parents:
        parser.error("--output-dir must not be inside the adapter directory")
    if output_dir.exists() and any(output_dir.iterdir()):
        parser.error("--output-dir must be empty or not yet exist")

    provenance_path = adapter_dir / "rusa-finetune-provenance.json"
    if not provenance_path.is_file():
        parser.error("adapter directory must contain rusa-finetune-provenance.json")
    try:
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        model_metadata = provenance["model"]
        model_id = model_metadata["id"]
        revision = model_metadata.get("resolved_revision") or model_metadata["configured_revision"]
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        parser.error(f"adapter provenance is invalid: {exc}")

    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit("Install requirements-training.txt before merging adapters") from exc

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        torch_dtype=dtype,
        device_map="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=False,
        use_safetensors=True,
    )
    model = PeftModel.from_pretrained(base_model, str(adapter_dir), is_trainable=False)
    merged = model.merge_and_unload(safe_merge=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(
        output_dir,
        safe_serialization=True,
        max_shard_size="4GB",
    )
    # The adapter directory is a local, provenance-checked artifact, never a Hub identifier.
    tokenizer = AutoTokenizer.from_pretrained(  # nosec B615
        str(adapter_dir), trust_remote_code=False, local_files_only=True
    )
    tokenizer.save_pretrained(output_dir)
    shutil.copy2(provenance_path, output_dir / provenance_path.name)
    print(f"Merged model saved to: {output_dir}")
    print("Next: create and pin rusa-model-manifest.json before serving this artifact.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
