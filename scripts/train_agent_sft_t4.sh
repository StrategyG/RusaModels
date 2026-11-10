#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CONFIG="configs/training/agent-sft-t4.yaml"
TRAIN_FILE="data/agent_sft/train.jsonl"
EVAL_FILE="data/agent_sft/eval.jsonl"
ADAPTER_DIR="artifacts/rusa-agent-qwen3-4b-qlora-t4/adapter"
EVAL_REPORT="artifacts/agent-sft-evaluation-t4.json"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi not found; install an NVIDIA driver/CUDA runtime." >&2
  exit 2
fi

nvidia-smi

python - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("ERROR: CUDA is unavailable; QLoRA training cannot run on CPU.")
props = torch.cuda.get_device_properties(torch.cuda.current_device())
name = props.name
vram_gib = props.total_memory / (1024 ** 3)
print(f"Detected GPU: {name}")
print(f"Visible VRAM: {vram_gib:.2f} GiB")
if "t4" not in name.lower():
    raise SystemExit(f"Expected an NVIDIA T4 GPU, found {name!r}.")
if vram_gib < 14:
    raise SystemExit(f"Expected at least 14 GiB visible VRAM, found {vram_gib:.2f} GiB.")
if torch.cuda.is_bf16_supported():
    print("BF16 is available; the T4 profile will still use FP16.")
else:
    print("FP16 profile confirmed for this GPU.")
PY

python rusamodels/training/validate_dataset.py \
  --train-file "$TRAIN_FILE" \
  --eval-file "$EVAL_FILE"

python rusamodels/training/train_sft.py --config "$CONFIG"

test -f "$ADAPTER_DIR/adapter_config.json" || {
  echo "ERROR: training completed without the expected PEFT adapter_config.json." >&2
  exit 3
}

python rusamodels/evaluation/evaluate_agent_sft.py \
  --adapter-dir "$ADAPTER_DIR" \
  --eval-file "$EVAL_FILE" \
  --config "$CONFIG" \
  --output "$EVAL_REPORT"

echo "Training and configured evaluation gates passed."
echo "Adapter weights: $ADAPTER_DIR"
echo "Evaluation report: $EVAL_REPORT"
