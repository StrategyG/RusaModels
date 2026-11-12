# RusaModels Agent SFT Training

## Scope and status

The initial RusaModels training package was migrated from RusaAi. It supports validation, QLoRA supervised fine-tuning, held-out contract/safety evaluation and adapter merging for the six RUSA worker roles.

The included 120-example training split and 36-example evaluation split are synthetic seed data. No trained RusaModels weights are included. Dataset availability is not evidence that training has run or that a model is production-ready.

## Model-selection boundary

The checked-in `agent-sft.yaml` and manual T4 workflow target Qwen3-4B and are an experimental GPU-only QLoRA track. They do not satisfy or replace the separate non-Qwen, CPU-first product-model decision in the RusaAi Training Engine v1.0 design. That design's provisional `BhinekaIntiLabs/bhineka-gpt` candidate remains pending architecture, provenance, license, tokenizer, custom-code, compatibility, and Indonesian/English evaluation audits. Do not promote the Qwen experiment or the provisional non-Qwen candidate as an approved production foundation model without a versioned checkpoint decision record.

The repository also does not yet contain the human-curated 800–1,200 prompt frozen release evaluation suite. The frozen suite validator and manifest/contamination contracts are implemented in `rusamodels.frozen_evaluation` and `rusamodels.dataset_contamination`; they validate a future reviewed suite but do not generate or certify its contents.

## CPU inference versus GPU training

The 16-GB Codespaces Model Server profile is for CPU inference only. The current QLoRA trainer explicitly requires CUDA and refuses CPU training. Run the optional training workflow in a dedicated, trusted CUDA environment (for example, a reviewed Colab T4 session or self-hosted GPU runner).

## Install and validate

```bash
python -m venv .venv-train
source .venv-train/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-training.txt

python rusamodels/training/validate_dataset.py \
  --train-file data/agent_sft/train.jsonl \
  --eval-file data/agent_sft/eval.jsonl
```

## Train and evaluate

```bash
python rusamodels/training/train_sft.py \
  --config configs/training/agent-sft.yaml

python rusamodels/evaluation/evaluate_agent_sft.py \
  --adapter-dir artifacts/rusa-agent-qwen3-4b-qlora/adapter \
  --eval-file data/agent_sft/eval.jsonl \
  --config configs/training/agent-sft.yaml \
  --output artifacts/agent-sft-evaluation.json
```

For a guided Colab run, open `notebooks/RusaModels_QLoRA_T4_Colab.ipynb`. For the constrained NVIDIA T4 experiment from a terminal or trusted self-hosted runner, use the separate `configs/training/agent-sft-t4.yaml` profile and `scripts/train_agent_sft_t4.sh`. The manual GitHub Actions workflow requires an explicitly registered, trusted self-hosted T4 runner and a manual confirmation input.

## Merge and release

```bash
python rusamodels/training/merge_adapter.py \
  --adapter-dir artifacts/rusa-agent-qwen3-4b-qlora/adapter \
  --output-dir artifacts/rusa-agent-qwen3-4b-merged

python scripts/create_safetensors_manifest.py \
  --model-dir artifacts/rusa-agent-qwen3-4b-merged \
  --model-id Qwen/Qwen3-4B-Instruct-2507+rusa-agent-sft \
  --revision eb25fbe4f35f7147763bc24445679d1c00588d89 \
  --license apache-2.0 \
  --tokenizer-id Qwen/Qwen3-4B-Instruct-2507 \
  --tokenizer-revision eb25fbe4f35f7147763bc24445679d1c00588d89
```

Keep adapters, merged weights, and private datasets out of Git. Register approved artifacts through the RusaModels registry and retain their checksums and provenance.

## Safety boundary

The versioned role/tool snapshot in `rusamodels/training/agent_contract.py` validates training examples and evaluation outputs only. It does not grant runtime permissions. RUSA AI's live Tool Gateway and Policy Engine remain the only runtime authorization boundary.

Promotion requires dataset provenance/license review, held-out contract and safety gates, human quality review, prompt-injection testing, exact artifact checksum verification, serving benchmarks, rollback evidence and explicit operator approval.
