# RusaModels agent SFT data

This directory supports data ingestion, cleaning, normalization, deduplication, provenance, and multilingual/multimodal preparation.

**Current status:** an initial synthetic agent-SFT seed corpus is available under `data/agent_sft/`. Broader production data collection and multimodal preparation remain planned. Dataset preparation does not mean model weights have been trained.

## Agent SFT data

- `curated/train.jsonl`: 120 examples, 20 for each supported worker role.
- `curated/eval.jsonl`: 36 held-out examples, 6 for each supported worker role.
- `curated/PROVENANCE.md`: source basis, construction method, role mapping, and limitations.

Each JSONL line matches `schemas/llm/agent-sft-example.schema.json`. Each example contains `id`, `agent_role`, `messages`, and `response`. The first message is `system`; the final message is `user`. The response is the exact JSON envelope consumed by `AutonomousCodeWorker`.

The current SFT runtime supports exactly these six roles: `analyst`, `architect`, `developer`, `debugger`, `tester`, and `security`. The wider architecture documents define additional agents, but they are not training roles until runtime contracts and permissions support them.

The validator rejects unknown roles, tool calls outside the shared role allowlist, unsafe file paths, forbidden `run_check` profiles, duplicate examples, and train/eval leakage. The trainer requires at least 100 reviewed train examples and 30 evaluation examples by default. The seed corpus meets the count guardrails; it is synthetic and must be reviewed and expanded before production use.

The evaluation split contains 12 blocked examples (33.3%) and 24 unblocked examples (66.7%) to exercise the configured safety metrics. Keep it held out.

## Data governance

The seed examples are synthetic and contain no production conversations, secrets, personal data, or live tenant data. Before production training, data owners must document source/license/provenance, redact credentials and personal data, remove private repository content unless its use is authorized, and review prompt-injection or harmful tool-use examples. Do not commit private datasets or generated weights to Git.

## Validate and train

```bash
python rusamodels/training/validate_dataset.py \
  --train-file llm-project/data/curated/train.jsonl \
  --eval-file llm-project/data/curated/eval.jsonl

python rusamodels/training/train_sft.py
```

The QLoRA trainer requires a dedicated CUDA GPU environment. The current committed dataset is a seed experiment corpus, not a trained model artifact.
