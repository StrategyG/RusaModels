# Carbon-CPU: from-scratch CPU training

This training path is separate from the existing QLoRA agent-SFT workflow. It initializes every Carbon-Tiny weight randomly and trains a decoder-only Transformer with next-byte prediction. It does not download or load Qwen, Llama, or any other pretrained model.

## Initial architecture

| Component | Carbon-Tiny v0.1 |
|---|---:|
| Trainable parameters | 3,214,080 (about 3.2M) |
| Layers | 4 |
| Hidden width | 256 |
| Query heads / KV heads | 4 / 2 (GQA) |
| FFN | SwiGLU, intermediate width 768 |
| Position encoding | RoPE |
| Normalization | RMSNorm |
| Vocabulary | 258 byte-level IDs |
| Maximum sequence length | 256 |
| Training objective | causal next-token prediction |
| Precision | FP32 |
| Device | CPU only |

This is a small engineering prototype, not the 30B Carbon target. An Android device with 8 GB RAM can validate and experiment with this small model, but cannot realistically pretrain the proposed 30B model from scratch.

## Environment

The trainer requires Python 3.11+, NumPy, and a PyTorch build compatible with the host OS and CPU. On Linux x86-64, a CPU PyTorch wheel is normally available from the official PyTorch installation selector. On Android/Termux, standard Linux wheels are not automatically compatible with Android's Bionic libc; use a Termux-compatible PyTorch package/build if one is available for the installed Termux repository and Python version. Do not install `requirements-training.txt` for this workflow: that file is for the existing CUDA QLoRA trainer and includes GPU-oriented dependencies.

Verify the environment:

```bash
python -c 'import numpy, torch; print("NumPy", numpy.__version__); print("PyTorch", torch.__version__); print("CUDA available:", torch.cuda.is_available())'
```

The Carbon trainer explicitly creates tensors and the model on `cpu`; it does not use CUDA even if a CUDA-enabled PyTorch build happens to be installed. The byte tokenizer is a reversible UTF-8 byte mapping with 256 byte IDs and reserved BOS/EOS IDs; it is not a learned subword tokenizer.

## Prepare data

Place an authorized UTF-8 text corpus at `data/carbon_cpu/corpus.txt`, or pass another file with `--data`. Set `data.dataset_id`, `data.dataset_version`, and a reviewed, non-placeholder `data.license_id` in the training config before a non-smoke run. The trainer records the corpus SHA-256 and dataset declaration; it does not legally validate the license. The trainer makes a 95/5 train/validation split when no separate evaluation file is supplied. Alternatively, pass `--eval-data path/to/heldout.txt` and declare `data.eval_license_id` if the evaluation corpus has a different license. The trainer rejects identical train/eval files or identical SHA-256 values.

A few kilobytes can validate code paths but will overfit immediately. For actual pretraining, curate a large, diverse corpus with documented source, license, language distribution, deduplication, and held-out evaluation. Keep the held-out set out of training.

## Train from random initialization

```bash
python scripts/train_carbon_cpu.py \
  --config configs/training/carbon-cpu-tiny.json \
  --data data/carbon_cpu/corpus.txt \
  --threads 4
```

For a short pipeline-only check (uses synthetic text and must not be described as useful model training):

```bash
python scripts/train_carbon_cpu.py --smoke-test --steps 3 --batch-size 1 --block-size 32
```

Resume from the latest checkpoint. Resume is accepted only when the model configuration, training contract, dataset fingerprint, and source checksums match. You may increase the requested total step count; changing the data, seed, batch/block settings, optimizer configuration, or thread count requires a new run:

```bash
python scripts/train_carbon_cpu.py \
  --config configs/training/carbon-cpu-tiny.json \
  --data data/carbon_cpu/corpus.txt \
  --resume artifacts/carbon-cpu-tiny-v0.1/checkpoint.pt
```

Override steps, batch size, context block, and thread count with `--steps`, `--batch-size`, `--block-size`, and `--threads`. `block_size` cannot exceed `model.max_seq_len`.

## Outputs

The output directory contains:

- `checkpoint.pt`: resumable model + optimizer state, dataset/training fingerprints, and Python/NumPy/PyTorch RNG state.
- `final.pt`: final model state and architecture config.
- `metrics.jsonl`: train and validation loss snapshots.
- `run_metadata.json`: CPU/software details, dataset/license declaration, corpus SHA-256, training-contract digest, artifact checksums, and smoke-test marker.
- `model_config.json`: architecture configuration.
- `train_tokens.u16` / `eval_tokens.u16`: local byte-token caches.

Do not commit corpora, token caches, or weights to Git. Back up checkpoints separately.

## Training claims and limitations

A successful run proves that this small architecture completed an optimization loop; it does not establish general reasoning, safety, instruction-following, or production quality. The existing repository's 120/36 synthetic agent-SFT examples are not a substitute for a pretraining corpus. Carbon-MoE, expert routing, PRM/value head, long-context training, quantization-aware training, and agent tools are not implemented by this Tiny trainer and must be developed and evaluated as separate stages.
