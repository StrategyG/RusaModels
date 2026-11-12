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

Use the Dataset Engine text exporter to produce separate train and validation corpora. The reference config requires both export sidecars by default (`data.require_export_manifest: true`). Set `data.dataset_id`, `data.dataset_version`, and `data.license_ids` to exactly match the export sidecars before a non-smoke run. The trainer verifies each text-file SHA-256, sidecar schema, dataset identity, split name, license declarations, tokenizer, and shared source release-manifest digest. It does not legally validate the declared licenses. For manually curated plain text, set `require_export_manifest` to false and provide a reviewed license declaration; this is a less strongly traceable path. The trainer rejects identical train/eval files or identical SHA-256 values.

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

Resume from the latest checkpoint. Resume is accepted only when the model configuration, training contract, dataset fingerprint, export-sidecar digests, and source checksums match. You may increase the requested total step count; changing the data, seed, batch/block settings, optimizer configuration, or thread count requires a new run:

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


## Neural Architecture Engine v1 and the 100M research blueprint

The Phase 3 architecture engine currently supports one explicit architecture family: a pre-norm, dense decoder-only Transformer with GQA, RoPE, RMSNorm, SwiGLU, tied embeddings, and bias-free projections. Unsupported architecture fields fail closed rather than being silently ignored.

The separate configs/architecture/rusaai-research-100m.json blueprint preserves the Training Engine v1.0 research target:

| Component | Research 100M target |
|---|---:|
| Layers | 12 |
| Hidden size | 768 |
| Query heads / KV heads | 12 / 4 |
| Head dimension | 64 |
| SwiGLU intermediate | 2,048 |
| Vocabulary | 32,000 |
| Context design target | 4,096 |
| Calculated trainable parameters | 100,092,672 |
| EOS token ID | 2 |

Validate the blueprint without allocating model weights:

~~~bash
python scripts/validate_architecture.py \\
  --config configs/architecture/rusaai-research-100m.json
~~~

Optionally instantiate it after installing the CPU model dependencies. The model factory has a default 120-million-parameter construction ceiling and validates every layer's projection shapes, RoPE cache, normalization dimensions/epsilon, and exact parameter count:

~~~bash
python scripts/validate_architecture.py \\
  --config configs/architecture/rusaai-research-100m.json \\
  --instantiate
~~~

Instantiation is not training. The blueprint remains design_ready_training_not_started; no 100M checkpoint or capability claim is implied. The existing Carbon-Tiny v0.1 model remains a separate 3.2M-parameter byte-level smoke/research model.

## Learned tokenizer for the 100M scratch track

Carbon-Tiny's reversible UTF-8 byte tokenizer is not a learned subword tokenizer. The new tokenizer manager trains a separate ByteLevel BPE artifact for the from-scratch 100M research model. It must not replace a selected foundation model's tokenizer during ordinary LoRA/adaptation.

Install the CPU research extra (includes PyTorch, NumPy, and Hugging Face Tokenizers):

~~~bash
python -m pip install -e ".[carbon-cpu]"
~~~

First export the verified Dataset Engine train split to plain text. The exporter creates a SHA-256 sidecar manifest:

~~~bash
python scripts/export_dataset_text.py \\
  --release /path/to/verified-dataset-release \\
  --split train \\
  --output /path/to/rusaai-train.txt
~~~

Train and publish the immutable tokenizer artifact:

~~~bash
python scripts/train_rusa_tokenizer.py \\
  --corpus /path/to/rusaai-train.txt \\
  --corpus-manifest /path/to/rusaai-train.txt.manifest.json \\
  --output-dir artifacts/tokenizers/rusaai-research-100m-bpe-32k \\
  --vocab-size 32000 \\
  --min-frequency 2
~~~

The trainer accepts only a verified Dataset Engine text-export sidecar with split=train, non-placeholder license identifiers, matching corpus digest/size, and a pinned release-manifest digest. It rejects validation, test, and frozen evaluation exports. The output contains tokenizer.json and rusa-tokenizer-manifest.json, including the tokenizer/library version, exact special-token IDs, vocabulary size, training corpus hashes, source dataset identity, and license declarations. The default special-token order is [UNK]=0, [BOS]=1, [EOS]=2, [PAD]=3, matching the 100M architecture blueprint. The 100M table in Training Engine v1.0 does not specify RoPE theta or RMSNorm epsilon: this blueprint currently uses theta=10000 from the Carbon implementation baseline and epsilon=1e-5 from the RusaAi Architecture v2.0 reference. Treat both as explicit provisional engineering defaults and revalidate them before a real training run.

The trainer requires the corpus to produce the requested vocabulary size; it does not pad an undersized vocabulary with synthetic entries. Its Indonesian, English, code, and JSON round-trip probes are compatibility smoke tests, not a morphology benchmark or evidence of language quality. Review segmentation quality on a representative, human-approved corpus before using the tokenizer for a research run.

The tokenizer artifact loader verifies the manifest, vocabulary, special-token IDs, file size, implementation type and SHA-256 before use. For a separately selected Hugging Face foundation tokenizer, use rusamodels.tokenizers.validate_chat_template after loading it with a pinned revision and trust_remote_code=false; this checks generation-prompt encoding, EOS/PAD IDs and decode behavior. The research BPE tokenizer itself has no chat template until a chat-format contract is deliberately added. Keep the corpus, tokenizer, and resulting model artifacts outside Git. Declared license IDs are provenance metadata, not legal validation.

