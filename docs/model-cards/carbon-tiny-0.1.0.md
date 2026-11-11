# Carbon-Tiny Model Card

**Model ID:** `carbon-tiny`  
**Version:** `0.1.0`  
**Status:** Research prototype — not production-approved  
**Architecture:** Decoder-only dense Transformer  
**Initialization:** Random initialization; no pretrained weights are loaded

## Summary

Carbon-Tiny is a small, from-scratch CPU research model used to validate RusaModels' architecture and training pipeline. It is not the RusaAi production foundation model and must not be represented as a production-quality assistant.

## Architecture

| Property | Value |
|---|---:|
| Trainable parameters | 3,214,080 |
| Transformer layers | 4 |
| Hidden dimension | 256 |
| Query heads | 4 |
| KV heads | 2 |
| Feed-forward dimension | 768 |
| Maximum context | 256 tokens |
| Vocabulary | 258 IDs |
| Tokenizer | UTF-8 byte-level |
| Attention | Causal GQA + RoPE |
| Normalization | RMSNorm |
| Feed-forward | SwiGLU |
| Embedding/output | Tied weights |

The tokenizer maps UTF-8 bytes to IDs 0–255 and reserves IDs 256 and 257 for EOS and BOS. It is reversible for valid UTF-8 input, but it is not a learned subword tokenizer.

## Intended use

- Validate the Carbon Transformer forward pass and CPU implementation.
- Exercise checkpoint save/resume and reproducibility controls.
- Validate Dataset Engine export-manifest integration.
- Run small research experiments on authorized text corpora.

## Out of scope

- Production conversational use.
- Safety-critical decisions.
- Claims of general reasoning, coding, or multilingual competence.
- Deployment as a RusaAi foundation model.
- Claims of model quality before a documented training run and benchmark report.

## Data and training requirements

Non-smoke training requires dataset identity, version, and reviewed license declarations. The reference configuration requires verified Dataset Engine train and validation text-export sidecars. The trainer records corpus hashes, export-manifest digests, training-contract digest, and RNG state.

The training code checks declared license metadata; it does not provide legal license validation. Dataset quality, privacy, provenance, and benchmark contamination must be evaluated separately.

## Evaluation status

No production benchmark result is claimed by this model card. A smoke test or a short CI training run validates software behavior only; it does not establish model capability.

Before any candidate can be considered for release, attach:
- Exact training dataset release and export manifest digests.
- Training configuration and code revision.
- Final checkpoint SHA-256.
- Held-out validation results.
- Frozen benchmark results.
- Safety and regression reports.
- CPU inference latency and peak-memory measurements on named hardware.

## Limitations

- Very small capacity (approximately 3.2M parameters).
- Short 256-token context.
- Byte-level tokenization produces long sequences for natural language.
- No KV-cache optimization in generation.
- No demonstrated production-scale training or evaluation.
- Quality and safety characteristics are unknown until measured.

## Release decision

**Current decision: Research only.** Do not promote this model to an approved or production lifecycle state until the required model evaluation and platform release gates have been satisfied.
