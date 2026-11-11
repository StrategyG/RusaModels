# ADR-TRAIN-001 — Foundation checkpoint selection

**Status:** DRAFT — no foundation checkpoint approved  
**Decision owner:** Rusaai model owner  
**Scope:** Product model foundation for RusaAi-Edge; this record does not approve a runtime artifact.

## 1. Context

RusaAi Architecture v2.0 defines the RusaAi-Edge 1.541B dense Transformer as a reference architecture and names `rusaai-edge-1.54b` as its architecture reference ID. It does not establish that an identical, compatible checkpoint is available.

RusaAi Training Engine v1.0 says that, until a checkpoint passes architecture and license audit, the product identity is `rusaai-edge-derived` with status `CHECKPOINT_CANDIDATE_PENDING_AUDIT`. The two source documents therefore describe different identity roles. This ADR keeps both names distinct:

- Architecture reference ID: `rusaai-edge-1.54b`
- Pre-audit product identity: `rusaai-edge-derived`
- Approved runtime model ID: **PENDING**
- Decision status: **PENDING**

The RusaModels manifest and Windows launcher preserve this distinction and block runtime startup until an approved runtime ID is recorded.

## 2. Non-negotiable selection gates

A foundation checkpoint must pass every applicable gate before approval:

1. Non-Qwen model lineage and provenance review.
2. License and acceptable-use review.
3. Tokenizer files, special tokens and chat-template review.
4. Actual architecture/configuration and parameter-count inspection.
5. Transformers loading test without unsafe unreviewed code.
6. Exact source revision pinning before any custom model code is executed; never enable `trust_remote_code=True` without source audit.
7. PEFT/LoRA module compatibility test if LoRA is considered.
8. GGUF conversion or existing GGUF artifact verification.
9. CPU-only inference test on the target device.
10. Indonesian and English baseline evaluation.
11. Security and supply-chain review.

If the selected checkpoint differs from the reference configuration, record the actual checkpoint configuration as authoritative and classify the result as `DERIVATIVE_ADAPTATION`, not `IDENTICAL`.

## 3. Candidates and current disposition

### 3.1 Provisional 8 GB pilot candidate — not approved

`BhinekaIntiLabs/bhineka-gpt`

The Training Engine v1.0 source and the current Hugging Face model card describe a Llama-style Indonesian-English causal model with 556,269,696 parameters, a 2,048-token context, 64K vocabulary, hidden size 1,152, 28 layers, 16 attention heads, 8 key/value heads, 3,072 feed-forward size, GQA, SwiGLU, RoPE and RMSNorm. The model card declares Apache-2.0 metadata and advertises GGUF/llama.cpp integration. The model card reports validation on 53,678,481 held-out tokens with loss 2.5355 and perplexity 12.6227, evaluated on CUDA; this is not a CPU-device benchmark or a complete instruction/safety evaluation.

**Research snapshot (2026-11-11):** [Current Hugging Face model card](https://huggingface.co/BhinekaIntiLabs/bhineka-gpt). The page's quick-start Transformers snippet loads without `trust_remote_code`, while its detailed usage section says the custom `bhineka` architecture may require custom modeling files and `trust_remote_code=True`. Treat this documentation inconsistency as an audit blocker. The current immutable repository revision, exact files, tokenizer/chat-template files, GGUF artifact filename and artifact checksum have not yet been pinned or independently verified.

Disposition: **PENDING_AUDIT / PILOT_CANDIDATE_ONLY**.

Mandatory caveats from the source:

- The card's Transformers instructions are inconsistent about custom `bhineka` code and `trust_remote_code=True`; inspect the exact files and pin the immutable revision before loading or executing any repository code.
- The published context is 2K, not the desired 4K/8K; keep 2K until extension is independently validated. Its 556.3M parameter count is below the architecture's 1B–1.5B 8 GB target range, so it remains a lightweight pilot candidate and not a match for that model-size target.
- Published validation-loss evaluation used CUDA and does not prove CPU training performance.
- Broader instruction-following, factuality and safety evaluations remain necessary.
- Published claims and benchmark numbers are not independent certification.
- PEFT target modules must be inspected, not assumed.
- Complete lineage must be reviewed to confirm that the checkpoint is not derived from Qwen.

This 556.3M candidate is an 8 GB pilot candidate, not evidence that the 1.541B reference architecture has been implemented.

### 3.2 Comparison candidates — not approved

The Training Engine v1.0 source lists the following as comparison candidates, not automatic replacements:

- Llama 3.2 1B
- OLMo 2 1B
- SmolLM2 1.7B
- Granite 3.3 2B

Each requires its own language-coverage, license, architecture, context, hardware and security review.

## 4. Evidence record — to be completed

| Evidence | Required value | Current status |
|---|---|---|
| Candidate repository | Exact repository identifier | NOT_SELECTED |
| Immutable revision | 40-character commit/revision | NOT_PINNED |
| Model files | Exact filenames and SHA-256 | NOT_RECORDED |
| Model lineage | Base model and provenance report | NOT_VERIFIED |
| License | Model-card metadata says Apache-2.0; inspect license file, training data and dependency terms | METADATA_DECLARED / REVIEW_PENDING |
| Actual architecture | Config and actual parameter count | NOT_AUDITED |
| Architecture relation | IDENTICAL or DERIVATIVE_ADAPTATION | NOT_AUDITED |
| Tokenizer | Repository, revision, files and special tokens | NOT_VALIDATED |
| Chat template | Exact template and validation result | NOT_VALIDATED |
| Custom code | Source audit and sandbox result | NOT_REVIEWED |
| PEFT/LoRA | Module compatibility, if applicable | NOT_TESTED |
| GGUF | Model card advertises GGUF/llama.cpp; verify exact file, format, quantization and SHA-256 | NOT_VERIFIED |
| CPU inference | Target-device test report | NOT_TESTED |
| Indonesian baseline | Frozen evaluation report | NOT_TESTED |
| English baseline | Frozen evaluation report | NOT_TESTED |
| Security/supply chain | Review report | NOT_REVIEWED |

## 5. Runtime identity decision

Do not change the runtime model ID by assumption.

| Role | Identifier | Status |
|---|---|---|
| Architecture reference | `rusaai-edge-1.54b` | SOURCE_DEFINED |
| Pre-audit product identity | `rusaai-edge-derived` | SOURCE_DEFINED |
| Approved runtime identity | To be entered after decision | PENDING |

The approved runtime identity must be written to the RusaModels model manifest and match `RUSAMODELS_INFERENCE_MODEL_ID`. Until then, the launcher must remain fail-closed.

## 6. Final decision

Select exactly one disposition after all required evidence is reviewed:

- `APPROVED`
- `REJECTED`
- `PILOT_ONLY`

**Selected repository:** PENDING  
**Selected immutable revision:** PENDING  
**Runtime model ID:** PENDING  
**Decision rationale:** PENDING  
**Reviewer / approval reference:** PENDING  
**Decision date:** PENDING

No checkpoint is approved by this draft ADR.
