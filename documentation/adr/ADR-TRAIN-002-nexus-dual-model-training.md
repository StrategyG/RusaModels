# ADR-TRAIN-002 — NEXUS Foundation and Separate Agent Training Architecture

- Date: 2026-11-15
- Status: **Proposed implementation companion; architecture direction accepted in RusaAi ADR-0003**
- Decision authority: RUSA AI Project Owner
- Implementation owner: RusaModels
- Related decision: [RusaAi ADR-0003 — NEXUS Foundation and Separate Agent Model Architecture](https://github.com/StrategyG/RusaAi/blob/docs/chunk-01-architecture-adr-reconciliation-20261113/docs/adr/ADR-0003-nexus-foundation-agent-architecture.md)
- Supersedes: the pretrained-checkpoint selection proposal in ADR-TRAIN-001

## 1. Scope and ownership

RusaModels owns NEXUS model identities, tokenizer/model training pipelines, immutable artifacts, model registry, evaluation records, inference runtimes, quantization, and release lifecycle. RusaAi remains the Control Plane: agent/task orchestration, trusted identity, policy, approvals, Tool Gateway, sandbox, and audit.

This ADR is a training/Model Plane implementation specification. It does not approve a model size, training run, dataset, compute spend, runtime artifact, or production release.

## 2. Binding model decisions

### 2.1 NEXUS Foundation

- Model ID (logical): `nexus-foundation`.
- Initialization: random initialization; train from scratch.
- Objective: causal next-token pretraining on a diverse, rights-cleared general-purpose corpus.
- No imported pretrained weights, Qwen lineage, Qwen teacher, or Qwen tokenizer.
- Independent architecture config, tokenizer, corpus lineage, run IDs, checkpoints, model card, evaluation suite, and release manifest.

### 2.2 NEXUS Agent

- Model ID (logical): `nexus-agent`.
- It is a separate model with its own checkpoint and independent training/release lineage, not a LoRA-only adapter or alias of Foundation.
- Initialization method remains `OPEN_DECISION`: separate from-scratch initialization or an independently approved non-Qwen starting checkpoint may be considered only after architecture, source lineage, license, data, compute, and evaluation review.
- Qwen-derived weights, tokenizer, teacher, or hidden dependency are prohibited.
- The Agent LLM may propose plans and tool calls; it cannot authorize execution. RusaAi Policy Engine, approvals, Tool Gateway, sandbox, and audit remain authoritative.

### 2.3 Existing Argon specialist

NEXUS Argon v5.8.2 remains a DevOps/SRE specialist candidate. Do not relabel or reuse it as Foundation or Agent. Any reuse of its data or code requires a new identity, rights/provenance, data-quality, and evaluation review.

## 3. Architecture and identity lock

Before training, each model must have a versioned identity record containing:

- model ID, version, role, owner, and intended/prohibited use;
- architecture type and immutable config digest;
- actual parameter count, layers, hidden size, attention/KV heads, FFN, context length, positional encoding, normalization, activation, and embedding/tie policy;
- tokenizer ID/version, algorithm, vocabulary, special tokens, normalization, chat template where relevant, and tokenizer digests;
- initialization method and lineage;
- dataset release IDs/versions and source/split manifest hashes;
- trainer source revision, dependency lock and SBOM, config digest, seed, optimizer, precision, scheduler, batch/sequence settings, and hardware profile;
- checkpoint/artifact file list, byte size, SHA-256, signatures, and evaluation report IDs.

Architecture sizing is not yet locked. Do not silently adopt the planned 96-layer/12,288-hidden blueprint or the separate RusaAi-Edge 1.54B reference as the NEXUS design. Parameter budget, context, vocabulary, and compute must be selected through a dedicated architecture and capacity review.

## 4. Dataset tracks and governance

Maintain separate immutable dataset releases:

| Dataset ID (logical) | Purpose | Not a substitute for |
|---|---|---|
| `nexus-foundation-pretrain` | General-purpose Foundation pretraining | Agent tool trajectories or narrow DevOps chunks |
| `nexus-foundation-instruct-safety` | Foundation instruction/safety post-training | Sealed evaluation data |
| `nexus-agent-trajectories` | Planning, tool selection, arguments, observations, recovery, verification, refusal, and escalation | General Foundation pretraining |
| `nexus-devops-knowledge` | Governed DevOps/SRE retrieval and later specialist adaptation/evaluation | General Foundation corpus |

Every release requires immutable source identity/revision, collection authority, rights/license basis, source hashes, sensitivity classification, privacy/secret redaction, transformation lineage, language/quality distribution, exact/near-deduplication, source/group-aware splits, sealed-test custody, and SHA-256 manifest. Track raw sources, unique source records, unique examples, chunks, tokens, approved records, and optimizer-consumed tokens separately. Related records/chunks must remain in one split.

A declared license is not legal approval. No confidential, credential-bearing, unreviewed, or evaluation data may enter training. Rights withdrawal must be traceable to dataset releases and affected downstream checkpoints.

The reported 50,000 DevOps chunks in the supplied NEXUS v5.8.2 package are not the sole Foundation corpus. The package reports 2,100 unique incident records and 3.2 million memmap tokens, but also reports 80 Argon optimizer steps, 160 memmap samples consumed, and 235,360 optimizer-processed tokens. Reconcile these figures and reproduce sample/token accounting before reusing the corpus. Chunk count alone does not prove unique-example coverage or general-purpose training volume.

## 5. Training strategy

### 5.1 Foundation from scratch

1. Freeze the architecture and compute envelope.
2. Train/configure the tokenizer using only the approved Foundation training split; freeze tokenizer artifacts and digests before pretraining.
3. Initialize model weights randomly and run causal next-token pretraining.
4. Record actual tokens consumed, samples/windows consumed, optimizer steps, skipped records, and data-loader coverage.
5. Preserve resumable checkpoints including model, optimizer, scheduler, RNG state, and dataset/config fingerprints.
6. Run separately governed instruction and safety post-training only after the base model has a frozen evaluation report.
7. Package immutable checkpoints, model card, dataset lineage, run logs, evaluation evidence, and signed release manifest.

### 5.2 Separate Agent model

1. Freeze a distinct Agent architecture and identity; decide its initialization method before training.
2. Build a dedicated Agent trajectory corpus with valid tool schemas, state transitions, observations, failures, bounded recovery, verification, abstention, and approval requests.
3. Use supervised training first. Preference/alignment methods require a separate proposal and approval.
4. Evaluate task success, planning quality, tool selection, structured-argument validity, recovery, verification, refusal/abstention, and prompt-injection resistance.
5. Keep runtime authorization external to the model and enforced by RusaAi.

## 6. Dependency and implementation baseline (2026-11-15)

The following RusaModels work exists only in open pull requests and must not be described as merged or production-ready:

| PR | Scope | Observed state | Required action before use |
|---|---|---|---|
| [#14](https://github.com/StrategyG/RusaModels/pull/14) | Carbon-Tiny from-scratch CPU trainer (3.2M parameters) | Open; CI success on PR head; base SHA is stale and GitHub reported mergeable=false | Rebase to current main, review, rerun exact-head CI; retain as tiny engineering smoke path only |
| [#17](https://github.com/StrategyG/RusaModels/pull/17) | Dataset engineering engine | Open; CI success on PR head; base SHA is stale and GitHub reported mergeable=false | Rebase, review data governance/scale limitations, rerun exact-head CI |
| [#18](https://github.com/StrategyG/RusaModels/pull/18) | Frozen benchmark engine | Open; CI success on PR head; based on older main | Rebase and validate benchmark contracts before use |
| [#20](https://github.com/StrategyG/RusaModels/pull/20) | Architecture and tokenizer engine | Draft; stacked on PR #14; no CI run observed | Resolve parent dependency, rebase/stack deliberately, run exact-head CI and review |

These PRs are candidate dependencies, not authorization to train. The Carbon-Tiny trainer proves only a small CPU optimization path. The research-100M architecture/tokenizer work is design/research and does not establish a selected NEXUS architecture or a production-scale training environment.

The existing Qwen QLoRA workflow and `agent-sft.yaml` remain a legacy, isolated experiment. They are not part of the NEXUS dependency graph and must not be run for NEXUS.

## 7. Training Authorization Gate (TAG)

No tokenizer or model training job may start until every item is `PASS` and linked to exact revisions/digests:

1. RusaAi CHUNK 00 and CHUNK 01 are closed `PASS`; relevant identity/security and data-governance dependencies are accepted.
2. Foundation and Agent have distinct model identity records; Foundation architecture/tokenizer are frozen; Agent initialization is explicitly decided; no Qwen lineage exists.
3. Dataset releases have source-rights approval, privacy/secret screening, quality/dedup reports, source/group-aware partitions, immutable manifests, and sealed held-out evaluation data.
4. Trainer source/dependencies are pinned; tests prove sample/token consumption, labels/masks, gradient updates, checkpoint integrity, deterministic/reproducible resume, and no data leakage.
5. Compute feasibility records parameter count, tokens, sequence length, batch/accumulation, optimizer/activation memory, hardware, storage, duration/cost, and recovery plan. A Colab T4 or CPU Tiny run is not presumed sufficient for full Foundation pretraining.
6. Training environment is isolated with least privilege, controlled egress, secret-free logs, SBOM, vulnerability disposition, artifact access controls, and backup.
7. Foundation and Agent benchmark suites, baselines, human-review process, metrics, and acceptance thresholds are frozen before training.
8. Model, data, security, and compute/budget owners approve the exact run plan.

Missing or stale evidence means `BLOCKED` or `NOT_VERIFIED`. No bulk ingestion, tokenizer training, or model training may be dispatched.

## 8. Training and release states

Training run: `PLANNED → AUTHORIZED → RUNNING → COMPLETED | FAILED | CANCELLED → EVALUATED`.

Registry release: map the NEXUS track to the existing RusaModels `draft → candidate → evaluated → approved → deprecated → archived` lifecycle. Do not bypass registry approval.

Foundation and Agent are evaluated and approved independently. A training run completing does not imply quality acceptance; quality acceptance does not imply runtime acceptance; runtime acceptance does not imply production approval. Rollback uses a previously approved immutable artifact and never overwrites checkpoints.

## 9. Acceptance gates

| Gate | Evidence | Acceptance |
|---|---|---|
| T0 Identity | Distinct model IDs, frozen architecture/tokenizer config, initialization policy | Foundation is verifiably random-initialized; Agent is separately identified; no Qwen |
| T1 Data | Immutable manifests, rights approvals, privacy, quality, dedup, split and sealed-test reports | No unapproved data or cross-split leakage |
| T2 Trainer | Pinned code/dependencies, sample/token accounting, smoke, resume and checkpoint tests | Actual consumption reconciles with dataset manifest; no false coverage claims |
| T3 Authorization | Completed TAG checklist and named approvals | All prerequisites pass before real training |
| T4 Training | Run logs, seeds, hardware, consumed tokens/samples, checkpoint hashes | Reproducible within documented limits |
| T5 Quality | Frozen independent Foundation/Agent benchmark, human review, baseline comparison | Predeclared thresholds pass; no critical safety blocker |
| T6 Runtime | Model-specific hardware/quantization/latency/memory/concurrency/restart report | Predeclared target profile is met |
| T7 Release | Signed manifests, SBOM, model cards, security review, rollback, owner approval | Each model independently approved |
| T8 Production | RusaAi CHUNK 13/14 evidence and sign-offs | Explicit GO only after all P0 gates pass |

Thresholds must be fixed before final evaluation and tied to workload/hardware. No universal numeric target is assumed in this ADR.

## 10. Open decisions

- Foundation parameter budget, exact dimensions, context, vocabulary, and general corpus token target.
- Foundation compute hardware, storage, budget, and run schedule.
- Agent initialization method, model size/architecture, and tokenizer sharing compatibility.
- Frozen benchmark definitions and numerical acceptance thresholds.

Until these are decided and TAG passes, NEXUS training remains blocked.
