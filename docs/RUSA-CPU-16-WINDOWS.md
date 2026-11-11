# RUSA-CPU-16 — Native Windows 11 Edge Profile

**Architecture:** RusaAi Unified CPU-First AI Architecture v2.0  
**Profile:** `rusa-cpu-16`  
**Profile version:** 2.0.0  
**Status:** Architecture target — not yet deployable  
**Platform target:** Windows 11 x64, Intel Core i5 (exact SKU to be detected), 16 GB physical RAM, CPU-only

This document supersedes the Qwen3-based v1.0 deployment instructions for the v2.0 profile. The v1.0 configuration is preserved at `docs/archive/RUSA-CPU-16-WINDOWS-v1.0-Qwen3.md` and `configs/runtime/archive/rusa-cpu-16-v1.0-qwen3.yaml`. Do not mix the two model baselines.

## 1. Model readiness gate

The v2.0 architecture reference model is **RusaAi-Edge-1.54B** (`rusaai-edge-1.54b`). The separate Training Engine v1.0 source uses `rusaai-edge-derived` as the pre-audit product model identity. This repository records both names and leaves `runtime_model_id_decision: PENDING`; the Windows launcher blocks startup until the user-approved runtime alias is written to the manifest and matches the configured model ID. The architecture source explicitly records:

- Checkpoint: `NOT_SELECTED`
- Tokenizer: `NOT_SELECTED`
- GGUF artifact: `NOT_BUILT`
- Training: `NOT_STARTED`
- Benchmark: `NOT_TESTED`

A compatible **non-Qwen open-weight checkpoint** must be selected, its license and immutable revision recorded, and its architecture/tokenizer/chat template audited before a runtime artifact can be built. The target design is not itself a checkpoint or a runnable model. If the selected checkpoint differs from the reference configuration, the actual checkpoint configuration becomes the source of truth and RusaAi-Edge must be described as a derivative/adaptation rather than an identical implementation.

The machine-readable target is:

- `configs/runtime/rusa-cpu-16.yaml` — device/deployment profile.
- `models/rusaai-edge/manifests/0.1.0-design.json` — model design manifest.

The Windows launcher intentionally fails closed while the manifest remains in its current design-only state. Do not bypass that gate or substitute Qwen3-4B.

## 2. v2.0 deployment baseline

| Component | v2.0 target |
|---|---|
| Operating system | Windows 11 64-bit |
| CPU | Intel Core i5; exact SKU, physical cores and logical processors detected |
| Physical RAM | 16 GB |
| Model ID | `rusaai-edge-1.54b` |
| Model architecture | Dense decoder-only Transformer; 1,541,277,696 target parameters |
| Checkpoint | Non-Qwen open-weight checkpoint; not yet selected |
| Artifact | GGUF Q4_K_M target; not yet built |
| Runtime | llama.cpp CPU build |
| Initial context | 4,096 tokens |
| Context target | 8,192 tokens, only after target-device RAM acceptance |
| Generation threads | 4 initially; reduce to detected physical core count when below 4 |
| Batch threads | 4 initially |
| HTTP threads | 2 initially |
| Batch / micro-batch | 128 / 64 initially |
| Resident model | 1 |
| Active generation | 1 |
| Pending inference queue | Maximum 3; owned by Rusaai |
| KV cache | Q8_0 only when supported and validated |
| GPU layers | 0 |
| llama.cpp | `127.0.0.1:8081` |
| RusaModels Model Server | `127.0.0.1:8090` |
| External/cloud inference fallback | Disabled by default |

The one active generation and three pending requests are an orchestration contract owned by **Rusaai**. RusaModels provides the model runtime and inference API; it does not own agent scheduling, queue priority, workflow decisions, tool execution, or policy.

## 3. Resource assumptions

The v2.0 architecture estimates Q4_K_M weights at approximately 0.9–1.1 GB for the reference 1.54B design. Estimated Q8_0 KV cache is approximately 128 MiB at 4K and 256 MiB at 8K. These are planning estimates, not measurements.

- Keep at least 2 GB free RAM during inference.
- Record peak RAM, minimum free RAM, CPU utilization and thermal throttling when telemetry is available.
- Do not reuse the v1.0 Qwen3-4B 2.5-GB weight estimate for RusaAi-Edge.
- Other runtime, operating-system and service memory must be measured on the target device.
- Start with 4K context. Enable 8K only after representative device acceptance.
- Thread counts are starting values and must be tuned against the exact CPU SKU and benchmark evidence.

The CPU/resource backpressure thresholds (CPU 70%/85%, RAM 75%/85%, free RAM below 2 GB) are owned by the Rusaai resource monitor/scheduler. RusaModels must not implement a second agent scheduler.

## 4. Model-plane and control-plane boundary

**RusaModels owns:**
- Model registry and model/artifact lifecycle.
- Checkpoint/artifact metadata and checksum verification.
- CPU runtime startup and model inference.
- Model load, inference and runtime telemetry.
- Versioned Model Server API and structured inference responses.

**Rusaai owns:**
- Supreme Orchestrator, agents, workflow and durable task queue.
- One-active-generation scheduling and up to three pending inference requests.
- Context Manager and reasoning controller.
- Carbon and Astra capability profiles (not separate model copies).
- SQLite/FTS5 memory and RAG.
- ORM, PRM-Lite, independent QA, Policy Engine and Tool Gateway.
- All tool execution and user-approval decisions.

RusaModels must not execute tools or decide workflow permissions. Carbon V1 preprocessing (OCR/ASR/document extraction) is not evidence of native multimodal model capability.

## 5. Prepare the deployment after model selection

Do not perform these steps with the current design-only manifest. They become valid only after the non-Qwen checkpoint, tokenizer and artifact have been approved and recorded.

1. Pin the upstream checkpoint to an immutable revision and record its license.
2. Audit model provenance and verify that the selected foundation lineage is non-Qwen.
3. Audit actual layers, hidden size, attention/GQA, FFN, vocabulary, tokenizer and chat template against the target design.
4. Record `architecture_relation: IDENTICAL` only when the actual configuration and parameter count match the reference. Otherwise record `DERIVATIVE_ADAPTATION`, the actual parameter count and a SHA-256 of the actual checkpoint configuration.
5. Build or obtain a legally usable GGUF artifact and quantize to Q4_K_M.
6. Record the exact artifact filename and SHA-256 in the model manifest.
7. Record tokenizer repository/revision and chat-template validation.
8. Set the manifest statuses to `SELECTED`, `VERIFIED`, `VALIDATED`, and `BUILT` only when each step has evidence, then set the top-level status to `READY_FOR_DEVICE_VALIDATION`.
9. For an 8K test, set `benchmark.context_8192_status: TESTING`, `RUSAMODELS_CONTEXT_SIZE=8192` and `RUSAMODELS_CONTEXT_8K_TEST_MODE=true`; run `scripts/verify_rusa_cpu_16_context.ps1`. Change the manifest status to `PASS` only after reviewing its report and confirming at least 2 GB free RAM; then set test mode to `false` and restart to verify the normal production gate.
10. Populate `.env` from `.env.rusa-cpu-16.example`, including the local artifact path and the same SHA-256 as the manifest.

The manifest gate in `scripts/start_rusa_cpu_16.ps1` checks the non-Qwen repository and lineage, immutable revision, license, audited architecture relation and actual parameter count/configuration hash, tokenizer/chat-template validation, GGUF Q4_K_M status, filename and SHA-256 agreement. It refuses to start while any required evidence is missing.

## 6. Local edge service configuration

The edge example uses:
- Local SQLite for RusaModels registry metadata and local artifact storage.
- One loopback-only llama.cpp upstream.
- A dedicated Rusaai inference principal with role `inference`.
- A separate internal service-token digest.
- No S3 credentials or external inference fallback.
- No second embedding runtime in this initial edge service configuration.

Copy the example only after the artifact gate is satisfied:

```powershell
Copy-Item .env.rusa-cpu-16.example .env
notepad .env
```

Generate independent high-entropy inference and service tokens. Keep raw tokens in Rusaai secret storage; only their SHA-256 digests belong in the RusaModels `.env`. Never commit raw tokens or the populated `.env`. Before running either acceptance script, expose the raw inference token from the secret store to the current PowerShell process as `RUSAMODELS_API_TOKEN`; do not put that raw token in `.env` or a committed script. The launcher temporarily clears inherited raw tokens from child services and restores the caller's environment when it exits.

Rusaai connects through the Model Access Gateway using:

- Base URL: `http://127.0.0.1:8090/openai/v1`
- Model ID: `rusaai-edge-1.54b`
- Bearer token: raw inference token held by Rusaai.

The llama.cpp and Model Server ports remain bound to loopback. Do not publish them to the public internet.

## 7. Start, stop and acceptance

After the model manifest and local artifact are complete:

```powershell
.\scripts\start_rusa_cpu_16.ps1
.\scripts\verify_rusa_cpu_16.ps1 -SequentialRequests 20
# For 8K acceptance only, after setting manifest status TESTING and the two 8K environment variables:
.\scripts\verify_rusa_cpu_16_context.ps1
.\scripts\stop_rusa_cpu_16.ps1
```

The launcher detects Windows/CPU/RAM, records Windows-reported instruction features, records the llama.cpp version, verifies the manifest and GGUF SHA-256, applies migrations, and starts the runtime with GPU offload disabled. It defaults to 4K. For 8K, the manifest must record `PASS`, or be explicitly set to `TESTING` while `RUSAMODELS_CONTEXT_8K_TEST_MODE=true`; the latter is acceptance-only and must not be treated as production approval.

The sequential acceptance script checks the authenticated model ID and performs 20 streaming requests distributed across prompt-body targets of 128, 1,024 and 4,096 tokens. It records TTFT, end-to-end latency and generation tokens/second per prompt size, plus sampled CPU/free RAM. The separate 8K context script tokenizes a long probe, sends a 7,400–7,600-token prompt, checks returned usage against the 8,192-token limit and samples system free RAM every 500 ms while the request and response body are pending. This is sampled system free RAM, not process-specific peak-RAM telemetry. Neither script replaces the architecture's full device acceptance.

Before calling the device production-ready, record evidence for:

- Non-Qwen checkpoint, license, immutable revision and actual configuration.
- Tokenizer and chat template validation.
- GGUF Q4_K_M artifact and SHA-256.
- CPU-only execution and model residency.
- Context 4K test; context 8K test before enabling 8K.
- 20 sequential requests and a 30-minute stable session.
- Minimum 2 GB free RAM, CPU utilization and thermal telemetry where available.
- One active generation and at most three pending requests in Rusaai.
- Restart recovery and structured inference/API behavior.
- TTFT and tokens/second for 128/1K/4K prompt bodies, latency, maximum observed system memory use, minimum sampled free RAM and queue wait/depth (queue telemetry from Rusaai).
- ORM, PRM-Lite, independent verification, Tool Gateway and Policy Engine evidence from Rusaai.

CI validates code, configuration, container builds, and PowerShell syntax. It does not select a checkpoint, build the RusaAi-Edge model, or prove target-device performance.

## References

- RusaAi Unified CPU-First AI Architecture v2.0, especially sections 5, 6, 7, 11, 13, 16–20, 21 and 23.
- [llama.cpp server parameters and metrics](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md).
- [Windows IsProcessorFeaturePresent API](https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-isprocessorfeaturepresent).
