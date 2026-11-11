# RUSA-CPU-16 — Native Windows 11 Deployment

**Profile:** rusa-cpu-16  
**Version:** 1.0.0  
**Status:** locked baseline; hardware acceptance is still required  
**Platform:** Windows 11 64-bit, Intel Core i5, nominal 16 GB RAM, CPU-only

This is the native local/edge deployment baseline. It is deliberately separate from the optional Docker/Codespaces CPU profile in CPU_INFERENCE.md. The Codespaces profile uses different context, quantization, resource limits and an optional embedding sidecar; do not copy those values into this baseline.

## 1. Locked runtime profile

| Component | Baseline |
|---|---|
| Model family | Qwen3-4B Instruct |
| Model repository | Qwen/Qwen3-4B-GGUF |
| Artifact | GGUF Q4_K_M |
| Runtime | llama.cpp llama-server.exe |
| Context | 8,192 tokens |
| Generation threads | 4 initially; reduce to detected physical core count when below 4 |
| Batch threads | 4 initially; same hardware-aware adjustment |
| HTTP threads | 2 |
| Concurrent generations | 1 |
| Pending inference queue | 3; owned by the Rusaai scheduler |
| KV cache | Q8_0 for K and V when supported |
| Batch / micro-batch | 512 / 128 |
| GPU layers | 0 |
| Model runtime | 127.0.0.1:8081 |
| RusaModels API | 127.0.0.1:8090 |
| Cloud inference fallback | Disabled |
| Separate embedding process | Disabled in v1.0 |

The exact Core i5 SKU is not present in the source baseline. Startup detects physical and logical cores and starts at min(4, physical cores). Do not raise thread count, context or concurrency without hardware-specific benchmark and acceptance evidence.

The matching machine-readable profile is configs/runtime/rusa-cpu-16.yaml.

## 2. Resource budget

The source configuration is a planning budget, not a measured benchmark:

| Component | Planned RAM |
|---|---:|
| Windows 11 and base applications | 5.0 GB |
| Qwen3-4B Q4_K_M weights | 2.5 GB |
| KV cache | 0.6 GB |
| Inference buffers | 1.2 GB |
| Rusaai services | 1.2 GB |
| RusaModels API, database and telemetry | 0.5 GB |
| Reserved free memory | 2.0 GB |
| Planned total | 13.0 GB |
| Flexible headroom | 3.0 GB |

The launcher refuses to start when detected physical RAM is below the nominal 16-GB class or free RAM is below 2 GB. These checks do not prove that the full workload will remain within budget. Measure peak RAM while the model is loaded and during the acceptance run.

Resource protection thresholds:
- CPU below 70%: normal operation.
- CPU 70–85%: reduce background work.
- CPU above 85% for a sustained period: defer non-priority tasks.
- RAM above 75%: do not load another model.
- RAM above 85%: reduce context and additional tasks.
- Free RAM below 2 GB: activate memory-protection mode.

## 3. Install prerequisites

1. Install Python 3.11 or newer.
2. Install llama.cpp using the approved Windows distribution. The baseline source lists winget install llama.cpp; verify that llama-server.exe is available.
3. Clone this repository and create a virtual environment:

    py -3.12 -m venv .venv
    .\.venv\Scripts\Activate.ps1
    python -m pip install --upgrade pip
    python -m pip install -e ".[dev]"

4. Prepare a local model directory outside Git. Download the official Qwen3-4B GGUF Q4_K_M artifact, record the upstream immutable revision and exact filename, and calculate SHA-256:

    Get-FileHash "C:\RusaModels\models\qwen3-4b-q4_k_m.gguf" -Algorithm SHA256

Do not assume a filename or revision. Use the exact artifact downloaded and approved. Keep model weights outside the repository; .local/ is ignored by Git.

## 4. Configure the edge service boundary

Copy the example and edit the local paths:

    Copy-Item .env.rusa-cpu-16.example .env
    notepad .env

The profile uses:
- RUSAMODELS_ENV=edge
- SQLite registry database and local artifact storage.
- One loopback-only llama.cpp upstream.
- One dedicated Rusaai inference principal.
- A separate internal service-token digest.
- No S3 credentials, embedding sidecar or external inference fallback.

Generate two independent high-entropy tokens in PowerShell. Keep the raw values in the Rusaai secret store; only their SHA-256 digests belong in the RusaModels .env file:

    $inferenceToken = python -c "import secrets; print(secrets.token_urlsafe(32))"
    $inferenceDigest = python -c "import hashlib,sys; print(hashlib.sha256(sys.argv[1].encode()).hexdigest())" $inferenceToken
    $serviceToken = python -c "import secrets; print(secrets.token_urlsafe(32))"
    $serviceDigest = python -c "import hashlib,sys; print(hashlib.sha256(sys.argv[1].encode()).hexdigest())" $serviceToken
    $principalRegistry = @{}
    $principalRegistry[$inferenceDigest] = @{ subject = "rusa-ai-inference"; role = "inference" }
    $env:RUSAMODELS_API_PRINCIPALS = ConvertTo-Json -InputObject $principalRegistry -Compress
    $env:RUSAMODELS_SERVICE_TOKEN_SHA256 = $serviceDigest
    $env:RUSAMODELS_API_TOKEN = $inferenceToken
    $env:RUSAMODELS_SERVICE_TOKEN = $serviceToken

Keep these variables available to the service process. Configure Rusaai with:
- OpenAI-compatible base URL: http://127.0.0.1:8090/openai/v1
- Model ID: rusa-cpu-4b
- Bearer token: the raw inference token
- Internal dependency-check header: X-Rusa-Service-Token with the raw service token

The inference token has role inference; it cannot manage models or read administrative audit data. The internal service token is a different credential. Do not commit either raw token or the populated .env.

## 5. Start and stop

From the repository root, in the PowerShell session containing the required environment variables:

    .\scripts\start_rusa_cpu_16.ps1

Startup:
1. Loads non-secret settings from .env without overriding current process environment variables.
2. Requires the edge environment, SQLite, local artifact storage, loopback inference URL and inference-role principal.
3. Detects CPU physical/logical cores, RAM and available memory.
4. Verifies the GGUF SHA-256 before loading it.
5. Applies Alembic migrations.
6. Starts llama.cpp bound to 127.0.0.1:8081 with the locked 8K/CPU-only parameters.
7. Starts the RusaModels API bound to 127.0.0.1:8090.
8. Waits for llama.cpp and API readiness.

Logs and process records are written under ignored .local/logs/ and .local/rusa-cpu-16/.

Stop both services with:

    .\scripts\stop_rusa_cpu_16.ps1

The API and llama.cpp ports must remain loopback-only. Do not publish either port to the public internet. If Rusaai runs on another host, this local profile is not directly reachable; use an explicitly reviewed private network design rather than opening the ports.

## 6. Smoke and sequential acceptance

Set RUSAMODELS_API_TOKEN in the current PowerShell session to the raw inference token, then run:

    .\scripts\verify_rusa_cpu_16.ps1 -SequentialRequests 20

The script verifies API readiness, confirms the model ID is exposed to the inference principal, sends up to 20 sequential bounded chat requests and checks free RAM before and after. This is a sequential smoke test, not a full performance benchmark; it does not prove context-window, queue, thermal or long-duration stability.

## 7. Reasoning and orchestration ownership

Rusaai owns the scheduler, context manager, reasoning selection, ORM, PRM-Lite, Tool Gateway and Policy Engine. RusaModels serves the configured model and does not execute tools or decide workflow permissions.

| Profile | Mode | Output budget | Candidates |
|---|---|---:|---:|
| LOW | No-think / light reasoning | 384 tokens | 1 |
| MEDIUM | Think | 768 tokens | 1 |
| HIGH | Think + limited search | 1,536 tokens | 2, sequentially |

Initial sampling values from the baseline:
- Thinking: temperature 0.6, top-p 0.95, top-k 20, min-p 0, presence penalty 1.5.
- Non-thinking: temperature 0.7, top-p 0.8.

Apply and validate these request parameters in Rusaai against the selected llama.cpp/model version. They are initial tuning values, not measured optimal settings. Keep one active generation; the three pending requests are queued by Rusaai, not by separate model processes.

The initial memory design is SQLite + FTS5 in Rusaai. Do not load a separate embedding model permanently in this 16-GB baseline. Add vector search only after retrieval-quality and RAM testing. Tool calls must pass through Rusaai Tool Gateway and deterministic Policy Engine; the model itself never authorizes or executes tools.

## 8. Acceptance gate before calling the device production-ready

- [ ] Exact Windows build, Intel Core i5 SKU, physical cores and logical processors recorded.
- [ ] Approved Qwen3-4B GGUF Q4_K_M revision, file and SHA-256 recorded.
- [ ] llama.cpp reports CPU-only execution and the model is resident.
- [ ] 8,192-token context tested with representative RUSA prompts.
- [ ] At least 20 sequential requests complete without crash.
- [ ] One active generation and a maximum of three pending requests verified in Rusaai.
- [ ] LOW, MEDIUM and HIGH reasoning profiles tested.
- [ ] Free RAM remains at least 2 GB during inference.
- [ ] ORM, PRM-Lite and independent verification return evidence.
- [ ] Rusaai calls RusaModels through the authenticated API contract.
- [ ] Tool execution is only through Tool Gateway and Policy Engine.
- [ ] Both runtime ports are bound to localhost only.
- [ ] Restart and recovery after process failure tested.
- [ ] TTFT, tokens/second, total latency, peak RAM, minimum free RAM, CPU use and queue wait time recorded.

Do not set a tokens/second target until the exact CPU SKU and runtime build have been benchmarked. CI validates code, configuration and PowerShell syntax; it does not run the model on the target Windows machine. Production certification still requires the device acceptance evidence above.
