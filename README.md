# RusaModels

Independent model infrastructure platform for the RusaAi ecosystem.

## Architecture

Standard RusaModels deployments expose two services: the Model Server and Quant Server. Rusaai remains the external Control Plane; RusaModels is the Model Plane. The native RUSA-CPU-16 Windows edge profile runs only the Model Server with local SQLite and local artifact storage:

- **Model Server** — model registry, lifecycle API and authenticated OpenAI-compatible inference gateway.
- **Quant Server** — model inspection, conversion and GGUF quantization service boundary.
- **Native Windows RUSA-CPU-16 v2.0** — architecture target for RusaAi-Edge-1.54B on Intel Core i5 / 16 GB RAM, llama.cpp and CPU-only inference. It is intentionally not deployable until a compatible non-Qwen checkpoint is selected, tokenizer/chat template are validated and a Q4_K_M GGUF artifact is built and checksummed.
- **RUSA-EDGE-8** — declarative Windows laptop / Android ARM64 profile for 8 GB physical RAM, 4K context, one active generation and CPU-only Q4_K_M inference. It remains a design target pending checkpoint selection and device benchmarks; see `docs/RUSA-EDGE-8.md`.
- **Codespaces CPU Runtime** — separate optional development profile with private llama.cpp and SentenceTransformers sidecars; it is not the Windows RUSA-CPU-16 baseline.
- **Shared RusaModels registry database** — PostgreSQL stores model metadata, versions, upload sessions and audit events for the Model Server and Quant Server in standard multi-service deployments.
- **Shared S3-compatible storage** — immutable model artifacts in standard multi-service deployments.
- **RUSA-CPU-16 edge storage** — local SQLite and local artifact files; no PostgreSQL or S3 dependency.

RusaAi is an authenticated service consumer. It does not connect directly to the RusaModels database or object storage.

## Implemented

- Separate FastAPI Model Server and Quant Server entry points plus an independent Quant Worker.
- Optional OpenAI-compatible chat, completion, embedding and model-list proxy endpoints on Model Server, with streaming support and a separately configured upstream runtime.
- PostgreSQL/SQLite adapters and Alembic migrations.
- S3-compatible storage for standard deployments and local artifact storage for development/edge deployments.
- Viewer, inference-only, Manager and Administrator roles, hashed bearer tokens and separate service-token authentication.
- Model catalog CRUD, version registration and administrator-only archival.
- Resumable multipart uploads, per-part checksums, object-level SHA-256 verification and streamed artifact downloads.
- Hugging Face Hub search and queued model download jobs with upstream license verification and explicit license acknowledgement.
- Queued F32/F16/BF16 safetensors conversion, GGUF F32/F16/BF16 conversion and Q8/Q6/Q5/Q4 GGUF quantization adapters.
- Independent chat and embedding upstream configuration, allowing local LLM and embedding runtimes to use separate CPU services.
- CPU-only Codespaces inference profile with private llama.cpp and local embedding containers, bounded CPU/RAM limits and no published runtime ports.
- Migrated agent SFT dataset validation, QLoRA training, held-out evaluation, adapter merge, GGUF checksum verification and inference benchmark tools.
- Verified model-version artifact download and pinned local embedding snapshot utilities.
- Immutable Hugging Face commit pinning, upstream license verification, and artifact manifests with SHA-256 digests.
- Separate Quant Worker with database leases, heartbeat renewal, stale-worker fencing, bounded retries for transient failures and cooperative cancellation.
- Container-aware RAM/disk preflight and cancellable conversion subprocesses outside API request processes.
- Append-only audit events enforced by database triggers.
- Docker Compose, separate Codespaces profiles and GitHub Actions CI.

## Not yet implemented

Dynamic model loading/deployment, multi-model routing, scheduling, automated training orchestration and dashboard. The standard Model Server profile still requires an explicitly configured upstream runtime; the native RUSA-CPU-16 v2.0 profile describes the fixed local llama.cpp target but deliberately blocks startup until the RusaAi-Edge non-Qwen checkpoint and GGUF artifact are approved and built. The optional CPU Codespaces profile supplies llama.cpp and embedding sidecars for development. The edge profile is not a dynamic model loader or multi-model router. The QLoRA training workflow is a separate CUDA-only candidate workflow and is not part of CPU inference. Safetensors conversion requires `pip install -e ".[quantization]"`. GGUF conversion requires a separately installed llama.cpp `convert_hf_to_gguf.py` and `llama-quantize`, configured with `LLAMA_CPP_CONVERT_SCRIPT` and `LLAMA_QUANTIZE_BIN`. The GGUF adapter allowlists Llama, Qwen2/Qwen3 and DeepSeek model types by default, with explicit configuration for extensions. Actual compatibility remains dependent on the installed upstream converter version. These features are not advertised as operational until implemented and tested.

## Local development

```bash
cp .env.example .env
docker compose up --build -d
```

Health endpoints:
- Model Server: `http://localhost:8000/health/live`
- Quant Server: `http://localhost:8001/health/live`

The Compose stack runs PostgreSQL and MinIO locally for development only. It is not the shared Codespaces control plane.

## Codespaces

Create a Model Server Codespace using `.devcontainer/model-server/devcontainer.json` or the CPU inference profile `.devcontainer/model-server-cpu/devcontainer.json`. Create a separate Quant Server Codespace using `.devcontainer/quant-server/devcontainer.json`. The CPU profile requests 4 CPUs, 16 GB RAM and starts private llama.cpp and local embedding services; keep GGUF and embedding model weights outside Git. Configure the shared RusaModels registry and artifact-store settings as documented in [docs/CODESPACES.md](docs/CODESPACES.md).

## Security

Never commit model weights, API tokens or credentials. Production requires PostgreSQL with `sslmode=verify-full` and HTTPS S3-compatible storage. Give Model Server read-only artifact permissions; give Quant Server and Quant Worker separate, narrowly scoped write credentials. Use a dedicated `inference` bearer principal for RusaAi. Apply schema changes with Alembic migrations.

## Documentation

- [Architecture](docs/architecture.md)
- [ADR-001](docs/ADR-001-shared-control-plane.md)
- [Codespaces setup](docs/CODESPACES.md)
- [Operations runbook](docs/OPERATIONS.md)
- [Production deployment](docs/PRODUCTION_DEPLOYMENT.md)
- [RusaAi integration](docs/RUSAAI-INTEGRATION.md)
- [RUSA-CPU-16 native Windows deployment](docs/RUSA-CPU-16-WINDOWS.md)
- [Codespaces CPU inference](docs/CPU_INFERENCE.md)
- [Training](docs/TRAINING.md)
- [Migration from RusaAi](docs/MIGRATION_FROM_RUSAAI.md)
