# RusaModels

Independent model infrastructure platform for the RUSA-AI ecosystem.

## Architecture

RusaModels is deployed as two independent services backed by one shared control plane:

- **Model Server** — model registry and lifecycle API; future inference gateway and router.
- **Quant Server** — model inspection, conversion and quantization service boundary.
- **Shared PostgreSQL** — registry metadata, model versions, upload sessions and audit events.
- **Shared S3-compatible storage** — immutable model artifacts.

RusaAi is an authenticated service consumer. It does not connect directly to the RusaModels database or object storage.

## Implemented

- Separate FastAPI Model Server and Quant Server entry points plus an independent Quant Worker.
- PostgreSQL/SQLite adapters and Alembic migrations.
- S3-compatible and development-only local artifact storage.
- Viewer, Manager and Administrator roles, hashed bearer tokens and service-token authentication.
- Model catalog CRUD, version registration and administrator-only archival.
- Resumable multipart uploads, per-part checksums, object-level SHA-256 verification and streamed artifact downloads.
- Hugging Face Hub search and queued model download jobs with upstream license verification and explicit license acknowledgement.
- Queued F32/F16/BF16 safetensors conversion, GGUF F32/F16/BF16 conversion and Q8/Q6/Q5/Q4 GGUF quantization adapters.
- Conservative CPU/RAM/disk preflight and a separate Quant Worker that runs jobs outside API request processes.
- Append-only audit events enforced by database triggers.
- Docker Compose, separate Codespaces profiles and GitHub Actions CI.

## Not yet implemented

Model inference, OpenAI-compatible endpoints, model routing, scheduling, benchmark/evaluation, training orchestration and dashboard. Safetensors conversion requires `pip install -e ".[quantization]"`. GGUF conversion requires a separately installed llama.cpp `convert_hf_to_gguf.py` and `llama-quantize`, configured with `LLAMA_CPP_CONVERT_SCRIPT` and `LLAMA_QUANTIZE_BIN`. The GGUF adapter allowlists Llama, Qwen2/Qwen3 and DeepSeek model types by default, with explicit configuration for extensions. Actual compatibility remains dependent on the installed upstream converter version. These features are not advertised as operational until implemented and tested.

## Local development

```bash
cp .env.example .env
docker compose up --build -d
```

Health endpoints:
- Model Server: `http://localhost:8000/health/live`
- Quant Server: `http://localhost:8001/health/live`

The Compose stack runs PostgreSQL and MinIO locally for development only. It is not the shared Codespaces control plane.

## Two Codespaces

Create one Codespace using `.devcontainer/model-server/devcontainer.json` and another using `.devcontainer/quant-server/devcontainer.json`. Configure both with the same external `DATABASE_URL`, `S3_ENDPOINT_URL`, `S3_BUCKET` and S3 credentials. See [docs/CODESPACES.md](docs/CODESPACES.md).

## Security

Never commit model weights, API tokens or credentials. Production requires PostgreSQL and S3-compatible storage. Give Model Server read-only artifact permissions and Quant Server narrowly scoped write permissions. Apply schema changes with Alembic migrations.

## Documentation

- [Architecture](docs/architecture.md)
- [ADR-001](docs/ADR-001-shared-control-plane.md)
- [Codespaces setup](docs/CODESPACES.md)
- [Operations runbook](docs/OPERATIONS.md)
- [RusaAi integration](docs/RUSAAI-INTEGRATION.md)
