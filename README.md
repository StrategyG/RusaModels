# RusaModels

Independent model infrastructure platform for the RUSA-AI ecosystem.

## Architecture

RusaModels is deployed as two independent services backed by one shared control plane:

- **Model Server** — model registry and lifecycle API; future inference gateway and router.
- **Quant Server** — model inspection, conversion and quantization API boundary.
- **Shared PostgreSQL** — registry metadata, model versions and audit events.
- **Shared S3-compatible storage** — immutable model artifacts.

RusaAi is an authenticated service consumer. It does not connect directly to the RusaModels database or object storage.

## Current implementation status

The repository currently contains:
- Separate FastAPI Model Server and Quant Server entry points.
- PostgreSQL/SQLite database adapters and an Alembic registry migration.
- S3-compatible and development-only local artifact storage adapters.
- Viewer, Manager and Administrator API roles.
- Hashed bearer-token authentication and a separate service-token boundary.
- Model catalog CRUD, model version registration and administrator-only archival.
- Append-only audit events enforced by database triggers.
- Docker Compose and separate 4-vCPU/16-GB Codespaces profiles.
- Initial automated tests and GitHub Actions CI.

**Not yet implemented:** model downloading, multipart artifact upload/download, conversion, GGUF quantization, Q8/Q6/Q5/Q4, model inference, OpenAI-compatible endpoints, routing, scheduling, benchmark/evaluation, training orchestration and dashboard. These must not be considered available until implemented and tested.

## Quick start

For a local development stack:

```bash
cp .env.example .env
docker compose up --build -d
docker compose run --rm migrate
```

Health endpoints:
- Model Server: `http://localhost:8000/health/live`
- Quant Server: `http://localhost:8001/health/live`

The Compose stack runs PostgreSQL and MinIO locally for development only. It is not the shared Codespaces control plane.

## Two Codespaces

Create one Codespace using `.devcontainer/model-server/devcontainer.json` and another using `.devcontainer/quant-server/devcontainer.json`. Configure both with the same external `DATABASE_URL`, `S3_ENDPOINT_URL`, `S3_BUCKET` and S3 credentials. See [docs/CODESPACES.md](docs/CODESPACES.md).

## Security

- Never commit model weights, API tokens or credentials.
- Configure API principals as SHA-256 token digests mapped to a subject and role.
- Production requires PostgreSQL and S3-compatible storage.
- Give Model Server read-only artifact permissions and Quant Server narrowly scoped write permissions.
- Apply schema changes with Alembic migrations, not application startup.
- See [docs/OPERATIONS.md](docs/OPERATIONS.md).

## Documentation

- [Architecture](docs/architecture.md)
- [ADR-001](docs/ADR-001-shared-control-plane.md)
- [Codespaces setup](docs/CODESPACES.md)
- [Operations runbook](docs/OPERATIONS.md)
- [RusaAi integration](docs/RUSAAI-INTEGRATION.md)
