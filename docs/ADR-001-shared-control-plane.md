# ADR-001: Independent services and shared control plane

- Status: Accepted
- Date: 10 November 2026

> **Terminology clarification:** “shared control plane” in this ADR means the RusaModels-internal PostgreSQL registry/control store shared by its Model Server and Quant Server. It is not the RUSA AI ecosystem Control Plane, which is owned by RusaAi. RusaModels remains the Model Plane.

## Context
Model lifecycle and inference are separated from conversion and quantization. Two Codespaces must share a single model registry and artifact store.

## Decision
1. Keep Model Server and Quant Server as independent FastAPI entry points and Codespaces profiles.
2. Use one shared PostgreSQL registry store for RusaModels Model Server and Quant Server, plus one S3-compatible artifact store.
3. Restrict SQLite and local files to development and tests.
4. Use Alembic migrations.
5. Enforce Viewer, Manager and Administrator roles in the APIs.
6. Use a separate service token for RusaAi integration.
7. Make audit events append-only.
8. Keep weights and secrets out of Git.
9. Use distinct production object-store identities: read-only Model Server and write-limited Quant Server.

## Consequences
Both Codespaces require identical shared endpoint settings. Database and object-store availability are shared dependencies. Conversion and quantization are not assumed to fit every model on 4-vCPU/16-GB CPU instances.
