# Operations Runbook

## Local development
1. Copy `.env.example` to `.env` and replace development passwords.
2. Start `docker compose up --build -d`.
3. Run health checks on ports 8000 and 8001.
4. Apply schema changes only with `alembic upgrade head`.

## Health
- `/health/live`: process is responsive.
- `/health/ready`: database, migrated tables and object storage are available.

## Production
- Set RUSAMODELS_ENV=production.
- Use PostgreSQL and S3-compatible object storage.
- Supply token digests through a secret manager.
- Use TLS, backups, point-in-time recovery, bucket versioning/retention and monitoring.
- Give Model Server read-only storage access and Quant Server scoped write access.
- Pin reviewed container image digests.
- Do not expose PostgreSQL or object-store administration publicly.

## Current scope
Model CRUD and version metadata APIs are available in the initial implementation. Artifact upload/download, resumable multipart transfer, conversion, quantization, inference, routing, scheduling and dashboard are later phases. No model weights are included.
