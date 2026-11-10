# Operations Runbook

## Local development
1. Copy `.env.example` to `.env` and replace development passwords.
2. Start the stack with `docker compose up --build -d`. The migration service applies Alembic migrations before the API services start.
3. Check Model Server on port 8000 and Quant Server on port 8001.
4. Apply later schema changes with `alembic upgrade head`; do not use ORM `create_all` outside isolated tests.

## Resumable artifact uploads
1. Create an upload session with the model/version endpoint and expected byte size plus SHA-256.
2. Upload numbered raw binary parts to `PUT /v1/uploads/{upload_id}/parts/{part_number}`. The maximum part size is 64 MiB.
3. Re-uploading a part number replaces that part, allowing interrupted transfers to resume.
4. Complete the upload. The server checks contiguous part numbers, expected size and the full SHA-256 by streaming the stored object back.
5. Download immutable artifacts from `GET /v1/artifacts/{artifact_id}/download`.

For S3, non-final multipart parts must be at least 5 MiB. The default configured maximum artifact size is 10 GiB. Use TLS for all non-local transfer paths. Abandoned multipart sessions need an operational cleanup policy; configure an S3 lifecycle rule to abort incomplete multipart uploads.

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
Model CRUD, immutable model-version registration, resumable artifact uploads, checksum verification and artifact download are implemented. Model discovery/download, conversion, quantization, inference, routing, scheduling, benchmark/evaluation, training orchestration and dashboard remain future phases.
