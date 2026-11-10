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

## Quantization worker

The Quant Server Codespace runs the API and the separate `services.quant_server.worker` process. Local Compose also starts `quant-worker`. The worker claims queued jobs from the shared PostgreSQL queue using a 30-second lease and a monotonically increasing lease epoch. A heartbeat renews the lease every 10 seconds. If a worker disappears, another worker can reclaim an expired lease; stale workers cannot commit results after ownership changes. Transient network and object-store failures are retried with exponential backoff and jitter, up to three attempts. Permanent validation and license failures are terminal. The worker downloads approved Hugging Face snapshots pinned to an immutable upstream commit or converts registered versions, verifies artifact checksums and registers immutable output artifacts.

Hugging Face downloads require an upstream model-card license identifier and explicit `license_accepted=true`. The worker compares the supplied license identifier to the upstream model card before downloading. Gated repositories additionally require an authorized `HF_TOKEN` secret.

Safetensors precision conversion requires the optional `quantization` dependencies; install them in the Quant Codespace with `python -m pip install --user -e ".[dev,quantization]"`. GGUF conversion additionally requires `LLAMA_CPP_CONVERT_SCRIPT` and `LLAMA_QUANTIZE_BIN` pointing to an installed llama.cpp converter and quantizer. The worker uses a temporary workspace, a separate subprocess for conversion commands, no shell invocation, a sanitized subprocess environment, a hard timeout and a conservative RAM/disk preflight. This is process isolation, not a separate virtual machine or network sandbox. A 4-vCPU/16-GB Codespace will reject jobs estimated to exceed available resources.

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
Model CRUD, immutable model-version registration, resumable artifact uploads, checksum verification, artifact download, Hugging Face discovery/download, worker leases/retries, safetensors conversion and GGUF conversion/quantization adapters are implemented. Inference, routing, scheduling, benchmark/evaluation, training orchestration and dashboard remain future phases. Actual conversion requires the optional dependencies and external llama.cpp tools documented above.
