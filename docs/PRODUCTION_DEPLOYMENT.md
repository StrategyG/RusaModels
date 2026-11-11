# RusaModels Production Deployment

**Status:** production deployment candidate; not a production certification.

This topology deploys the Model Server, Quant Server, and Quant Worker against
managed PostgreSQL and S3-compatible object storage. It deliberately does not
start local PostgreSQL or MinIO. Do not use the development `docker-compose.yml`
as a production credential or availability model.

## 1. Required infrastructure

- Managed PostgreSQL with TLS certificates trusted by the containers.
- A versioned S3-compatible bucket with server-side encryption, lifecycle
  retention, and incomplete multipart-upload cleanup.
- A secret manager for database DSNs, API principal digests, service-token
  digest, S3 credentials, Hugging Face token, and upstream model keys.
- A reverse proxy that terminates TLS and forwards only to the loopback-bound
  Model Server and Quant Server ports.
- A persistent worker workspace with sufficient free disk for the largest
  source and converted model artifacts.
- Monitoring for API latency/errors, database connections, S3 failures, queue
  age, worker leases/retries, conversion duration, disk, memory, and cost.

The Model Server needs read-only artifact access. The Quant Server and Quant
Worker receive separate read and write S3 credentials because upload
finalization verifies the stored object by reading it back. Prefer workload
identity where supported. If static read credentials are configured, the
Quant Server and Worker must also receive a distinct write key pair. Never
reuse a write credential in the Model Server container.

## 2. Database identities

Use distinct database logins for:
- schema migration (DDL owner; used only by the one-shot migration command);
- Model Server (read model metadata and write audit events);
- Quant Server (only the model/version/upload/job operations it serves);
- Quant Worker (only the queue, lease, artifact metadata, and audit operations
  required to process jobs).

All staging/production DSNs must use `sslmode=verify-full`. Do not use a
PostgreSQL superuser or schema owner in an application container. Grant only
the required table/sequence privileges; keep audit triggers append-only.

## 3. API identities

Configure `RUSAMODELS_API_PRINCIPALS` as a JSON object keyed by lowercase
SHA-256 digests of raw bearer tokens. Give RusaAi a dedicated principal with
role `inference`; do not give an agent runtime Manager or Administrator
credentials. Keep the raw token only in RusaAi's secret manager.

Configure `RUSAMODELS_SERVICE_TOKEN_SHA256` separately for the internal
`X-Rusa-Service-Token` dependency-check route. It is not the OpenAI-compatible
inference token.

The Model Server validates its principal registry before serving traffic. If
an inference runtime is configured, startup requires at least one
`inference`-role principal. Production inference and embedding upstreams must
use HTTPS and authenticated service keys.

## 4. Prepare the host and secrets

Create the workspace for the non-root container user (UID/GID 1000):

```bash
sudo install -d -o 1000 -g 1000 -m 0750 /var/lib/rusamodels/workspace
sudo install -d -o 1000 -g 1000 -m 0750 /var/lib/rusamodels/workspace/tmp
```

Populate the deployment environment through the secret manager. The image
reference must be an immutable digest, for example
`ghcr.io/strategyg/rusamodels@sha256:<64-lowercase-hex>`. Do not use a tag
such as `latest`.

Required variables:

- `RUSAMODELS_IMAGE`
- `RUSAMODELS_MIGRATION_DATABASE_URL`
- `RUSAMODELS_MODEL_DATABASE_URL`
- `RUSAMODELS_QUANT_DATABASE_URL`
- `RUSAMODELS_WORKER_DATABASE_URL`
- `RUSAMODELS_S3_BUCKET`
- `RUSAMODELS_API_PRINCIPALS`
- `RUSAMODELS_SERVICE_TOKEN_SHA256`
- `RUSAMODELS_INFERENCE_BASE_URL`
- `RUSAMODELS_INFERENCE_MODEL_ID`
- `RUSAMODELS_INFERENCE_API_KEY`
- `RUSAMODELS_WORKSPACE_ROOT`

S3 read/write static key pairs, embedding settings, and `HF_TOKEN` are
optional according to the selected workload identity and enabled features.
If a custom S3 endpoint is configured, it must be credential-free HTTPS.

## 5. Validate and deploy

Run the migration explicitly with the DDL-only identity, then start the
services:

```bash
docker compose -f docker-compose.production.yml config --quiet
docker compose -f docker-compose.production.yml --profile migration run --rm migrate
docker compose -f docker-compose.production.yml pull
docker compose -f docker-compose.production.yml up -d
docker compose -f docker-compose.production.yml ps
```

The Compose file binds API ports to loopback only:
- Model Server: `127.0.0.1:8000`
- Quant Server: `127.0.0.1:8001`

Expose them only through an authenticated TLS reverse proxy and firewall.
The worker has no published port. The Compose file uses read-only root
filesystems, non-root execution from the image, dropped Linux capabilities,
no-new-privileges, resource limits, and bounded log rotation.

## 6. Acceptance and rollback

Before enabling production traffic:
1. Verify `/health/ready` through the TLS ingress for both servers.
2. Verify the RusaAi inference principal can call `/openai/v1/models` and
   one bounded chat/embedding request, and cannot access model-management or
   audit-administration routes.
3. Verify Model Server cannot write to the artifact bucket.
4. Verify Quant Server can complete an upload and checksum verification.
5. Verify Quant Worker can claim, heartbeat, cancel, and finish a test job.
6. Restore the database and artifact bucket from backup in an isolated
   environment; record measured RPO/RTO.
7. Exercise a previous immutable image digest and database rollback plan.
8. Obtain Security, SRE, QA, Data Governance, and process-owner approval.

This Compose file is not an HA topology and does not itself configure TLS,
backups, bucket versioning, IAM policies, network firewalls, branch protection,
image signing, SBOM/provenance verification, load tests, or disaster recovery.
Those controls require evidence from the target infrastructure before
production certification.
