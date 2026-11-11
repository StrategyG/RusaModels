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
- An API gateway/reverse proxy that terminates TLS, forwards only to the
  loopback-bound Model Server and Quant Server ports, and enforces a WAF,
  per-principal/IP rate limits, request-body limits, connection/concurrency
  limits, and upstream timeouts. A per-IP Nginx reference configuration is
  provided at `deploy/nginx/rusamodels.conf.example`; per-principal quotas
  still require an API gateway or shared rate-limit service.
- Network egress controls for the Quant Worker. Restrict downloads to approved
  model sources and do not allow arbitrary egress from conversion subprocesses.
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


## 4. Model supply chain and conversion isolation

Hugging Face downloads are resolved to an immutable commit SHA and checked
against the declared upstream license. Conversion accepts Safetensors weights
only; pickle-based `.bin` and `.pt` checkpoints are rejected. The converter
uses `trust_remote_code=False` and local-only model loading.

Safetensors reduces deserialization risk but does not make model files
trustworthy. Model weights and tokenizer metadata remain untrusted input.
Before production use, run conversion in an ephemeral, isolated job sandbox
(for example, a dedicated one-shot container with a read-only root filesystem,
no database credentials, no long-lived object-store credentials, restricted
network egress, bounded CPU/memory/disk/PIDs, and only scoped input/output
capabilities). The current Compose worker is a long-running service with
database and object-store access; container hardening alone is not a per-job
sandbox. Treat this isolation as a production launch gate.


## 5. Prepare the host and secrets

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
- `RUSAMODELS_WORKER_IMAGE` (built with the `quantization` extra)
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

## 6. Validate and deploy

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

Expose the public inference surface only through an authenticated TLS reverse
proxy and firewall. Keep Quant Server on a management/VPN ingress; do not
publish its management API to the public internet. The Nginx example applies
per-IP request and connection limits, a bounded inference request size, and
an allowlist for the management API. Replace its example hostnames, TLS paths,
and VPN CIDRs before use. It does not provide per-principal billing quotas.
The worker image is separate because it includes the optional `quantization`
dependencies; keep the larger ML dependency set out of the Model Server image.
GGUF conversion additionally requires the compatible llama.cpp converter and
quantizer to be installed in that worker image at the configured paths. The
worker has no published port. The Compose file uses read-only root
filesystems, non-root execution from the image, dropped Linux capabilities,
no-new-privileges, resource limits, and bounded log rotation.

## 7. Acceptance and rollback

Before enabling production traffic:
1. Verify `/health/ready` through the TLS ingress for both servers. Model
   Server readiness checks the configured upstream model catalog and confirms
   that the configured chat and embedding model IDs are actually exposed.
2. Verify the RusaAi inference principal can call `/openai/v1/models` and
   one bounded chat/embedding request, and cannot access model-management or
   audit-administration routes.
3. Verify ingress rate limits, WAF policy, request-body caps, concurrency
   limits, and timeout behavior under load.
4. Verify Model Server cannot write to the artifact bucket.
5. Verify Quant Server can complete an upload and checksum verification.
6. Verify the isolated conversion sandbox cannot read application secrets,
   access the database, or make unrestricted outbound network requests.
7. Verify Quant Worker can claim, heartbeat, cancel, and finish a test job.
8. Restore the database and artifact bucket from backup in an isolated
   environment; record measured RPO/RTO.
9. Exercise a previous immutable image digest and database rollback plan.
10. Verify signed image provenance and attach an SBOM to each released image.
11. Obtain Security, SRE, QA, Data Governance, and process-owner approval.

This Compose file is not an HA topology and does not itself configure TLS,
backups, bucket versioning, IAM policies, network firewalls, branch protection,
image signing, SBOM/provenance verification, load tests, or disaster recovery.
Those controls require evidence from the target infrastructure before
production certification.


## 8. GitHub governance gate

The repository's current `main` branch is not protected and has no repository
ruleset. Configure this in GitHub repository settings before production:

- Require pull requests before merging to `main`.
- Require at least one independent approval and require CODEOWNERS review.
- Dismiss stale approvals when new commits are pushed.
- Require the `test` status check from the RusaModels CI workflow.
- Require branches to be up to date before merging.
- Block force pushes and branch deletion.
- Enable secret scanning, push protection, and Dependabot security updates
  where available.

The GitHub connection used for this engineering pass can inspect repository
settings but cannot change branch-protection/ruleset administration. A
repository administrator must apply and verify these settings.

## 9. Release security gates and known exceptions

- Build images from reviewed commits and publish immutable digests only.
- Scan both Model Server and Quant Worker images. Fix all fixable CRITICAL and
  HIGH findings before release; document any remaining unfixable findings
  with package, CVE, exposure analysis, owner, and review date. Do not suppress
  findings solely to obtain a green workflow.
- Publish an SBOM and build provenance for each image and verify the signature
  before deployment. The stable-semver-tag workflow
  `.github/workflows/publish-images.yml` accepts only a release commit
  reachable from `main`, builds both images, publishes SBOM and provenance
  attestations, and signs their immutable digests.
- Pin Python dependencies to security-supported versions and run
  `pip check` in the final image.
- Keep training adapters experimental until the full held-out evaluation,
  safety regression, and human promotion gates pass. Fine-tuning is not a
  prerequisite for the base inference service.
