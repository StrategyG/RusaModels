# Two Codespaces Setup

Create two Codespaces from this repository:
- Model Server: select `.devcontainer/model-server/devcontainer.json`.
- Quant Server: select `.devcontainer/quant-server/devcontainer.json`.

Each profile requests 4 CPUs, 16 GB memory and 32 GB storage. Choose the corresponding 4-core / 16-GB Codespaces machine type when prompted.

Configure identical repository Codespaces secrets for both:
- DATABASE_URL — shared PostgreSQL URL
- S3_ENDPOINT_URL — shared S3-compatible endpoint
- S3_REGION
- S3_BUCKET
- S3_ACCESS_KEY_ID
- S3_SECRET_ACCESS_KEY
- RUSAMODELS_API_PRINCIPALS — JSON map of SHA-256 token digests to subject/role
- RUSAMODELS_SERVICE_TOKEN_SHA256 — digest of the service token

Do not use per-Codespace local databases or object stores. Run `alembic upgrade head` once against the shared database before checking readiness.

Generate a token digest:
```bash
python -c 'import getpass,hashlib; print(hashlib.sha256(getpass.getpass("API token: ").encode()).hexdigest())'
```

Example principal JSON:
```json
{"<64-character-sha256-digest>":{"subject":"platform-admin","role":"administrator"}}
```
Store raw tokens only in the secret manager.
