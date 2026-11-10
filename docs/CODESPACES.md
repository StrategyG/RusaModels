# Two Codespaces Setup

Create two Codespaces from this repository:
- Model Server: select `.devcontainer/model-server/devcontainer.json`.
- Quant Server: select `.devcontainer/quant-server/devcontainer.json`.

Each profile requests 4 CPUs, 16 GB memory and 32 GB storage. Choose the corresponding 4-core / 16-GB Codespaces machine type when prompted.

## Shared Codespaces secrets

Configure these secrets for both Codespaces:
- `DATABASE_URL` — shared PostgreSQL URL
- `S3_ENDPOINT_URL` — shared S3-compatible endpoint
- `S3_REGION`
- `S3_BUCKET`
- `RUSAMODELS_API_PRINCIPALS` — JSON map of SHA-256 token digests to subject/role
- `RUSAMODELS_SERVICE_TOKEN_SHA256` — digest of the service token

## Model Server-only secrets

- `S3_READ_ACCESS_KEY_ID` and `S3_READ_SECRET_ACCESS_KEY` — read-only artifact permissions
- `RUSAMODELS_INFERENCE_BASE_URL` — optional reachable OpenAI-compatible runtime base URL, without the `/v1` suffix
- `RUSAMODELS_INFERENCE_MODEL_ID` — public model ID exposed by the proxy
- `RUSAMODELS_INFERENCE_API_KEY` — optional upstream bearer key in development; required with HTTPS in production
- `RUSAMODELS_INFERENCE_TIMEOUT_SECONDS` — optional timeout, default 300

## Quant Server-only secrets and configuration

- `S3_WRITE_ACCESS_KEY_ID` and `S3_WRITE_SECRET_ACCESS_KEY` — write-limited artifact permissions
- `HF_TOKEN` — optional Hugging Face token for gated/private model repositories
- `LLAMA_CPP_CONVERT_SCRIPT` — optional absolute path to llama.cpp conversion script in Quant Codespace
- `LLAMA_QUANTIZE_BIN` — optional llama-quantize binary path

Do not give Model Server write access to the artifact bucket. Do not give Quant Server the inference upstream key. Do not use per-Codespace local databases or object stores. Run `alembic upgrade head` once against the shared database before checking readiness.

Generate a token digest:
```bash
python -c 'import getpass,hashlib; print(hashlib.sha256(getpass.getpass("API token: ").encode()).hexdigest())'
```

Example principal JSON:
```json
{"<64-character-sha256-digest>":{"subject":"platform-admin","role":"administrator"}}
```

Store raw tokens only in the secret manager.
