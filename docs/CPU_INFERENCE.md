# RusaModels CPU Inference on GitHub Codespaces

## Scope

This profile runs one local GGUF chat model and one local embedding model in a 4-vCPU / 16-GB Codespace. The Model Server API proxies requests to those private containers. No GPU and no external AI inference API are used.

The model files are not committed to Git. This profile is for development and validation; Codespaces can stop or expire and is not a permanent production host.

## Runtime topology

- `model-server`: authenticated RusaModels control plane and OpenAI-compatible API.
- `llama-cpp`: private GGUF chat inference, `--n-gpu-layers 0`.
- `embedding-server`: private SentenceTransformers CPU embedding API.
- Both runtimes are reachable only on the Compose network. Only Model Server port 8000 is forwarded.

## Prepare local model artifacts

### Chat model

Register/download/convert a compatible causal-LM checkpoint through the RusaModels model catalog and Quant Server. Choose a GGUF artifact such as Q5_K_M only after checking the actual file size and model compatibility.

Download a registered model version using a Manager/Administrator token:

```bash
export RUSAMODELS_API_TOKEN='...'
python scripts/download_model_version.py \
  --model-id MODEL_ID \
  --version-id VERSION_ID \
  --output-dir .local/models/rusa-general-4b
```

Find the resulting `.gguf` file and set its absolute path. Pin its checksum:

```sha
sha256sum .local/models/rusa-general-4b/*.gguf
```

```bash
export RUSAMODELS_GGUF_MODEL_PATH="$PWD/.local/models/rusa-general-4b/model-q5_k_m.gguf"
export RUSAMODELS_GGUF_MODEL_SHA256='<approved-sha256>'
python scripts/verify_gguf_model.py \
  --model "$RUSAMODELS_GGUF_MODEL_PATH" \
  --sha256 "$RUSAMODELS_GGUF_MODEL_SHA256"
```

The filename is an example; use the exact artifact filename downloaded from the registry.

### Embedding model

The default CPU profile uses `intfloat/multilingual-e5-small`, with 384-dimensional vectors. Review the upstream model card/license and pin an immutable 40-character revision before downloading:

```bash
python scripts/download_embedding_snapshot.py \
  --repo-id intfloat/multilingual-e5-small \
  --revision '<40-character-commit-sha>' \
  --output-dir .local/models/multilingual-e5-small \
  --license-accepted
```

The snapshot downloader requires a pinned revision and explicit license acknowledgement. Keep the model directory outside Git.

### Codespaces secrets/environment

Configure the CPU Model Server Codespace with:
- `RUSAMODELS_GGUF_MODEL_PATH`: absolute path to the verified GGUF file.
- `RUSAMODELS_EMBEDDING_MODEL_PATH`: absolute path to the local embedding model directory.
- `RUSAMODELS_EMBEDDING_INTERNAL_TOKEN`: random secret of at least 32 characters. The Model Server and embedding container receive the same value.
- Existing Model Server secrets: database URL, S3 endpoint/bucket, API principal digests, service-token digest and read-only S3 credentials.

Use `.devcontainer/model-server-cpu/devcontainer.json`. It requests 4 CPUs, 16 GB RAM and 64 GB storage. The standard Model Server and Quant Server Codespaces profiles remain available.

## Start and verify

The CPU Codespaces profile starts the model-server, llama.cpp and embedding-server containers. The GGUF model is mounted read-only; embedding weights are also mounted read-only.

Check the Model Server:
```bash
curl --fail-with-body http://127.0.0.1:8000/health/ready
```

List configured models with a Manager/Administrator API token:
```bash
curl --fail-with-body http://127.0.0.1:8000/openai/v1/models \
  -H "Authorization: Bearer $RUSAMODELS_API_TOKEN"
```

Chat inference:
```bash
curl --fail-with-body http://127.0.0.1:8000/openai/v1/chat/completions \
  -H "Authorization: Bearer $RUSAMODELS_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"model":"rusa-general-4b","messages":[{"role":"user","content":"Explain RUSA AI briefly."}],"temperature":0,"max_tokens":128}'
```

Embedding inference:
```bash
curl --fail-with-body http://127.0.0.1:8000/openai/v1/embeddings \
  -H "Authorization: Bearer $RUSAMODELS_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"model":"rusa-embedding-small","input":["RUSA autonomous agent architecture"],"input_type":"passage"}'
```

For multilingual-e5, use `input_type=passage` for document chunks and `input_type=query` for retrieval queries. The embedding API returns 384-dimensional normalized vectors.

## Resource controls

Initial limits:
- GGUF model container: up to 8 GB memory, 3 CPU quota, 4K context, one parallel slot.
- Embedding container: up to 2 GB memory, one CPU quota, one Torch thread.
- Model Server API: existing Codespaces container and API limits.
- No model is loaded inside RUSA AI workers.

These are starting limits, not measured guarantees. Benchmark the exact model, prompt lengths, context, Codespaces CPU and memory before increasing limits. Monitor cgroup memory, CPU, latency and restarts.

## RUSA AI integration

Configure RUSA AI's OpenAI-compatible model client with:
- Base URL: `https://<RusaModels-Codespace-forwarded-host>/openai/v1`
- API key: a dedicated Manager/Administrator token for the RusaModels Model Server.
- Chat model: `rusa-general-4b`.

Configure its embedding client with the same Model Server base URL and model `rusa-embedding-small`, dimensions `384`, and a dedicated API token. The RUSA AI Knowledge/RAG schema must use the same vector dimension; changing from a previous 1536-dimensional embedding requires a controlled database migration and re-embedding existing documents.

Codespaces forwarded ports used by Colab must be reachable from that runtime. If using public port visibility, keep strong bearer authentication, HTTPS, rate limits and short-lived credentials; never expose the llama.cpp or embedding ports directly. A private Codespaces port is not directly reachable from an unauthenticated Colab runtime.

## Limitations

- Codespaces is an ephemeral development environment, not an always-on production server.
- The Model Server readiness endpoint checks the control plane; perform actual chat and embedding smoke tests to confirm both runtimes have loaded.
- CPU QLoRA training is not part of this profile. The migrated SFT pipeline is an optional CUDA training workflow.
- No external AI inference fallback is configured.
