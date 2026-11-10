# RusaAi Integration

RusaModels is a separate service boundary. RusaAi should call the service API and must not receive database credentials or object-store credentials.

RusaModels now exposes authenticated model registry, model-version, artifact download and job-status APIs. The Quant Server accepts queued Hugging Face downloads and conversion jobs; the Quant Worker executes them with leases, heartbeat renewal, bounded retries and cooperative cancellation. The Model Server exposes Hub search and read-only artifact access.

- `GET /v1/jobs/{job_id}` — inspect job state and retry/cancellation metadata.
- `POST /v1/jobs/{job_id}/cancel` — administrator-only idempotent cancellation; queued jobs cancel immediately and running jobs receive a cooperative cancellation request.
- GET /internal/v1/platform/dependencies
- Header: X-Rusa-Service-Token
- The token is verified against RUSAMODELS_SERVICE_TOKEN_SHA256.

The Model Server optionally exposes an OpenAI-compatible proxy under `/openai/v1` when `RUSAMODELS_INFERENCE_BASE_URL` and `RUSAMODELS_INFERENCE_MODEL_ID` are configured. It supports model listing, chat completions, completions, embeddings and streaming. The dedicated `inference` role, Manager, and Administrator roles can invoke it. For RusaAi runtime traffic, issue a dedicated principal with role `inference`; its only permission is `inference:invoke`. It cannot create or modify models, run quantization jobs, archive models, or read administrative audit records. This is a single-upstream proxy: dynamic model loading, routing, tool-call compatibility verification, rate limiting and usage metering remain future work. A dependency check does not prove that a model is loaded or compatible with tool calling.


## OpenAI SDK example

Configure the Model Server with a reachable OpenAI-compatible upstream, then point RusaAi at the proxy. The upstream URL is a server-side setting; RusaAi never receives the upstream runtime key.

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://MODEL-SERVER-HOST/openai/v1",
    api_key="RUSAMODELS_INFERENCE_SERVICE_TOKEN",
)

response = client.chat.completions.create(
    model="qwen-local",
    messages=[{"role": "user", "content": "Hello"}],
)
print(response.choices[0].message.content)
```

For streaming, pass `stream=True`. Use a dedicated `inference`-role token for runtime traffic. Manager and Administrator credentials should remain reserved for model operations and administration. The configured model ID is the only model exposed by this initial single-upstream proxy.


## Local CPU model and embedding runtimes

The optional Codespaces CPU profile connects the Model Server proxy to two private local services:
- Chat/completion requests are forwarded to llama.cpp serving the configured GGUF model.
- Embedding requests are forwarded to a separate local SentenceTransformers service.

The RusaModels public API remains the single authenticated boundary:
- `POST /openai/v1/chat/completions` → local llama.cpp `/v1/chat/completions`.
- `POST /openai/v1/embeddings` → local embedding service `/v1/embeddings`.
- `GET /openai/v1/models` lists the configured generation and embedding model IDs.

RUSA AI should configure its OpenAI-compatible chat client with `https://<model-server-host>/openai/v1`, a dedicated RusaModels `inference`-role token, and model ID `rusa-general-4b`. The inference role is limited to the OpenAI-compatible inference routes. Its embedding provider should use the same API base URL, the same protected API boundary, model ID `rusa-embedding-small`, and dimension 384.

The default embedding model is multilingual-e5-small. It expects `input_type=passage` for document chunks and `input_type=query` for retrieval queries. The RUSA AI embedding client/RAG call sites must send this field explicitly. The migrated RusaModels embedding endpoint accepts it; this repository change does not alter RUSA AI's existing RAG package.

**Vector schema migration is required before switching an existing RUSA AI knowledge database from a 1536-dimensional provider to this 384-dimensional model.** Recreate or migrate the vector column/index using the selected dimension and re-embed every stored chunk. Do not mix embeddings of different dimensions or models in one collection.

For Colab integration, use the HTTPS Codespaces forwarded Model Server URL. Public port visibility is protected only by the RusaModels bearer token; keep that token secret, use a dedicated principal, and never expose the llama.cpp or embedding container ports. Private Codespaces ports require GitHub-authenticated access and are not directly reachable from an ordinary Colab runtime.

See [CPU inference](CPU_INFERENCE.md) for model artifact preparation and runtime startup.
