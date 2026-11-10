# RusaAi Integration

RusaModels is a separate service boundary. RusaAi should call the service API and must not receive database credentials or object-store credentials.

RusaModels now exposes authenticated model registry, model-version, artifact download and job-status APIs. The Quant Server accepts queued Hugging Face downloads and conversion jobs; the Quant Worker executes them with leases, heartbeat renewal, bounded retries and cooperative cancellation. The Model Server exposes Hub search and read-only artifact access.

- `GET /v1/jobs/{job_id}` — inspect job state and retry/cancellation metadata.
- `POST /v1/jobs/{job_id}/cancel` — administrator-only idempotent cancellation; queued jobs cancel immediately and running jobs receive a cooperative cancellation request.
- GET /internal/v1/platform/dependencies
- Header: X-Rusa-Service-Token
- The token is verified against RUSAMODELS_SERVICE_TOKEN_SHA256.

The OpenAI-compatible inference endpoint and model routing contract are not yet implemented. A dependency check does not prove that a model is loaded or compatible with tool calling.
