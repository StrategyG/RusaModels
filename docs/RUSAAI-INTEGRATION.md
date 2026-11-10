# RusaAi Integration

RusaModels is a separate service boundary. RusaAi should call the service API and must not receive database credentials or object-store credentials.

RusaModels now exposes authenticated model registry, model-version, artifact download and job-status APIs. The Quant Server accepts queued Hugging Face downloads and conversion jobs; the Quant Worker executes them. The Model Server exposes Hub search and read-only artifact access.
- GET /internal/v1/platform/dependencies
- Header: X-Rusa-Service-Token
- The token is verified against RUSAMODELS_SERVICE_TOKEN_SHA256.

The OpenAI-compatible inference endpoint and model routing contract are not yet implemented. A dependency check does not prove that a model is loaded or compatible with tool calling.
