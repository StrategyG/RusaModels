# RusaAi Integration

RusaModels is a separate service boundary. RusaAi should call the service API and must not receive database credentials or object-store credentials.

Phase 1 dependency endpoint:
- GET /internal/v1/platform/dependencies
- Header: X-Rusa-Service-Token
- The token is verified against RUSAMODELS_SERVICE_TOKEN_SHA256.

The OpenAI-compatible inference endpoint and model routing contract are not yet implemented. A dependency check does not prove that a model is loaded or compatible with tool calling.
