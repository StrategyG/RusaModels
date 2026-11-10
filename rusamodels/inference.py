"""OpenAI-compatible proxy for an explicitly configured inference runtime."""
from collections.abc import AsyncIterator
from typing import Any

import httpx
from fastapi import HTTPException
from fastapi.responses import Response, StreamingResponse


def _runtime_configuration(settings) -> tuple[str, str, dict[str, str]]:
    if not settings.inference_base_url or not settings.inference_model_id:
        raise HTTPException(503, "Inference runtime is not configured")
    headers = {"Content-Type": "application/json"}
    if settings.inference_api_key:
        headers["Authorization"] = (
            f"Bearer {settings.inference_api_key.get_secret_value()}"
        )
    return (
        settings.inference_base_url.rstrip("/"),
        settings.inference_model_id,
        headers,
    )


def openai_model_list(settings) -> dict[str, Any]:
    _base_url, model_id, _headers = _runtime_configuration(settings)
    return {
        "object": "list",
        "data": [
            {
                "id": model_id,
                "object": "model",
                "created": 0,
                "owned_by": "rusamodels",
            }
        ],
    }


async def proxy_openai_request(
    settings,
    payload: dict[str, Any],
    endpoint: str,
    request_id: str,
) -> Response:
    base_url, model_id, headers = _runtime_configuration(settings)
    requested_model = payload.get("model")
    if not isinstance(requested_model, str) or not requested_model:
        raise HTTPException(422, "An inference model ID is required")
    if requested_model not in {model_id, "default"}:
        raise HTTPException(404, "Inference model is not configured")
    forwarded = {**payload, "model": model_id}
    if request_id:
        headers["X-Request-ID"] = request_id
    url = f"{base_url}/v1/{endpoint.lstrip('/')}"
    client = httpx.AsyncClient(timeout=settings.inference_timeout_seconds)
    try:
        request = client.build_request("POST", url, json=forwarded, headers=headers)
        upstream = await client.send(request, stream=bool(payload.get("stream")))
    except httpx.TimeoutException as exc:
        await client.aclose()
        raise HTTPException(504, "Inference runtime timed out") from exc
    except httpx.HTTPError as exc:
        await client.aclose()
        raise HTTPException(502, "Inference runtime is unavailable") from exc

    response_headers = {
        key: value
        for key, value in upstream.headers.items()
        if key.lower() in {"content-type", "cache-control", "x-request-id"}
    }
    if payload.get("stream"):
        if upstream.status_code >= 400:
            body = await upstream.aread()
            status_code = upstream.status_code
            await upstream.aclose()
            await client.aclose()
            return Response(
                content=body,
                status_code=status_code,
                headers=response_headers,
                media_type=response_headers.get("content-type"),
            )

        async def stream_body() -> AsyncIterator[bytes]:
            try:
                async for chunk in upstream.aiter_raw():
                    yield chunk
            finally:
                await upstream.aclose()
                await client.aclose()

        return StreamingResponse(
            stream_body(),
            status_code=upstream.status_code,
            media_type=response_headers.get("content-type", "text/event-stream"),
            headers=response_headers,
        )

    body = await upstream.aread()
    status_code = upstream.status_code
    await upstream.aclose()
    await client.aclose()
    return Response(
        content=body,
        status_code=status_code,
        headers=response_headers,
        media_type=response_headers.get("content-type"),
    )
