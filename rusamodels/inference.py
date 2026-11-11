"""OpenAI-compatible proxy for configured local model runtimes."""
from collections.abc import AsyncIterator
from typing import Any

import httpx
from fastapi import HTTPException
from fastapi.responses import Response, StreamingResponse


def _runtime_configuration(settings, endpoint: str) -> tuple[str, str, dict[str, str], int]:
    if endpoint == "embeddings" and settings.embedding_base_url:
        base_url = settings.embedding_base_url
        model_id = settings.embedding_model_id
        api_key = settings.embedding_api_key
        timeout = settings.embedding_timeout_seconds
    else:
        base_url = settings.inference_base_url
        model_id = settings.inference_model_id
        api_key = settings.inference_api_key
        timeout = settings.inference_timeout_seconds
    if not base_url or not model_id:
        raise HTTPException(503, f"{endpoint} runtime is not configured")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key.get_secret_value()}"
    return base_url.rstrip("/"), model_id, headers, timeout


def openai_model_list(settings) -> dict[str, Any]:
    _base_url, model_id, _headers, _timeout = _runtime_configuration(settings, "chat/completions")
    models = [{
        "id": model_id,
        "object": "model",
        "created": 0,
        "owned_by": "rusamodels",
    }]
    if settings.embedding_base_url and settings.embedding_model_id:
        models.append({
            "id": settings.embedding_model_id,
            "object": "model",
            "created": 0,
            "owned_by": "rusamodels",
        })
    return {"object": "list", "data": models}


async def proxy_openai_request(
    settings,
    payload: dict[str, Any],
    endpoint: str,
    request_id: str,
) -> Response:
    base_url, model_id, headers, timeout = _runtime_configuration(settings, endpoint)
    requested_model = payload.get("model")
    if not isinstance(requested_model, str) or not requested_model:
        raise HTTPException(422, "An inference model ID is required")
    if requested_model not in {model_id, "default"}:
        raise HTTPException(404, "Inference model is not configured")
    forwarded = {**payload, "model": model_id}
    if request_id:
        headers["X-Request-ID"] = request_id
    url = f"{base_url}/v1/{endpoint.lstrip('/')}"
    client = httpx.AsyncClient(timeout=timeout, trust_env=False)
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


async def check_runtime_readiness(settings, endpoint: str) -> None:
    """Verify that the configured upstream is reachable and exposes the pinned model ID."""
    base_url, model_id, headers, _timeout = _runtime_configuration(settings, endpoint)
    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(5.0, connect=2.0),
            trust_env=False,
        ) as client:
            response = await client.get(f"{base_url}/v1/models", headers=headers)
            response.raise_for_status()
            payload = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        raise RuntimeError(f"{endpoint} runtime readiness check failed") from exc

    items = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(items, list):
        raise RuntimeError(f"{endpoint} runtime returned an invalid model catalog")
    available = {
        item.get("id") for item in items
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    }
    if model_id not in available:
        raise RuntimeError(f"Configured {endpoint} model is not exposed by the upstream runtime")
