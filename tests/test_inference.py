import httpx
from fastapi.testclient import TestClient
from pydantic import SecretStr

from rusamodels.api import create_service_app


def auth(token):
    return {"Authorization": f"Bearer {token}"}


class FakeResponse:
    status_code = 200
    headers = {"content-type": "application/json", "x-request-id": "runtime-req"}

    async def aread(self):
        return b'{"id":"chatcmpl-test","object":"chat.completion"}'

    async def aiter_raw(self):
        yield b"data: {\"id\":\"chatcmpl-stream\"}\n\n"
        yield b"data: [DONE]\n\n"

    async def aclose(self):
        return None


class FakeAsyncClient:
    last_request = None
    last_stream = None
    fail_with_timeout = False

    def __init__(self, timeout):
        self.timeout = timeout

    def build_request(self, method, url, json, headers):
        return {"method": method, "url": url, "json": json, "headers": headers}

    async def send(self, request, stream=False):
        type(self).last_request = request
        type(self).last_stream = stream
        if type(self).fail_with_timeout:
            raise httpx.TimeoutException("simulated runtime timeout")
        return FakeResponse()

    async def aclose(self):
        return None


def _model_client(api_client, settings):
    app = create_service_app(
        "model-server",
        settings,
        engine=api_client.app.state.engine,
        storage=api_client.app.state.storage,
    )
    return TestClient(app)


def test_openai_chat_proxy_forwards_model_and_private_upstream_key(
    api_client, monkeypatch
):
    settings = api_client.app.state.settings
    settings.inference_base_url = "http://runtime.internal:8088"
    settings.inference_model_id = "qwen-local"
    settings.inference_api_key = SecretStr("runtime-secret")
    monkeypatch.setattr("rusamodels.inference.httpx.AsyncClient", FakeAsyncClient)

    with _model_client(api_client, settings) as client:
        response = client.post(
            "/openai/v1/chat/completions",
            json={
                "model": "default",
                "messages": [{"role": "user", "content": "hello"}],
            },
            headers=auth("manager-test-token"),
        )

    assert response.status_code == 200
    assert response.json()["id"] == "chatcmpl-test"
    assert FakeAsyncClient.last_request["url"] == (
        "http://runtime.internal:8088/v1/chat/completions"
    )
    assert FakeAsyncClient.last_request["json"]["model"] == "qwen-local"
    assert FakeAsyncClient.last_request["headers"]["Authorization"] == (
        "Bearer runtime-secret"
    )
    assert FakeAsyncClient.last_request["headers"]["X-Request-ID"]
    assert FakeAsyncClient.last_stream is False


def test_openai_model_list_is_compatible_and_requires_manager_role(api_client):
    settings = api_client.app.state.settings
    settings.inference_base_url = "http://runtime.internal:8088"
    settings.inference_model_id = "qwen-local"

    with _model_client(api_client, settings) as client:
        denied = client.get("/openai/v1/models", headers=auth("viewer-test-token"))
        listed = client.get("/openai/v1/models", headers=auth("manager-test-token"))

    assert denied.status_code == 403
    assert listed.status_code == 200
    assert listed.json()["object"] == "list"
    assert listed.json()["data"][0]["id"] == "qwen-local"
    assert listed.json()["data"][0]["object"] == "model"


def test_openai_proxy_rejects_unknown_model_without_calling_runtime(
    api_client, monkeypatch
):
    settings = api_client.app.state.settings
    settings.inference_base_url = "http://runtime.internal:8088"
    settings.inference_model_id = "qwen-local"
    FakeAsyncClient.last_request = None
    monkeypatch.setattr("rusamodels.inference.httpx.AsyncClient", FakeAsyncClient)

    with _model_client(api_client, settings) as client:
        response = client.post(
            "/openai/v1/chat/completions",
            json={"model": "unapproved-model", "messages": []},
            headers=auth("manager-test-token"),
        )

    assert response.status_code == 404
    assert FakeAsyncClient.last_request is None


def test_openai_proxy_supports_server_sent_event_streaming(api_client, monkeypatch):
    settings = api_client.app.state.settings
    settings.inference_base_url = "http://runtime.internal:8088"
    settings.inference_model_id = "qwen-local"
    monkeypatch.setattr("rusamodels.inference.httpx.AsyncClient", FakeAsyncClient)

    with _model_client(api_client, settings) as client:
        response = client.post(
            "/openai/v1/chat/completions",
            json={
                "model": "qwen-local",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": True,
            },
            headers=auth("manager-test-token"),
        )

    assert response.status_code == 200
    assert "chatcmpl-stream" in response.text
    assert "[DONE]" in response.text
    assert FakeAsyncClient.last_stream is True


def test_openai_proxy_maps_runtime_timeout_to_gateway_timeout(api_client, monkeypatch):
    settings = api_client.app.state.settings
    settings.inference_base_url = "http://runtime.internal:8088"
    settings.inference_model_id = "qwen-local"
    FakeAsyncClient.fail_with_timeout = True
    monkeypatch.setattr("rusamodels.inference.httpx.AsyncClient", FakeAsyncClient)

    try:
        with _model_client(api_client, settings) as client:
            response = client.post(
                "/openai/v1/completions",
                json={"model": "qwen-local", "prompt": "hello"},
                headers=auth("manager-test-token"),
            )
        assert response.status_code == 504
    finally:
        FakeAsyncClient.fail_with_timeout = False


def test_openai_proxy_returns_service_unavailable_when_unconfigured(api_client):
    settings = api_client.app.state.settings
    settings.inference_base_url = None
    settings.inference_model_id = None

    with _model_client(api_client, settings) as client:
        response = client.get("/openai/v1/models", headers=auth("manager-test-token"))

    assert response.status_code == 503


def test_inference_audit_records_metadata_without_prompt_content(api_client, monkeypatch):
    settings = api_client.app.state.settings
    settings.inference_base_url = "http://runtime.internal:8088"
    settings.inference_model_id = "qwen-local"
    monkeypatch.setattr("rusamodels.inference.httpx.AsyncClient", FakeAsyncClient)

    with _model_client(api_client, settings) as client:
        response = client.post(
            "/openai/v1/chat/completions",
            json={
                "model": "qwen-local",
                "messages": [{"role": "user", "content": "private prompt must not be logged"}],
            },
            headers=auth("manager-test-token"),
        )
        assert response.status_code == 200
        audit = client.get("/v1/platform/audit-events", headers=auth("admin-test-token"))

    event = next(
        item for item in audit.json()["items"]
        if item["action"] == "inference.requested"
    )
    assert event["metadata"]["model"] == "qwen-local"
    assert "messages" not in event["metadata"]
    assert "private prompt must not be logged" not in str(event)



def test_embeddings_proxy_uses_dedicated_local_embedding_runtime(api_client, monkeypatch):
    settings = api_client.app.state.settings
    settings.inference_base_url = "http://llm.internal:8080"
    settings.inference_model_id = "rusa-general-4b"
    settings.embedding_base_url = "http://embedding.internal:8081"
    settings.embedding_model_id = "rusa-embedding-small"
    settings.embedding_api_key = SecretStr("embedding-runtime-secret")
    monkeypatch.setattr("rusamodels.inference.httpx.AsyncClient", FakeAsyncClient)

    with _model_client(api_client, settings) as client:
        response = client.post(
            "/openai/v1/embeddings",
            json={
                "model": "rusa-embedding-small",
                "input": ["RUSA AI agent architecture"],
                "input_type": "passage",
            },
            headers=auth("manager-test-token"),
        )

    assert response.status_code == 200
    assert FakeAsyncClient.last_request["url"] == (
        "http://embedding.internal:8081/v1/embeddings"
    )
    assert FakeAsyncClient.last_request["json"]["model"] == "rusa-embedding-small"
    assert FakeAsyncClient.last_request["headers"]["Authorization"] == (
        "Bearer embedding-runtime-secret"
    )


def test_openai_model_list_includes_local_embedding_model(api_client):
    settings = api_client.app.state.settings
    settings.inference_base_url = "http://llm.internal:8080"
    settings.inference_model_id = "rusa-general-4b"
    settings.embedding_base_url = "http://embedding.internal:8081"
    settings.embedding_model_id = "rusa-embedding-small"

    with _model_client(api_client, settings) as client:
        response = client.get("/openai/v1/models", headers=auth("manager-test-token"))

    assert response.status_code == 200
    assert [item["id"] for item in response.json()["data"]] == [
        "rusa-general-4b",
        "rusa-embedding-small",
    ]
