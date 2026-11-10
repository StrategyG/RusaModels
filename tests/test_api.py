import hashlib


def auth(token):
    return {"Authorization": f"Bearer {token}"}


def test_health_and_request_id(api_client):
    response = api_client.get("/health/live", headers={"X-Request-ID": "req-test-1"})
    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == "req-test-1"
    assert api_client.get("/health/ready").status_code == 200


def test_model_crud_and_versioning(api_client):
    assert api_client.get("/v1/models").status_code == 401
    denied = api_client.post("/v1/models", json={"name": "demo"}, headers=auth("viewer-test-token"))
    assert denied.status_code == 403
    created = api_client.post("/v1/models", json={"name": "Qwen Demo", "architecture": "Qwen"},
                              headers=auth("manager-test-token"))
    assert created.status_code == 201
    model_id = created.json()["id"]
    version = api_client.post(f"/v1/models/{model_id}/versions",
                              json={"version": "v1", "precision": "F16"},
                              headers=auth("manager-test-token"))
    assert version.status_code == 201
    assert api_client.get(f"/v1/models/{model_id}", headers=auth("viewer-test-token")).status_code == 200


def test_admin_only_archive_and_audit(api_client):
    model = api_client.post("/v1/models", json={"name": "Archive me"},
                            headers=auth("manager-test-token")).json()
    denied = api_client.delete(f"/v1/models/{model['id']}", headers=auth("manager-test-token"))
    assert denied.status_code == 403
    allowed = api_client.delete(f"/v1/models/{model['id']}", headers=auth("admin-test-token"))
    assert allowed.status_code == 204
    assert api_client.get(f"/v1/models/{model['id']}", headers=auth("viewer-test-token")).status_code == 404
    audit = api_client.get("/v1/platform/audit-events", headers=auth("admin-test-token"))
    assert audit.status_code == 200
    assert any(item["action"] == "authorization.denied" for item in audit.json()["items"])


def create_model_version(api_client):
    model = api_client.post("/v1/models", json={"name": "Artifact test"},
                            headers=auth("manager-test-token")).json()
    version = api_client.post(f"/v1/models/{model['id']}/versions",
                              json={"version": "v1", "precision": "F16"},
                              headers=auth("manager-test-token")).json()
    return model["id"], version["id"]


def test_resumable_upload_and_verified_download(api_client):
    model_id, version_id = create_model_version(api_client)
    payload = b"small artifact for multipart test"
    digest = hashlib.sha256(payload).hexdigest()
    started = api_client.post(
        f"/v1/models/{model_id}/versions/{version_id}/uploads",
        json={"filename": "../../model.gguf", "expected_size": len(payload),
              "expected_sha256": digest, "media_type": "application/octet-stream"},
        headers=auth("manager-test-token"),
    )
    assert started.status_code == 201
    upload_id = started.json()["id"]
    part = api_client.put(
        f"/v1/uploads/{upload_id}/parts/1",
        content=payload,
        headers={**auth("manager-test-token"), "Content-Type": "application/octet-stream"},
    )
    assert part.status_code == 200
    completed = api_client.post(
        f"/v1/uploads/{upload_id}/complete", headers=auth("manager-test-token")
    )
    assert completed.status_code == 200
    artifact_id = completed.json()["id"]
    assert completed.json()["sha256"] == digest
    assert completed.json()["original_filename"] == "model.gguf"

    listed = api_client.get(
        f"/v1/models/{model_id}/versions/{version_id}/artifacts",
        headers=auth("viewer-test-token"),
    )
    assert listed.status_code == 200
    downloaded = api_client.get(
        f"/v1/artifacts/{artifact_id}/download", headers=auth("viewer-test-token")
    )
    assert downloaded.status_code == 200
    assert downloaded.content == payload
    assert downloaded.headers["x-checksum-sha256"] == digest


def test_upload_checksum_mismatch_is_rejected(api_client):
    model_id, version_id = create_model_version(api_client)
    payload = b"tampered"
    started = api_client.post(
        f"/v1/models/{model_id}/versions/{version_id}/uploads",
        json={"filename": "weights.bin", "expected_size": len(payload),
              "expected_sha256": "0" * 64},
        headers=auth("manager-test-token"),
    )
    upload_id = started.json()["id"]
    part = api_client.put(
        f"/v1/uploads/{upload_id}/parts/1", content=payload,
        headers=auth("manager-test-token"),
    )
    assert part.status_code == 200
    complete = api_client.post(
        f"/v1/uploads/{upload_id}/complete", headers=auth("manager-test-token")
    )
    assert complete.status_code == 422
    status = api_client.get(f"/v1/uploads/{upload_id}", headers=auth("manager-test-token"))
    assert status.json()["status"] == "failed"


def test_admin_only_upload_abort(api_client):
    model_id, version_id = create_model_version(api_client)
    payload = b"partial"
    started = api_client.post(
        f"/v1/models/{model_id}/versions/{version_id}/uploads",
        json={"filename": "partial.bin", "expected_size": 100,
              "expected_sha256": hashlib.sha256(payload).hexdigest()},
        headers=auth("manager-test-token"),
    )
    upload_id = started.json()["id"]
    denied = api_client.delete(f"/v1/uploads/{upload_id}", headers=auth("manager-test-token"))
    assert denied.status_code == 403
    aborted = api_client.delete(f"/v1/uploads/{upload_id}", headers=auth("admin-test-token"))
    assert aborted.status_code == 204


def test_service_authentication(api_client):
    assert api_client.get("/internal/v1/platform/dependencies").status_code == 401
    response = api_client.get("/internal/v1/platform/dependencies",
                              headers={"X-Rusa-Service-Token": "service-test-token"})
    assert response.status_code == 200


def test_hub_download_job_is_queued_with_license_acknowledgement(api_client):
    rejected = api_client.post(
        "/v1/hub/downloads",
        json={
            "repo_id": "Qwen/Qwen2.5-0.5B-Instruct",
            "revision": "main",
            "model_name": "Qwen 0.5B",
            "version": "hf-main",
            "license_id": "apache-2.0",
            "license_accepted": False,
        },
        headers=auth("manager-test-token"),
    )
    assert rejected.status_code == 400
    accepted = api_client.post(
        "/v1/hub/downloads",
        json={
            "repo_id": "Qwen/Qwen2.5-0.5B-Instruct",
            "revision": "main",
            "model_name": "Qwen 0.5B",
            "version": "hf-main",
            "license_id": "apache-2.0",
            "license_accepted": True,
            "architecture": "Qwen2ForCausalLM",
        },
        headers=auth("manager-test-token"),
    )
    assert accepted.status_code == 202
    assert accepted.json()["job"]["job_type"] == "download"
    assert accepted.json()["job"]["status"] == "queued"
    audit = api_client.get("/v1/platform/audit-events", headers=auth("admin-test-token"))
    queued_event = next(
        item for item in audit.json()["items"] if item["action"] == "hub.download_queued"
    )
    assert queued_event["actor_subject"] == "test-manager"
    assert queued_event["metadata"]["license_id"] == "apache-2.0"
    assert queued_event["metadata"]["license_accepted"] is True


def test_conversion_job_queue_and_admin_cancel(api_client):
    model_id, version_id = create_model_version(api_client)
    payload = b"placeholder source weights"
    digest = hashlib.sha256(payload).hexdigest()
    started = api_client.post(
        f"/v1/models/{model_id}/versions/{version_id}/uploads",
        json={"filename": "model.safetensors", "expected_size": len(payload),
              "expected_sha256": digest},
        headers=auth("manager-test-token"),
    )
    upload_id = started.json()["id"]
    api_client.put(
        f"/v1/uploads/{upload_id}/parts/1", content=payload,
        headers=auth("manager-test-token"),
    )
    completed = api_client.post(
        f"/v1/uploads/{upload_id}/complete", headers=auth("manager-test-token")
    )
    assert completed.status_code == 200

    queued = api_client.post(
        f"/v1/models/{model_id}/versions/{version_id}/conversion-jobs",
        json={"output_version": "v2", "target_format": "safetensors",
              "target_precision": "F16"},
        headers=auth("manager-test-token"),
    )
    assert queued.status_code == 202
    job_id = queued.json()["id"]
    assert queued.json()["job_type"] == "convert"
    assert queued.json()["status"] == "queued"
    assert api_client.get(f"/v1/jobs/{job_id}", headers=auth("viewer-test-token")).status_code == 200
    cancelled = api_client.post(f"/v1/jobs/{job_id}/cancel", headers=auth("admin-test-token"))
    assert cancelled.status_code == 200
    assert cancelled.json()["status"] == "cancelled"


def test_model_server_hub_search_and_artifact_write_boundary(api_client, monkeypatch):
    from fastapi.testclient import TestClient
    from rusamodels.api import create_service_app

    def fake_search(query, limit):
        return [{"repo_id": "Qwen/demo", "license_id": "apache-2.0"}]

    monkeypatch.setattr("rusamodels.api.search_models", fake_search)
    model_app = create_service_app(
        "model-server",
        api_client.app.state.settings,
        engine=api_client.app.state.engine,
        storage=api_client.app.state.storage,
    )
    with TestClient(model_app) as model_client:
        search = model_client.get("/v1/hub/search?q=Qwen", headers=auth("viewer-test-token"))
        assert search.status_code == 200
        assert search.json()["items"][0]["repo_id"] == "Qwen/demo"
        model_id, version_id = create_model_version(api_client)
        denied = model_client.post(
            f"/v1/models/{model_id}/versions/{version_id}/uploads",
            json={"filename": "file.bin", "expected_size": 1,
                  "expected_sha256": hashlib.sha256(b"x").hexdigest()},
            headers=auth("manager-test-token"),
        )
        assert denied.status_code == 404



def test_admin_can_request_running_job_cancellation_idempotently(api_client):
    from rusamodels.models import ModelJob

    accepted = api_client.post(
        "/v1/hub/downloads",
        json={
            "repo_id": "Qwen/Qwen2.5-0.5B-Instruct",
            "revision": "main",
            "model_name": "Qwen 0.5B cancel test",
            "version": "cancel-test",
            "license_id": "apache-2.0",
            "license_accepted": True,
        },
        headers=auth("manager-test-token"),
    )
    assert accepted.status_code == 202
    job_id = accepted.json()["job"]["id"]
    db = api_client.app.state.sessions()
    job = db.get(ModelJob, job_id)
    job.status = "running"
    job.lease_owner = "worker-test"
    job.lease_epoch = 1
    db.commit()
    db.close()

    first = api_client.post(
        f"/v1/jobs/{job_id}/cancel", headers=auth("admin-test-token")
    )
    assert first.status_code == 200
    assert first.json()["status"] == "running"
    assert first.json()["cancel_requested"] is True

    second = api_client.post(
        f"/v1/jobs/{job_id}/cancel", headers=auth("admin-test-token")
    )
    assert second.status_code == 200
    assert second.json()["cancel_requested"] is True

    audit = api_client.get("/v1/platform/audit-events", headers=auth("admin-test-token"))
    assert sum(
        item["action"] == "model_job.cancel_requested"
        for item in audit.json()["items"]
    ) == 1


def test_inference_service_role_can_call_only_openai_compatible_routes(api_client, monkeypatch):
    settings = api_client.app.state.settings
    settings.inference_base_url = "http://runtime.internal:8088"
    settings.inference_model_id = "qwen-local"

    class FakeResponse:
        status_code = 200
        headers = {"content-type": "application/json"}

        async def aread(self):
            return b'{"id":"chatcmpl-service","object":"chat.completion"}'

        async def aclose(self):
            return None

    class FakeClient:
        def __init__(self, timeout):
            self.timeout = timeout

        def build_request(self, method, url, json, headers):
            return {"method": method, "url": url, "json": json, "headers": headers}

        async def send(self, request, stream=False):
            return FakeResponse()

        async def aclose(self):
            return None

    monkeypatch.setattr("rusamodels.inference.httpx.AsyncClient", FakeClient)
    inference_headers = auth("inference-test-token")
    assert api_client.get("/openai/v1/models", headers=inference_headers).status_code == 200
    chat = api_client.post(
        "/openai/v1/chat/completions",
        json={"model": "qwen-local", "messages": [{"role": "user", "content": "hello"}]},
        headers=inference_headers,
    )
    assert chat.status_code == 200
    assert chat.json()["id"] == "chatcmpl-service"

    # The same principal must not be able to manage models or read administrative audit.
    assert api_client.get("/v1/models", headers=inference_headers).status_code == 403
    assert api_client.post(
        "/v1/models", json={"name": "unauthorized"}, headers=inference_headers
    ).status_code == 403
    assert api_client.get(
        "/v1/platform/audit-events", headers=inference_headers
    ).status_code == 403
