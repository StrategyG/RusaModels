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
    output_version_id = queued.json()["output_version_id"]
    assert queued.json()["job_type"] == "convert"
    assert queued.json()["status"] == "queued"
    assert api_client.get(f"/v1/jobs/{job_id}", headers=auth("viewer-test-token")).status_code == 200
    cancelled = api_client.post(f"/v1/jobs/{job_id}/cancel", headers=auth("admin-test-token"))
    assert cancelled.status_code == 200
    assert cancelled.json()["status"] == "cancelled"

    # A queued job and its output model version must reach the same terminal state.
    from rusamodels.models import ModelVersion

    db = api_client.app.state.sessions()
    try:
        output_version = db.get(ModelVersion, output_version_id)
        assert output_version is not None
        assert output_version.status == "cancelled"
    finally:
        db.close()


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

def test_upload_mutations_lock_the_upload_session_row(api_client, monkeypatch):
    """PostgreSQL row locks serialize concurrent part, complete and abort requests."""
    from sqlalchemy.dialects import postgresql
    from sqlalchemy.orm import Session

    model_id, version_id = create_model_version(api_client)
    payload = b"multipart-lock"
    digest = hashlib.sha256(payload).hexdigest()

    def start_upload():
        response = api_client.post(
            f"/v1/models/{model_id}/versions/{version_id}/uploads",
            json={
                "filename": "locked.bin",
                "expected_size": len(payload),
                "expected_sha256": digest,
            },
            headers=auth("manager-test-token"),
        )
        assert response.status_code == 201
        return response.json()["id"]

    statements = []
    original_scalar = Session.scalar

    def capture_scalar(session, statement, *args, **kwargs):
        statements.append(statement)
        return original_scalar(session, statement, *args, **kwargs)

    monkeypatch.setattr(Session, "scalar", capture_scalar)

    def assert_upload_row_lock():
        compiled = [
            str(statement.compile(dialect=postgresql.dialect())).upper()
            for statement in statements
        ]
        assert any(
            "FROM UPLOAD_SESSIONS" in statement and "FOR UPDATE" in statement
            for statement in compiled
        ), "upload mutation must acquire a PostgreSQL row lock"

    upload_id = start_upload()
    part = api_client.put(
        f"/v1/uploads/{upload_id}/parts/1",
        content=payload,
        headers=auth("manager-test-token"),
    )
    assert part.status_code == 200
    assert_upload_row_lock()

    statements.clear()
    completed = api_client.post(
        f"/v1/uploads/{upload_id}/complete",
        headers=auth("manager-test-token"),
    )
    assert completed.status_code == 200
    assert_upload_row_lock()

    statements.clear()
    abort_id = start_upload()
    aborted = api_client.delete(
        f"/v1/uploads/{abort_id}",
        headers=auth("admin-test-token"),
    )
    assert aborted.status_code == 204
    assert_upload_row_lock()


def test_postgres_upload_session_row_lock_blocks_competing_transaction(postgres_api_client):
    """A real PostgreSQL transaction must hold the upload row lock until commit."""
    import pytest
    from sqlalchemy import select
    from sqlalchemy.exc import OperationalError
    from rusamodels.models import UploadSession

    model_id, version_id = create_model_version(postgres_api_client)
    payload = b"postgres-row-lock"
    started = postgres_api_client.post(
        f"/v1/models/{model_id}/versions/{version_id}/uploads",
        json={
            "filename": "postgres-lock.bin",
            "expected_size": len(payload),
            "expected_sha256": hashlib.sha256(payload).hexdigest(),
        },
        headers=auth("manager-test-token"),
    )
    assert started.status_code == 201
    upload_id = started.json()["id"]

    sessions = postgres_api_client.app.state.sessions
    locker = sessions()
    contender = sessions()
    statement = select(UploadSession).where(UploadSession.id == upload_id)
    try:
        locker.begin()
        locked = locker.scalar(statement.with_for_update())
        assert locked is not None and locked.status == "open"

        contender.begin()
        with pytest.raises(OperationalError) as error:
            contender.scalar(statement.with_for_update(nowait=True))
        assert getattr(error.value.orig, "sqlstate", None) == "55P03"
        contender.rollback()

        locker.commit()
        contender.begin()
        released = contender.scalar(statement.with_for_update(nowait=True))
        assert released is not None and released.status == "open"
        contender.commit()
    finally:
        locker.rollback()
        contender.rollback()
        locker.close()
        contender.close()
