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


def _upload_artifact(api_client, model_id, version_id, filename, payload):
    digest = hashlib.sha256(payload).hexdigest()
    started = api_client.post(
        f"/v1/models/{model_id}/versions/{version_id}/uploads",
        json={
            "filename": filename,
            "expected_size": len(payload),
            "expected_sha256": digest,
        },
        headers=auth("manager-test-token"),
    )
    assert started.status_code == 201
    upload_id = started.json()["id"]
    part = api_client.put(
        f"/v1/uploads/{upload_id}/parts/1",
        content=payload,
        headers=auth("manager-test-token"),
    )
    assert part.status_code == 200
    completed = api_client.post(
        f"/v1/uploads/{upload_id}/complete",
        headers=auth("manager-test-token"),
    )
    assert completed.status_code == 200
    return completed.json()


def _model_manifest(model_id, version, filename, payload, report_id="eval-v1"):
    return {
        "schema_version": "1.0",
        "model_id": model_id,
        "model_version": version,
        "model_type": "causal_lm",
        "architecture": "decoder_only_transformer",
        "architecture_config": {"layers": 2, "hidden_size": 64},
        "tokenizer": {"id": "rusa-tokenizer", "version": "v1"},
        "source": {
            "repository": "StrategyG/RusaModels",
            "revision": "a" * 40,
            "license": "apache-2.0",
            "usage_restrictions": [],
        },
        "training": {
            "dataset_release": "rusa-synthetic-v1",
            "config_digest": "b" * 64,
            "code_revision": "abcdef1234567",
        },
        "artifacts": [{
            "name": filename,
            "format": "safetensors",
            "quantization": None,
            "size_bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }],
        "evaluation": {"report_id": report_id},
    }


def test_model_manifest_is_immutable_and_approval_is_audited(api_client):
    model_id, version_id = create_model_version(api_client)
    payload = b"immutable model weights"
    _upload_artifact(api_client, model_id, version_id, "model.safetensors", payload)
    manifest = _model_manifest(model_id, "v1", "model.safetensors", payload)

    attached = api_client.put(
        f"/v1/models/{model_id}/versions/{version_id}/manifest",
        json=manifest,
        headers=auth("manager-test-token"),
    )
    assert attached.status_code == 200
    assert len(attached.json()["sha256"]) == 64

    candidate = api_client.post(
        f"/v1/models/{model_id}/versions/{version_id}/lifecycle",
        json={"status": "candidate"},
        headers=auth("manager-test-token"),
    )
    assert candidate.status_code == 200
    assert candidate.json()["lifecycle_status"] == "candidate"

    evaluated = api_client.post(
        f"/v1/models/{model_id}/versions/{version_id}/lifecycle",
        json={"status": "evaluated"},
        headers=auth("manager-test-token"),
    )
    assert evaluated.status_code == 200

    denied = api_client.post(
        f"/v1/models/{model_id}/versions/{version_id}/lifecycle",
        json={"status": "approved"},
        headers=auth("manager-test-token"),
    )
    assert denied.status_code == 403

    approved = api_client.post(
        f"/v1/models/{model_id}/versions/{version_id}/lifecycle",
        json={"status": "approved"},
        headers=auth("admin-test-token"),
    )
    assert approved.status_code == 200
    assert approved.json()["lifecycle_status"] == "approved"
    assert approved.json()["approved_by"] == "test-admin"

    reread = api_client.get(
        f"/v1/models/{model_id}/versions/{version_id}/manifest",
        headers=auth("viewer-test-token"),
    )
    assert reread.status_code == 200
    assert reread.json()["manifest"] == manifest

    replacement = {**manifest, "architecture": "different_architecture"}
    immutable = api_client.put(
        f"/v1/models/{model_id}/versions/{version_id}/manifest",
        json=replacement,
        headers=auth("manager-test-token"),
    )
    assert immutable.status_code == 409

    audit = api_client.get(
        "/v1/platform/audit-events", headers=auth("admin-test-token")
    )
    assert any(
        item["action"] == "versions.lifecycle.transition"
        and item["metadata"]["to"] == "approved"
        for item in audit.json()["items"]
    )


def test_new_approval_supersedes_previous_and_can_be_rolled_back(api_client):
    model = api_client.post(
        "/v1/models", json={"name": "Promotion chain"},
        headers=auth("manager-test-token"),
    ).json()
    model_id = model["id"]

    first = api_client.post(
        f"/v1/models/{model_id}/versions",
        json={"version": "v1"},
        headers=auth("manager-test-token"),
    ).json()
    first_payload = b"version one"
    _upload_artifact(api_client, model_id, first["id"], "model.bin", first_payload)
    first_manifest = _model_manifest(model_id, "v1", "model.bin", first_payload)
    assert api_client.put(
        f"/v1/models/{model_id}/versions/{first['id']}/manifest",
        json=first_manifest, headers=auth("manager-test-token"),
    ).status_code == 200
    for status in ("candidate", "evaluated", "approved"):
        response = api_client.post(
            f"/v1/models/{model_id}/versions/{first['id']}/lifecycle",
            json={"status": status}, headers=auth("admin-test-token"),
        )
        assert response.status_code == 200

    second = api_client.post(
        f"/v1/models/{model_id}/versions",
        json={"version": "v2"},
        headers=auth("manager-test-token"),
    ).json()
    second_payload = b"version two"
    _upload_artifact(api_client, model_id, second["id"], "model.bin", second_payload)
    second_manifest = _model_manifest(model_id, "v2", "model.bin", second_payload, "eval-v2")
    assert api_client.put(
        f"/v1/models/{model_id}/versions/{second['id']}/manifest",
        json=second_manifest, headers=auth("manager-test-token"),
    ).status_code == 200
    for status in ("candidate", "evaluated", "approved"):
        promoted = api_client.post(
            f"/v1/models/{model_id}/versions/{second['id']}/lifecycle",
            json={"status": status}, headers=auth("admin-test-token"),
        )
        assert promoted.status_code == 200

    assert promoted.json()["supersedes_version_id"] == first["id"]
    first_state = api_client.get(
        f"/v1/models/{model_id}/versions",
        headers=auth("viewer-test-token"),
    ).json()
    assert next(item for item in first_state if item["id"] == first["id"])[
        "lifecycle_status"
    ] == "deprecated"

    rollback = api_client.post(
        f"/v1/models/{model_id}/versions/{first['id']}/lifecycle",
        json={"status": "approved"},
        headers=auth("admin-test-token"),
    )
    assert rollback.status_code == 200
    assert rollback.json()["supersedes_version_id"] == second["id"]


def test_manifest_attachment_rejects_artifact_checksum_mismatch(api_client):
    model_id, version_id = create_model_version(api_client)
    payload = b"actual model bytes"
    _upload_artifact(api_client, model_id, version_id, "model.bin", payload)
    manifest = _model_manifest(model_id, "v1", "model.bin", payload)
    manifest["artifacts"][0]["sha256"] = "0" * 64
    attached = api_client.put(
        f"/v1/models/{model_id}/versions/{version_id}/manifest",
        json=manifest,
        headers=auth("manager-test-token"),
    )
    assert attached.status_code == 409


def test_manifest_attachment_waits_for_all_artifact_uploads(api_client):
    model_id, version_id = create_model_version(api_client)
    payload = b"pending model bytes"
    digest = hashlib.sha256(payload).hexdigest()
    started = api_client.post(
        f"/v1/models/{model_id}/versions/{version_id}/uploads",
        json={
            "filename": "pending.bin",
            "expected_size": len(payload),
            "expected_sha256": digest,
        },
        headers=auth("manager-test-token"),
    )
    assert started.status_code == 201
    manifest = _model_manifest(model_id, "v1", "pending.bin", payload)
    pending = api_client.put(
        f"/v1/models/{model_id}/versions/{version_id}/manifest",
        json=manifest,
        headers=auth("manager-test-token"),
    )
    assert pending.status_code == 409

    upload_id = started.json()["id"]
    part = api_client.put(
        f"/v1/uploads/{upload_id}/parts/1",
        content=payload,
        headers=auth("manager-test-token"),
    )
    assert part.status_code == 200
    completed = api_client.post(
        f"/v1/uploads/{upload_id}/complete",
        headers=auth("manager-test-token"),
    )
    assert completed.status_code == 200
    attached = api_client.put(
        f"/v1/models/{model_id}/versions/{version_id}/manifest",
        json=manifest,
        headers=auth("manager-test-token"),
    )
    assert attached.status_code == 200


def test_artifact_retention_holds_and_expiry_control_purge(api_client):
    model_id, version_id = create_model_version(api_client)
    artifact = _upload_artifact(
        api_client, model_id, version_id, "temporary.bin", b"temporary artifact"
    )
    artifact_id = artifact["id"]

    missing_expiry = api_client.patch(
        f"/v1/artifacts/{artifact_id}/retention",
        json={"retention_class": "temporary"},
        headers=auth("admin-test-token"),
    )
    assert missing_expiry.status_code == 422

    held = api_client.patch(
        f"/v1/artifacts/{artifact_id}/retention",
        json={"retention_until": "2000-01-01T00:00:00Z", "legal_hold": True},
        headers=auth("admin-test-token"),
    )
    assert held.status_code == 200
    assert api_client.delete(
        f"/v1/artifacts/{artifact_id}", headers=auth("admin-test-token")
    ).status_code == 409

    released = api_client.patch(
        f"/v1/artifacts/{artifact_id}/retention",
        json={"legal_hold": False},
        headers=auth("admin-test-token"),
    )
    assert released.status_code == 200
    purged = api_client.delete(
        f"/v1/artifacts/{artifact_id}", headers=auth("admin-test-token")
    )
    assert purged.status_code == 204

    missing = api_client.get(
        f"/v1/artifacts/{artifact_id}/download",
        headers=auth("viewer-test-token"),
    )
    assert missing.status_code == 404
    listed = api_client.get(
        f"/v1/models/{model_id}/versions/{version_id}/artifacts",
        headers=auth("viewer-test-token"),
    )
    assert listed.status_code == 200
    assert listed.json()[0]["deleted_at"] is not None


def test_artifact_download_fails_closed_when_stored_bytes_are_tampered(api_client):
    from pathlib import Path

    model_id, version_id = create_model_version(api_client)
    payload = b"verified artifact bytes"
    artifact = _upload_artifact(
        api_client, model_id, version_id, "tamper.bin", payload
    )
    storage = api_client.app.state.storage
    object_path = storage.root / artifact["storage_key"]
    Path(object_path).write_bytes(b"tampered bytes")

    response = api_client.get(
        f"/v1/artifacts/{artifact['id']}/download",
        headers=auth("viewer-test-token"),
    )
    assert response.status_code == 502
