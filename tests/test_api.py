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


def test_service_authentication(api_client):
    assert api_client.get("/internal/v1/platform/dependencies").status_code == 401
    response = api_client.get("/internal/v1/platform/dependencies",
                              headers={"X-Rusa-Service-Token": "service-test-token"})
    assert response.status_code == 200
