import json
import os
import pytest
from fastapi.testclient import TestClient
from rusamodels.api import create_service_app
from rusamodels.config import Settings
from rusamodels.database import build_engine
from rusamodels.models import Base
from rusamodels.security import sha256_hex


@pytest.fixture
def api_client(tmp_path):
    token_map = {
        "viewer-test-token": {"subject": "test-viewer", "role": "viewer"},
        "manager-test-token": {"subject": "test-manager", "role": "manager"},
        "inference-test-token": {"subject": "rusa-ai-inference", "role": "inference"},
        "admin-test-token": {"subject": "test-admin", "role": "administrator"},
    }
    principals = {sha256_hex(token): value for token, value in token_map.items()}
    settings = Settings(environment="test", database_url=f"sqlite:///{tmp_path / 'test.db'}",
                        storage_backend="local", local_storage_path=tmp_path / "objects",
                        api_principals_json=json.dumps(principals),
                        service_token_sha256=sha256_hex("service-test-token"))
    engine = build_engine(settings.database_url)
    Base.metadata.create_all(engine)
    app = create_service_app("quant-server", settings, engine=engine)
    with TestClient(app) as client:
        yield client
    engine.dispose()


@pytest.fixture
def postgres_api_client(tmp_path):
    database_url = os.getenv("RUSAMODELS_TEST_DATABASE_URL")
    if not database_url:
        pytest.skip("RUSAMODELS_TEST_DATABASE_URL is not configured")

    token_map = {
        "viewer-test-token": {"subject": "test-viewer", "role": "viewer"},
        "manager-test-token": {"subject": "test-manager", "role": "manager"},
        "inference-test-token": {"subject": "rusa-ai-inference", "role": "inference"},
        "admin-test-token": {"subject": "test-admin", "role": "administrator"},
    }
    principals = {sha256_hex(token): value for token, value in token_map.items()}
    settings = Settings(
        environment="test",
        database_url=database_url,
        storage_backend="local",
        local_storage_path=tmp_path / "postgres-objects",
        api_principals_json=json.dumps(principals),
        service_token_sha256=sha256_hex("service-test-token"),
    )
    engine = build_engine(settings.database_url)
    # This fixture is used only against the disposable PostgreSQL service in CI.
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    app = create_service_app("quant-server", settings, engine=engine)
    try:
        with TestClient(app) as client:
            yield client
    finally:
        Base.metadata.drop_all(engine)
        engine.dispose()
