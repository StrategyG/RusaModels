import pytest
from pydantic import ValidationError

from rusamodels.config import Settings


def production_settings(**overrides):
    values = {
        "environment": "production",
        "database_url": "postgresql+psycopg://rusamodels:secret@db.example/rusamodels?sslmode=verify-full",
        "storage_backend": "s3",
        "api_principals_json": '{"manager":"digest"}',
        "service_token_sha256": "a" * 64,
        "inference_base_url": "https://runtime.example",
        "inference_model_id": "qwen-prod",
        "inference_api_key": "upstream-secret",
    }
    values.update(overrides)
    return Settings(**values)


def test_blank_optional_inference_settings_are_disabled():
    settings = Settings(
        environment="test",
        inference_base_url=" ",
        inference_model_id="",
        inference_api_key="",
    )
    assert settings.inference_base_url is None
    assert settings.inference_model_id is None
    assert settings.inference_api_key is None


def test_inference_url_and_model_id_must_be_configured_together():
    with pytest.raises(ValidationError, match="configured together"):
        Settings(
            environment="test",
            inference_base_url="http://runtime.example",
        )


def test_production_inference_requires_https_and_upstream_key():
    with pytest.raises(ValidationError, match="HTTPS"):
        production_settings(inference_base_url="http://runtime.example")
    with pytest.raises(ValidationError, match="HTTPS"):
        production_settings(inference_api_key="")
    assert production_settings().inference_base_url == "https://runtime.example"


def test_inference_timeout_has_a_bounded_range():
    with pytest.raises(ValidationError, match="between 1 and 3600"):
        Settings(environment="test", inference_timeout_seconds=3601)


def test_inference_base_url_must_not_include_an_api_path():
    with pytest.raises(ValidationError, match="API path prefix"):
        Settings(
            environment="test",
            inference_base_url="http://runtime.example/v1",
            inference_model_id="qwen-local",
        )


def test_embedding_url_and_model_id_must_be_configured_together():
    with pytest.raises(ValidationError, match="configured together"):
        Settings(environment="test", embedding_base_url="http://embedding.internal")


def test_production_embedding_requires_https_and_upstream_key():
    with pytest.raises(ValidationError, match="HTTPS"):
        production_settings(
            embedding_base_url="http://embedding.example",
            embedding_model_id="e5-small",
            embedding_api_key="embedding-secret",
        )
    with pytest.raises(ValidationError, match="HTTPS"):
        production_settings(
            embedding_base_url="https://embedding.example",
            embedding_model_id="e5-small",
            embedding_api_key="",
        )
    settings = production_settings(
        embedding_base_url="https://embedding.example",
        embedding_model_id="e5-small",
        embedding_api_key="embedding-secret",
    )
    assert settings.embedding_model_id == "e5-small"



def test_staging_and_production_require_full_database_identity_verification():
    with pytest.raises(ValidationError, match="sslmode=verify-full"):
        production_settings(
            database_url="postgresql+psycopg://rusamodels:secret@db.example/rusamodels?sslmode=require"
        )


def test_staging_and_production_reject_insecure_s3_endpoint():
    with pytest.raises(ValidationError, match="credential-free HTTPS"):
        production_settings(s3_endpoint_url="http://s3.internal")


def test_staging_and_production_accept_https_s3_endpoint():
    settings = production_settings(s3_endpoint_url="https://s3.internal")
    assert settings.s3_endpoint_url == "https://s3.internal"


def test_s3_static_credentials_must_be_configured_as_pairs():
    with pytest.raises(ValidationError, match="read access key ID and secret"):
        production_settings(s3_access_key_id="read-key")
    with pytest.raises(ValidationError, match="write access key ID and secret"):
        production_settings(
            s3_write_access_key_id="write-key",
            s3_write_secret_access_key="",
        )


def test_service_token_digest_must_be_lowercase_sha256():
    with pytest.raises(ValidationError, match="lowercase SHA-256"):
        production_settings(service_token_sha256="not-a-digest")


def edge_settings(**overrides):
    values = {
        "environment": "edge",
        "database_url": "sqlite:///./.local/rusamodels.db",
        "storage_backend": "local",
        "local_storage_path": ".local/artifacts",
        "api_principals_json": (
            '{"' + "a" * 64 + '":{"subject":"rusa-ai-inference","role":"inference"}}'
        ),
        "service_token_sha256": "b" * 64,
        "inference_base_url": "http://127.0.0.1:8081",
        "inference_model_id": "rusa-cpu-4b",
        "inference_timeout_seconds": 360,
    }
    values.update(overrides)
    return Settings(**values)


def test_edge_profile_accepts_only_local_sqlite_and_loopback_inference():
    settings = edge_settings()
    assert settings.environment == "edge"
    assert settings.database_url.startswith("sqlite:///")
    assert settings.storage_backend == "local"
    assert settings.inference_base_url == "http://127.0.0.1:8081"
    assert settings.inference_api_key is None


def test_edge_profile_rejects_nonlocal_database_or_object_storage():
    with pytest.raises(ValidationError, match="local SQLite"):
        edge_settings(database_url="postgresql+psycopg://user:pass@db.example/rusa")
    with pytest.raises(ValidationError, match="local artifact storage"):
        edge_settings(storage_backend="s3")


def test_edge_profile_rejects_remote_or_https_inference_upstreams():
    with pytest.raises(ValidationError, match="loopback only"):
        edge_settings(inference_base_url="http://runtime.example")
    with pytest.raises(ValidationError, match="loopback only"):
        edge_settings(inference_base_url="https://127.0.0.1:8081")


def test_edge_profile_rejects_separate_embedding_runtime():
    with pytest.raises(ValidationError, match="separate embedding runtime"):
        edge_settings(
            embedding_base_url="http://127.0.0.1:8082",
            embedding_model_id="rusa-embedding-small",
        )


def test_edge_profile_requires_an_inference_role_and_service_token():
    from rusamodels.api import create_service_app

    viewer_only = (
        '{"' + "c" * 64 + '":{"subject":"local-viewer","role":"viewer"}}'
    )
    with pytest.raises(RuntimeError, match="dedicated role=inference"):
        create_service_app("model-server", edge_settings(api_principals_json=viewer_only))
    with pytest.raises(RuntimeError, match="SERVICE_TOKEN_SHA256"):
        create_service_app("model-server", edge_settings(service_token_sha256=None))


def test_edge_profile_does_not_start_the_quant_server():
    from rusamodels.api import create_service_app

    with pytest.raises(RuntimeError, match="runs only the Model Server"):
        create_service_app("quant-server", edge_settings())


def test_edge_profile_rejects_s3_endpoints_and_credentials():
    with pytest.raises(ValidationError, match="must not configure S3"):
        edge_settings(s3_endpoint_url="https://s3.example")
    with pytest.raises(ValidationError, match="must not configure S3"):
        edge_settings(
            s3_access_key_id="read-key",
            s3_secret_access_key="read-secret",
        )


def test_edge_profile_requires_a_configured_local_runtime():
    with pytest.raises(ValidationError, match="requires one local inference runtime"):
        edge_settings(inference_base_url=None, inference_model_id=None)
