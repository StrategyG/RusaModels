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
