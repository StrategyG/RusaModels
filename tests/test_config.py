import pytest
from pydantic import ValidationError

from rusamodels.config import Settings


def production_settings(**overrides):
    values = {
        "environment": "production",
        "database_url": "postgresql+psycopg://rusamodels:secret@db.example/rusamodels",
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
