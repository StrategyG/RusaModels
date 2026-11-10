"""Validated configuration shared by Model Server and Quant Server."""
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", case_sensitive=False, populate_by_name=True)

    environment: Literal["development", "test", "staging", "production"] = Field(
        "development", validation_alias="RUSAMODELS_ENV"
    )
    database_url: str = Field("sqlite:///./.local/rusamodels.db", validation_alias="DATABASE_URL")
    storage_backend: Literal["local", "s3"] = Field("local", validation_alias="STORAGE_BACKEND")
    local_storage_path: Path = Field(Path(".local/artifacts"), validation_alias="LOCAL_STORAGE_PATH")
    s3_endpoint_url: str | None = Field(None, validation_alias="S3_ENDPOINT_URL")
    s3_region: str = Field("us-east-1", validation_alias="S3_REGION")
    s3_bucket: str = Field("rusamodels-artifacts", validation_alias="S3_BUCKET")
    s3_access_key_id: str | None = Field(None, validation_alias="S3_ACCESS_KEY_ID")
    s3_secret_access_key: SecretStr | None = Field(None, validation_alias="S3_SECRET_ACCESS_KEY")
    api_principals_json: str = Field("{}", validation_alias="RUSAMODELS_API_PRINCIPALS")
    service_token_sha256: str | None = Field(None, validation_alias="RUSAMODELS_SERVICE_TOKEN_SHA256")
    hf_token: SecretStr | None = Field(None, validation_alias="HF_TOKEN")
    worker_poll_seconds: int = Field(5, validation_alias="RUSAMODELS_WORKER_POLL_SECONDS")
    inference_base_url: str | None = Field(None, validation_alias="RUSAMODELS_INFERENCE_BASE_URL")
    inference_model_id: str | None = Field(None, validation_alias="RUSAMODELS_INFERENCE_MODEL_ID")
    inference_api_key: SecretStr | None = Field(None, validation_alias="RUSAMODELS_INFERENCE_API_KEY")
    inference_timeout_seconds: int = Field(300, validation_alias="RUSAMODELS_INFERENCE_TIMEOUT_SECONDS")
    embedding_base_url: str | None = Field(None, validation_alias="RUSAMODELS_EMBEDDING_BASE_URL")
    embedding_model_id: str | None = Field(None, validation_alias="RUSAMODELS_EMBEDDING_MODEL_ID")
    embedding_api_key: SecretStr | None = Field(None, validation_alias="RUSAMODELS_EMBEDDING_API_KEY")
    embedding_timeout_seconds: int = Field(120, validation_alias="RUSAMODELS_EMBEDDING_TIMEOUT_SECONDS")
    max_upload_bytes: int = Field(10 * 1024**3, validation_alias="MAX_UPLOAD_BYTES")

    @field_validator("max_upload_bytes")
    @classmethod
    def positive_upload_limit(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("MAX_UPLOAD_BYTES must be positive")
        return value

    @field_validator(
        "inference_base_url", "inference_model_id", "inference_api_key", "embedding_base_url", "embedding_model_id", "embedding_api_key", mode="before"
    )
    @classmethod
    def empty_optional_runtime_settings_to_none(cls, value):
        if isinstance(value, str):
            value = value.strip()
            return value or None
        return value

    @field_validator("inference_timeout_seconds")
    @classmethod
    def positive_inference_timeout(cls, value: int) -> int:
        if value <= 0 or value > 3600:
            raise ValueError("RUSAMODELS_INFERENCE_TIMEOUT_SECONDS must be between 1 and 3600")
        return value

    @field_validator("embedding_timeout_seconds")
    @classmethod
    def positive_embedding_timeout(cls, value: int) -> int:
        if value <= 0 or value > 3600:
            raise ValueError("RUSAMODELS_EMBEDDING_TIMEOUT_SECONDS must be between 1 and 3600")
        return value

    @model_validator(mode="after")
    def enforce_production_adapters(self):
        if self.environment in {"staging", "production"}:
            if not self.database_url.startswith(("postgresql://", "postgresql+")):
                raise ValueError("Staging and production require PostgreSQL")
            if self.storage_backend != "s3":
                raise ValueError("Staging and production require S3-compatible storage")
        if bool(self.inference_base_url) != bool(self.inference_model_id):
            raise ValueError("Inference base URL and model ID must be configured together")
        if self.inference_base_url:
            from urllib.parse import urlsplit

            parsed = urlsplit(self.inference_base_url)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                raise ValueError("RUSAMODELS_INFERENCE_BASE_URL must be an HTTP(S) URL")
            if parsed.username or parsed.password or parsed.query or parsed.fragment:
                raise ValueError("Inference base URL must not contain credentials, query or fragment")
            if parsed.path not in {"", "/"}:
                raise ValueError("Inference base URL must not contain an API path prefix")
            if self.environment == "production":
                if parsed.scheme != "https" or not self.inference_api_key:
                    raise ValueError("Production inference requires HTTPS and an upstream API key")
        if bool(self.embedding_base_url) != bool(self.embedding_model_id):
            raise ValueError("Embedding base URL and model ID must be configured together")
        if self.embedding_base_url:
            from urllib.parse import urlsplit

            parsed = urlsplit(self.embedding_base_url)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                raise ValueError("RUSAMODELS_EMBEDDING_BASE_URL must be an HTTP(S) URL")
            if parsed.username or parsed.password or parsed.query or parsed.fragment:
                raise ValueError("Embedding base URL must not contain credentials, query or fragment")
            if parsed.path not in {"", "/"}:
                raise ValueError("Embedding base URL must not contain an API path prefix")
            if self.environment == "production":
                if parsed.scheme != "https" or not self.embedding_api_key:
                    raise ValueError("Production embedding requires HTTPS and an upstream API key")
        if self.environment == "production":
            if not self.service_token_sha256 or not self.api_principals_json.strip() or self.api_principals_json.strip() == "{}":
                raise ValueError("Production requires service and API credentials")
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
