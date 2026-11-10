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
    max_upload_bytes: int = Field(10 * 1024**3, validation_alias="MAX_UPLOAD_BYTES")

    @field_validator("max_upload_bytes")
    @classmethod
    def positive_upload_limit(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("MAX_UPLOAD_BYTES must be positive")
        return value

    @model_validator(mode="after")
    def enforce_production_adapters(self):
        if self.environment in {"staging", "production"}:
            if not self.database_url.startswith(("postgresql://", "postgresql+")):
                raise ValueError("Staging and production require PostgreSQL")
            if self.storage_backend != "s3":
                raise ValueError("Staging and production require S3-compatible storage")
        if self.environment == "production":
            if not self.service_token_sha256 or not self.api_principals_json.strip() or self.api_principals_json.strip() == "{}":
                raise ValueError("Production requires service and API credentials")
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
