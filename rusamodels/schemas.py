"""Validated API request and response schemas."""
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator


class ModelCreate(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    publisher: str | None = Field(default=None, max_length=255)
    architecture: str | None = Field(default=None, max_length=128)
    license_id: str | None = Field(default=None, max_length=128)
    source_uri: HttpUrl | None = None


class ModelPatch(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=255)
    publisher: str | None = Field(default=None, max_length=255)
    architecture: str | None = Field(default=None, max_length=128)
    license_id: str | None = Field(default=None, max_length=128)


class VersionCreate(BaseModel):
    version: str = Field(min_length=1, max_length=128)
    source_revision: str | None = Field(default=None, max_length=255)
    precision: str | None = Field(default=None, max_length=32)
    manifest_sha256: str | None = Field(default=None, pattern=r"^[0-9a-fA-F]{64}$")


class UploadCreate(BaseModel):
    filename: str = Field(min_length=1, max_length=512)
    expected_size: int = Field(gt=0)
    expected_sha256: str = Field(pattern=r"^[0-9a-fA-F]{64}$")
    media_type: str = Field(
        default="application/octet-stream",
        max_length=255,
        pattern=r"^[A-Za-z0-9.+-]+/[A-Za-z0-9.+-]+$",
    )


class ModelOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    name: str
    publisher: str | None
    architecture: str | None
    license_id: str | None
    source_uri: str | None
    status: str
    created_by: str
    created_at: datetime | None


class VersionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    model_id: str
    version: str
    source_revision: str | None
    precision: str | None
    status: str
    manifest_sha256: str | None
    created_by: str
    created_at: datetime | None


class UploadOut(BaseModel):
    id: str
    version_id: str
    filename: str
    expected_size: int
    expected_sha256: str
    status: str
    part_size_limit: int


class ArtifactOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    version_id: str
    original_filename: str | None
    media_type: str | None
    size_bytes: int
    sha256: str
    immutable: bool
    created_at: datetime | None


class HubDownloadCreate(BaseModel):
    repo_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$", max_length=255)
    revision: str = Field(default="main", min_length=1, max_length=255)
    model_name: str = Field(min_length=1, max_length=255)
    version: str = Field(default="hf-main", min_length=1, max_length=128)
    license_id: str = Field(min_length=1, max_length=128)
    license_accepted: bool
    architecture: str | None = Field(default=None, max_length=128)

    @field_validator("license_id")
    @classmethod
    def normalize_license_id(cls, value: str) -> str:
        normalized = value.strip().lower()
        if not normalized:
            raise ValueError("license_id must not be blank")
        return normalized


class ConversionCreate(BaseModel):
    output_version: str = Field(min_length=1, max_length=128)
    target_format: str = Field(pattern=r"^(safetensors|gguf)$")
    target_precision: str = Field(default="F16", pattern=r"^(F32|F16|BF16)$")
    quantization: str | None = Field(default=None, pattern=r"^(Q8|Q6|Q5|Q4)$")


class JobOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    model_id: str
    input_version_id: str | None
    output_version_id: str | None
    job_type: str
    status: str
    progress_percent: int
    attempt_count: int
    max_attempts: int
    next_attempt_at: datetime | None
    error_code: str | None
    created_by: str
    created_at: datetime | None
    started_at: datetime | None
    finished_at: datetime | None
