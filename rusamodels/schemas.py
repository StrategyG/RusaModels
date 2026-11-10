"""Validated API request and response schemas."""
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, HttpUrl


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
    media_type: str = Field(default="application/octet-stream", max_length=255)


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
    original_filename: str
    media_type: str | None
    size_bytes: int
    sha256: str
    immutable: bool
    created_at: datetime | None
