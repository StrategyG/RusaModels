"""Validated request and response schemas."""
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
