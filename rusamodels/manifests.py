"""Strict, versioned model manifest contract and canonical digest helpers."""
from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class _StrictManifestModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ManifestSource(_StrictManifestModel):
    repository: str = Field(min_length=1, max_length=512)
    revision: str = Field(pattern=r"^[a-fA-F0-9]{7,64}$")
    license: str = Field(min_length=1, max_length=128)
    usage_restrictions: list[str] = Field(default_factory=list, max_length=64)

    @field_validator("usage_restrictions")
    @classmethod
    def validate_usage_restrictions(cls, value: list[str]) -> list[str]:
        if any(not item.strip() or len(item) > 512 for item in value):
            raise ValueError(
                "usage restrictions must be non-empty strings up to 512 characters"
            )
        return value


class ManifestTokenizer(_StrictManifestModel):
    id: str = Field(min_length=1, max_length=512)
    version: str = Field(min_length=1, max_length=255)


class ManifestTraining(_StrictManifestModel):
    dataset_release: str = Field(min_length=1, max_length=255)
    config_digest: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    code_revision: str = Field(pattern=r"^[a-fA-F0-9]{7,64}$")

    @field_validator("config_digest")
    @classmethod
    def normalize_config_digest(cls, value: str) -> str:
        return value.lower()


class ManifestArtifact(_StrictManifestModel):
    name: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,254}$")
    format: str = Field(min_length=1, max_length=64)
    quantization: str | None = Field(default=None, max_length=64)
    size_bytes: int = Field(gt=0)
    sha256: str = Field(pattern=r"^[a-fA-F0-9]{64}$")

    @field_validator("sha256")
    @classmethod
    def normalize_sha256(cls, value: str) -> str:
        return value.lower()


class ManifestEvaluation(_StrictManifestModel):
    report_id: str = Field(min_length=1, max_length=255)


class ModelManifest(_StrictManifestModel):
    schema_version: Literal["1.0"]
    model_id: str = Field(min_length=1, max_length=255)
    model_version: str = Field(min_length=1, max_length=128)
    model_type: str = Field(min_length=1, max_length=128)
    architecture: str = Field(min_length=1, max_length=128)
    architecture_config: dict[str, Any] = Field(default_factory=dict)
    tokenizer: ManifestTokenizer
    source: ManifestSource
    training: ManifestTraining | None = None
    artifacts: list[ManifestArtifact] = Field(min_length=1, max_length=1000)
    evaluation: ManifestEvaluation | None = None

    @field_validator("architecture_config")
    @classmethod
    def require_json_safe_architecture_config(cls, value: dict[str, Any]) -> dict[str, Any]:
        try:
            json.dumps(value, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("architecture_config must contain only finite JSON values") from exc
        return value

    @model_validator(mode="after")
    def unique_artifact_names(self) -> ModelManifest:
        names = [item.name for item in self.artifacts]
        if len(names) != len(set(names)):
            raise ValueError("manifest artifact names must be unique")
        return self


def canonical_manifest_bytes(manifest: ModelManifest) -> bytes:
    payload = manifest.model_dump(mode="json")
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_manifest_sha256(manifest: ModelManifest) -> str:
    return hashlib.sha256(canonical_manifest_bytes(manifest)).hexdigest()


def manifest_artifacts_match(manifest: ModelManifest, artifacts: list[Any]) -> bool:
    """Require the manifest to describe every live artifact exactly once."""
    actual_items = [item for item in artifacts if item.deleted_at is None]
    if any(not item.immutable for item in actual_items):
        return False
    if len(actual_items) != len(manifest.artifacts):
        return False
    actual = {
        item.original_filename: (item.size_bytes, item.sha256.lower())
        for item in actual_items
    }
    if len(actual) != len(actual_items):
        return False
    expected = {
        item.name: (item.size_bytes, item.sha256.lower())
        for item in manifest.artifacts
    }
    return actual == expected
