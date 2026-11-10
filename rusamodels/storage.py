"""Immutable artifact storage adapters."""
import hashlib
import os
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import boto3
from botocore.exceptions import ClientError


@dataclass(frozen=True)
class StorageWriteResult:
    key: str
    sha256: str
    size_bytes: int


def validate_storage_key(key: str) -> str:
    if not isinstance(key, str) or not key or key.strip() != key:
        raise ValueError("Storage key must be a non-empty relative path")
    if "\\" in key or "\x00" in key:
        raise ValueError("Storage key contains invalid characters")
    path = PurePosixPath(key)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("Storage key contains traversal segments")
    return str(path)


class LocalArtifactStorage:
    backend = "local"

    def __init__(self, root: Path):
        self.root = root.resolve()

    def check_ready(self):
        self.root.mkdir(parents=True, exist_ok=True)
        if not os.access(self.root, os.W_OK | os.X_OK):
            raise RuntimeError("Local artifact storage is not writable")

    def write_bytes(self, key: str, payload: bytes, expected_sha256: str | None = None) -> StorageWriteResult:
        safe = validate_storage_key(key)
        target = (self.root / safe).resolve()
        if self.root not in target.parents:
            raise ValueError("Artifact path escapes storage root")
        digest = hashlib.sha256(payload).hexdigest()
        if expected_sha256 and not hashlib.compare_digest(digest, expected_sha256.lower()):
            raise ValueError("SHA-256 mismatch")
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            with target.open("xb") as handle:
                handle.write(payload)
        except FileExistsError:
            raise FileExistsError("Artifact keys are immutable") from None
        return StorageWriteResult(safe, digest, len(payload))

    def read_bytes(self, key: str) -> bytes:
        safe = validate_storage_key(key)
        target = (self.root / safe).resolve()
        if self.root not in target.parents:
            raise ValueError("Artifact path escapes storage root")
        return target.read_bytes()


class S3ArtifactStorage:
    backend = "s3"

    def __init__(self, settings):
        self.bucket = settings.s3_bucket
        self.client = boto3.client(
            "s3", endpoint_url=settings.s3_endpoint_url, region_name=settings.s3_region,
            aws_access_key_id=settings.s3_access_key_id,
            aws_secret_access_key=settings.s3_secret_access_key.get_secret_value() if settings.s3_secret_access_key else None,
        )

    def check_ready(self):
        self.client.head_bucket(Bucket=self.bucket)

    def write_bytes(self, key: str, payload: bytes, expected_sha256: str | None = None) -> StorageWriteResult:
        safe = validate_storage_key(key)
        digest = hashlib.sha256(payload).hexdigest()
        if expected_sha256 and not hmac_compare(digest, expected_sha256.lower()):
            raise ValueError("SHA-256 mismatch")
        try:
            self.client.head_object(Bucket=self.bucket, Key=safe)
        except ClientError as exc:
            code = str(exc.response.get("Error", {}).get("Code", ""))
            if code not in {"404", "NoSuchKey", "NotFound"}:
                raise
        else:
            raise FileExistsError("Artifact keys are immutable")
        self.client.put_object(Bucket=self.bucket, Key=safe, Body=payload,
                               ContentLength=len(payload), Metadata={"sha256": digest}, IfNoneMatch="*")
        return StorageWriteResult(safe, digest, len(payload))

    def read_bytes(self, key: str) -> bytes:
        return self.client.get_object(Bucket=self.bucket, Key=validate_storage_key(key))["Body"].read()


def hmac_compare(left: str, right: str) -> bool:
    import hmac
    return hmac.compare_digest(left, right)


def create_storage(settings):
    if settings.storage_backend == "s3":
        return S3ArtifactStorage(settings)
    if settings.environment not in {"development", "test"}:
        raise RuntimeError("Local storage is allowed only in development and tests")
    return LocalArtifactStorage(settings.local_storage_path)
