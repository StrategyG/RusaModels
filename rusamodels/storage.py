"""Immutable local and S3 multipart artifact storage adapters."""
import hashlib
import hmac
import os
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import BinaryIO, Iterator

import boto3
from boto3.s3.transfer import TransferConfig
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
        self.multipart_root = self.root / ".multipart"

    def check_ready(self):
        self.root.mkdir(parents=True, exist_ok=True)
        self.multipart_root.mkdir(parents=True, exist_ok=True)
        if not os.access(self.root, os.W_OK | os.X_OK):
            raise RuntimeError("Local artifact storage is not writable")

    def write_bytes(self, key: str, payload: bytes, expected_sha256: str | None = None) -> StorageWriteResult:
        safe = validate_storage_key(key)
        target = (self.root / safe).resolve()
        if self.root not in target.parents:
            raise ValueError("Artifact path escapes storage root")
        digest = hashlib.sha256(payload).hexdigest()
        if expected_sha256 and not hmac.compare_digest(digest, expected_sha256.lower()):
            raise ValueError("SHA-256 mismatch")
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            with target.open("xb") as handle:
                handle.write(payload)
        except FileExistsError:
            raise FileExistsError("Artifact keys are immutable") from None
        except Exception:
            target.unlink(missing_ok=True)
            raise
        return StorageWriteResult(safe, digest, len(payload))

    def create_multipart(self, key: str, content_type: str, metadata: dict[str, str]) -> str:
        validate_storage_key(key)
        upload_id = str(uuid.uuid4())
        (self.multipart_root / upload_id).mkdir(parents=True, exist_ok=False)
        return upload_id

    def upload_part(self, upload_id: str, part_number: int, payload: bytes) -> str:
        folder = self._multipart_folder(upload_id)
        if not 1 <= part_number <= 10000:
            raise ValueError("part_number must be between 1 and 10000")
        part = folder / f"{part_number:05d}.part"
        part.write_bytes(payload)
        return hashlib.md5(payload, usedforsecurity=False).hexdigest()

    def complete_multipart(self, key: str, upload_id: str, parts: list[dict]) -> None:
        safe = validate_storage_key(key)
        folder = self._multipart_folder(upload_id)
        target = (self.root / safe).resolve()
        if self.root not in target.parents:
            raise ValueError("Artifact path escapes storage root")
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            with target.open("xb") as output:
                for part in parts:
                    number = int(part["PartNumber"])
                    with (folder / f"{number:05d}.part").open("rb") as source:
                        shutil.copyfileobj(source, output, length=1024 * 1024)
        except FileExistsError:
            raise FileExistsError("Artifact keys are immutable") from None
        except Exception:
            target.unlink(missing_ok=True)
            raise
        shutil.rmtree(folder, ignore_errors=True)

    def abort_multipart(self, upload_id: str) -> None:
        shutil.rmtree(self._multipart_folder(upload_id), ignore_errors=True)

    def _multipart_folder(self, upload_id: str) -> Path:
        try:
            safe_id = str(uuid.UUID(upload_id))
        except ValueError as exc:
            raise ValueError("Invalid storage upload identifier") from exc
        folder = (self.multipart_root / safe_id).resolve()
        if self.multipart_root.resolve() not in folder.parents:
            raise ValueError("Multipart path escapes storage root")
        if not folder.is_dir():
            raise FileNotFoundError("Multipart upload not found")
        return folder

    def read_stream(self, key: str) -> Iterator[bytes]:
        target = (self.root / validate_storage_key(key)).resolve()
        if self.root not in target.parents:
            raise ValueError("Artifact path escapes storage root")
        with target.open("rb") as source:
            while chunk := source.read(1024 * 1024):
                yield chunk

    def delete(self, key: str) -> None:
        target = (self.root / validate_storage_key(key)).resolve()
        if self.root not in target.parents:
            raise ValueError("Artifact path escapes storage root")
        target.unlink(missing_ok=True)


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
        if expected_sha256 and not hmac.compare_digest(digest, expected_sha256.lower()):
            raise ValueError("SHA-256 mismatch")
        self.client.put_object(Bucket=self.bucket, Key=safe, Body=payload,
                               ContentLength=len(payload), Metadata={"sha256": digest}, IfNoneMatch="*")
        return StorageWriteResult(safe, digest, len(payload))

    def create_multipart(self, key: str, content_type: str, metadata: dict[str, str]) -> str:
        safe = validate_storage_key(key)
        result = self.client.create_multipart_upload(
            Bucket=self.bucket, Key=safe, ContentType=content_type, Metadata=metadata
        )
        return result["UploadId"]

    def upload_part(self, upload_id: str, part_number: int, payload: bytes) -> str:
        result = self.client.upload_part(
            Bucket=self.bucket, Key=self._active_key(upload_id), UploadId=upload_id,
            PartNumber=part_number, Body=payload,
        )
        return result["ETag"]

    def _active_key(self, upload_id: str) -> str:
        # The API persists the object key with the upload session and passes it to complete/abort.
        raise NotImplementedError("Use upload_part_for_key with the persisted storage key")

    def upload_part_for_key(self, key: str, upload_id: str, part_number: int, payload: bytes) -> str:
        result = self.client.upload_part(
            Bucket=self.bucket, Key=validate_storage_key(key), UploadId=upload_id,
            PartNumber=part_number, Body=payload,
        )
        return result["ETag"]

    def complete_multipart(self, key: str, upload_id: str, parts: list[dict]) -> None:
        self.client.complete_multipart_upload(
            Bucket=self.bucket, Key=validate_storage_key(key), UploadId=upload_id,
            MultipartUpload={"Parts": parts},
        )

    def abort_multipart(self, key: str, upload_id: str) -> None:
        self.client.abort_multipart_upload(
            Bucket=self.bucket, Key=validate_storage_key(key), UploadId=upload_id
        )

    def read_stream(self, key: str) -> Iterator[bytes]:
        body = self.client.get_object(Bucket=self.bucket, Key=validate_storage_key(key))["Body"]
        try:
            while chunk := body.read(1024 * 1024):
                yield chunk
        finally:
            body.close()

    def delete(self, key: str) -> None:
        self.client.delete_object(Bucket=self.bucket, Key=validate_storage_key(key))


def create_storage(settings):
    if settings.storage_backend == "s3":
        return S3ArtifactStorage(settings)
    if settings.environment not in {"development", "test"}:
        raise RuntimeError("Local storage is allowed only in development and tests")
    return LocalArtifactStorage(settings.local_storage_path)
