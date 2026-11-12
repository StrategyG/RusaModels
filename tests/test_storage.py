import hashlib
import io
from types import SimpleNamespace

import pytest
from pydantic import SecretStr

from rusamodels.storage import (
    LocalArtifactStorage,
    S3ArtifactStorage,
    validate_storage_key,
)


def test_write_checks_hash_and_is_immutable(tmp_path):
    store = LocalArtifactStorage(tmp_path / "objects")
    data = b"small model artifact"
    result = store.write_bytes("models/demo/v1/manifest.json", data)
    assert result.sha256 == hashlib.sha256(data).hexdigest()
    assert store.read_bytes(result.key) == data
    with pytest.raises(FileExistsError):
        store.write_bytes(result.key, b"replacement")


@pytest.mark.parametrize("key", ["../secret", "/absolute", "models/../../secret", r"a\\..\\b"])
def test_rejects_path_traversal(key):
    with pytest.raises(ValueError):
        validate_storage_key(key)


def test_rejects_checksum_mismatch(tmp_path):
    with pytest.raises(ValueError, match="SHA-256"):
        LocalArtifactStorage(tmp_path).write_bytes("artifact.bin", b"data", "0" * 64)



def test_s3_storage_separates_read_and_write_credentials(monkeypatch):
    clients = []

    class FakeS3Client:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.calls = []
            clients.append(self)

        def get_object(self, **kwargs):
            self.calls.append(("get_object", kwargs))
            return {"Body": io.BytesIO(b"artifact")}

        def put_object(self, **kwargs):
            self.calls.append(("put_object", kwargs))

    monkeypatch.setattr(
        "rusamodels.storage.boto3.client",
        lambda service, **kwargs: FakeS3Client(**kwargs),
    )
    settings = SimpleNamespace(
        s3_bucket="models",
        s3_endpoint_url="https://s3.example",
        s3_region="us-east-1",
        s3_access_key_id="read-key",
        s3_secret_access_key=SecretStr("read-secret"),
        s3_write_access_key_id="write-key",
        s3_write_secret_access_key=SecretStr("write-secret"),
    )

    storage = S3ArtifactStorage(settings)
    assert storage.read_client is not storage.write_client
    assert storage.read_client.kwargs["aws_access_key_id"] == "read-key"
    assert storage.write_client.kwargs["aws_access_key_id"] == "write-key"

    assert storage.read_bytes("models/demo/v1/model.gguf") == b"artifact"
    storage.write_bytes("models/demo/v1/manifest.json", b"manifest")

    assert [name for name, _ in storage.read_client.calls] == ["get_object"]
    assert [name for name, _ in storage.write_client.calls] == ["put_object"]


def test_s3_storage_reuses_workload_identity_when_write_key_is_absent(monkeypatch):
    clients = []

    monkeypatch.setattr(
        "rusamodels.storage.boto3.client",
        lambda service, **kwargs: clients.append(kwargs) or object(),
    )
    settings = SimpleNamespace(
        s3_bucket="models",
        s3_endpoint_url=None,
        s3_region="us-east-1",
        s3_access_key_id=None,
        s3_secret_access_key=None,
        s3_write_access_key_id=None,
        s3_write_secret_access_key=None,
    )
    storage = S3ArtifactStorage(settings)
    assert storage.read_client is storage.write_client
    assert len(clients) == 1
