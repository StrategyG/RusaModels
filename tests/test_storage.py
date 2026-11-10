import hashlib
import pytest
from rusamodels.storage import LocalArtifactStorage, validate_storage_key


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
