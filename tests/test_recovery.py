import hashlib
import shutil
import sqlite3

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from rusamodels.manifests import ModelManifest, canonical_manifest_sha256
from rusamodels.models import Artifact, Base, ModelCatalog, ModelVersion
from rusamodels.recovery import verify_artifact_inventory
from rusamodels.storage import LocalArtifactStorage


def test_sqlite_and_local_artifact_backup_restore_is_verifiable(tmp_path):
    source_db = tmp_path / "source.db"
    backup_db = tmp_path / "backup.db"
    source_storage_root = tmp_path / "source-objects"
    backup_storage_root = tmp_path / "backup-objects"
    source_engine = create_engine(f"sqlite:///{source_db}")
    Base.metadata.create_all(source_engine)
    source_storage = LocalArtifactStorage(source_storage_root)

    payload = b"artifact included in the recovery snapshot"
    key = "models/model-1/version-1/model.bin"
    stored = source_storage.write_bytes(key, payload)
    with Session(source_engine) as session:
        session.add(ModelCatalog(
            id="model-1", name="Recovery test", status="active", created_by="test"
        ))
        version = ModelVersion(
            id="version-1", model_id="model-1", version="v1", created_by="test"
        )
        artifact = Artifact(
            id="artifact-1",
            version_id="version-1",
            storage_key=stored.key,
            original_filename="model.bin",
            media_type="application/octet-stream",
            size_bytes=stored.size_bytes,
            sha256=stored.sha256,
            created_by="test",
            provenance={"origin": "recovery-test"},
            retention_class="standard",
            immutable=True,
        )
        manifest = ModelManifest.model_validate({
            "schema_version": "1.0",
            "model_id": "model-1",
            "model_version": "v1",
            "model_type": "causal_lm",
            "architecture": "decoder_only_transformer",
            "tokenizer": {"id": "rusa-tokenizer", "version": "v1"},
            "source": {
                "repository": "StrategyG/RusaModels",
                "revision": "a" * 40,
                "license": "apache-2.0",
            },
            "artifacts": [{
                "name": "model.bin",
                "format": "safetensors",
                "size_bytes": stored.size_bytes,
                "sha256": stored.sha256,
            }],
        })
        version.manifest_json = manifest.model_dump(mode="json")
        version.manifest_sha256 = canonical_manifest_sha256(manifest)
        session.add_all([version, artifact])
        session.commit()

    source_connection = sqlite3.connect(source_db)
    backup_connection = sqlite3.connect(backup_db)
    try:
        source_connection.backup(backup_connection)
    finally:
        backup_connection.close()
        source_connection.close()
    shutil.copytree(source_storage_root, backup_storage_root)
    source_engine.dispose()

    restored_engine = create_engine(f"sqlite:///{backup_db}")
    restored_storage = LocalArtifactStorage(backup_storage_root)
    with Session(restored_engine) as session:
        report = verify_artifact_inventory(session, restored_storage)
    restored_engine.dispose()

    assert report == {
        "artifact_count": 1,
        "verified_count": 1,
        "model_manifest_count": 1,
        "verified_manifest_count": 1,
        "failed_count": 0,
        "errors": [],
    }


def test_recovery_verifier_reports_missing_or_tampered_artifacts(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'recovery.db'}")
    Base.metadata.create_all(engine)
    storage = LocalArtifactStorage(tmp_path / "objects")
    payload = b"original"
    stored = storage.write_bytes("models/m/v/a.bin", payload)
    with Session(engine) as session:
        session.add(ModelCatalog(
            id="m", name="Recovery mismatch", status="active", created_by="test"
        ))
        session.add(ModelVersion(
            id="v", model_id="m", version="v1", created_by="test"
        ))
        session.add(Artifact(
            id="a",
            version_id="v",
            storage_key=stored.key,
            original_filename="a.bin",
            media_type="application/octet-stream",
            size_bytes=len(payload),
            sha256=hashlib.sha256(payload).hexdigest(),
            created_by="test",
            provenance={"origin": "recovery-test"},
            retention_class="standard",
            immutable=True,
        ))
        session.commit()
        (storage.root / stored.key).write_bytes(b"tampered")
        report = verify_artifact_inventory(session, storage)
    engine.dispose()

    assert report["artifact_count"] == 1
    assert report["verified_count"] == 0
    assert report["model_manifest_count"] == 0
    assert report["verified_manifest_count"] == 0
    assert report["failed_count"] == 1
    assert report["errors"][0]["error"] == "integrity_mismatch"
