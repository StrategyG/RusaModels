import hashlib
import json

from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import DBAPIError


def test_migration_and_audit_append_only(tmp_path, monkeypatch):
    url = f"sqlite:///{tmp_path / 'migration.db'}"
    monkeypatch.setenv("RUSAMODELS_ENV", "test")
    monkeypatch.setenv("DATABASE_URL", url)
    from rusamodels.config import get_settings
    get_settings.cache_clear()
    command.upgrade(Config("alembic.ini"), "head")
    engine = create_engine(url)
    tables = set(inspect(engine).get_table_names())
    assert {"model_catalog", "model_versions", "artifacts", "upload_sessions",
            "upload_parts", "model_jobs", "audit_events", "alembic_version"} <= tables
    with engine.connect() as connection:
        version_columns = {
            column["name"] for column in inspect(engine).get_columns("model_versions")
        }
        artifact_columns = {
            column["name"] for column in inspect(engine).get_columns("artifacts")
        }
        columns = {column["name"] for column in inspect(engine).get_columns("model_jobs")}
    assert {"manifest_json", "manifest_sha256", "artifact_index_sha256",
            "lifecycle_status", "approved_by", "approved_at",
            "supersedes_version_id"} <= version_columns
    version_indexes = {
        index["name"] for index in inspect(engine).get_indexes("model_versions")
    }
    assert "uq_model_versions_single_approved" in version_indexes
    assert {"created_by", "source_job_id", "provenance", "retention_class",
            "retention_until", "legal_hold", "verified_at", "deleted_at"} <= artifact_columns
    assert {"attempt_count", "max_attempts", "lease_owner", "lease_epoch",
            "lease_expires_at", "last_heartbeat_at", "next_attempt_at",
            "cancel_requested"} <= columns
    with engine.begin() as connection:
        connection.execute(text(
            "INSERT INTO audit_events (id, actor_subject, actor_role, action, resource, outcome, request_id, metadata) "
            "VALUES ('a1','ci','administrator','test','/','success','req-1','{}')"))
    try:
        with engine.begin() as connection:
            connection.execute(text("UPDATE audit_events SET outcome='changed' WHERE id='a1'"))
    except DBAPIError:
        pass
    else:
        raise AssertionError("audit UPDATE was not blocked")
    with engine.begin() as connection:
        connection.execute(text(
            "INSERT INTO model_catalog (id, name, status, created_by) "
            "VALUES ('m1', 'manifest-test', 'active', 'ci')"
        ))
        connection.execute(text(
            "INSERT INTO model_versions "
            "(id, model_id, version, status, lifecycle_status, created_by) "
            "VALUES ('v1', 'm1', 'v1', 'registered', 'draft', 'ci')"
        ))
        connection.execute(text(
            "UPDATE model_versions SET manifest_json='{\"schema_version\":\"1.0\"}', "
            "manifest_sha256='" + "a" * 64 + "' WHERE id='v1'"
        ))
        connection.execute(text(
            "INSERT INTO artifacts "
            "(id, version_id, storage_key, original_filename, media_type, size_bytes, "
            "sha256, immutable, created_by, provenance, retention_class, legal_hold) "
            "VALUES ('a1', 'v1', 'models/m1/v1/model.bin', 'model.bin', "
            "'application/octet-stream', 1, '" + "b" * 64 + "', 1, 'ci', '{}', 'standard', 0)"
        ))
    try:
        with engine.begin() as connection:
            connection.execute(text(
                "UPDATE model_versions SET manifest_sha256='" + "c" * 64 + "' WHERE id='v1'"
            ))
    except DBAPIError:
        pass
    else:
        raise AssertionError("model manifest mutation was not blocked")
    try:
        with engine.begin() as connection:
            connection.execute(text(
                "UPDATE artifacts SET sha256='" + "d" * 64 + "' WHERE id='a1'"
            ))
    except DBAPIError:
        pass
    else:
        raise AssertionError("artifact identity mutation was not blocked")
    with engine.begin() as connection:
        connection.execute(text("UPDATE artifacts SET legal_hold=1 WHERE id='a1'"))
        connection.execute(text("UPDATE artifacts SET deleted_at=CURRENT_TIMESTAMP WHERE id='a1'"))
    try:
        with engine.begin() as connection:
            connection.execute(text("UPDATE artifacts SET deleted_at=NULL WHERE id='a1'"))
    except DBAPIError:
        pass
    else:
        raise AssertionError("artifact purge tombstone was reversible")

    engine.dispose()
    get_settings.cache_clear()



def test_legacy_manifest_digest_is_classified_without_losing_expected_digest(
    tmp_path, monkeypatch
):
    url = f"sqlite:///{tmp_path / 'legacy-digest.db'}"
    monkeypatch.setenv("RUSAMODELS_ENV", "test")
    monkeypatch.setenv("DATABASE_URL", url)
    from rusamodels.config import get_settings

    get_settings.cache_clear()
    config = Config("alembic.ini")
    command.upgrade(config, "0005_model_job_cancellation")
    engine = create_engine(url)

    legacy_artifacts = [{
        "filename": "model.bin",
        "size_bytes": 4,
        "sha256": "b" * 64,
    }]
    artifact_index_digest = hashlib.sha256(
        json.dumps(
            legacy_artifacts, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()
    expected_model_manifest_digest = "c" * 64

    with engine.begin() as connection:
        connection.execute(text(
            "INSERT INTO model_catalog (id, name, status, created_by) "
            "VALUES ('m1', 'legacy-digest', 'active', 'ci')"
        ))
        connection.execute(text(
            "INSERT INTO model_versions "
            "(id, model_id, version, status, manifest_sha256, created_by) "
            "VALUES ('v1', 'm1', 'v1', 'ready', :digest, 'ci')"
        ), {"digest": artifact_index_digest})
        connection.execute(text(
            "INSERT INTO artifacts "
            "(id, version_id, storage_key, original_filename, media_type, size_bytes, "
            "sha256, immutable) VALUES "
            "('a1', 'v1', 'models/m1/v1/model.bin', 'model.bin', "
            "'application/octet-stream', 4, :sha256, 1)"
        ), {"sha256": "b" * 64})
        connection.execute(text(
            "INSERT INTO model_versions "
            "(id, model_id, version, status, manifest_sha256, created_by) "
            "VALUES ('v2', 'm1', 'v2', 'registered', :digest, 'ci')"
        ), {"digest": expected_model_manifest_digest})
    engine.dispose()

    command.upgrade(config, "head")
    engine = create_engine(url)
    with engine.connect() as connection:
        migrated = connection.execute(text(
            "SELECT id, manifest_sha256, artifact_index_sha256 "
            "FROM model_versions ORDER BY id"
        )).mappings().all()
    by_id = {item["id"]: item for item in migrated}

    assert by_id["v1"]["manifest_sha256"] is None
    assert by_id["v1"]["artifact_index_sha256"] == artifact_index_digest
    assert by_id["v2"]["manifest_sha256"] == expected_model_manifest_digest
    assert by_id["v2"]["artifact_index_sha256"] is None

    engine.dispose()
    get_settings.cache_clear()
