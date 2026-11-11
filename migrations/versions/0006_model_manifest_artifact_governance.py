"""Add immutable model manifests, lifecycle and artifact retention metadata."""
import hashlib
import json

from alembic import op
import sqlalchemy as sa

revision = "0006_model_manifest_artifact_governance"
down_revision = "0005_model_job_cancellation"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        "model_versions", sa.Column("artifact_index_sha256", sa.String(64), nullable=True)
    )
    # Older workers used manifest_sha256 for a canonical file-list digest.
    # Move only values that can be reproduced from the stored artifact inventory;
    # preserve other pre-registered digests as expected model-manifest digests.
    bind = op.get_bind()
    versions = bind.execute(
        sa.text(
            "SELECT id, manifest_sha256 FROM model_versions "
            "WHERE manifest_sha256 IS NOT NULL"
        )
    ).mappings().all()
    for version in versions:
        artifact_rows = bind.execute(
            sa.text(
                "SELECT original_filename, size_bytes, sha256 FROM artifacts "
                "WHERE version_id = :version_id"
            ),
            {"version_id": version["id"]},
        ).mappings().all()
        legacy_index = [
            {
                "filename": item["original_filename"],
                "size_bytes": item["size_bytes"],
                "sha256": item["sha256"],
            }
            for item in sorted(
                artifact_rows, key=lambda item: item["original_filename"] or ""
            )
        ]
        legacy_digest = hashlib.sha256(
            json.dumps(legacy_index, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        if legacy_digest == version["manifest_sha256"].lower():
            bind.execute(
                sa.text(
                    "UPDATE model_versions SET artifact_index_sha256 = :digest, "
                    "manifest_sha256 = NULL WHERE id = :version_id"
                ),
                {"digest": legacy_digest, "version_id": version["id"]},
            )
    op.add_column("model_versions", sa.Column("manifest_json", sa.JSON(), nullable=True))
    op.add_column(
        "model_versions",
        sa.Column("lifecycle_status", sa.String(32), nullable=False, server_default="draft"),
    )
    op.add_column("model_versions", sa.Column("approved_by", sa.String(255), nullable=True))
    op.add_column(
        "model_versions", sa.Column("approved_at", sa.DateTime(timezone=True), nullable=True)
    )
    op.add_column(
        "model_versions",
        sa.Column(
            "supersedes_version_id",
            sa.String(36),
            sa.ForeignKey("model_versions.id", ondelete="RESTRICT"),
            nullable=True,
        ),
    )
    op.create_index(
        "ix_model_versions_lifecycle_status", "model_versions", ["lifecycle_status"]
    )
    op.create_index(
        "uq_model_versions_single_approved",
        "model_versions",
        ["model_id"],
        unique=True,
        sqlite_where=sa.text("lifecycle_status = 'approved'"),
        postgresql_where=sa.text("lifecycle_status = 'approved'"),
    )

    op.add_column("artifacts", sa.Column("created_by", sa.String(255), nullable=True))
    op.add_column(
        "artifacts",
        sa.Column(
            "source_job_id",
            sa.String(36),
            sa.ForeignKey("model_jobs.id", ondelete="RESTRICT"),
            nullable=True,
        ),
    )
    op.add_column(
        "artifacts",
        sa.Column(
            "provenance",
            sa.JSON(),
            nullable=False,
            server_default=sa.text("'{}'"),
        ),
    )
    op.add_column(
        "artifacts",
        sa.Column("retention_class", sa.String(32), nullable=False, server_default="standard"),
    )
    op.add_column(
        "artifacts",
        sa.Column("retention_until", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "artifacts",
        sa.Column("legal_hold", sa.Boolean(), nullable=False, server_default=sa.false()),
    )
    op.add_column(
        "artifacts", sa.Column("verified_at", sa.DateTime(timezone=True), nullable=True)
    )
    op.add_column("artifacts", sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True))
    op.create_index(
        "ix_artifacts_retention_due",
        "artifacts",
        ["retention_until", "legal_hold", "deleted_at"],
    )
    dialect = op.get_bind().dialect.name
    if dialect == "postgresql":
        op.execute("""
        CREATE FUNCTION reject_model_manifest_mutation() RETURNS trigger AS $function$
        BEGIN
            IF OLD.manifest_json IS NOT NULL AND (
                NEW.manifest_json::text IS DISTINCT FROM OLD.manifest_json::text
                OR NEW.manifest_sha256 IS DISTINCT FROM OLD.manifest_sha256
            ) THEN
                RAISE EXCEPTION 'model version manifest is immutable';
            END IF;
            RETURN NEW;
        END;
        $function$ LANGUAGE plpgsql;
        """)
        op.execute("""
        CREATE TRIGGER trg_model_manifest_immutable
        BEFORE UPDATE ON model_versions
        FOR EACH ROW EXECUTE FUNCTION reject_model_manifest_mutation();
        """)
        op.execute("""
        CREATE FUNCTION reject_artifact_identity_mutation() RETURNS trigger AS $function$
        BEGIN
            IF NEW.version_id IS DISTINCT FROM OLD.version_id
                OR NEW.storage_key IS DISTINCT FROM OLD.storage_key
                OR NEW.original_filename IS DISTINCT FROM OLD.original_filename
                OR NEW.media_type IS DISTINCT FROM OLD.media_type
                OR NEW.size_bytes IS DISTINCT FROM OLD.size_bytes
                OR NEW.sha256 IS DISTINCT FROM OLD.sha256
                OR NEW.created_by IS DISTINCT FROM OLD.created_by
                OR NEW.source_job_id IS DISTINCT FROM OLD.source_job_id
                OR NEW.provenance::text IS DISTINCT FROM OLD.provenance::text
                OR NEW.immutable IS DISTINCT FROM OLD.immutable
            THEN
                RAISE EXCEPTION 'artifact identity and provenance are immutable';
            END IF;
            IF OLD.deleted_at IS NOT NULL AND NEW.deleted_at IS NULL THEN
                RAISE EXCEPTION 'artifact purge tombstones are immutable';
            END IF;
            RETURN NEW;
        END;
        $function$ LANGUAGE plpgsql;
        """)
        op.execute("""
        CREATE TRIGGER trg_artifact_identity_immutable
        BEFORE UPDATE ON artifacts
        FOR EACH ROW EXECUTE FUNCTION reject_artifact_identity_mutation();
        """)
    elif dialect == "sqlite":
        op.execute("""
        CREATE TRIGGER trg_model_manifest_immutable
        BEFORE UPDATE ON model_versions
        WHEN OLD.manifest_json IS NOT NULL AND (
            NEW.manifest_json IS NOT OLD.manifest_json
            OR NEW.manifest_sha256 IS NOT OLD.manifest_sha256
        )
        BEGIN
            SELECT RAISE(ABORT, 'model version manifest is immutable');
        END;
        """)
        op.execute("""
        CREATE TRIGGER trg_artifact_identity_immutable
        BEFORE UPDATE ON artifacts
        WHEN NEW.version_id IS NOT OLD.version_id
            OR NEW.storage_key IS NOT OLD.storage_key
            OR NEW.original_filename IS NOT OLD.original_filename
            OR NEW.media_type IS NOT OLD.media_type
            OR NEW.size_bytes IS NOT OLD.size_bytes
            OR NEW.sha256 IS NOT OLD.sha256
            OR NEW.created_by IS NOT OLD.created_by
            OR NEW.source_job_id IS NOT OLD.source_job_id
            OR NEW.provenance IS NOT OLD.provenance
            OR NEW.immutable IS NOT OLD.immutable
            OR (OLD.deleted_at IS NOT NULL AND NEW.deleted_at IS NULL)
        BEGIN
            SELECT RAISE(ABORT, 'artifact identity and provenance are immutable');
        END;
        """)


def downgrade():
    dialect = op.get_bind().dialect.name
    if dialect == "postgresql":
        op.execute("DROP TRIGGER IF EXISTS trg_artifact_identity_immutable ON artifacts")
        op.execute("DROP FUNCTION IF EXISTS reject_artifact_identity_mutation()")
        op.execute("DROP TRIGGER IF EXISTS trg_model_manifest_immutable ON model_versions")
        op.execute("DROP FUNCTION IF EXISTS reject_model_manifest_mutation()")
    elif dialect == "sqlite":
        op.execute("DROP TRIGGER IF EXISTS trg_artifact_identity_immutable")
        op.execute("DROP TRIGGER IF EXISTS trg_model_manifest_immutable")
    op.drop_index("ix_artifacts_retention_due", table_name="artifacts")
    for column in (
        "deleted_at",
        "verified_at",
        "legal_hold",
        "retention_until",
        "retention_class",
        "provenance",
        "source_job_id",
        "created_by",
    ):
        op.drop_column("artifacts", column)
    op.drop_index("uq_model_versions_single_approved", table_name="model_versions")
    op.drop_index("ix_model_versions_lifecycle_status", table_name="model_versions")
    op.execute(
        "UPDATE model_versions SET manifest_sha256 = artifact_index_sha256 "
        "WHERE manifest_sha256 IS NULL AND artifact_index_sha256 IS NOT NULL"
    )
    for column in (
        "supersedes_version_id",
        "approved_at",
        "approved_by",
        "lifecycle_status",
        "manifest_json",
        "artifact_index_sha256",
    ):
        op.drop_column("model_versions", column)
