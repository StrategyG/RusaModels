"""Initial registry, version, artifact and append-only audit schema."""
from alembic import op
import sqlalchemy as sa

revision = "0001_platform_foundation"
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "model_catalog",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("publisher", sa.String(255)),
        sa.Column("architecture", sa.String(128)),
        sa.Column("license_id", sa.String(128)),
        sa.Column("source_uri", sa.String(2048)),
        sa.Column("status", sa.String(32), nullable=False, server_default="active"),
        sa.Column("created_by", sa.String(255), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_model_catalog_name", "model_catalog", ["name"])
    op.create_table(
        "model_versions",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("model_id", sa.String(36), sa.ForeignKey("model_catalog.id", ondelete="RESTRICT"), nullable=False),
        sa.Column("version", sa.String(128), nullable=False),
        sa.Column("source_revision", sa.String(255)),
        sa.Column("precision", sa.String(32)),
        sa.Column("status", sa.String(32), nullable=False, server_default="registered"),
        sa.Column("manifest_sha256", sa.String(64)),
        sa.Column("created_by", sa.String(255), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.UniqueConstraint("model_id", "version", name="uq_model_version"),
    )
    op.create_table(
        "artifacts",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("version_id", sa.String(36), sa.ForeignKey("model_versions.id", ondelete="RESTRICT"), nullable=False),
        sa.Column("storage_key", sa.String(1024), nullable=False, unique=True),
        sa.Column("media_type", sa.String(255)),
        sa.Column("size_bytes", sa.Integer(), nullable=False),
        sa.Column("sha256", sa.String(64), nullable=False),
        sa.Column("immutable", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_artifacts_version_id", "artifacts", ["version_id"])
    op.create_table(
        "audit_events",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("actor_subject", sa.String(255), nullable=False),
        sa.Column("actor_role", sa.String(32), nullable=False),
        sa.Column("action", sa.String(128), nullable=False),
        sa.Column("resource", sa.String(1024), nullable=False),
        sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("outcome", sa.String(32), nullable=False),
        sa.Column("request_id", sa.String(128), nullable=False),
        sa.Column("metadata", sa.JSON(), nullable=False),
    )
    for name, column in [
        ("ix_audit_events_occurred_at", "occurred_at"),
        ("ix_audit_events_actor_subject", "actor_subject"),
        ("ix_audit_events_request_id", "request_id"),
    ]:
        op.create_index(name, "audit_events", [column])
    dialect = op.get_bind().dialect.name
    if dialect == "postgresql":
        op.execute("""
        CREATE FUNCTION reject_audit_event_mutation() RETURNS trigger AS $$
        BEGIN RAISE EXCEPTION 'audit_events is append-only'; END;
        $$ LANGUAGE plpgsql;
        """)
        for operation in ("UPDATE", "DELETE"):
            op.execute(f"""
            CREATE TRIGGER trg_audit_events_no_{operation.lower()}
            BEFORE {operation} ON audit_events
            FOR EACH ROW EXECUTE FUNCTION reject_audit_event_mutation();
            """)
    elif dialect == "sqlite":
        for operation in ("UPDATE", "DELETE"):
            op.execute(f"""
            CREATE TRIGGER trg_audit_events_no_{operation.lower()}
            BEFORE {operation} ON audit_events
            BEGIN SELECT RAISE(ABORT, 'audit_events is append-only'); END;
            """)


def downgrade():
    dialect = op.get_bind().dialect.name
    if dialect == "postgresql":
        op.execute("DROP TRIGGER IF EXISTS trg_audit_events_no_update ON audit_events")
        op.execute("DROP TRIGGER IF EXISTS trg_audit_events_no_delete ON audit_events")
        op.execute("DROP FUNCTION IF EXISTS reject_audit_event_mutation()")
    elif dialect == "sqlite":
        op.execute("DROP TRIGGER IF EXISTS trg_audit_events_no_update")
        op.execute("DROP TRIGGER IF EXISTS trg_audit_events_no_delete")
    for name in ("ix_audit_events_request_id", "ix_audit_events_actor_subject", "ix_audit_events_occurred_at"):
        op.drop_index(name, table_name="audit_events")
    op.drop_table("audit_events")
    op.drop_index("ix_artifacts_version_id", table_name="artifacts")
    op.drop_table("artifacts")
    op.drop_table("model_versions")
    op.drop_index("ix_model_catalog_name", table_name="model_catalog")
    op.drop_table("model_catalog")
