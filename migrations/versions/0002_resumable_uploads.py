"""Add resumable upload sessions and part metadata."""
from alembic import op
import sqlalchemy as sa

revision = "0002_resumable_uploads"
down_revision = "0001_platform_foundation"
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table("artifacts") as batch:
        batch.add_column(sa.Column("original_filename", sa.String(255), nullable=True))
        batch.alter_column("size_bytes", existing_type=sa.Integer(), type_=sa.BigInteger())
    op.create_table(
        "upload_sessions",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("version_id", sa.String(36), sa.ForeignKey("model_versions.id", ondelete="RESTRICT"), nullable=False),
        sa.Column("storage_key", sa.String(1024), nullable=False, unique=True),
        sa.Column("storage_upload_id", sa.String(255), nullable=False),
        sa.Column("original_filename", sa.String(255), nullable=False),
        sa.Column("media_type", sa.String(255), nullable=False),
        sa.Column("expected_size", sa.BigInteger(), nullable=False),
        sa.Column("expected_sha256", sa.String(64), nullable=False),
        sa.Column("status", sa.String(32), nullable=False, server_default="open"),
        sa.Column("created_by", sa.String(255), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_upload_sessions_version_id", "upload_sessions", ["version_id"])
    op.create_table(
        "upload_parts",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("upload_id", sa.String(36), sa.ForeignKey("upload_sessions.id", ondelete="CASCADE"), nullable=False),
        sa.Column("part_number", sa.Integer(), nullable=False),
        sa.Column("etag", sa.String(255), nullable=False),
        sa.Column("size_bytes", sa.BigInteger(), nullable=False),
        sa.Column("sha256", sa.String(64), nullable=False),
        sa.UniqueConstraint("upload_id", "part_number", name="uq_upload_part_number"),
    )


def downgrade():
    op.drop_table("upload_parts")
    op.drop_index("ix_upload_sessions_version_id", table_name="upload_sessions")
    op.drop_table("upload_sessions")
    with op.batch_alter_table("artifacts") as batch:
        batch.drop_column("original_filename")
        batch.alter_column("size_bytes", existing_type=sa.BigInteger(), type_=sa.Integer())
