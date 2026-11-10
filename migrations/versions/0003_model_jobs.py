"""Add persistent model jobs for downloads and conversions."""
from alembic import op
import sqlalchemy as sa

revision = "0003_model_jobs"
down_revision = "0002_resumable_uploads"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "model_jobs",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("model_id", sa.String(36), sa.ForeignKey("model_catalog.id", ondelete="RESTRICT"), nullable=False),
        sa.Column("input_version_id", sa.String(36), sa.ForeignKey("model_versions.id", ondelete="RESTRICT")),
        sa.Column("output_version_id", sa.String(36), sa.ForeignKey("model_versions.id", ondelete="RESTRICT")),
        sa.Column("job_type", sa.String(32), nullable=False),
        sa.Column("parameters", sa.JSON(), nullable=False),
        sa.Column("status", sa.String(32), nullable=False, server_default="queued"),
        sa.Column("progress_percent", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("error_code", sa.String(128)),
        sa.Column("created_by", sa.String(255), nullable=False),
        sa.Column("created_by_role", sa.String(32), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("started_at", sa.DateTime(timezone=True)),
        sa.Column("finished_at", sa.DateTime(timezone=True)),
    )
    op.create_index("ix_model_jobs_status_created_at", "model_jobs", ["status", "created_at"])
    op.create_index("ix_model_jobs_created_by", "model_jobs", ["created_by"])


def downgrade():
    op.drop_index("ix_model_jobs_created_by", table_name="model_jobs")
    op.drop_index("ix_model_jobs_status_created_at", table_name="model_jobs")
    op.drop_table("model_jobs")
