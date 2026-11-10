"""Add leases, heartbeats and bounded retries to model jobs."""
from alembic import op
import sqlalchemy as sa

revision = "0004_model_job_leases"
down_revision = "0003_model_jobs"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        "model_jobs",
        sa.Column("attempt_count", sa.Integer(), nullable=False, server_default="0"),
    )
    op.add_column(
        "model_jobs",
        sa.Column("max_attempts", sa.Integer(), nullable=False, server_default="3"),
    )
    op.add_column("model_jobs", sa.Column("lease_owner", sa.String(255), nullable=True))
    op.add_column(
        "model_jobs",
        sa.Column("lease_epoch", sa.Integer(), nullable=False, server_default="0"),
    )
    op.add_column("model_jobs", sa.Column("lease_expires_at", sa.DateTime(timezone=True)))
    op.add_column("model_jobs", sa.Column("last_heartbeat_at", sa.DateTime(timezone=True)))
    op.add_column("model_jobs", sa.Column("next_attempt_at", sa.DateTime(timezone=True)))
    op.create_index("ix_model_jobs_lease_expires_at", "model_jobs", ["lease_expires_at"])


def downgrade():
    op.drop_index("ix_model_jobs_lease_expires_at", table_name="model_jobs")
    op.drop_column("model_jobs", "next_attempt_at")
    op.drop_column("model_jobs", "last_heartbeat_at")
    op.drop_column("model_jobs", "lease_expires_at")
    op.drop_column("model_jobs", "lease_epoch")
    op.drop_column("model_jobs", "lease_owner")
    op.drop_column("model_jobs", "max_attempts")
    op.drop_column("model_jobs", "attempt_count")
