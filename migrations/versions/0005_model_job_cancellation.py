"""Add cooperative cancellation flag to model jobs."""
from alembic import op
import sqlalchemy as sa

revision = "0005_model_job_cancellation"
down_revision = "0004_model_job_leases"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        "model_jobs",
        sa.Column("cancel_requested", sa.Boolean(), nullable=False, server_default=sa.false()),
    )


def downgrade():
    op.drop_column("model_jobs", "cancel_requested")
