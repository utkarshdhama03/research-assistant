"""queue, plan approval and worker: new run statuses, auto_approve, cancel_requested, worker_heartbeats

Revision ID: 3c9a7f21d4b8
Revises: e6b36bf2589f
Create Date: 2026-11-19

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "3c9a7f21d4b8"
down_revision: str | Sequence[str] | None = "e6b36bf2589f"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

OLD_STATUSES = ("pending", "running", "completed", "failed", "partial")
NEW_STATUSES = ("queued", "awaiting_approval", "cancelled")


def _status_check(values: Sequence[str]) -> str:
    return "status IN (" + ", ".join(f"'{v}'" for v in values) + ")"


def upgrade() -> None:
    op.drop_constraint(op.f("ck_research_runs_status"), "research_runs", type_="check")
    op.create_check_constraint(
        op.f("ck_research_runs_status"), "research_runs", _status_check(OLD_STATUSES + NEW_STATUSES)
    )
    op.add_column(
        "research_runs",
        sa.Column("auto_approve", sa.Boolean(), server_default=sa.text("false"), nullable=False),
    )
    op.add_column(
        "research_runs",
        sa.Column("cancel_requested", sa.Boolean(), server_default=sa.text("false"), nullable=False),
    )
    op.create_index(
        op.f("ix_research_runs_created_at_id"), "research_runs", ["created_at", "id"], unique=False
    )
    op.create_table(
        "worker_heartbeats",
        sa.Column("id", sa.UUID(), server_default=sa.text("gen_random_uuid()"), nullable=False),
        sa.Column("pid", sa.Integer(), nullable=False),
        sa.Column(
            "started_at", postgresql.TIMESTAMP(timezone=True), server_default=sa.text("now()"), nullable=False
        ),
        sa.Column(
            "last_seen", postgresql.TIMESTAMP(timezone=True), server_default=sa.text("now()"), nullable=False
        ),
        sa.Column("current_run_id", sa.UUID(), nullable=True),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_worker_heartbeats")),
    )


def downgrade() -> None:
    op.drop_table("worker_heartbeats")
    op.drop_index(op.f("ix_research_runs_created_at_id"), table_name="research_runs")
    op.drop_column("research_runs", "cancel_requested")
    op.drop_column("research_runs", "auto_approve")
    # Rows in a status the old constraint does not know become failed, with the old status recorded.
    op.execute(
        "UPDATE research_runs SET error = 'downgraded from ' || status, status = 'failed', "
        "finished_at = COALESCE(finished_at, now()) "
        f"WHERE status IN ({', '.join(repr(s) for s in NEW_STATUSES)})"
    )
    op.drop_constraint(op.f("ck_research_runs_status"), "research_runs", type_="check")
    op.create_check_constraint(op.f("ck_research_runs_status"), "research_runs", _status_check(OLD_STATUSES))
