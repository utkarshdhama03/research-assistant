"""sub_questions.premise_check: the first sub-question may check the question's premise

Revision ID: b5d2e8f04a17
Revises: 8e41b0c7a2f5
Create Date: 2026-11-21

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "b5d2e8f04a17"
down_revision: str | Sequence[str] | None = "8e41b0c7a2f5"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "sub_questions",
        sa.Column("premise_check", sa.Boolean(), server_default=sa.text("false"), nullable=False),
    )


def downgrade() -> None:
    op.drop_column("sub_questions", "premise_check")
