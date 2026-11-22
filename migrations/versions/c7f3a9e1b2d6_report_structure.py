"""reports.structure: the report as data (sections, sentences with markers, references, claims)

Revision ID: c7f3a9e1b2d6
Revises: b5d2e8f04a17
Create Date: 2026-11-22

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "c7f3a9e1b2d6"
down_revision: str | Sequence[str] | None = "b5d2e8f04a17"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Nullable: reports written before this revision have Markdown only.
    op.add_column("reports", sa.Column("structure", postgresql.JSONB(), nullable=True))


def downgrade() -> None:
    op.drop_column("reports", "structure")
