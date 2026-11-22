"""graph v1 columns: extracted text, sub-question queries, report citations

Revision ID: 069c24a3c8f9
Revises: b077085a6e91
Create Date: 2026-11-17 00:06:42.075631

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "069c24a3c8f9"
down_revision: str | Sequence[str] | None = "b077085a6e91"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # chunk char offsets refer to this text
    op.add_column("sources", sa.Column("extracted_text", sa.Text(), nullable=True))
    op.add_column(
        "sub_questions",
        sa.Column(
            "search_queries",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'[]'::jsonb"),
            nullable=False,
        ),
    )
    # one entry per [n] in the report: {"n", "source_id", "chunk_ids"}
    op.add_column(
        "reports",
        sa.Column(
            "citations",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'[]'::jsonb"),
            nullable=False,
        ),
    )


def downgrade() -> None:
    op.drop_column("reports", "citations")
    op.drop_column("sub_questions", "search_queries")
    op.drop_column("sources", "extracted_text")
