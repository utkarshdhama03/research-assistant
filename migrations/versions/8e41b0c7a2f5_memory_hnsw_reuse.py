"""memory: HNSW index on chunks.embedding, sources.reused_from_source_id

Revision ID: 8e41b0c7a2f5
Revises: 3c9a7f21d4b8
Create Date: 2026-11-19

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "8e41b0c7a2f5"
down_revision: str | Sequence[str] | None = "3c9a7f21d4b8"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("sources", sa.Column("reused_from_source_id", sa.UUID(), nullable=True))
    op.create_index(
        "ix_chunks_embedding_hnsw",
        "chunks",
        ["embedding"],
        unique=False,
        postgresql_using="hnsw",
        postgresql_with={"m": 16, "ef_construction": 64},
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )


def downgrade() -> None:
    op.drop_index("ix_chunks_embedding_hnsw", table_name="chunks", postgresql_using="hnsw")
    op.drop_column("sources", "reused_from_source_id")
