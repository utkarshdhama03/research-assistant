"""claims.ordinal: creation order of claims within a run

Revision ID: e6b36bf2589f
Revises: 069c24a3c8f9
Create Date: 2026-11-17

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "e6b36bf2589f"
down_revision: str | Sequence[str] | None = "069c24a3c8f9"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("claims", sa.Column("ordinal", sa.Integer(), nullable=True))
    # Existing claims (if any) get 0, 1, ... per run; their insertion order is unknown, so id order.
    op.execute(
        """
        UPDATE claims c SET ordinal = n.rn
        FROM (SELECT id, row_number() OVER (PARTITION BY run_id ORDER BY id) - 1 AS rn FROM claims) n
        WHERE c.id = n.id
        """
    )
    op.alter_column("claims", "ordinal", nullable=False)
    op.create_unique_constraint(op.f("uq_claims_run_id_ordinal"), "claims", ["run_id", "ordinal"])


def downgrade() -> None:
    op.drop_constraint(op.f("uq_claims_run_id_ordinal"), "claims", type_="unique")
    op.drop_column("claims", "ordinal")
