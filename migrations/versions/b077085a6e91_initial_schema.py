"""initial schema

Revision ID: b077085a6e91
Revises:
Create Date: 2026-11-16 16:40:03.508188

"""

from collections.abc import Sequence

import pgvector.sqlalchemy
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "b077085a6e91"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # The Docker init script creates it too; keep the migration self-sufficient.
    # Downgrade leaves the extension in place (other databases/tools may rely on it).
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.create_table(
        "research_runs",
        sa.Column("id", sa.UUID(), server_default=sa.text("gen_random_uuid()"), nullable=False),
        sa.Column("question", sa.Text(), nullable=False),
        sa.Column("status", sa.String(length=32), server_default="pending", nullable=False),
        sa.Column(
            "config",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "status IN ('pending', 'running', 'completed', 'failed', 'partial')",
            name=op.f("ck_research_runs_status"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_research_runs")),
    )
    op.create_table(
        "reports",
        sa.Column("id", sa.UUID(), server_default=sa.text("gen_random_uuid()"), nullable=False),
        sa.Column("run_id", sa.UUID(), nullable=False),
        sa.Column("markdown", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(
            ["run_id"], ["research_runs.id"], name=op.f("fk_reports_run_id_research_runs"), ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_reports")),
    )
    op.create_index(op.f("ix_reports_run_id"), "reports", ["run_id"], unique=False)
    op.create_table(
        "run_events",
        sa.Column("id", sa.UUID(), server_default=sa.text("gen_random_uuid()"), nullable=False),
        sa.Column("run_id", sa.UUID(), nullable=False),
        sa.Column("node", sa.Text(), nullable=False),
        sa.Column("event", sa.Text(), nullable=False),
        sa.Column(
            "payload",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("clock_timestamp()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["run_id"],
            ["research_runs.id"],
            name=op.f("fk_run_events_run_id_research_runs"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_run_events")),
    )
    op.create_index(
        op.f("ix_run_events_run_id_created_at"), "run_events", ["run_id", "created_at"], unique=False
    )
    op.create_table(
        "sources",
        sa.Column("id", sa.UUID(), server_default=sa.text("gen_random_uuid()"), nullable=False),
        sa.Column("run_id", sa.UUID(), nullable=False),
        sa.Column("url", sa.Text(), nullable=False),
        sa.Column("final_url", sa.Text(), nullable=True),
        sa.Column("title", sa.Text(), nullable=True),
        sa.Column("content_type", sa.Text(), nullable=True),
        sa.Column("content_hash", sa.Text(), nullable=True),
        sa.Column("fetched_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("error", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(
            ["run_id"], ["research_runs.id"], name=op.f("fk_sources_run_id_research_runs"), ondelete="CASCADE"
        ),
        sa.CheckConstraint("status IN ('fetched', 'failed', 'skipped')", name=op.f("ck_sources_status")),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_sources")),
        sa.UniqueConstraint("run_id", "url", name=op.f("uq_sources_run_id_url")),
    )
    op.create_index(op.f("ix_sources_content_hash"), "sources", ["content_hash"], unique=False)
    op.create_table(
        "sub_questions",
        sa.Column("id", sa.UUID(), server_default=sa.text("gen_random_uuid()"), nullable=False),
        sa.Column("run_id", sa.UUID(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(
            ["run_id"],
            ["research_runs.id"],
            name=op.f("fk_sub_questions_run_id_research_runs"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_sub_questions")),
        sa.UniqueConstraint("run_id", "ordinal", name=op.f("uq_sub_questions_run_id_ordinal")),
    )
    op.create_table(
        "chunks",
        sa.Column("id", sa.UUID(), server_default=sa.text("gen_random_uuid()"), nullable=False),
        sa.Column("source_id", sa.UUID(), nullable=False),
        sa.Column("run_id", sa.UUID(), nullable=False),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("char_start", sa.Integer(), nullable=False),
        sa.Column("char_end", sa.Integer(), nullable=False),
        sa.Column("embedding", pgvector.sqlalchemy.vector.VECTOR(dim=768), nullable=False),
        sa.CheckConstraint("char_start >= 0 AND char_end >= char_start", name=op.f("ck_chunks_char_range")),
        sa.ForeignKeyConstraint(
            ["run_id"], ["research_runs.id"], name=op.f("fk_chunks_run_id_research_runs"), ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["source_id"], ["sources.id"], name=op.f("fk_chunks_source_id_sources"), ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_chunks")),
        sa.UniqueConstraint("source_id", "ordinal", name=op.f("uq_chunks_source_id_ordinal")),
    )
    op.create_index(op.f("ix_chunks_run_id"), "chunks", ["run_id"], unique=False)
    op.create_table(
        "claims",
        sa.Column("id", sa.UUID(), server_default=sa.text("gen_random_uuid()"), nullable=False),
        sa.Column("run_id", sa.UUID(), nullable=False),
        sa.Column("sub_question_id", sa.UUID(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("verdict", sa.String(length=32), nullable=True),
        sa.Column("rationale", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(
            ["run_id"], ["research_runs.id"], name=op.f("fk_claims_run_id_research_runs"), ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["sub_question_id"],
            ["sub_questions.id"],
            name=op.f("fk_claims_sub_question_id_sub_questions"),
            ondelete="CASCADE",
        ),
        sa.CheckConstraint(
            "verdict IN ('supported', 'partial', 'unsupported', 'contradicted')",
            name=op.f("ck_claims_verdict"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_claims")),
    )
    op.create_index(op.f("ix_claims_run_id"), "claims", ["run_id"], unique=False)
    op.create_index(op.f("ix_claims_sub_question_id"), "claims", ["sub_question_id"], unique=False)
    op.create_table(
        "claim_evidence",
        sa.Column("claim_id", sa.UUID(), nullable=False),
        sa.Column("chunk_id", sa.UUID(), nullable=False),
        sa.Column("quote", sa.Text(), nullable=False),
        sa.Column("quote_start", sa.Integer(), nullable=False),
        sa.Column("quote_end", sa.Integer(), nullable=False),
        sa.CheckConstraint(
            "quote_start >= 0 AND quote_end > quote_start", name=op.f("ck_claim_evidence_quote_range")
        ),
        sa.ForeignKeyConstraint(
            ["chunk_id"], ["chunks.id"], name=op.f("fk_claim_evidence_chunk_id_chunks"), ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["claim_id"], ["claims.id"], name=op.f("fk_claim_evidence_claim_id_claims"), ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("claim_id", "chunk_id", name=op.f("pk_claim_evidence")),
    )
    op.create_index(op.f("ix_claim_evidence_chunk_id"), "claim_evidence", ["chunk_id"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_claim_evidence_chunk_id"), table_name="claim_evidence")
    op.drop_table("claim_evidence")
    op.drop_index(op.f("ix_claims_sub_question_id"), table_name="claims")
    op.drop_index(op.f("ix_claims_run_id"), table_name="claims")
    op.drop_table("claims")
    op.drop_index(op.f("ix_chunks_run_id"), table_name="chunks")
    op.drop_table("chunks")
    op.drop_table("sub_questions")
    op.drop_index(op.f("ix_sources_content_hash"), table_name="sources")
    op.drop_table("sources")
    op.drop_index(op.f("ix_run_events_run_id_created_at"), table_name="run_events")
    op.drop_table("run_events")
    op.drop_index(op.f("ix_reports_run_id"), table_name="reports")
    op.drop_table("reports")
    op.drop_table("research_runs")
