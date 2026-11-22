"""SQLAlchemy models: runs, sub-questions, sources, chunks, claims, evidence, reports and events.

Status and verdict columns are VARCHAR + CHECK (not Postgres ENUM types) mapped to Python enums.
Every table hangs off research_runs with ON DELETE CASCADE.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import StrEnum
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    CheckConstraint,
    DateTime,
    Enum,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    PrimaryKeyConstraint,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy import (
    text as sql_text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

# Fixed by the initial migration. Changing the embedding model's dimension needs a new migration.
EMBED_DIM = 768

NAMING_CONVENTION = {
    "ix": "ix_%(table_name)s_%(column_0_N_name)s",
    "uq": "uq_%(table_name)s_%(column_0_N_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class RunStatus(StrEnum):
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    AWAITING_APPROVAL = "awaiting_approval"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"
    CANCELLED = "cancelled"


TERMINAL_RUN_STATUSES = frozenset(
    {RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.PARTIAL, RunStatus.CANCELLED}
)


class SourceStatus(StrEnum):
    FETCHED = "fetched"
    FAILED = "failed"
    SKIPPED = "skipped"


class Verdict(StrEnum):
    SUPPORTED = "supported"
    PARTIAL = "partial"
    UNSUPPORTED = "unsupported"
    CONTRADICTED = "contradicted"


def _str_enum(enum_cls: type[StrEnum], name: str) -> Enum:
    """VARCHAR + named CHECK constraint; stores the enum *values*."""
    return Enum(
        enum_cls,
        name=name,
        native_enum=False,
        create_constraint=True,
        validate_strings=True,
        length=32,
        values_callable=lambda e: [m.value for m in e],
    )


def _uuid_pk() -> Mapped[uuid.UUID]:
    return mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, server_default=func.gen_random_uuid()
    )


def _run_fk() -> Mapped[uuid.UUID]:
    return mapped_column(
        UUID(as_uuid=True), ForeignKey("research_runs.id", ondelete="CASCADE"), nullable=False
    )


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING_CONVENTION)
    type_annotation_map = {
        datetime: DateTime(timezone=True),
        dict[str, Any]: JSONB,
        str: Text,
    }


class ResearchRun(Base):
    __tablename__ = "research_runs"
    # Keyset pagination (newest first) and oldest-first queue claims.
    __table_args__ = (Index(None, "created_at", "id"),)

    id: Mapped[uuid.UUID] = _uuid_pk()
    question: Mapped[str]
    status: Mapped[RunStatus] = mapped_column(
        _str_enum(RunStatus, "status"), default=RunStatus.PENDING, server_default=RunStatus.PENDING.value
    )
    config: Mapped[dict[str, Any]] = mapped_column(default=dict, server_default=sql_text("'{}'::jsonb"))
    error: Mapped[str | None]
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    finished_at: Mapped[datetime | None]
    # Skip the plan-approval pause.
    auto_approve: Mapped[bool] = mapped_column(default=False, server_default=sql_text("false"))
    # Set by the API for a running run; checked before each unit of work.
    cancel_requested: Mapped[bool] = mapped_column(default=False, server_default=sql_text("false"))


class WorkerHeartbeat(Base):
    """One row per live worker process, refreshed every WORKER_HEARTBEAT_S (read by GET /health)."""

    __tablename__ = "worker_heartbeats"

    id: Mapped[uuid.UUID] = _uuid_pk()
    pid: Mapped[int] = mapped_column(Integer)
    started_at: Mapped[datetime] = mapped_column(server_default=func.now())
    last_seen: Mapped[datetime] = mapped_column(server_default=func.now())
    current_run_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))


class SubQuestion(Base):
    __tablename__ = "sub_questions"
    __table_args__ = (UniqueConstraint("run_id", "ordinal"),)

    id: Mapped[uuid.UUID] = _uuid_pk()
    run_id: Mapped[uuid.UUID] = _run_fk()
    text: Mapped[str]
    ordinal: Mapped[int] = mapped_column(Integer)
    search_queries: Mapped[list[str]] = mapped_column(
        JSONB, default=list, server_default=sql_text("'[]'::jsonb")
    )
    # The first sub-question may check a fact the question asserts (false-premise handling).
    premise_check: Mapped[bool] = mapped_column(default=False, server_default=sql_text("false"))


class Source(Base):
    __tablename__ = "sources"
    __table_args__ = (
        UniqueConstraint("run_id", "url"),
        Index(None, "content_hash"),
    )

    id: Mapped[uuid.UUID] = _uuid_pk()
    run_id: Mapped[uuid.UUID] = _run_fk()
    url: Mapped[str]
    final_url: Mapped[str | None]
    title: Mapped[str | None]
    content_type: Mapped[str | None]
    content_hash: Mapped[str | None]
    fetched_at: Mapped[datetime | None]
    status: Mapped[SourceStatus] = mapped_column(_str_enum(SourceStatus, "status"))
    error: Mapped[str | None]
    # Normalized extracted text; chunk char offsets refer to this string.
    extracted_text: Mapped[str | None]
    # Set when this source was copied from an earlier run (memory reuse). No foreign key: the
    # original's run may be deleted later without affecting the copy.
    reused_from_source_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))


class Chunk(Base):
    __tablename__ = "chunks"
    __table_args__ = (
        UniqueConstraint("source_id", "ordinal"),
        Index(None, "run_id"),
        CheckConstraint("char_start >= 0 AND char_end >= char_start", name="char_range"),
        # Approximate search across runs (memory). Per-run retrieval stays exact: nearest_chunks
        # filters the run in a MATERIALIZED CTE, which this index cannot serve.
        Index(
            "ix_chunks_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )

    id: Mapped[uuid.UUID] = _uuid_pk()
    source_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("sources.id", ondelete="CASCADE"), nullable=False
    )
    # Denormalised from sources.run_id so run-scoped search and evidence checks need no join.
    run_id: Mapped[uuid.UUID] = _run_fk()
    ordinal: Mapped[int] = mapped_column(Integer)
    text: Mapped[str]
    char_start: Mapped[int] = mapped_column(Integer)
    char_end: Mapped[int] = mapped_column(Integer)
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBED_DIM))


class Claim(Base):
    __tablename__ = "claims"
    __table_args__ = (
        Index(None, "run_id"),
        Index(None, "sub_question_id"),
        UniqueConstraint("run_id", "ordinal"),
    )

    id: Mapped[uuid.UUID] = _uuid_pk()
    run_id: Mapped[uuid.UUID] = _run_fk()
    # Order of creation within the run (0, 1, ...); used for dedupe order, the evidence appendix and display.
    ordinal: Mapped[int] = mapped_column(Integer)
    sub_question_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("sub_questions.id", ondelete="CASCADE"), nullable=False
    )
    text: Mapped[str]
    verdict: Mapped[Verdict | None] = mapped_column(_str_enum(Verdict, "verdict"))
    rationale: Mapped[str | None]


class ClaimEvidence(Base):
    __tablename__ = "claim_evidence"
    __table_args__ = (
        PrimaryKeyConstraint("claim_id", "chunk_id"),
        Index(None, "chunk_id"),
        CheckConstraint("quote_start >= 0 AND quote_end > quote_start", name="quote_range"),
    )

    claim_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("claims.id", ondelete="CASCADE")
    )
    chunk_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("chunks.id", ondelete="CASCADE")
    )
    quote: Mapped[str]
    quote_start: Mapped[int] = mapped_column(Integer)
    quote_end: Mapped[int] = mapped_column(Integer)


class Report(Base):
    __tablename__ = "reports"
    __table_args__ = (Index(None, "run_id"),)

    id: Mapped[uuid.UUID] = _uuid_pk()
    run_id: Mapped[uuid.UUID] = _run_fk()
    markdown: Mapped[str]
    # One entry per [n] in the report: {"n": int, "source_id": str, "chunk_ids": [str, ...]}.
    citations: Mapped[list[dict[str, Any]]] = mapped_column(
        JSONB, default=list, server_default=sql_text("'[]'::jsonb")
    )
    # The report as data (graph/citations.py assemble_report): sections, sentences with their markers,
    # references, cited claims. NULL for reports written before 2026-11-22 (the API serves their Markdown).
    structure: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())


class RunEvent(Base):
    __tablename__ = "run_events"
    __table_args__ = (Index(None, "run_id", "created_at"),)

    id: Mapped[uuid.UUID] = _uuid_pk()
    run_id: Mapped[uuid.UUID] = _run_fk()
    node: Mapped[str]
    event: Mapped[str]
    payload: Mapped[dict[str, Any]] = mapped_column(default=dict, server_default=sql_text("'{}'::jsonb"))
    # clock_timestamp(), not now(): now() is the transaction start, so events written in one
    # transaction would share a timestamp and lose their order.
    created_at: Mapped[datetime] = mapped_column(server_default=func.clock_timestamp())
