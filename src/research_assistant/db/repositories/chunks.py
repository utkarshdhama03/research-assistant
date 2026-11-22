"""Chunks with embeddings, and exact nearest-chunk search within one run."""

import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from sqlalchemy import Float, Select, select
from sqlalchemy.orm import Session

from research_assistant.db.errors import NotFoundError
from research_assistant.db.models import EMBED_DIM, Chunk, Source


@dataclass(frozen=True)
class ChunkInput:
    ordinal: int
    text: str
    char_start: int
    char_end: int
    embedding: Sequence[float]


@dataclass(frozen=True)
class ChunkHit:
    chunk_id: uuid.UUID
    source_id: uuid.UUID
    text: str
    char_start: int
    char_end: int
    url: str
    final_url: str | None
    distance: float


def _check_dim(vec: Sequence[float]) -> None:
    if len(vec) != EMBED_DIM:
        raise ValueError(f"embedding has {len(vec)} dimensions, expected {EMBED_DIM}")


def add_chunks(session: Session, source_id: uuid.UUID, chunks: Sequence[ChunkInput]) -> list[Chunk]:
    """Store chunks for a source. run_id is copied from the source."""
    source = session.get(Source, source_id)
    if source is None:
        raise NotFoundError(f"source {source_id} not found")
    for c in chunks:
        _check_dim(c.embedding)
    rows = [
        Chunk(
            source_id=source_id,
            run_id=source.run_id,
            ordinal=c.ordinal,
            text=c.text,
            char_start=c.char_start,
            char_end=c.char_end,
            embedding=list(c.embedding),
        )
        for c in chunks
    ]
    session.add_all(rows)
    session.flush()
    return rows


def nearest_chunks(session: Session, run_id: uuid.UUID, query: Sequence[float], k: int) -> list[ChunkHit]:
    """Exact cosine search over one run's chunks.

    The run's chunks are selected in a MATERIALIZED CTE and sorted outside it. A materialized CTE has
    no index, so the planner can never answer this with the approximate HNSW index on
    chunks.embedding (which exists for cross-run memory search only); the result is exact.
    """
    return [
        ChunkHit(
            chunk_id=r.id,
            source_id=r.source_id,
            text=r.text,
            char_start=r.char_start,
            char_end=r.char_end,
            url=r.url,
            final_url=r.final_url,
            distance=float(r.distance),
        )
        for r in session.execute(nearest_chunks_statement(run_id, query, k))
    ]


def nearest_chunks_statement(run_id: uuid.UUID, query: Sequence[float], k: int) -> Select[Any]:
    """The statement behind nearest_chunks (exposed so a test can EXPLAIN it)."""
    if k < 1:
        raise ValueError("k must be >= 1")
    _check_dim(query)
    run_chunks = (
        select(Chunk.id, Chunk.source_id, Chunk.text, Chunk.char_start, Chunk.char_end, Chunk.embedding)
        .where(Chunk.run_id == run_id)
        .cte("run_chunks")
        .prefix_with("MATERIALIZED", dialect="postgresql")
    )
    distance = run_chunks.c.embedding.op("<=>", return_type=Float)(list(query)).label("distance")
    stmt = (
        select(
            run_chunks.c.id,
            run_chunks.c.source_id,
            run_chunks.c.text,
            run_chunks.c.char_start,
            run_chunks.c.char_end,
            Source.url,
            Source.final_url,
            distance,
        )
        .join(Source, Source.id == run_chunks.c.source_id)
        .order_by(distance, run_chunks.c.id)
        .limit(k)
    )
    return stmt
