"""Cross-run memory: approximate nearest sources from earlier runs, and copy-on-reuse into a run.

The search uses the HNSW index on chunks.embedding with pgvector's iterative scan, so the filters
(other runs, fetched sources, maximum age) are applied while scanning instead of after a fixed
number of candidates. Chunks are over-fetched and grouped per source (best chunk wins).
"""

import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sqlalchemy import Float, Select, func, or_, select, text
from sqlalchemy.orm import Session

from research_assistant.db.models import EMBED_DIM, Chunk, Source, SourceStatus


@dataclass(frozen=True)
class MemoryHit:
    source_id: uuid.UUID
    run_id: uuid.UUID
    url: str
    final_url: str | None
    title: str | None
    fetched_at: datetime
    content_hash: str | None
    similarity: float  # cosine similarity of the source's best chunk to the query (1 - distance)
    best_chunk: str  # text of that chunk (the excerpt shown to select)


def memory_statement(
    run_id: uuid.UUID, query: Sequence[float], *, max_age_days: int, limit: int
) -> Select[Any]:
    """Nearest chunks of fetched, recent sources in other runs (exposed so a test can EXPLAIN it)."""
    if len(query) != EMBED_DIM:
        raise ValueError(f"embedding has {len(query)} dimensions, expected {EMBED_DIM}")
    distance = Chunk.embedding.op("<=>", return_type=Float)(list(query))
    return (
        select(
            Chunk.source_id,
            Source.run_id,
            Source.url,
            Source.final_url,
            Source.title,
            Source.fetched_at,
            Source.content_hash,
            Chunk.text,
            distance.label("distance"),
        )
        .join(Source, Source.id == Chunk.source_id)
        .where(
            Chunk.run_id != run_id,
            Source.status == SourceStatus.FETCHED,
            Source.fetched_at.is_not(None),
            Source.fetched_at >= func.now() - func.make_interval(0, 0, 0, max_age_days),
        )
        .order_by(distance)
        .limit(limit)
    )


def nearest_sources(
    session: Session, run_id: uuid.UUID, query: Sequence[float], *, max_age_days: int, overfetch: int
) -> list[MemoryHit]:
    """Distinct sources from other runs, most similar first (one entry per source: its best chunk)."""
    limit = int(overfetch)
    if not 1 <= limit <= 1000:
        raise ValueError("overfetch must be 1..1000")
    # SET LOCAL takes no bind parameters; limit is a checked int. Scoped to this transaction.
    session.execute(text("SET LOCAL hnsw.iterative_scan = relaxed_order"))
    session.execute(text(f"SET LOCAL hnsw.ef_search = {limit}"))
    rows = session.execute(memory_statement(run_id, query, max_age_days=max_age_days, limit=limit)).all()
    best: dict[uuid.UUID, MemoryHit] = {}
    for r in sorted(rows, key=lambda r: float(r.distance)):  # relaxed_order may be slightly out of order
        if r.source_id in best:
            continue
        best[r.source_id] = MemoryHit(
            source_id=r.source_id,
            run_id=r.run_id,
            url=r.url,
            final_url=r.final_url,
            title=r.title,
            fetched_at=r.fetched_at,
            content_hash=r.content_hash,
            similarity=round(1.0 - float(r.distance), 4),
            best_chunk=r.text,
        )
    return list(best.values())


def reusable_by_url(
    session: Session, run_id: uuid.UUID, url: str, *, max_age_days: int
) -> tuple[uuid.UUID, uuid.UUID, datetime] | None:
    """The most recently fetched source of another run for this (normalized) URL, matched on url or
    final_url, if it is no older than max_age_days: (source_id, run_id, fetched_at)."""
    row = session.execute(
        select(Source.id, Source.run_id, Source.fetched_at)
        .where(
            Source.run_id != run_id,
            Source.status == SourceStatus.FETCHED,
            Source.fetched_at.is_not(None),
            Source.fetched_at >= func.now() - func.make_interval(0, 0, 0, max_age_days),
            or_(Source.url == url, Source.final_url == url),
        )
        .order_by(Source.fetched_at.desc(), Source.id)
        .limit(1)
    ).first()
    if row is None or row[2] is None:  # fetched_at IS NOT NULL is in the query; this satisfies typing
        return None
    return row[0], row[1], row[2]


@dataclass(frozen=True)
class CopiedSource:
    source_id: uuid.UUID
    from_source_id: uuid.UUID
    from_run_id: uuid.UUID
    chunks: int


class MemoryCopyError(Exception):
    """The copy could not be completed (no chunks copied); raised so the caller's transaction rolls back."""


def copy_source(session: Session, source_id: uuid.UUID, run_id: uuid.UUID) -> CopiedSource | None:
    """Copy a fetched source and its chunks (text, offsets, ordinals, embeddings) into run_id.
    Returns None if the original no longer exists or is not a fetched source.

    The original is read FOR KEY SHARE, so deleting its run (which cascades to the source and its
    chunks) waits until this transaction ends: without the lock, a delete committed between this read
    and the chunk INSERT ... SELECT (READ COMMITTED: a new snapshot per statement) left a copy with no
    chunks. A copy that still gets no chunks raises MemoryCopyError."""
    original = session.get(Source, source_id, with_for_update={"key_share": True})
    if original is None or original.status is not SourceStatus.FETCHED:
        return None
    copy = Source(
        run_id=run_id,
        url=original.url,
        final_url=original.final_url,
        title=original.title,
        content_type=original.content_type,
        content_hash=original.content_hash,
        fetched_at=original.fetched_at,
        status=SourceStatus.FETCHED,
        extracted_text=original.extracted_text,
        reused_from_source_id=original.id,
    )
    session.add(copy)
    session.flush()
    result = session.execute(
        text(
            "INSERT INTO chunks (id, source_id, run_id, ordinal, text, char_start, char_end, embedding) "
            "SELECT gen_random_uuid(), :new_source, :run, ordinal, text, char_start, char_end, embedding "
            "FROM chunks WHERE source_id = :original"
        ),
        {"new_source": copy.id, "run": run_id, "original": original.id},
    )
    chunks = int(getattr(result, "rowcount", 0) or 0)
    if chunks == 0:
        raise MemoryCopyError(f"source {original.id} had no chunks to copy")
    return CopiedSource(copy.id, original.id, original.run_id, chunks)
