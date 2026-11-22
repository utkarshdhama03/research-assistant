"""Fetched (or attempted) sources for a run."""

import uuid
from datetime import datetime

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from research_assistant.db.errors import NotFoundError
from research_assistant.db.models import Source, SourceStatus


def add_source(
    session: Session,
    run_id: uuid.UUID,
    url: str,
    *,
    status: SourceStatus,
    final_url: str | None = None,
    title: str | None = None,
    content_type: str | None = None,
    content_hash: str | None = None,
    fetched_at: datetime | None = None,
    error: str | None = None,
    extracted_text: str | None = None,
) -> Source:
    """Insert a source, or return the existing one for (run_id, url).

    `url` must already be normalized; normalization is the fetch tool's job (Phase 2). Two spellings
    of the same URL are stored as two sources. If the source already exists it is returned unchanged
    and the other arguments are ignored.
    """
    stmt = (
        insert(Source)
        .values(
            id=uuid.uuid4(),
            run_id=run_id,
            url=url,
            status=status,
            final_url=final_url,
            title=title,
            content_type=content_type,
            content_hash=content_hash,
            fetched_at=fetched_at,
            error=error,
            extracted_text=extracted_text,
        )
        .on_conflict_do_nothing(index_elements=[Source.run_id, Source.url])
        .returning(Source.id)
    )
    source_id = session.scalar(stmt)
    if source_id is None:
        source_id = session.scalar(select(Source.id).where(Source.run_id == run_id, Source.url == url))
    source = session.get(Source, source_id) if source_id is not None else None
    if source is None:  # only possible if the row was deleted concurrently
        raise NotFoundError(f"source ({run_id}, {url}) vanished after insert")
    return source


def find_source_by_url(session: Session, run_id: uuid.UUID, url: str) -> Source | None:
    return session.scalar(select(Source).where(Source.run_id == run_id, Source.url == url))


def find_source_by_content_hash(session: Session, run_id: uuid.UUID, content_hash: str) -> Source | None:
    """The fetched source in this run with this content hash, if any (oldest first)."""
    stmt = (
        select(Source)
        .where(
            Source.run_id == run_id,
            Source.content_hash == content_hash,
            Source.status == SourceStatus.FETCHED,
        )
        .order_by(Source.fetched_at, Source.id)
        .limit(1)
    )
    return session.scalar(stmt)


def list_sources(session: Session, run_id: uuid.UUID) -> list[Source]:
    return list(
        session.scalars(select(Source).where(Source.run_id == run_id).order_by(Source.fetched_at, Source.url))
    )


def count_sources(session: Session, run_id: uuid.UUID, *, reused: bool | None = None) -> int:
    """Source rows of the run. reused=False: fetch attempts (every attempt is stored as a row);
    reused=True: sources copied from earlier runs; None: all."""
    stmt = select(func.count()).select_from(Source).where(Source.run_id == run_id)
    if reused is not None:
        column = Source.reused_from_source_id
        stmt = stmt.where(column.is_not(None) if reused else column.is_(None))
    return int(session.scalar(stmt) or 0)


def seen_urls(session: Session, run_id: uuid.UUID) -> dict[str, tuple[uuid.UUID, SourceStatus]]:
    return {s.url: (s.id, s.status) for s in session.scalars(select(Source).where(Source.run_id == run_id))}
