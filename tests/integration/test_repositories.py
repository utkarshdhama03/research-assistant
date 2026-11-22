import math
import uuid
from datetime import UTC, datetime

import pytest
from sqlalchemy import delete, func, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from research_assistant.db.errors import (
    ChunkNotInRunError,
    NotFoundError,
    QuoteMismatchError,
    SubQuestionNotInRunError,
)
from research_assistant.db.models import (
    EMBED_DIM,
    Chunk,
    Claim,
    ClaimEvidence,
    Report,
    ResearchRun,
    RunEvent,
    RunStatus,
    Source,
    SourceStatus,
    SubQuestion,
    Verdict,
)
from research_assistant.db.repositories.chunks import ChunkInput, add_chunks, nearest_chunks
from research_assistant.db.repositories.claims import add_claim, add_claim_evidence
from research_assistant.db.repositories.events import append_event, list_events
from research_assistant.db.repositories.reports import save_report
from research_assistant.db.repositories.runs import (
    add_sub_questions,
    create_run,
    get_run,
    update_run_status,
)
from research_assistant.db.repositories.sources import add_source

CHUNK_TEXT = "Paris is the capital of France. It lies on the Seine."


def vec(angle_deg: float) -> list[float]:
    """Unit vector in the first two dimensions; cosine distance to vec(0) is 1 - cos(angle)."""
    v = [0.0] * EMBED_DIM
    v[0], v[1] = math.cos(math.radians(angle_deg)), math.sin(math.radians(angle_deg))
    return v


def make_source(session: Session, run_id: uuid.UUID, url: str = "https://example.org/a") -> Source:
    return add_source(session, run_id, url, status=SourceStatus.FETCHED)


def make_chunks(session: Session, source: Source, angles: list[float]) -> list[Chunk]:
    return add_chunks(
        session,
        source.id,
        [ChunkInput(i, f"chunk {a}", 0, 10, vec(a)) for i, a in enumerate(angles)],
    )


# ---------- runs ----------
def test_create_and_get_run(session: Session) -> None:
    run = create_run(session, "What is X?", {"max_sub_questions": 3})
    session.expire_all()
    got = get_run(session, run.id)
    assert got is not None
    assert got.question == "What is X?"
    assert got.status is RunStatus.PENDING
    assert got.config == {"max_sub_questions": 3}
    assert got.created_at.tzinfo is not None
    assert got.finished_at is None and got.error is None
    assert get_run(session, uuid.uuid4()) is None


def test_update_run_status(session: Session) -> None:
    run = create_run(session, "q")
    running = update_run_status(session, run.id, RunStatus.RUNNING)
    assert running.status is RunStatus.RUNNING and running.finished_at is None

    failed = update_run_status(session, run.id, RunStatus.FAILED, error="search unavailable")
    assert failed.status is RunStatus.FAILED
    assert failed.error == "search unavailable"
    assert isinstance(failed.finished_at, datetime) and failed.finished_at.tzinfo is not None

    with pytest.raises(NotFoundError):
        update_run_status(session, uuid.uuid4(), RunStatus.RUNNING)


def test_run_status_check_constraint(session: Session) -> None:
    with pytest.raises(IntegrityError), session.begin_nested():
        session.execute(text("INSERT INTO research_runs (question, status) VALUES ('q', 'bogus')"))


def test_add_sub_questions_ordinals_continue(session: Session) -> None:
    run = create_run(session, "q")
    first = add_sub_questions(session, run.id, ["a", "b"])
    second = add_sub_questions(session, run.id, ["c"])
    assert [(s.text, s.ordinal) for s in first + second] == [("a", 0), ("b", 1), ("c", 2)]
    with pytest.raises(NotFoundError):
        add_sub_questions(session, uuid.uuid4(), ["x"])


# ---------- sources ----------
def test_add_source_is_idempotent_on_run_and_url(session: Session) -> None:
    run = create_run(session, "q")
    fetched_at = datetime(2026, 11, 16, 12, 0, tzinfo=UTC)
    s1 = add_source(
        session,
        run.id,
        "https://example.org/a",
        status=SourceStatus.FETCHED,
        title="A",
        content_hash="h1",
        fetched_at=fetched_at,
    )
    s2 = add_source(session, run.id, "https://example.org/a", status=SourceStatus.FAILED, title="other")
    assert s2.id == s1.id
    assert s2.status is SourceStatus.FETCHED and s2.title == "A" and s2.fetched_at == fetched_at
    assert session.scalar(select(func.count()).select_from(Source).where(Source.run_id == run.id)) == 1

    other_url = add_source(session, run.id, "https://example.org/b", status=SourceStatus.SKIPPED)
    other_run = add_source(
        session, create_run(session, "q2").id, "https://example.org/a", status=SourceStatus.FETCHED
    )
    assert len({s1.id, other_url.id, other_run.id}) == 3


# ---------- chunks ----------
def test_add_chunks_copies_run_id_and_round_trips_embedding(session: Session) -> None:
    run = create_run(session, "q")
    source = make_source(session, run.id)
    [chunk] = add_chunks(session, source.id, [ChunkInput(0, CHUNK_TEXT, 100, 100 + len(CHUNK_TEXT), vec(30))])
    session.expire_all()
    got = session.get(Chunk, chunk.id)
    assert got is not None
    assert got.run_id == run.id and got.source_id == source.id
    assert got.text == CHUNK_TEXT and (got.char_start, got.char_end) == (100, 153)
    assert len(got.embedding) == EMBED_DIM
    assert got.embedding[:2] == pytest.approx(vec(30)[:2], abs=1e-6)


def test_add_chunks_rejects_bad_input(session: Session) -> None:
    run = create_run(session, "q")
    source = make_source(session, run.id)
    with pytest.raises(ValueError, match="dimensions"):
        add_chunks(session, source.id, [ChunkInput(0, "t", 0, 1, [1.0, 0.0])])
    with pytest.raises(NotFoundError):
        add_chunks(session, uuid.uuid4(), [ChunkInput(0, "t", 0, 1, vec(0))])
    make_chunks(session, source, [0])
    with pytest.raises(IntegrityError), session.begin_nested():
        make_chunks(session, source, [10])  # ordinal 0 again


def test_nearest_chunks_orders_by_cosine_distance(session: Session) -> None:
    run = create_run(session, "q")
    source = make_source(session, run.id)
    chunks = make_chunks(session, source, [40, 10, 25])
    hits = nearest_chunks(session, run.id, vec(0), k=3)
    assert [h.chunk_id for h in hits] == [chunks[1].id, chunks[2].id, chunks[0].id]
    assert [h.distance for h in hits] == pytest.approx([1 - math.cos(math.radians(a)) for a in (10, 25, 40)])
    assert all(h.url == "https://example.org/a" and h.source_id == source.id for h in hits)


def test_nearest_chunks_returns_exactly_k_from_the_run_even_if_other_runs_are_closer(
    session: Session,
) -> None:
    run = create_run(session, "q")
    mine = make_chunks(session, make_source(session, run.id), [30, 35, 40, 45, 50])
    other_run = create_run(session, "other")
    make_chunks(session, make_source(session, other_run.id), [i * 0.1 for i in range(50)])  # all closer

    hits = nearest_chunks(session, run.id, vec(0), k=3)
    assert [h.chunk_id for h in hits] == [c.id for c in mine[:3]]

    assert len(nearest_chunks(session, run.id, vec(0), k=10)) == 5  # fewer chunks than k
    assert nearest_chunks(session, create_run(session, "empty").id, vec(0), k=3) == []
    with pytest.raises(ValueError):
        nearest_chunks(session, run.id, vec(0), k=0)


# ---------- claims ----------
def test_add_claim(session: Session) -> None:
    run = create_run(session, "q")
    [sq] = add_sub_questions(session, run.id, ["sub"])
    claim = add_claim(session, run.id, sq.id, "Paris is the capital.", Verdict.SUPPORTED, "stated")
    session.expire_all()
    got = session.get(Claim, claim.id)
    assert got is not None and got.verdict is Verdict.SUPPORTED and got.sub_question_id == sq.id

    other = create_run(session, "other")
    with pytest.raises(SubQuestionNotInRunError):
        add_claim(session, other.id, sq.id, "x")
    with pytest.raises(NotFoundError):
        add_claim(session, run.id, uuid.uuid4(), "x")


@pytest.fixture
def claim_and_chunk(session: Session) -> tuple[Claim, Chunk]:
    run = create_run(session, "q")
    [sq] = add_sub_questions(session, run.id, ["sub"])
    claim = add_claim(session, run.id, sq.id, "Paris is the capital of France.")
    [chunk] = add_chunks(
        session, make_source(session, run.id).id, [ChunkInput(0, CHUNK_TEXT, 0, len(CHUNK_TEXT), vec(0))]
    )
    return claim, chunk


def test_add_claim_evidence_accepts_verbatim_quote(
    session: Session, claim_and_chunk: tuple[Claim, Chunk]
) -> None:
    claim, chunk = claim_and_chunk
    quote = "Paris is the capital of France."
    ev = add_claim_evidence(session, claim.id, chunk.id, quote, 0, len(quote))
    session.expire_all()
    got = session.get(ClaimEvidence, (claim.id, chunk.id))
    assert got is not None and got.quote == quote and (got.quote_start, got.quote_end) == (0, 31)
    assert ev.claim_id == claim.id


@pytest.mark.parametrize(
    ("quote", "start", "end"),
    [
        ("Paris is the capital of Spain.", 0, 30),  # wrong quote text
        ("Paris is the capital of France.", 1, 32),  # right text, shifted offsets
        ("It lies on the Seine.", 32, 100),  # end past chunk length
        ("Paris", 5, 0),  # inverted range
        ("", 0, 0),  # empty quote
    ],
)
def test_add_claim_evidence_rejects_mismatched_quote(
    session: Session, claim_and_chunk: tuple[Claim, Chunk], quote: str, start: int, end: int
) -> None:
    claim, chunk = claim_and_chunk
    with pytest.raises(QuoteMismatchError):
        add_claim_evidence(session, claim.id, chunk.id, quote, start, end)


def test_add_claim_evidence_rejects_chunk_from_another_run(
    session: Session, claim_and_chunk: tuple[Claim, Chunk]
) -> None:
    claim, _ = claim_and_chunk
    other_run = create_run(session, "other")
    [foreign] = add_chunks(
        session, make_source(session, other_run.id).id, [ChunkInput(0, CHUNK_TEXT, 0, 53, vec(0))]
    )
    quote = "Paris is the capital of France."
    with pytest.raises(ChunkNotInRunError):
        add_claim_evidence(session, claim.id, foreign.id, quote, 0, len(quote))


# ---------- reports and events ----------
def test_save_report(session: Session) -> None:
    run = create_run(session, "q")
    report = save_report(session, run.id, "# Report\n\nText [1].")
    assert report.created_at.tzinfo is not None
    assert session.get(Report, report.id) is not None
    with pytest.raises(NotFoundError):
        save_report(session, uuid.uuid4(), "x")


def test_events_keep_write_order_within_one_transaction(session: Session) -> None:
    run = create_run(session, "q")
    for i in range(5):
        append_event(session, run.id, "plan", f"step-{i}", {"i": i})
    events = list_events(session, run.id)
    assert [e.event for e in events] == [f"step-{i}" for i in range(5)]
    assert len({e.created_at for e in events}) == 5  # clock_timestamp(), not transaction time
    assert events[0].payload == {"i": 0}


# ---------- cascade ----------
def _populate(session: Session) -> uuid.UUID:
    run = create_run(session, "q")
    [sq] = add_sub_questions(session, run.id, ["sub"])
    [chunk] = add_chunks(session, make_source(session, run.id).id, [ChunkInput(0, CHUNK_TEXT, 0, 53, vec(0))])
    claim = add_claim(session, run.id, sq.id, "claim")
    add_claim_evidence(session, claim.id, chunk.id, "Paris", 0, 5)
    save_report(session, run.id, "report")
    append_event(session, run.id, "write", "done")
    return run.id


def test_deleting_a_run_cascades_to_everything_below_it(session: Session) -> None:
    doomed, kept = _populate(session), _populate(session)
    tables = [SubQuestion, Source, Chunk, Claim, Report, RunEvent]

    def counts(run_id: uuid.UUID) -> list[int]:
        per_table = [
            session.scalar(select(func.count()).select_from(t).where(t.run_id == run_id)) or 0  # type: ignore[attr-defined]
            for t in tables
        ]
        evidence = session.scalar(
            select(func.count())
            .select_from(ClaimEvidence)
            .join(Claim, Claim.id == ClaimEvidence.claim_id)
            .where(Claim.run_id == run_id)
        )
        return [*per_table, evidence or 0]

    assert counts(doomed) == [1] * 7
    session.execute(delete(ResearchRun).where(ResearchRun.id == doomed))
    session.expire_all()
    assert counts(doomed) == [0] * 7
    assert session.scalar(select(func.count()).select_from(ClaimEvidence)) == 1  # the kept run's
    assert counts(kept) == [1] * 7
