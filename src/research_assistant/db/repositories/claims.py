"""Claims and their evidence. add_claim_evidence enforces the citation invariant in code."""

import uuid
from collections.abc import Sequence

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from research_assistant.db.errors import (
    ChunkNotInRunError,
    NotFoundError,
    QuoteMismatchError,
    SubQuestionNotInRunError,
)
from research_assistant.db.models import Chunk, Claim, ClaimEvidence, SubQuestion, Verdict


def add_claim(
    session: Session,
    run_id: uuid.UUID,
    sub_question_id: uuid.UUID,
    text: str,
    verdict: Verdict | None = None,
    rationale: str | None = None,
) -> Claim:
    sub_question = session.get(SubQuestion, sub_question_id)
    if sub_question is None:
        raise NotFoundError(f"sub-question {sub_question_id} not found")
    if sub_question.run_id != run_id:
        raise SubQuestionNotInRunError(f"sub-question {sub_question_id} is not in run {run_id}")
    claim = Claim(
        run_id=run_id,
        sub_question_id=sub_question_id,
        text=text,
        verdict=verdict,
        rationale=rationale,
        ordinal=_next_ordinal(session, run_id),
    )
    session.add(claim)
    session.flush()
    return claim


def _next_ordinal(session: Session, run_id: uuid.UUID) -> int:
    last = session.scalar(select(func.max(Claim.ordinal)).where(Claim.run_id == run_id))
    return 0 if last is None else last + 1


def add_claims(
    session: Session, run_id: uuid.UUID, sub_question_id: uuid.UUID, texts: Sequence[str]
) -> list[Claim]:
    """Add unverified claims (verdict NULL) in order; ordinals continue across the run."""
    return [add_claim(session, run_id, sub_question_id, t) for t in texts]


def update_claim_verdict(
    session: Session, claim_id: uuid.UUID, verdict: Verdict, rationale: str | None
) -> Claim:
    claim = session.get(Claim, claim_id)
    if claim is None:
        raise NotFoundError(f"claim {claim_id} not found")
    claim.verdict = verdict
    claim.rationale = rationale
    session.flush()
    return claim


def list_claims(session: Session, run_id: uuid.UUID) -> list[Claim]:
    """All claims of a run in ordinal order."""
    return list(session.scalars(select(Claim).where(Claim.run_id == run_id).order_by(Claim.ordinal)))


def evidence_for_claims(
    session: Session, claim_ids: Sequence[uuid.UUID]
) -> dict[uuid.UUID, list[ClaimEvidence]]:
    rows = session.scalars(select(ClaimEvidence).where(ClaimEvidence.claim_id.in_(list(claim_ids))))
    out: dict[uuid.UUID, list[ClaimEvidence]] = {}
    for row in rows:
        out.setdefault(row.claim_id, []).append(row)
    return out


def add_claim_evidence(
    session: Session,
    claim_id: uuid.UUID,
    chunk_id: uuid.UUID,
    quote: str,
    quote_start: int,
    quote_end: int,
) -> ClaimEvidence:
    """Link a claim to a chunk with a verbatim quote.

    Raises ChunkNotInRunError if the chunk is from another run, and QuoteMismatchError unless
    quote is non-empty and quote == chunk.text[quote_start:quote_end].
    """
    claim = session.get(Claim, claim_id)
    if claim is None:
        raise NotFoundError(f"claim {claim_id} not found")
    chunk = session.get(Chunk, chunk_id)
    if chunk is None:
        raise NotFoundError(f"chunk {chunk_id} not found")
    if chunk.run_id != claim.run_id:
        raise ChunkNotInRunError(f"chunk {chunk_id} is in run {chunk.run_id}, claim is in {claim.run_id}")
    if not quote:
        raise QuoteMismatchError("quote is empty")
    if not 0 <= quote_start < quote_end <= len(chunk.text):
        raise QuoteMismatchError(
            f"offsets [{quote_start}, {quote_end}) out of range for chunk of length {len(chunk.text)}"
        )
    if chunk.text[quote_start:quote_end] != quote:
        raise QuoteMismatchError(f"quote does not match chunk text at [{quote_start}, {quote_end})")
    evidence = ClaimEvidence(
        claim_id=claim_id, chunk_id=chunk_id, quote=quote, quote_start=quote_start, quote_end=quote_end
    )
    session.add(evidence)
    session.flush()
    return evidence
