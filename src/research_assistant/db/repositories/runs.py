"""Research runs and their sub-questions."""

import uuid
from collections.abc import Sequence
from typing import Any

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from research_assistant.db.errors import NotFoundError
from research_assistant.db.models import TERMINAL_RUN_STATUSES, ResearchRun, RunStatus, SubQuestion


def create_run(
    session: Session,
    question: str,
    config: dict[str, Any] | None = None,
    *,
    status: RunStatus = RunStatus.PENDING,
    auto_approve: bool = False,
) -> ResearchRun:
    run = ResearchRun(question=question, config=config or {}, status=status, auto_approve=auto_approve)
    session.add(run)
    session.flush()
    return run


def get_run(session: Session, run_id: uuid.UUID) -> ResearchRun | None:
    return session.get(ResearchRun, run_id)


def update_run_status(
    session: Session, run_id: uuid.UUID, status: RunStatus, error: str | None = None
) -> ResearchRun:
    """Set the status. Terminal statuses (completed/failed/partial) also set finished_at."""
    run = session.get(ResearchRun, run_id)
    if run is None:
        raise NotFoundError(f"run {run_id} not found")
    run.status = status
    run.error = error
    if status in TERMINAL_RUN_STATUSES:
        run.finished_at = func.now()  # database clock, like created_at
    session.flush()
    session.refresh(run, ["finished_at"])
    return run


def add_sub_questions(
    session: Session,
    run_id: uuid.UUID,
    texts: Sequence[str],
    search_queries: Sequence[Sequence[str]] | None = None,
    premise_checks: Sequence[bool] | None = None,
) -> list[SubQuestion]:
    """Append sub-questions in order; ordinals continue after any existing ones.

    search_queries and premise_checks, if given, hold one entry per text.
    """
    if session.get(ResearchRun, run_id) is None:
        raise NotFoundError(f"run {run_id} not found")
    queries = [list(q) for q in search_queries] if search_queries is not None else [[] for _ in texts]
    if len(queries) != len(texts):
        raise ValueError("search_queries must have one entry per sub-question")
    flags = list(premise_checks) if premise_checks is not None else [False] * len(texts)
    if len(flags) != len(texts):
        raise ValueError("premise_checks must have one entry per sub-question")
    last = session.scalar(select(func.max(SubQuestion.ordinal)).where(SubQuestion.run_id == run_id))
    start = 0 if last is None else last + 1
    items = [
        SubQuestion(run_id=run_id, text=t, ordinal=start + i, search_queries=q, premise_check=f)
        for i, (t, q, f) in enumerate(zip(texts, queries, flags, strict=True))
    ]
    session.add_all(items)
    session.flush()
    return items


def list_sub_questions(session: Session, run_id: uuid.UUID) -> list[SubQuestion]:
    return list(
        session.scalars(select(SubQuestion).where(SubQuestion.run_id == run_id).order_by(SubQuestion.ordinal))
    )


def replace_sub_questions(
    session: Session,
    run_id: uuid.UUID,
    texts: Sequence[str],
    search_queries: Sequence[Sequence[str]],
    premise_checks: Sequence[bool] | None = None,
) -> list[SubQuestion]:
    """Replace a run's sub-questions (an edited plan). Only for a run awaiting plan approval, before
    research: claims reference sub-questions with ON DELETE CASCADE and would be deleted with them."""
    session.execute(delete(SubQuestion).where(SubQuestion.run_id == run_id))
    session.flush()
    return add_sub_questions(session, run_id, texts, search_queries, premise_checks)


def append_search_queries(
    session: Session, sub_question_id: uuid.UUID, queries: Sequence[str]
) -> SubQuestion:
    sub_question = session.get(SubQuestion, sub_question_id)
    if sub_question is None:
        raise NotFoundError(f"sub-question {sub_question_id} not found")
    sub_question.search_queries = [*sub_question.search_queries, *queries]
    session.flush()
    return sub_question
