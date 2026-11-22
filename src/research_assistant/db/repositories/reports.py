"""Final Markdown reports."""

import uuid
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from research_assistant.db.errors import NotFoundError
from research_assistant.db.models import Report, ResearchRun


def save_report(
    session: Session,
    run_id: uuid.UUID,
    markdown: str,
    citations: list[dict[str, Any]] | None = None,
    structure: dict[str, Any] | None = None,
) -> Report:
    if session.get(ResearchRun, run_id) is None:
        raise NotFoundError(f"run {run_id} not found")
    report = Report(run_id=run_id, markdown=markdown, citations=citations or [], structure=structure)
    session.add(report)
    session.flush()
    session.refresh(report, ["created_at"])
    return report


def latest_report(session: Session, run_id: uuid.UUID) -> Report | None:
    stmt = (
        select(Report).where(Report.run_id == run_id).order_by(Report.created_at.desc(), Report.id).limit(1)
    )
    return session.scalar(stmt)
