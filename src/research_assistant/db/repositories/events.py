"""Append-only audit trail of node events per run."""

import uuid
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from research_assistant.db.models import RunEvent


def append_event(
    session: Session, run_id: uuid.UUID, node: str, event: str, payload: dict[str, Any] | None = None
) -> RunEvent:
    row = RunEvent(run_id=run_id, node=node, event=event, payload=payload or {})
    session.add(row)
    session.flush()
    return row


def list_events(session: Session, run_id: uuid.UUID) -> list[RunEvent]:
    """Events in the order they were written (created_at uses clock_timestamp())."""
    stmt = select(RunEvent).where(RunEvent.run_id == run_id).order_by(RunEvent.created_at, RunEvent.id)
    return list(session.scalars(stmt))
