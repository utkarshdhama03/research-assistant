"""The run queue (status = queued) and the worker's heartbeat row."""

import os
import uuid

from sqlalchemy import delete, exists, func, select
from sqlalchemy.orm import Session

from research_assistant.db.models import ResearchRun, RunEvent, RunStatus, WorkerHeartbeat
from research_assistant.db.repositories.events import append_event
from research_assistant.db.repositories.runs import update_run_status

CLAIM_EVENT = "worker_claimed"


def claim_next_run(session: Session, worker_id: uuid.UUID | None = None) -> uuid.UUID | None:
    """Claim the oldest queued run: lock it (SKIP LOCKED, so two claimers never get the same run),
    set it running and log worker_claimed. Commit the session to release the lock."""
    run_id = session.scalar(
        select(ResearchRun.id)
        .where(ResearchRun.status == RunStatus.QUEUED)
        .order_by(ResearchRun.created_at, ResearchRun.id)
        .limit(1)
        .with_for_update(skip_locked=True)
    )
    if run_id is None:
        return None
    update_run_status(session, run_id, RunStatus.RUNNING)
    append_event(
        session,
        run_id,
        "run",
        CLAIM_EVENT,
        {"worker_id": str(worker_id) if worker_id else None, "pid": os.getpid()},
    )
    return run_id


def lock_run(session: Session, run_id: uuid.UUID) -> ResearchRun | None:
    """The run row, locked until the session commits (serializes the API with the worker's claim)."""
    return session.get(ResearchRun, run_id, with_for_update=True)


def orphaned_worker_runs(session: Session) -> list[uuid.UUID]:
    """Runs left in running by a worker (they have a worker_claimed event). Inline runs (CLI, eval)
    are never claimed and are not returned."""
    claimed = exists().where(RunEvent.run_id == ResearchRun.id, RunEvent.event == CLAIM_EVENT)
    stmt = select(ResearchRun.id).where(ResearchRun.status == RunStatus.RUNNING, claimed)
    return list(session.scalars(stmt.with_for_update(skip_locked=True)))


def beat(session: Session, worker_id: uuid.UUID, current_run_id: uuid.UUID | None) -> None:
    row = session.get(WorkerHeartbeat, worker_id)
    if row is None:
        session.add(WorkerHeartbeat(id=worker_id, pid=os.getpid(), current_run_id=current_run_id))
    else:
        row.last_seen = func.now()
        row.current_run_id = current_run_id
    session.flush()


def remove_heartbeat(session: Session, worker_id: uuid.UUID) -> None:
    session.execute(delete(WorkerHeartbeat).where(WorkerHeartbeat.id == worker_id))


def latest_heartbeat(session: Session) -> tuple[WorkerHeartbeat, float] | None:
    """The most recent heartbeat and its age in seconds (database clock)."""
    row = session.execute(
        select(WorkerHeartbeat, func.extract("epoch", func.now() - WorkerHeartbeat.last_seen))
        .order_by(WorkerHeartbeat.last_seen.desc())
        .limit(1)
    ).first()
    return (row[0], float(row[1])) if row is not None else None
