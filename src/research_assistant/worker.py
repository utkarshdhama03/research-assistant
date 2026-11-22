"""`ra worker`: one process that executes queued runs one at a time.

- Single instance: holds a Postgres advisory lock for its lifetime (freed by Postgres if it crashes).
- On start, runs left in running by a worker (worker_claimed event) are marked failed "worker restarted".
- Claims the oldest queued run (FOR UPDATE SKIP LOCKED) and executes it. A run that pauses for plan
  approval leaves the worker free for the next queued run; once approved it is queued again and
  resumed from its checkpoint.
- Writes a heartbeat row every WORKER_HEARTBEAT_S from a background thread.
- First Ctrl+C (or Ctrl+Break): claim nothing new; the current run stops at its next check and is
  marked failed "worker stopped". Second: stop at once (KeyboardInterrupt), same marking.
"""

import logging
import threading
import uuid
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, ExitStack, contextmanager
from typing import Any

import psycopg
from langgraph.checkpoint.base import BaseCheckpointSaver
from sqlalchemy.orm import Session

from research_assistant.config import Settings
from research_assistant.db.models import RunStatus
from research_assistant.db.repositories.events import append_event
from research_assistant.db.repositories.queue import (
    beat,
    claim_next_run,
    lock_run,
    orphaned_worker_runs,
    remove_heartbeat,
)
from research_assistant.db.repositories.runs import update_run_status
from research_assistant.db.session import libpq_url, session_scope
from research_assistant.graph.context import Deps
from research_assistant.graph.runner import WORKER_STOPPED_REASON, RunOutcome, execute_run

log = logging.getLogger(__name__)

WORKER_LOCK_KEY = 72_657_107  # pg advisory lock key for "one worker per database"
RESTARTED_REASON = "worker restarted"

# deps_factory(stop_requested) -> context manager yielding the run's dependencies
DepsFactory = Callable[[Callable[[], bool]], AbstractContextManager[Deps]]
Checkpointer = BaseCheckpointSaver[Any]


class AnotherWorkerRunning(RuntimeError):
    pass


@contextmanager
def single_worker_lock(database_url: str) -> Iterator[None]:
    """Hold the worker advisory lock on a dedicated connection; raise if another worker holds it."""
    with psycopg.connect(libpq_url(database_url), autocommit=True) as conn:
        row = conn.execute("SELECT pg_try_advisory_lock(%s)", (WORKER_LOCK_KEY,)).fetchone()
        if not (row and row[0]):
            raise AnotherWorkerRunning("another worker is running (advisory lock held)")
        try:
            yield
        finally:
            conn.execute("SELECT pg_advisory_unlock(%s)", (WORKER_LOCK_KEY,))


class Worker:
    def __init__(
        self,
        settings: Settings,
        session_factory: Callable[[], Session],
        deps_factory: DepsFactory,
        *,
        log_line: Callable[[str], None] = print,
    ) -> None:
        self.settings = settings
        self.session_factory = session_factory
        self.deps_factory = deps_factory
        self.log_line = log_line
        self.id = uuid.uuid4()
        self.current_run_id: uuid.UUID | None = None
        self._stop = threading.Event()

    # -- stop ------------------------------------------------------------------------------------

    def request_stop(self) -> None:
        self._stop.set()

    def stopping(self) -> bool:
        return self._stop.is_set()

    # -- startup ---------------------------------------------------------------------------------

    def recover(self, checkpointer: Checkpointer | None) -> list[uuid.UUID]:
        """Mark runs a crashed worker left in running as failed. Call only while holding the lock."""
        with session_scope(self.session_factory) as session:
            orphans = orphaned_worker_runs(session)
            for run_id in orphans:
                update_run_status(session, run_id, RunStatus.FAILED, error=RESTARTED_REASON)
                append_event(session, run_id, "run", "run_aborted", {"reason": RESTARTED_REASON})
        for run_id in orphans:
            if checkpointer is not None:
                checkpointer.delete_thread(str(run_id))
            self.log_line(f"recovered {run_id}: failed ({RESTARTED_REASON})")
        return orphans

    # -- one run ---------------------------------------------------------------------------------

    def run_once(self, checkpointer: Checkpointer) -> RunOutcome | None:
        """Claim and execute the oldest queued run. None if the queue is empty or we are stopping."""
        if self.stopping():
            return None
        with session_scope(self.session_factory) as session:
            run_id = claim_next_run(session, self.id)
        if run_id is None:
            return None
        self.current_run_id = run_id
        self.log_line(f"claimed {run_id}")
        try:
            with self.deps_factory(self.stopping) as deps:
                outcome = execute_run(run_id, deps, checkpointer)
        except KeyboardInterrupt:
            self._mark_stopped(run_id, checkpointer)
            raise
        finally:
            self.current_run_id = None
        self.log_line(f"{run_id}: {outcome.status}" + (f" - {outcome.reason}" if outcome.reason else ""))
        return outcome

    def _mark_stopped(self, run_id: uuid.UUID, checkpointer: Checkpointer) -> None:
        """Hard stop (second Ctrl+C) in the middle of a run: it cannot be resumed, so it fails."""
        with session_scope(self.session_factory) as session:
            run = lock_run(session, run_id)
            if run is None or run.status is not RunStatus.RUNNING:
                return
            update_run_status(session, run_id, RunStatus.FAILED, error=WORKER_STOPPED_REASON)
            append_event(session, run_id, "run", "run_aborted", {"reason": WORKER_STOPPED_REASON})
        checkpointer.delete_thread(str(run_id))
        self.log_line(f"{run_id}: failed ({WORKER_STOPPED_REASON})")

    # -- heartbeat -------------------------------------------------------------------------------

    def heartbeat_once(self) -> None:
        with session_scope(self.session_factory) as session:
            beat(session, self.id, self.current_run_id)

    def _heartbeat_loop(self, done: threading.Event) -> None:
        while not done.is_set():
            try:
                self.heartbeat_once()
            except Exception:  # a missed beat must not kill the worker; /health will show it stale
                log.exception("heartbeat failed")
            done.wait(self.settings.worker_heartbeat_s)

    # -- main loop -------------------------------------------------------------------------------

    def serve(self, checkpointer_factory: Callable[[], AbstractContextManager[Checkpointer]]) -> int:
        """Run until stopped. Returns the process exit code (2: another worker is running)."""
        done = threading.Event()
        beats = threading.Thread(target=self._heartbeat_loop, args=(done,), name="heartbeat", daemon=True)
        with ExitStack() as stack:
            try:
                stack.enter_context(single_worker_lock(self.settings.database_url))
            except AnotherWorkerRunning as e:
                self.log_line(str(e))
                return 2
            stack.callback(self._remove_heartbeat)
            stack.callback(beats.join, 5)
            stack.callback(done.set)
            checkpointer = stack.enter_context(checkpointer_factory())
            setup = getattr(checkpointer, "setup", None)
            if callable(setup):
                setup()  # creates or migrates the checkpointer's tables; idempotent
            self.recover(checkpointer)
            beats.start()
            self.log_line(f"worker {self.id} ready (poll every {self.settings.worker_poll_s} s)")
            try:
                while not self.stopping():
                    if self.run_once(checkpointer) is None:
                        self._stop.wait(self.settings.worker_poll_s)
            except KeyboardInterrupt:
                self.log_line("stopping now (second interrupt)")
        self.log_line("worker stopped")
        return 0

    def _remove_heartbeat(self) -> None:
        try:
            with session_scope(self.session_factory) as session:
                remove_heartbeat(session, self.id)
        except Exception:
            log.exception("could not remove the heartbeat row")
