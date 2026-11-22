"""Queue claims, the worker, plan approval across a worker restart, cancellation and stopping."""

import threading
import time
import uuid
from typing import Any

import pytest
from sqlalchemy import Engine, select, text
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import sessionmaker

from research_assistant.db.models import TERMINAL_RUN_STATUSES, Claim, ResearchRun, RunStatus, WorkerHeartbeat
from research_assistant.db.repositories.queue import claim_next_run
from research_assistant.db.session import session_scope
from research_assistant.graph.checkpoint import postgres_checkpointer, strict_serializer
from research_assistant.graph.runner import CANCELLED_REASON, WORKER_STOPPED_REASON
from research_assistant.graph.state import GraphState
from research_assistant.worker import RESTARTED_REASON, AnotherWorkerRunning, single_worker_lock
from tests.integration.conftest import TrackingSessionFactory
from tests.integration.graph_fakes import PLAN, RFC9309, supported
from tests.integration.phase7_support import (
    checkpoint_rows,
    client_for,
    event_names,
    event_payloads,
    make_app,
    new_run,
    run_status,
    worker_for,
)
from tests.integration.test_graph import Harness


@pytest.fixture
def h(committing_factory: TrackingSessionFactory) -> Harness:
    return Harness(committing_factory)


EDITED = [
    {"question": s["question"], "search_queries": list(s["search_queries"])} for s in PLAN["sub_questions"]
]
EDITED[2] = {"question": "What does the robots.txt standard say?", "search_queries": ["robots txt standard"]}


# -- queue -----------------------------------------------------------------------------------


def test_claims_oldest_queued_first(h: Harness) -> None:
    first, second = new_run(h), new_run(h)
    new_run(h, status=RunStatus.AWAITING_APPROVAL)
    with session_scope(h.factory) as session:
        assert claim_next_run(session) == first
    with session_scope(h.factory) as session:
        assert claim_next_run(session) == second
    with session_scope(h.factory) as session:
        assert claim_next_run(session) is None
    assert run_status(h, first)[0] is RunStatus.RUNNING
    assert event_names(h, first) == ["worker_claimed"]


def test_two_claimers_never_get_the_same_run(h: Harness) -> None:
    first, second = new_run(h), new_run(h)
    a, b = h.factory(), h.factory()
    try:
        assert claim_next_run(a) == first  # locked, not committed
        # Without SKIP LOCKED, b would wait on a's row lock; the timeout turns that into a failure.
        b.execute(text("SET LOCAL lock_timeout = '2s'"))
        assert claim_next_run(b) == second
        b.commit()
        a.commit()
    finally:
        a.close()
        b.close()
    assert run_status(h, first)[0] is run_status(h, second)[0] is RunStatus.RUNNING


# -- worker basics ---------------------------------------------------------------------------


def test_auto_approve_run_completes_without_pause(h: Harness, checkpointer: Any, engine: Engine) -> None:
    run_id = new_run(h, auto_approve=True)
    outcome = worker_for(h).run_once(checkpointer)
    assert outcome is not None and outcome.status is RunStatus.COMPLETED
    names = event_names(h, run_id)
    assert "plan_auto_approved" in names and "awaiting_approval" not in names
    assert checkpoint_rows(engine, run_id) == 0  # deleted at the terminal status


def test_recovery_marks_only_worker_claimed_runs(h: Harness, checkpointer: Any) -> None:
    claimed = new_run(h)
    with session_scope(h.factory) as session:
        assert claim_next_run(session) == claimed  # a worker took it, then "crashed"
    inline = new_run(h, status=RunStatus.RUNNING, auto_approve=True)  # `ra research` / eval in progress
    recovered = worker_for(h).recover(checkpointer)
    assert recovered == [claimed]
    assert run_status(h, claimed) == (RunStatus.FAILED, RESTARTED_REASON)
    assert event_names(h, claimed)[-1] == "run_aborted"
    assert run_status(h, inline)[0] is RunStatus.RUNNING


def test_only_one_worker_holds_the_lock(db_url: str) -> None:
    with single_worker_lock(db_url):
        with pytest.raises(AnotherWorkerRunning):
            with single_worker_lock(db_url):
                pass
    with single_worker_lock(db_url):  # released when the first worker exits
        pass


def test_serve_loop_heartbeat_and_stop(h: Harness, checkpointer: Any, db_url: str, engine: Engine) -> None:
    settings = h.settings.model_copy(
        update={"database_url": db_url, "worker_poll_s": 0.05, "worker_heartbeat_s": 0.05}
    )
    # Everything the test and the worker's own threads do uses a plain factory; only the run's code
    # uses the tracking one, whose "no open session during I/O" check counts sessions of all threads.
    plain = sessionmaker(bind=engine, expire_on_commit=False)

    def status() -> RunStatus:
        with session_scope(plain) as session:
            return session.get_one(ResearchRun, run_id).status

    worker = worker_for(h, settings=settings, session_factory=plain)
    run_id = new_run(h, auto_approve=True)
    result: list[int] = []
    thread = threading.Thread(
        target=lambda: result.append(worker.serve(lambda: postgres_checkpointer(db_url)))
    )
    thread.start()
    try:
        deadline = time.monotonic() + 60
        while status() not in TERMINAL_RUN_STATUSES and time.monotonic() < deadline:
            time.sleep(0.05)
        with session_scope(plain) as session:
            assert session.get(WorkerHeartbeat, worker.id) is not None
        second = worker_for(h, settings=settings, session_factory=plain)
        assert second.serve(lambda: postgres_checkpointer(db_url)) == 2  # the lock is held
    finally:
        worker.request_stop()
        thread.join(timeout=30)
    assert result == [0] and status() is RunStatus.COMPLETED
    with session_scope(plain) as session:
        assert session.get(WorkerHeartbeat, worker.id) is None  # removed on clean exit


# -- plan approval ---------------------------------------------------------------------------


def test_pause_restart_edit_approve_resumes_from_checkpoint(
    h: Harness, checkpointer: Any, db_url: str, engine: Engine
) -> None:
    h.search.results["robots txt standard"] = [RFC9309]
    run_id = new_run(h)
    paused = worker_for(h).run_once(checkpointer)
    assert paused is not None and paused.status is RunStatus.AWAITING_APPROVAL
    assert run_status(h, run_id)[0] is RunStatus.AWAITING_APPROVAL
    assert h.model.count("plan") == 1 and h.model.count("select") == 0
    assert checkpoint_rows(engine, run_id) > 0

    client = client_for(make_app(h, db_url))
    detail = client.get(f"/runs/{run_id}").json()
    assert [s["question"] for s in detail["sub_questions"]] == [s["question"] for s in PLAN["sub_questions"]]
    response = client.post(f"/runs/{run_id}/approve", json={"sub_questions": EDITED})
    assert response.status_code == 200 and response.json()["status"] == "queued"

    # A new worker with a new checkpointer connection and a new graph (a restart).
    with postgres_checkpointer(db_url) as fresh:
        outcome = worker_for(h).run_once(fresh)
    assert outcome is not None and outcome.status is RunStatus.COMPLETED, outcome
    assert h.model.count("plan") == 1  # resumed after the plan, not rerun
    assert "robots txt standard" in h.search.queries and "robots exclusion" not in h.search.queries
    [edit] = event_payloads(h, run_id, "plan_edited")
    assert edit["before"][2]["search_queries"] == ["robots exclusion"]
    assert edit["after"][2] == {
        "question": EDITED[2]["question"],
        "search_queries": ["robots txt standard"],
        "premise_check": False,
    }
    names = event_names(h, run_id)
    assert names.index("awaiting_approval") < names.index("plan_approved") < names.index("run_resumed")
    assert "plan_resumed" in names and names.count("run_start") == 1
    assert checkpoint_rows(engine, run_id) == 0
    report = client.get(f"/runs/{run_id}/report.md").text
    assert "What does the robots.txt standard say?" in report


def test_reject_cancels_and_drops_checkpoint(
    h: Harness, checkpointer: Any, db_url: str, engine: Engine
) -> None:
    run_id = new_run(h)
    worker_for(h).run_once(checkpointer)
    client = client_for(make_app(h, db_url))
    response = client.post(f"/runs/{run_id}/reject", json={})
    assert response.status_code == 200 and response.json()["status"] == "cancelled"
    assert run_status(h, run_id) == (RunStatus.CANCELLED, "plan rejected")
    assert checkpoint_rows(engine, run_id) == 0
    assert worker_for(h).run_once(checkpointer) is None  # nothing left to resume


def test_checkpoint_serializer_is_strict_and_round_trips_state() -> None:
    a, b = uuid.uuid4(), uuid.uuid4()
    state = GraphState(run_id=a, sub_question_ids=[a, b], queries_searched={a: 1}, claim_chunks={b: [a]})
    serde = strict_serializer()
    restored = serde.loads_typed(serde.dumps_typed(state.model_dump()))
    assert GraphState.model_validate(restored) == state


# -- cancel and stop -------------------------------------------------------------------------


def test_cancel_running_run_stops_at_next_check(h: Harness, checkpointer: Any, db_url: str) -> None:
    run_id = new_run(h, auto_approve=True)
    client = client_for(make_app(h, db_url))
    calls = 0

    def verify(call: Any) -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            response = client.post(f"/runs/{run_id}/cancel", json={})
            assert response.status_code == 202 and response.json()["cancel_requested"] is True
        return supported(call)

    h.model.verify = verify
    outcome = worker_for(h).run_once(checkpointer)
    assert outcome is not None and outcome.status is RunStatus.CANCELLED
    assert run_status(h, run_id) == (RunStatus.CANCELLED, CANCELLED_REASON)
    assert calls == 1  # no further verify call after the cancel
    with session_scope(h.factory) as session:
        claims = list(session.scalars(select(Claim).where(Claim.run_id == run_id)))
        assert claims and sum(c.verdict is not None for c in claims) == 1  # what was stored is kept
        assert session.get(ResearchRun, run_id) is not None
    assert client.get(f"/runs/{run_id}/report.md").status_code == 404
    assert "node_stopped" in event_names(h, run_id)


def test_cancel_queued_and_awaiting_is_immediate(
    h: Harness, checkpointer: Any, db_url: str, engine: Engine
) -> None:
    client = client_for(make_app(h, db_url))
    queued = client.post("/runs", json={"question": "What is an okapi?"}).json()["id"]
    response = client.post(f"/runs/{queued}/cancel", json={})
    assert response.status_code == 200 and response.json()["status"] == "cancelled"
    awaiting = new_run(h)
    worker_for(h).run_once(checkpointer)
    assert run_status(h, awaiting)[0] is RunStatus.AWAITING_APPROVAL
    response = client.post(f"/runs/{awaiting}/cancel", json={})
    assert response.status_code == 200 and run_status(h, awaiting) == (RunStatus.CANCELLED, CANCELLED_REASON)
    assert checkpoint_rows(engine, awaiting) == 0
    assert event_names(h, awaiting)[-1] == "run_finished"
    assert client.post(f"/runs/{awaiting}/cancel", json={}).status_code == 409  # already terminal


def test_stop_fails_the_current_run_but_keeps_an_awaiting_one(h: Harness, checkpointer: Any) -> None:
    awaiting = new_run(h)
    worker = worker_for(h)
    worker.run_once(checkpointer)
    running = new_run(h, auto_approve=True)

    def verify(call: Any) -> str:
        worker.request_stop()  # first Ctrl+C while the run is in verify
        return supported(call)

    h.model.verify = verify
    outcome = worker.run_once(checkpointer)
    assert outcome is not None and outcome.status is RunStatus.FAILED
    assert run_status(h, running) == (RunStatus.FAILED, WORKER_STOPPED_REASON)
    assert run_status(h, awaiting)[0] is RunStatus.AWAITING_APPROVAL
    assert worker.run_once(checkpointer) is None  # stopping: claims nothing new


def test_hard_stop_marks_the_run_failed(h: Harness, checkpointer: Any) -> None:
    run_id = new_run(h, auto_approve=True)

    def verify(call: Any) -> str:
        raise KeyboardInterrupt  # second Ctrl+C

    h.model.verify = verify
    with pytest.raises(KeyboardInterrupt):
        worker_for(h).run_once(checkpointer)
    assert run_status(h, run_id) == (RunStatus.FAILED, WORKER_STOPPED_REASON)


def test_lock_timeout_guard_is_real(h: Harness) -> None:
    """The SKIP LOCKED test relies on lock_timeout turning a blocked claim into an error."""
    run_id = new_run(h)
    a, b = h.factory(), h.factory()
    try:
        a.execute(select(ResearchRun).where(ResearchRun.id == run_id).with_for_update())
        b.execute(text("SET LOCAL lock_timeout = '200ms'"))
        with pytest.raises(OperationalError):
            b.execute(select(ResearchRun).where(ResearchRun.id == run_id).with_for_update())
    finally:
        b.rollback()
        a.rollback()
        a.close()
        b.close()
