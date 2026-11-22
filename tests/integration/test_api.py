"""HTTP API against the test database: endpoints, errors, local-only guard, health, event streams."""

import threading
import time
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import anyio
import httpx
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import sessionmaker

from research_assistant.api.app import create_app
from research_assistant.db.models import RunStatus, WorkerHeartbeat
from research_assistant.db.repositories.events import append_event
from research_assistant.db.repositories.reports import save_report
from research_assistant.db.repositories.runs import update_run_status
from research_assistant.db.session import session_scope
from tests.integration.conftest import TrackingSessionFactory
from tests.integration.graph_fakes import PLAN
from tests.integration.phase7_support import (
    API_PORT,
    AsgiStream,
    client_for,
    event_payloads,
    health_transport,
    make_app,
    new_run,
    parse_sse,
    run_status,
)
from tests.integration.test_graph import Harness


@pytest.fixture
def h(committing_factory: TrackingSessionFactory) -> Harness:
    return Harness(committing_factory)


@pytest.fixture
def client(h: Harness, db_url: str, checkpoint_tables: None) -> TestClient:
    return client_for(make_app(h, db_url))


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def plan_body(sub_questions: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    subs = sub_questions or [
        {"question": s["question"], "search_queries": list(s["search_queries"])}
        for s in PLAN["sub_questions"]
    ]
    return {"sub_questions": subs}


def assert_error(response: httpx.Response, status: int, code: str) -> dict[str, Any]:
    assert response.status_code == status, response.text
    body = response.json()
    assert set(body) == {"error"} and body["error"]["code"] == code and body["error"]["message"]
    return body["error"]  # type: ignore[no-any-return]


def finished_run_with_events(h: Harness, n: int = 3) -> tuple[uuid.UUID, list[uuid.UUID]]:
    run_id = new_run(h, status=RunStatus.RUNNING, auto_approve=True)
    with session_scope(h.factory) as session:
        ids = [append_event(session, run_id, "plan", f"step_{i}", {"i": i}).id for i in range(n)]
        update_run_status(session, run_id, RunStatus.COMPLETED)
        ids.append(append_event(session, run_id, "run", "run_finished", {"status": "completed"}).id)
    return run_id, ids


# -- create, list, detail, report ------------------------------------------------------------


def test_create_run_is_queued(client: TestClient, h: Harness) -> None:
    response = client.post("/runs", json={"question": "   What is an okapi?  ", "auto_approve": True})
    assert response.status_code == 202
    run_id = uuid.UUID(response.json()["id"])
    assert response.json()["status"] == "queued"
    detail = client.get(f"/runs/{run_id}").json()
    assert detail["question"] == "What is an okapi?" and detail["auto_approve"] is True
    assert detail["sub_questions"] == [] and detail["report_markdown"] is None
    assert run_status(h, run_id) == (RunStatus.QUEUED, None)


@pytest.mark.parametrize(
    "body",
    [
        {"question": "   "},
        {"question": "abc"},
        {"question": "x" * 501},
        {"question": "What is an okapi?", "extra": 1},
        {"auto_approve": True},
        {"question": 5},
    ],
)
def test_create_run_validation(client: TestClient, body: dict[str, Any]) -> None:
    error = assert_error(client.post("/runs", json=body), 422, "validation_error")
    assert "x" * 50 not in str(error)  # the input is not echoed back


def test_list_runs_keyset_pagination(client: TestClient, h: Harness) -> None:
    created = [new_run(h, question=f"Question number {i}?") for i in range(5)]
    seen: list[str] = []
    cursor = None
    pages = 0
    while True:
        params: dict[str, Any] = {"limit": 2, **({"cursor": cursor} if cursor else {})}
        page = client.get("/runs", params=params).json()
        seen += [item["id"] for item in page["items"]]
        pages += 1
        cursor = page["next_cursor"]
        if cursor is None:
            break
    assert pages == 3 and len(seen) == len(set(seen)) == 5
    assert seen == [str(r) for r in reversed(created)]  # newest first


@pytest.mark.parametrize("params", [{"limit": 0}, {"limit": 101}, {"cursor": "not-a-cursor"}, {"limit": "x"}])
def test_list_runs_validation(client: TestClient, params: dict[str, Any]) -> None:
    assert_error(client.get("/runs", params=params), 422, "validation_error")


def test_detail_shows_plan_timings_claims_and_report(client: TestClient, h: Harness) -> None:
    run_id = new_run(h, status=RunStatus.RUNNING, with_plan=True)
    with session_scope(h.factory) as session:
        append_event(session, run_id, "plan", "node_end", {"duration_s": 1.5})
        append_event(session, run_id, "research", "node_error", {"duration_s": 0.25})
        save_report(session, run_id, "# Report\n\ntext [1]", [])
    detail = client.get(f"/runs/{run_id}").json()
    assert [s["search_queries"] for s in detail["sub_questions"]] == [
        s["search_queries"] for s in PLAN["sub_questions"]
    ]
    assert detail["node_timings"] == [
        {"node": "plan", "event": "node_end", "duration_s": 1.5},
        {"node": "research", "event": "node_error", "duration_s": 0.25},
    ]
    assert detail["claims_by_verdict"] == {} and detail["report_markdown"].startswith("# Report")
    report = client.get(f"/runs/{run_id}/report.md")
    assert report.status_code == 200 and report.headers["content-type"] == "text/markdown; charset=utf-8"
    assert report.text == "# Report\n\ntext [1]"


def test_not_found(client: TestClient, h: Harness) -> None:
    missing = uuid.uuid4()
    for response in (
        client.get(f"/runs/{missing}"),
        client.get(f"/runs/{missing}/report.md"),
        client.get(f"/runs/{missing}/events"),
        client.post(f"/runs/{missing}/approve", json={}),
        client.post(f"/runs/{missing}/reject", json={}),
        client.post(f"/runs/{missing}/cancel", json={}),
        client.get("/nope"),
    ):
        assert_error(response, 404, "not_found")
    no_report = new_run(h)
    assert_error(client.get(f"/runs/{no_report}/report.md"), 404, "not_found")
    assert_error(client.get("/runs/not-a-uuid"), 422, "validation_error")


# -- approve / reject / cancel: status rules and plan validation -----------------------------


def test_wrong_status_is_409(client: TestClient, h: Harness) -> None:
    queued = new_run(h)
    completed = new_run(h, status=RunStatus.COMPLETED)
    running = new_run(h, status=RunStatus.RUNNING)
    error = assert_error(client.post(f"/runs/{queued}/approve", json={}), 409, "conflict")
    assert "queued" in error["message"]
    assert_error(client.post(f"/runs/{running}/reject", json={}), 409, "conflict")
    assert_error(client.post(f"/runs/{completed}/cancel", json={}), 409, "conflict")
    assert run_status(h, queued)[0] is RunStatus.QUEUED


def test_approve_without_edit(client: TestClient, h: Harness) -> None:
    run_id = new_run(h, status=RunStatus.AWAITING_APPROVAL, with_plan=True)
    response = client.post(f"/runs/{run_id}/approve", json={})
    assert response.status_code == 200 and response.json()["status"] == "queued"
    assert event_payloads(h, run_id, "plan_approved") == [{"edited": False}]
    assert event_payloads(h, run_id, "plan_edited") == []


def _subs(*queries: list[str]) -> list[dict[str, Any]]:
    return [
        {"question": f"Distinct sub-question number {i}?", "search_queries": q} for i, q in enumerate(queries)
    ]


@pytest.mark.parametrize(
    "subs",
    [
        _subs(["alpha one"], ["beta two"]),  # 2 sub-questions (< 3)
        _subs(["alpha one", "alpha one"], ["beta two"], ["gamma three"]),  # duplicate query
        _subs(["a1 x", "a2 y", "a3 z", "a4 w"], ["beta two"], ["gamma three"]),  # 4 queries (> 3)
        _subs([], ["beta two"], ["gamma three"]),  # no query
        _subs(["alpha one"], ["beta two"], ["gamma three"], ["delta four"]),  # 4 > MAX_SUB_QUESTIONS (3)
        [{"question": "hi", "search_queries": ["alpha one"]}, *_subs(["beta two"], ["gamma three"])],  # short
    ],
)
def test_invalid_edited_plan_is_422(client: TestClient, h: Harness, subs: list[dict[str, Any]]) -> None:
    run_id = new_run(h, status=RunStatus.AWAITING_APPROVAL, with_plan=True)
    assert_error(
        client.post(f"/runs/{run_id}/approve", json={"sub_questions": subs}), 422, "validation_error"
    )
    assert run_status(h, run_id)[0] is RunStatus.AWAITING_APPROVAL


def test_edited_plan_near_duplicate_queries_are_dropped_and_logged(client: TestClient, h: Harness) -> None:
    run_id = new_run(h, status=RunStatus.AWAITING_APPROVAL, with_plan=True)
    subs = _subs(["okapi habitat congo"], ["congo okapi habitat", "dns reserved names"], ["robots txt rules"])
    response = client.post(f"/runs/{run_id}/approve", json={"sub_questions": subs})
    assert response.status_code == 200
    [edit] = event_payloads(h, run_id, "plan_edited")
    assert edit["dropped_queries"] == ["congo okapi habitat"]
    assert edit["after"][1]["search_queries"] == ["dns reserved names"]
    detail = client.get(f"/runs/{run_id}").json()
    assert [s["question"] for s in detail["sub_questions"]] == [s["question"] for s in subs]


# -- local-only guard ------------------------------------------------------------------------


@pytest.mark.parametrize(
    "host", ["evil.example:8000", "evil.example", "127.0.0.1:9999", "localhost", "[::1]:8000"]
)
def test_foreign_host_rejected(client: TestClient, host: str) -> None:
    """DNS rebinding: a page on another name that resolves to 127.0.0.1 sends its own Host header."""
    assert_error(client.get("/runs", headers={"host": host}), 400, "bad_request")


def test_local_hosts_accepted(h: Harness, db_url: str) -> None:
    client = client_for(make_app(h, db_url))
    for host in (f"127.0.0.1:{API_PORT}", f"localhost:{API_PORT}"):
        assert client.get("/runs", headers={"host": host}).status_code == 200


def test_post_requires_json_content_type(client: TestClient, h: Harness) -> None:
    run_id = new_run(h)
    plain = client.post(f"/runs/{run_id}/cancel", content=b"{}", headers={"content-type": "text/plain"})
    assert_error(plain, 415, "unsupported_media_type")
    form = client.post(
        "/runs", content=b"question=hi", headers={"content-type": "application/x-www-form-urlencoded"}
    )
    assert_error(form, 415, "unsupported_media_type")
    missing = client.post("/runs", content=b'{"question": "What is an okapi?"}')
    assert_error(missing, 415, "unsupported_media_type")
    assert run_status(h, run_id)[0] is RunStatus.QUEUED
    ok = client.post(
        "/runs",
        content=b'{"question": "What is an okapi?"}',
        headers={"content-type": "application/json; charset=utf-8"},
    )
    assert ok.status_code == 202


def test_cors_only_for_the_local_ui(client: TestClient) -> None:
    def preflight(origin: str) -> httpx.Response:
        return client.options(
            "/runs",
            headers={
                "origin": origin,
                "access-control-request-method": "POST",
                "access-control-request-headers": "content-type",
            },
        )

    for origin in ("http://127.0.0.1:5173", "http://localhost:5173"):
        assert preflight(origin).headers.get("access-control-allow-origin") == origin
    for origin in ("http://evil.example", "http://localhost:3000"):
        assert "access-control-allow-origin" not in preflight(origin).headers


def test_unexpected_error_is_generic_500(h: Harness, db_url: str) -> None:
    app = make_app(h, db_url)

    def boom() -> None:
        raise RuntimeError("SELECT secret FROM research_runs WHERE password = 'x'")

    app.add_api_route("/boom", boom)
    response = client_for(app, raise_server_exceptions=False).get("/boom")
    error = assert_error(response, 500, "internal_error")
    assert (
        "SELECT" not in response.text
        and "Traceback" not in response.text
        and error["message"] == "internal server error"
    )


# -- health ----------------------------------------------------------------------------------


def _beat(h: Harness, age_s: float) -> None:
    with session_scope(h.factory) as session:
        session.add(WorkerHeartbeat(pid=1, last_seen=datetime.now(UTC) - timedelta(seconds=age_s)))


def test_health_all_up(h: Harness, db_url: str, checkpointer: Any) -> None:
    _beat(h, 1)
    response = client_for(make_app(h, db_url)).get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok" and all(body[k]["ok"] for k in ("db", "ollama", "searxng", "worker"))


def test_health_dead_ollama_and_stale_worker(h: Harness, db_url: str, checkpointer: Any) -> None:
    _beat(h, 120)
    app = make_app(h, db_url, transport=health_transport(ollama=False))
    response = client_for(app).get("/health")
    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "degraded" and body["db"]["ok"] and body["searxng"]["ok"]
    assert body["ollama"] == {"ok": False, "detail": "unreachable (ConnectError)"}
    assert body["worker"]["ok"] is False and body["worker"]["detail"].startswith("last heartbeat")


def test_health_dead_database(h: Harness, checkpointer: Any) -> None:
    def broken() -> Any:
        raise OperationalError("SELECT 1", {}, Exception("connection refused"))

    app = create_app(
        h.settings.model_copy(update={"api_port": API_PORT}),
        broken,
        http_client=httpx.Client(transport=health_transport()),
        delete_checkpoint=lambda _run_id: None,
        extra_hosts=("testserver",),
    )
    body = client_for(app).get("/health").json()
    assert (
        body["db"] == {"ok": False, "detail": "unreachable (OperationalError)"}
        and body["status"] == "degraded"
    )


# -- event stream ----------------------------------------------------------------------------


def test_events_replay_then_end(client: TestClient, h: Harness) -> None:
    run_id, ids = finished_run_with_events(h)
    response = client.get(f"/runs/{run_id}/events")
    assert response.status_code == 200 and response.headers["content-type"].startswith("text/event-stream")
    events = parse_sse(response.text)
    assert [e.get("id") for e in events[:-1]] == [str(i) for i in ids]
    assert [e["event"] for e in events] == ["step_0", "step_1", "step_2", "run_finished", "end"]
    assert events[0]["data"]["payload"] == {"i": 0} and events[-1]["data"] == {"status": "completed"}


def test_events_resume_after_last_event_id(client: TestClient, h: Harness) -> None:
    run_id, ids = finished_run_with_events(h)
    events = parse_sse(client.get(f"/runs/{run_id}/events", headers={"last-event-id": str(ids[1])}).text)
    assert [e["event"] for e in events] == ["step_2", "run_finished", "end"]
    assert_error(
        client.get(f"/runs/{run_id}/events", headers={"last-event-id": "nope"}), 422, "validation_error"
    )
    other, other_ids = finished_run_with_events(h, 1)
    foreign = client.get(f"/runs/{run_id}/events", headers={"last-event-id": str(other_ids[0])})
    assert_error(foreign, 422, "validation_error")


def test_events_stream_live_until_terminal(client: TestClient, h: Harness) -> None:
    run_id = new_run(h, status=RunStatus.RUNNING, auto_approve=True)

    def produce() -> None:
        for i in range(3):
            time.sleep(0.1)
            with session_scope(h.factory) as session:
                append_event(session, run_id, "research", f"live_{i}", {})
        with session_scope(h.factory) as session:
            update_run_status(session, run_id, RunStatus.PARTIAL, error="x")
            append_event(session, run_id, "run", "run_finished", {"status": "partial"})

    producer = threading.Thread(target=produce)
    producer.start()
    with client.stream("GET", f"/runs/{run_id}/events") as response:
        body = "".join(response.iter_text())
    producer.join()
    events = parse_sse(body)
    assert [e["event"] for e in events] == ["live_0", "live_1", "live_2", "run_finished", "end"]
    assert events[-1]["data"] == {"status": "partial"}


@pytest.mark.anyio
async def test_disconnect_ends_stream_and_polling(h: Harness, db_url: str) -> None:
    app = make_app(h, db_url)
    state = app.state.api
    run_id = new_run(h, status=RunStatus.RUNNING, auto_approve=True)
    with session_scope(h.factory) as session:
        append_event(session, run_id, "plan", "node_start", {})
    stream = AsgiStream(app, f"/runs/{run_id}/events")
    async with anyio.create_task_group() as tg:
        tg.start_soon(stream.run)
        with anyio.fail_after(10):
            await stream.first_body.wait()
        assert stream.status == 200 and state.streams.active == 1
        stream.disconnect()
        with anyio.fail_after(10):
            await stream.finished.wait()
    assert state.streams.active == 0
    polls = state.polls
    await anyio.sleep(0.3)  # six poll intervals
    assert state.polls == polls  # nothing polls the database any more
    assert parse_sse(stream.text)[0]["event"] == "node_start"


@pytest.mark.anyio
async def test_stream_limit_and_pool_not_exhausted(h: Harness, db_url: str) -> None:
    # A pool smaller than the stream limit: if a stream held a connection, GET /runs would time out.
    engine = create_engine(db_url, pool_size=2, max_overflow=0, pool_timeout=2)
    try:
        factory = sessionmaker(bind=engine, expire_on_commit=False)
        app = create_app(
            h.settings.model_copy(update={"api_port": API_PORT, "sse_poll_s": 0.05, "sse_max_streams": 4}),
            factory,
            http_client=httpx.Client(transport=health_transport()),
            delete_checkpoint=lambda _run_id: None,
            extra_hosts=("testserver",),
        )
        run_id = new_run(h, status=RunStatus.RUNNING, auto_approve=True)
        with session_scope(h.factory) as session:
            append_event(session, run_id, "plan", "node_start", {})
        streams = [AsgiStream(app, f"/runs/{run_id}/events") for _ in range(4)]
        async with anyio.create_task_group() as tg:
            for s in streams:
                tg.start_soon(s.run)
            with anyio.fail_after(10):
                for s in streams:
                    await s.first_body.wait()
            assert app.state.api.streams.active == 4
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as ac:
                with anyio.fail_after(10):
                    listing = await ac.get("/runs")
                    extra = await ac.get(f"/runs/{run_id}/events")
            assert listing.status_code == 200
            assert extra.status_code == 503 and extra.json()["error"]["code"] == "unavailable"
            for s in streams:
                s.disconnect()
        assert app.state.api.streams.active == 0
    finally:
        engine.dispose()


# -- question search -------------------------------------------------------------------------


def test_search_runs_by_question(client: TestClient, h: Harness) -> None:
    hundred = new_run(h, question="Is 100% renewable power feasible?")
    under = new_run(h, question="What does snake_case mean in Python?")
    plain = new_run(h, question="What is snake case in Python?")
    quote = new_run(h, question="Why can't O'Brien's code run?")
    backslash = new_run(h, question=r"What does C:\temp mean on Windows?")

    def ids(q: str, **params: Any) -> list[str]:
        response = client.get("/runs", params={"q": q, **params})
        assert response.status_code == 200, response.text
        return [item["id"] for item in response.json()["items"]]

    assert ids("100%") == [str(hundred)]  # % is literal, not a wildcard
    assert ids("snake_case") == [str(under)]  # _ is literal: "snake case" does not match
    assert ids("SNAKE") == [str(plain), str(under)]  # case-insensitive, newest first
    assert ids("o'brien") == [str(quote)]
    assert ids(r"c:\temp") == [str(backslash)]
    assert ids("c:\temp") == []  # a TAB (\t) is not a backslash
    for injection in ("%' OR 1=1 --", "'; DROP TABLE research_runs; --"):
        assert ids(injection) == [], injection
    # A lone wildcard character matches only questions that literally contain it.
    assert ids("%") == [str(hundred)] and ids("_") == [str(under)] and ids("\\") == [str(backslash)]
    assert run_status(h, hundred)[0] is RunStatus.QUEUED  # the table is intact
    first = client.get("/runs", params={"q": "python", "limit": 1}).json()
    second = client.get("/runs", params={"q": "python", "limit": 1, "cursor": first["next_cursor"]}).json()
    assert [first["items"][0]["id"], second["items"][0]["id"]] == [str(plain), str(under)]
    assert second["next_cursor"] is None


@pytest.mark.parametrize("q", ["", "x" * 201])
def test_search_query_length(client: TestClient, q: str) -> None:
    assert_error(client.get("/runs", params={"q": q}), 422, "validation_error")
