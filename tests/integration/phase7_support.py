"""Helpers for the API, queue and worker tests: app factory, worker factory, ASGI stream driver."""

import dataclasses
import json
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

import anyio
import httpx
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import Engine, text
from sqlalchemy.orm import Session

from research_assistant.api.app import checkpoint_deleter, create_app
from research_assistant.config import Settings
from research_assistant.db.models import RunStatus
from research_assistant.db.repositories.events import list_events
from research_assistant.db.repositories.runs import add_sub_questions, create_run
from research_assistant.db.session import session_scope
from research_assistant.graph.context import Deps
from research_assistant.graph.runner import run_config
from research_assistant.worker import Worker
from tests.integration.graph_fakes import PLAN
from tests.integration.test_graph import QUESTION, Harness

API_PORT = 8000


def health_transport(*, ollama: bool = True, searxng: bool = True) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        up = ollama if request.url.path == "/api/version" else searxng
        if not up:
            raise httpx.ConnectError("connection refused", request=request)
        return httpx.Response(200, json={"version": "test"})

    return httpx.MockTransport(handler)


def make_app(
    h: Harness, db_url: str, *, transport: httpx.MockTransport | None = None, **settings: Any
) -> FastAPI:
    s = h.settings.model_copy(update={"api_port": API_PORT, "sse_poll_s": 0.05, **settings})
    return create_app(
        s,
        h.factory,
        http_client=httpx.Client(transport=transport or health_transport()),
        delete_checkpoint=checkpoint_deleter(db_url),
        extra_hosts=("testserver",),
    )


def client_for(app: FastAPI, **kwargs: Any) -> TestClient:
    return TestClient(app, **kwargs)


def worker_for(
    h: Harness,
    logs: list[str] | None = None,
    settings: Settings | None = None,
    session_factory: Callable[[], Session] | None = None,
) -> Worker:
    """A worker whose runs use the harness fakes. session_factory: the worker's own sessions (claims,
    heartbeat); the runs always use the harness's tracking factory."""

    @contextmanager
    def deps_factory(stop_requested: Callable[[], bool]) -> Iterator[Deps]:
        yield dataclasses.replace(h.deps(), stop_requested=stop_requested)

    return Worker(
        settings or h.settings,
        session_factory or h.factory,
        deps_factory,
        log_line=(logs if logs is not None else []).append,
    )


def new_run(
    h: Harness,
    *,
    status: RunStatus = RunStatus.QUEUED,
    auto_approve: bool = False,
    question: str = QUESTION,
    with_plan: bool = False,
) -> uuid.UUID:
    with session_scope(h.factory) as session:
        run = create_run(session, question, run_config(h.settings), status=status, auto_approve=auto_approve)
        if with_plan:
            subs = PLAN["sub_questions"]
            add_sub_questions(
                session, run.id, [s["question"] for s in subs], [s["search_queries"] for s in subs]
            )
        return run.id


def run_status(h: Harness, run_id: uuid.UUID) -> tuple[RunStatus, str | None]:
    with session_scope(h.factory) as session:
        row = session.execute(
            text("SELECT status, error FROM research_runs WHERE id = :id"), {"id": run_id}
        ).one()
        return RunStatus(row[0]), row[1]


def event_names(h: Harness, run_id: uuid.UUID) -> list[str]:
    with session_scope(h.factory) as session:
        return [e.event for e in list_events(session, run_id)]


def event_payloads(h: Harness, run_id: uuid.UUID, name: str) -> list[dict[str, Any]]:
    with session_scope(h.factory) as session:
        return [e.payload for e in list_events(session, run_id) if e.event == name]


def checkpoint_rows(engine: Engine, run_id: uuid.UUID) -> int:
    with engine.connect() as conn:
        return int(
            conn.execute(
                text("SELECT count(*) FROM checkpoints WHERE thread_id = :t"), {"t": str(run_id)}
            ).scalar()
            or 0
        )


def parse_sse(body: str) -> list[dict[str, Any]]:
    """SSE wire text -> [{event, id, data}] (comments skipped)."""
    out = []
    for block in body.replace("\r\n", "\n").split("\n\n"):
        fields: dict[str, Any] = {}
        for line in block.split("\n"):
            if not line or line.startswith(":"):
                continue
            key, _, value = line.partition(": ")
            fields[key] = value
        if fields:
            fields["data"] = json.loads(fields["data"]) if "data" in fields else None
            out.append(fields)
    return out


class AsgiStream:
    """Drive one GET at the ASGI level: collect body chunks, and disconnect on demand."""

    def __init__(self, app: FastAPI, path: str, headers: dict[str, str] | None = None) -> None:
        self.app = app
        self.path = path
        self.headers = {"host": "testserver", **(headers or {})}
        self.status: int | None = None
        self.chunks: list[bytes] = []
        self.started = anyio.Event()
        self.first_body = anyio.Event()
        self.finished = anyio.Event()
        self._disconnect = anyio.Event()
        self._sent_request = False

    def disconnect(self) -> None:
        self._disconnect.set()

    @property
    def text(self) -> str:
        return b"".join(self.chunks).decode()

    async def _receive(self) -> dict[str, Any]:
        if not self._sent_request:
            self._sent_request = True
            return {"type": "http.request", "body": b"", "more_body": False}
        await self._disconnect.wait()
        return {"type": "http.disconnect"}

    async def _send(self, message: dict[str, Any]) -> None:
        if message["type"] == "http.response.start":
            self.status = message["status"]
            self.started.set()
        elif message["type"] == "http.response.body":
            if message.get("body"):
                self.chunks.append(message["body"])
                self.first_body.set()

    async def run(self) -> None:
        scope = {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.3"},
            "http_version": "1.1",
            "method": "GET",
            "scheme": "http",
            "path": self.path,
            "raw_path": self.path.encode(),
            "query_string": b"",
            "root_path": "",
            "headers": [(k.encode(), v.encode()) for k, v in self.headers.items()],
            "client": ("127.0.0.1", 50000),
            "server": ("testserver", 80),
        }
        try:
            await self.app(scope, self._receive, self._send)
        finally:
            self.finished.set()
