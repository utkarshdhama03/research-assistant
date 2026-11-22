"""GET /runs/{id}/events: server-sent events.

Replays the run's events after Last-Event-ID, then polls the database every SSE_POLL_S. Each poll
runs in a worker thread with its own short session, so no session or connection is held while the
stream sleeps. Ends with `event: end` once the run is terminal (the runner commits run_finished in
the same transaction as the terminal status, so nothing is missed). Stops when the client
disconnects. At most SSE_MAX_STREAMS streams are open at once; more get 503.
"""

import json
import time
import uuid
from collections.abc import AsyncIterator
from datetime import datetime
from typing import Annotated, Any

import anyio
from fastapi import APIRouter, Header, Request
from fastapi.responses import StreamingResponse
from fastapi.sse import format_sse_event
from sqlalchemy import select, tuple_
from starlette.background import BackgroundTask

from research_assistant.api.errors import ApiError
from research_assistant.api.runs import State
from research_assistant.api.state import ApiState, Lease
from research_assistant.db.models import TERMINAL_RUN_STATUSES, ResearchRun, RunEvent
from research_assistant.db.session import session_scope

router = APIRouter()
KEEPALIVE_S = 15.0
Cursor = tuple[datetime, uuid.UUID] | None


def _start_cursor(state: ApiState, run_id: uuid.UUID, last_event_id: str | None) -> Cursor:
    with session_scope(state.session_factory) as session:
        if session.get(ResearchRun, run_id) is None:
            raise ApiError(404, f"run {run_id} not found")
        if last_event_id is None:
            return None
        try:
            event_id = uuid.UUID(last_event_id)
        except ValueError as e:
            raise ApiError(422, "Last-Event-ID is not an event id") from e
        event = session.get(RunEvent, event_id)
        if event is None or event.run_id != run_id:
            raise ApiError(422, "Last-Event-ID is not an event of this run")
        return event.created_at, event.id


def _poll(state: ApiState, run_id: uuid.UUID, cursor: Cursor) -> tuple[list[dict[str, Any]], str | None]:
    """Status first, then events: a terminal status read here means all its events are visible."""
    state.polls += 1
    with session_scope(state.session_factory) as session:
        status = session.scalar(select(ResearchRun.status).where(ResearchRun.id == run_id))
        stmt = select(RunEvent).where(RunEvent.run_id == run_id)
        if cursor is not None:
            stmt = stmt.where(tuple_(RunEvent.created_at, RunEvent.id) > tuple_(*cursor))
        rows = session.scalars(stmt.order_by(RunEvent.created_at, RunEvent.id))
        events = [
            {
                "id": e.id,
                "created_at": e.created_at,
                "data": {
                    "node": e.node,
                    "event": e.event,
                    "payload": e.payload,
                    "created_at": e.created_at.isoformat(),
                },
            }
            for e in rows
        ]
    terminal = status is None or status in TERMINAL_RUN_STATUSES  # None: the run was deleted
    return events, (str(status) if status is not None else "deleted") if terminal else None


async def _stream(
    state: ApiState, request: Request, run_id: uuid.UUID, cursor: Cursor, lease: Lease
) -> AsyncIterator[bytes]:
    try:
        last_sent = time.monotonic()
        while True:
            if await request.is_disconnected():
                return
            events, final_status = await anyio.to_thread.run_sync(_poll, state, run_id, cursor)
            for e in events:
                yield format_sse_event(
                    data_str=json.dumps(e["data"]), event=e["data"]["event"], id=str(e["id"])
                )
                cursor = (e["created_at"], e["id"])
                last_sent = time.monotonic()
            if final_status is not None:
                yield format_sse_event(data_str=json.dumps({"status": final_status}), event="end")
                return
            if time.monotonic() - last_sent >= KEEPALIVE_S:
                yield b": ping\n\n"
                last_sent = time.monotonic()
            await anyio.sleep(state.settings.sse_poll_s)
    finally:
        lease.release()


@router.get("/runs/{run_id}/events")
async def events(
    run_id: uuid.UUID,
    request: Request,
    state: State,
    last_event_id: Annotated[str | None, Header()] = None,
) -> StreamingResponse:
    cursor = await anyio.to_thread.run_sync(_start_cursor, state, run_id, last_event_id)
    if not state.streams.acquire():
        raise ApiError(503, f"too many open event streams (SSE_MAX_STREAMS={state.streams.limit})")
    lease = Lease(state.streams)
    return StreamingResponse(
        _stream(state, request, run_id, cursor, lease),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        # Releases the slot even if the generator never started (client gone before the first byte).
        background=BackgroundTask(lease.release),
    )
