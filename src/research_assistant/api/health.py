"""GET /health: database, Ollama, SearXNG and the worker heartbeat. 200 if all are up, else 503.

Without a session (when AUTH_ENABLED) only the overall status is returned; the details need a session.
"""

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from sqlalchemy import text

from research_assistant.api.auth import SessionsDep, authenticated
from research_assistant.api.runs import State
from research_assistant.api.schemas import ComponentHealth, Health, HealthStatus, WorkerHealth
from research_assistant.api.state import ApiState
from research_assistant.db.repositories.queue import latest_heartbeat
from research_assistant.db.session import session_scope

router = APIRouter()
HTTP_TIMEOUT_S = 2.0


def _http(state: ApiState, url: str) -> ComponentHealth:
    try:
        response = state.http_client.get(url, timeout=HTTP_TIMEOUT_S)
    except Exception as e:  # any transport failure means "down"; the message names only the error type
        return ComponentHealth(ok=False, detail=f"unreachable ({type(e).__name__})")
    if response.status_code != 200:
        return ComponentHealth(ok=False, detail=f"HTTP {response.status_code}")
    return ComponentHealth(ok=True)


def _database_and_worker(state: ApiState) -> tuple[ComponentHealth, WorkerHealth]:
    try:
        with session_scope(state.session_factory) as session:
            session.execute(text("SELECT 1"))
            beat = latest_heartbeat(session)
    except Exception as e:
        down = f"unreachable ({type(e).__name__})"
        return ComponentHealth(ok=False, detail=down), WorkerHealth(ok=False, detail="database unreachable")
    if beat is None:
        return ComponentHealth(ok=True), WorkerHealth(ok=False, detail="no worker heartbeat")
    row, age = beat
    fresh = age <= state.settings.worker_stale_s
    return ComponentHealth(ok=True), WorkerHealth(
        ok=fresh,
        detail=None if fresh else f"last heartbeat {age:.0f} s ago",
        last_seen=row.last_seen,
        age_s=round(age, 1),
        current_run_id=row.current_run_id,
    )


@router.get(
    "/health",
    response_model=Health | HealthStatus,
    responses={503: {"model": Health | HealthStatus}},
)
def health(state: State, request: Request, store: SessionsDep) -> Health | HealthStatus | JSONResponse:
    db, worker = _database_and_worker(state)
    s = state.settings
    ollama = _http(state, f"{s.ollama_url.rstrip('/')}/api/version")
    searxng = _http(state, f"{s.searxng_url.rstrip('/')}/healthz")
    ok = db.ok and ollama.ok and searxng.ok and worker.ok
    status = "ok" if ok else "degraded"
    result: Health | HealthStatus = HealthStatus(status=status)
    if authenticated(request, store):
        result = Health(status=status, db=db, ollama=ollama, searxng=searxng, worker=worker)
    if ok:
        return result
    return JSONResponse(result.model_dump(mode="json"), status_code=503)
