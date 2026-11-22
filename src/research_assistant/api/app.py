"""FastAPI app factory and `ra serve`.

`ra serve` binds API_HOST (default 127.0.0.1) and refuses a non-loopback host unless AUTH_ENABLED.
Every route except /auth/login, /auth/session, /health and the UI shell needs a session (see auth.py).
"""

import ipaddress
import time
import uuid
from collections.abc import Callable, Iterable
from typing import Any

import httpx
from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from research_assistant import __version__
from research_assistant.api import auth, events, health, runs, ui
from research_assistant.api.auth import Clock, Sessions, require_session
from research_assistant.api.errors import install_error_handlers
from research_assistant.api.local_only import LocalOnlyMiddleware
from research_assistant.api.state import ApiState, StreamSlots
from research_assistant.config import Settings
from research_assistant.graph.checkpoint import postgres_checkpointer
from research_assistant.logs import LOG_DATEFMT, LOG_FORMAT

CORS_ORIGINS = ("http://127.0.0.1:5173", "http://localhost:5173")  # Vite dev server (Phase 8 UI)


def checkpoint_deleter(database_url: str) -> Callable[[uuid.UUID], None]:
    def delete(run_id: uuid.UUID) -> None:
        with postgres_checkpointer(database_url) as saver:
            saver.delete_thread(str(run_id))

    return delete


def create_app(
    settings: Settings,
    session_factory: Callable[[], Session],
    *,
    http_client: httpx.Client | None = None,
    delete_checkpoint: Callable[[uuid.UUID], None] | None = None,
    extra_hosts: Iterable[str] = (),
    clock: Clock = time.time,
) -> FastAPI:
    # FastAPI's OpenTelemetry hooks stay off: nothing may leave the machine.
    # /docs and /openapi.json only without login (local development): they would answer without a session.
    app = FastAPI(
        title="Research Assistant",
        version=__version__,
        openapi_url=None if settings.auth_enabled else "/openapi.json",
        telemetry={"auto_configure": False, "tracing": False, "metrics": False, "logs": False},
    )
    app.state.api = ApiState(
        settings=settings,
        session_factory=session_factory,
        http_client=http_client or httpx.Client(trust_env=False),
        delete_checkpoint=delete_checkpoint or checkpoint_deleter(settings.database_url),
        streams=StreamSlots(settings.sse_max_streams),
    )
    app.state.sessions = Sessions(settings, clock)
    install_error_handlers(app)
    protected = [Depends(require_session)]
    app.include_router(auth.router)
    app.include_router(runs.router, dependencies=protected)
    app.include_router(events.router, dependencies=protected)
    app.include_router(health.router)  # answers without a session, with the overall status only
    if settings.ui_dist_dir is not None:
        ui.mount(app, settings.ui_dist_dir)
    hosts = [
        f"127.0.0.1:{settings.api_port}",
        f"localhost:{settings.api_port}",
        *settings.api_allowed_hosts,
        *extra_hosts,
    ]
    app.add_middleware(LocalOnlyMiddleware, allowed_hosts=hosts)
    # Added last, so outermost: CORS headers also on the guard's 400/415 responses for the local UI.
    # Credentials: the Vite dev UI on :5173 sends the session cookie to the API on :8000.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(CORS_ORIGINS),
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "Last-Event-ID"],
    )
    return app


def is_loopback(host: str) -> bool:
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host.strip("[]")).is_loopback
    except ValueError:
        return False


def bind_error(settings: Settings) -> str | None:
    """Why `ra serve` must not start with these settings, or None."""
    if not is_loopback(settings.api_host) and not settings.auth_enabled:
        return (
            f"refusing to listen on {settings.api_host}: a non-loopback API_HOST needs AUTH_ENABLED=true "
            "and an ACCESS_KEY (or use API_HOST=127.0.0.1)"
        )
    return None


def uvicorn_log_config() -> dict[str, Any]:
    """uvicorn's loggers in the same one-line stdout format as the rest of the app."""
    formatter = {"()": "research_assistant.logs.UtcFormatter", "fmt": LOG_FORMAT, "datefmt": LOG_DATEFMT}
    handler = {"class": "logging.StreamHandler", "formatter": "default", "stream": "ext://sys.stdout"}
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {"default": formatter},
        "handlers": {"default": handler},
        "loggers": {
            "uvicorn": {"handlers": ["default"], "level": "INFO", "propagate": False},
            "uvicorn.error": {"level": "INFO"},
            "uvicorn.access": {"handlers": ["default"], "level": "INFO", "propagate": False},
        },
    }


def serve(settings: Settings, session_factory: Callable[[], Session]) -> None:
    import uvicorn

    uvicorn.run(
        create_app(settings, session_factory),
        host=settings.api_host,
        port=settings.api_port,
        log_config=uvicorn_log_config(),
        proxy_headers=False,  # no proxy in front; never trust X-Forwarded-* from clients
    )
