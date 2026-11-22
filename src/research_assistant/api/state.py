"""What the routes share: settings, sessions, the health HTTP client, checkpoint cleanup, SSE slots."""

import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field

import httpx
from fastapi import Request
from sqlalchemy.orm import Session

from research_assistant.config import Settings


class StreamSlots:
    """Counts open event streams; acquire() fails once SSE_MAX_STREAMS are open."""

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.active = 0
        self._lock = threading.Lock()

    def acquire(self) -> bool:
        with self._lock:
            if self.active >= self.limit:
                return False
            self.active += 1
            return True

    def release(self) -> None:
        with self._lock:
            self.active -= 1


@dataclass
class Lease:
    """One acquired stream slot. release() is idempotent: the generator and the response both call it."""

    slots: StreamSlots
    released: bool = False
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def release(self) -> None:
        with self._lock:
            if self.released:
                return
            self.released = True
        self.slots.release()


@dataclass
class ApiState:
    settings: Settings
    session_factory: Callable[[], Session]
    http_client: httpx.Client  # health checks of Ollama and SearXNG
    delete_checkpoint: Callable[[uuid.UUID], None]
    streams: StreamSlots
    polls: int = 0  # event-stream database polls (observable in tests)


def api_state(request: Request) -> ApiState:
    state = request.app.state.api
    assert isinstance(state, ApiState)
    return state
