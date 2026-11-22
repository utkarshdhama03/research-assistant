"""Per-run context: dependencies, deadlines, problems found, and event logging.

Nothing here is graph state. Database access goes through RunContext.db(), a short session that
must never be held open across an LLM call, a fetch or an embedding call.
"""

import dataclasses
import time
import uuid
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Protocol

from sqlalchemy import select
from sqlalchemy.orm import Session

from research_assistant.config import Settings
from research_assistant.db.models import ResearchRun
from research_assistant.db.repositories.events import append_event
from research_assistant.db.session import session_scope
from research_assistant.llm.client import LLMCallRecord, LLMClient, OnCall
from research_assistant.tools.fetch import FetchResult
from research_assistant.tools.search import SearchProvider


class FetcherLike(Protocol):
    def fetch(self, url: str) -> FetchResult: ...


class EmbedderLike(Protocol):
    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...

    def embed_for_clustering(self, texts: Sequence[str]) -> list[list[float]]: ...

    def warm_up(self) -> float: ...


EventSink = Callable[[str, str, dict[str, Any]], None]


class RunStopped(Exception):
    """The run must stop before its next unit of work. What it stored so far is kept."""


class RunCancelled(RunStopped):
    """research_runs.cancel_requested was set (POST /runs/{id}/cancel)."""


class WorkerStopping(RunStopped):
    """The worker process is shutting down."""


@dataclass
class Deps:
    settings: Settings
    session_factory: Callable[[], Session]
    make_llm: Callable[[OnCall], LLMClient]  # the runner passes the per-run on_call logger
    search: SearchProvider
    fetcher: FetcherLike
    embedder: EmbedderLike
    clock: Callable[[], float] = time.monotonic
    on_event: EventSink | None = None  # progress output (CLI); called after the event is stored
    stop_requested: Callable[[], bool] | None = None  # the worker's shutdown flag
    sleep: Callable[[float], None] = time.sleep  # search pacing (tests pass a fake that advances the clock)
    # eval --record: called with (query, SearchTrace) after every logical search
    search_recorder: Callable[[str, Any], None] | None = None


@dataclass(frozen=True)
class NodeResult:
    update: dict[str, Any]
    summary: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunContext:
    deps: Deps
    run_id: uuid.UUID
    question: str
    started_at: float
    llm: LLMClient = dataclasses.field(init=False)
    current_node: str = "run"
    issues: list[str] = field(default_factory=list)  # problems that make the run partial
    plan_failure: str | None = None
    auto_approve: bool = True  # False: pause for plan approval (needs a checkpointer)
    last_search_at: float | None = None

    def __post_init__(self) -> None:
        self.llm = self.deps.make_llm(self.log_llm_call)

    @property
    def settings(self) -> Settings:
        return self.deps.settings

    def now(self) -> float:
        return self.deps.clock()

    @property
    def run_deadline_at(self) -> float:
        return self.started_at + self.settings.run_deadline_s

    @property
    def research_deadline_at(self) -> float:
        return self.started_at + self.settings.run_deadline_s * self.settings.research_budget_share

    @property
    def verify_deadline_at(self) -> float:
        """analyze, verify and gap stop starting new work here; the rest is left for writing."""
        return self.started_at + self.settings.run_deadline_s * self.settings.verify_budget_share

    @contextmanager
    def db(self) -> Iterator[Session]:
        with session_scope(self.deps.session_factory) as session:
            yield session

    def event(self, event: str, payload: dict[str, Any] | None = None, *, node: str | None = None) -> None:
        node = node or self.current_node
        data = payload or {}
        with self.db() as session:
            append_event(session, self.run_id, node, event, data)
        if self.deps.on_event is not None:
            self.deps.on_event(node, event, data)

    def pace_search(self) -> None:
        """Wait until SEARCH_MIN_INTERVAL_S has passed since the previous search of this run."""
        interval = self.settings.search_min_interval_s
        if self.last_search_at is not None and interval > 0:
            wait = self.last_search_at + interval - self.now()
            if wait > 0:
                self.deps.sleep(wait)
        self.last_search_at = self.now()

    def ensure_not_stopped(self) -> None:
        """Called before each unit of work (node, search, fetch, LLM call): raises WorkerStopping if the
        worker is shutting down, RunCancelled if a cancel was requested for this run."""
        if self.deps.stop_requested is not None and self.deps.stop_requested():
            raise WorkerStopping("worker stopped")
        with self.db() as session:
            requested = session.scalar(
                select(ResearchRun.cancel_requested).where(ResearchRun.id == self.run_id)
            )
        if requested:
            raise RunCancelled("cancelled by user")

    def issue(self, message: str, **payload: Any) -> None:
        """Record a problem that makes the run partial (and log it as an event). Repeats count once."""
        if message not in self.issues:
            self.issues.append(message)
        self.event("problem", {"message": message, **payload})

    def log_llm_call(self, record: LLMCallRecord) -> None:
        self.event("llm_call", dataclasses.asdict(record))
