"""Eval record/replay: fix the search and fetch inputs of an eval so two runs can be compared.

record: every logical search (query -> the final normalized results, plus each provider call) is appended
to <eval_dir>/search_cache.jsonl.
replay: searches are served from that file (exact query match; anything else is a replay_miss) and
fetches from the sources stored for the recorded runs (by normalized URL; a fetched page is served as its
stored extracted text, a recorded failure is raised again, anything else is a replay_miss). Nothing in
replay mode touches the network for search or fetch. The LLM and the embedder still run, so outputs can
differ slightly between runs.
"""

import json
import re
import threading
import uuid
from collections.abc import Callable, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from research_assistant.db.models import Source, SourceStatus
from research_assistant.db.session import session_scope
from research_assistant.tools.fetch import FetchError, FetchErrorReason, FetchResult
from research_assistant.tools.search import SearchError, SearchResult, SearchTrace
from research_assistant.tools.url import normalize_url

SEARCH_CACHE = "search_cache.jsonl"
REPLAY_MISS = FetchErrorReason.REPLAY_MISS.value
_DUPLICATE_OF = re.compile(r"duplicate_content: same text as source ([0-9a-f-]{36})")
_STATUS = re.compile(r"HTTP (\d{3})")


class SearchRecorder:
    """Appends each logical search to search_cache.jsonl (thread-safe; one JSON object per line)."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()

    def __call__(self, query: str, trace: SearchTrace) -> None:
        line = {
            "query": query,
            "provider": trace.calls[-1].provider if trace.calls else None,
            "results": [asdict(r) for r in trace.results],
            "calls": [asdict(c) for c in trace.calls],
            "error": str(trace.error) if trace.error else None,
        }
        with self._lock, self.path.open("a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")


def load_search_cache(path: Path) -> dict[str, list[SearchResult]]:
    """query -> results of its first recorded search."""
    cache: dict[str, list[SearchResult]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        cache.setdefault(entry["query"], [SearchResult(**r) for r in entry["results"]])
    return cache


class ReplaySearch:
    name = "replay"

    def __init__(self, cache: dict[str, list[SearchResult]]) -> None:
        self.cache = cache

    def search(self, query: str, max_results: int) -> list[SearchResult]:
        if query not in self.cache:
            raise SearchError(REPLAY_MISS)
        return list(self.cache[query][:max_results])


class ReplayFetcher:
    """Serves fetches from the sources stored for the recorded runs."""

    def __init__(self, session_factory: Callable[[], Session], run_ids: Sequence[uuid.UUID]) -> None:
        self.session_factory = session_factory
        self.run_ids = list(run_ids)

    def fetch(self, url: str) -> FetchResult:
        try:
            key = normalize_url(url)
        except ValueError:
            key = url
        with session_scope(self.session_factory) as session:
            rows = list(
                session.scalars(
                    select(Source)
                    .where(Source.run_id.in_(self.run_ids), or_(Source.url == key, Source.final_url == key))
                    .order_by(Source.fetched_at.desc().nulls_last())
                )
            )
            fetched = next((r for r in rows if r.status is SourceStatus.FETCHED and r.extracted_text), None)
            if fetched is not None:
                return self._page(url, fetched.final_url or fetched.url, fetched.extracted_text or "")
            for row in rows:
                if row.status is SourceStatus.SKIPPED:
                    duplicate = _DUPLICATE_OF.search(row.error or "")
                    original = session.get(Source, uuid.UUID(duplicate.group(1))) if duplicate else None
                    text = original.extracted_text if original is not None else ""
                    return self._page(url, row.final_url or row.url, text or "")
                if row.status is SourceStatus.FAILED:
                    raise self._recorded_failure(url, row.error or "")
        raise FetchError(FetchErrorReason.REPLAY_MISS, "not fetched in the recorded runs", url=url)

    @staticmethod
    def _page(url: str, final_url: str, text: str) -> FetchResult:
        # The stored text is already extracted and normalized; served as plain text it extracts to itself.
        return FetchResult(url, final_url, 200, "text/plain; charset=utf-8", text.encode("utf-8"))

    @staticmethod
    def _recorded_failure(url: str, error: str) -> FetchError:
        reason_text, _, message = error.partition(": ")
        try:
            reason = FetchErrorReason(reason_text)
        except ValueError:
            reason = FetchErrorReason.REPLAY_MISS
        status = _STATUS.search(message)
        return FetchError(
            reason, message or error, url=url, status_code=int(status.group(1)) if status else None
        )


def recorded_run_ids(eval_dir: Path) -> list[uuid.UUID]:
    results: dict[str, Any] = json.loads((eval_dir / "results.json").read_text(encoding="utf-8"))
    return [uuid.UUID(e["run_id"]) for e in results["runs"].values() if e.get("run_id")]
