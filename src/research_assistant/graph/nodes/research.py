"""research: per active sub-question search -> select -> fetch -> extract -> chunk -> embed -> store.

A selected result that fails to fetch, is skipped at extraction, or duplicates content already in
the run is backfilled with the next-ranked unselected result (up to MAX_BACKFILLS_PER_SUB_QUESTION).
Before every fetch (picks, backfills, gap rounds), with memory on, a fresh fetched source of
another run with the same URL is copied instead (memory_reused, reason url_match).
A source whose rows the database rejects (DataError, IntegrityError) is rolled back on its own,
recorded as failed with reason storage_error and backfilled like a failed fetch.
Memory (MEMORY_ENABLED, first round only): sources of earlier runs similar to the sub-question are
offered to select next to the web results; a chosen one is copied into this run (source + chunks)
instead of fetched. Reused sources count toward MAX_REUSED_SOURCES_PER_RUN, not toward
MAX_SOURCES_PER_RUN (fetch attempts); with the reuse budget spent, such pages are fetched from the web.
The fetch budget and the URLs already seen are read from the database, so they hold across gap
rounds. No session is open during search, LLM, fetch or embed calls.
"""

import uuid
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from urllib.parse import urlsplit

from sqlalchemy.exc import DataError, IntegrityError

from research_assistant.db.models import Source, SourceStatus, SubQuestion
from research_assistant.db.repositories.chunks import ChunkInput, add_chunks
from research_assistant.db.repositories.memory import MemoryCopyError, copy_source, reusable_by_url
from research_assistant.db.repositories.sources import (
    add_source,
    count_sources,
    find_source_by_content_hash,
    seen_urls,
)
from research_assistant.graph.context import NodeResult, RunContext
from research_assistant.graph.memory import MemoryCandidate, lookup, merge
from research_assistant.graph.state import GraphState
from research_assistant.llm.aliases import AliasMap
from research_assistant.llm.errors import LLMError
from research_assistant.llm.prompt_loader import load_prompt, untrusted
from research_assistant.llm.schemas import selection_schema
from research_assistant.tools.chunk import chunk_text
from research_assistant.tools.embed import EmbedError
from research_assistant.tools.extract import ExtractStatus, extract
from research_assistant.tools.fetch import FetchError, FetchErrorReason
from research_assistant.tools.search import SearchResult, dedupe_results, traced_search


@dataclass(frozen=True)
class _SubQuestion:
    id: uuid.UUID
    ordinal: int
    text: str
    queries: list[str]
    premise_check: bool = False


class Outcome(StrEnum):
    FETCHED = "fetched"
    DUPLICATE = "duplicate"
    FETCH_FAILED = "fetch_failed"
    EXTRACT_SKIPPED = "extract_skipped"
    EMBED_FAILED = "embed_failed"
    SEEN_UNUSABLE = "seen_unusable"
    HOST_BLOCKED = "host_blocked"
    STORAGE_FAILED = "storage_failed"
    REUSED = "reused"  # copied from an earlier run (memory)
    MEMORY_UNAVAILABLE = "memory_unavailable"  # the earlier source is gone, or its copy failed


# Outcomes that let the next-ranked unselected result take the pick's place (not embed failures).
BACKFILL_OUTCOMES = frozenset(
    {
        Outcome.DUPLICATE,
        Outcome.FETCH_FAILED,
        Outcome.STORAGE_FAILED,
        Outcome.MEMORY_UNAVAILABLE,
        Outcome.EXTRACT_SKIPPED,
        Outcome.SEEN_UNUSABLE,
        Outcome.HOST_BLOCKED,
    }
)
# HTTP statuses after which a host is not fetched again in the run.
BLOCKING_STATUSES = frozenset({403, 429})


def host_of(url: str) -> str:
    return (urlsplit(url).hostname or "").lower()


def load_sub_questions(ctx: RunContext, ids: list[uuid.UUID]) -> list[_SubQuestion]:
    with ctx.db() as session:
        rows = [session.get(SubQuestion, i) for i in ids]
        return [
            _SubQuestion(r.id, r.ordinal, r.text, list(r.search_queries), r.premise_check)
            for r in rows
            if r is not None
        ]


def research_node(ctx: RunContext, state: GraphState) -> NodeResult:
    s = ctx.settings
    deadline = min(ctx.research_deadline_at, ctx.now() + s.node_timeout_s)
    sub_questions = load_sub_questions(ctx, state.active_sub_question_ids)
    mapping = {k: list(v) for k, v in state.sources_by_sub_question.items()}
    searched = dict(state.queries_searched)
    blocked = set(state.blocked_hosts)
    stats: Counter[str] = Counter()
    with ctx.db() as session:
        # Fetch attempts (web sources) count toward MAX_SOURCES_PER_RUN; sources copied from earlier
        # runs count toward MAX_REUSED_SOURCES_PER_RUN instead.
        attempts = count_sources(session, state.run_id, reused=False)
        reused = count_sources(session, state.run_id, reused=True)
        seen: dict[str, uuid.UUID | None] = {
            url: (sid if status is SourceStatus.FETCHED else None)
            for url, (sid, status) in seen_urls(session, state.run_id).items()
        }
    budget_logged = False

    def budget_reached() -> None:
        nonlocal budget_logged
        if not budget_logged:
            ctx.event("source_budget_reached", {"max_sources_per_run": s.max_sources_per_run})
            budget_logged = True

    for index, sq in enumerate(sub_questions):
        ctx.ensure_not_stopped()  # before this sub-question's search and select call
        own = mapping.setdefault(sq.id, [])
        if ctx.now() >= deadline:
            ctx.issue(
                f"deadline reached during research; "
                f"{len(sub_questions) - index} sub-question(s) not researched",
                elapsed_s=round(ctx.now() - ctx.started_at, 1),
            )
            break
        if attempts >= s.max_sources_per_run:
            # Budget spent: no search or select calls for the remaining sub-questions.
            budget_reached()
            stats["skipped_budget"] += 1
            ctx.issue(
                f"sub-question {sq.ordinal + 1} ({sq.text!r}): not researched, "
                f"source budget of {s.max_sources_per_run} reached",
                sub_question_id=str(sq.id),
            )
            continue
        first_round = searched.get(sq.id, 0) == 0
        memory: list[MemoryCandidate] = []
        if s.memory_enabled and first_round:
            if reused < s.max_reused_sources_per_run:
                memory = lookup(ctx, sq.id, sq.text, seen.keys())
                stats["memory_offered"] += len(memory)
            else:
                stats["memory_lookup_skipped"] += 1
                ctx.event("reuse_budget_spent", {"max_reused_sources_per_run": s.max_reused_sources_per_run})
        memory_by_url = {m.result.url: m for m in memory}
        candidates = merge(_search(ctx, sq.queries[searched.get(sq.id, 0) :]), memory)
        searched[sq.id] = len(sq.queries)
        stats["candidates"] += len(candidates)
        picks = _select(ctx, sq, candidates) if candidates else []
        pick_urls = {p.url for p in picks}
        reserve = [c for c in candidates if c.url not in pick_urls]  # rank order
        queue = list(picks)
        backfills = 0
        while queue:
            result = queue.pop(0)
            outcome: Outcome
            if result.url in seen:
                existing = seen[result.url]
                if existing is not None:
                    if existing not in own:
                        own.append(existing)
                        stats["reused"] += 1
                    continue
                outcome = Outcome.SEEN_UNUSABLE
            elif ctx.now() >= deadline:
                break
            else:
                ctx.ensure_not_stopped()  # before each reuse or fetch
                # With the reuse budget spent, a memory candidate or URL match is fetched from the web.
                origin = (
                    _memory_origin(ctx, state.run_id, result, memory_by_url)
                    if reused < s.max_reused_sources_per_run
                    else None
                )
                if origin is not None:  # copy from an earlier run instead of fetching
                    outcome, source_id = _reuse(ctx, state.run_id, result, origin, stats)
                    if outcome is Outcome.REUSED:
                        reused += 1  # reuse budget, not the fetch budget
                        seen[result.url] = source_id
                elif attempts >= s.max_sources_per_run:
                    budget_reached()
                    break
                elif host_of(result.url) in blocked:
                    outcome, source_id = Outcome.HOST_BLOCKED, None  # no fetch, no source row, no budget use
                    stats["host_blocked_skips"] += 1
                else:
                    attempts += 1
                    outcome, source_id = _process(ctx, state.run_id, result, stats, blocked)
                    seen[result.url] = source_id if outcome is Outcome.FETCHED else None
                if source_id is not None and source_id not in own:
                    own.append(source_id)
            if outcome in BACKFILL_OUTCOMES and backfills < s.max_backfills_per_sub_question:
                replacement = next(
                    (
                        c
                        for c in reserve
                        if c.url not in seen and (c.url in memory_by_url or host_of(c.url) not in blocked)
                    ),
                    None,
                )
                if replacement is not None:
                    reserve.remove(replacement)
                    queue.append(replacement)
                    backfills += 1
                    stats["backfilled"] += 1
                    ctx.event(
                        "backfill",
                        {
                            "sub_question_id": str(sq.id),
                            "replaces": result.url,
                            "because": str(outcome),
                            "url": replacement.url,
                        },
                    )
        if not own:
            reason = "no search results" if not candidates else "no source with usable text"
            ctx.issue(f"sub-question {sq.ordinal + 1} ({sq.text!r}): {reason}", sub_question_id=str(sq.id))

    summary = {
        "sources_total": attempts + reused,
        "fetch_attempts_total": attempts,
        "reused_total": reused,
        **stats,
    }
    return NodeResult(
        {"sources_by_sub_question": mapping, "queries_searched": searched, "blocked_hosts": sorted(blocked)},
        summary,
    )


@dataclass(frozen=True)
class _Origin:
    source_id: uuid.UUID
    reason: str  # "selected": a memory candidate chosen in select; "url_match": same URL in memory
    similarity: float | None
    fetched_at: datetime


def _memory_origin(
    ctx: RunContext, run_id: uuid.UUID, result: SearchResult, memory_by_url: dict[str, MemoryCandidate]
) -> _Origin | None:
    """Where to copy this result from instead of fetching it, if anywhere. Checked before every fetch
    (picks, backfills, gap rounds): a chosen memory candidate, or else, with memory on, a fresh fetched
    source of another run with the same URL, whatever its similarity."""
    candidate = memory_by_url.get(result.url)
    if candidate is not None:
        return _Origin(candidate.source_id, "selected", candidate.similarity, candidate.fetched_at)
    if not ctx.settings.memory_enabled:
        return None
    with ctx.db() as session:
        match = reusable_by_url(session, run_id, result.url, max_age_days=ctx.settings.memory_max_age_days)
    return _Origin(match[0], "url_match", None, match[2]) if match is not None else None


def _reuse(
    ctx: RunContext, run_id: uuid.UUID, result: SearchResult, origin: _Origin, stats: Counter[str]
) -> tuple[Outcome, uuid.UUID | None]:
    """Copy an earlier run's source and its chunks into this run (one transaction). A copy whose text
    is already in this run (same content hash) is not made; the existing source is used."""
    try:
        with ctx.db() as session:
            # FOR KEY SHARE: deleting the original's run waits until the copy is committed.
            original = session.get(Source, origin.source_id, with_for_update={"key_share": True})
            duplicate = (
                find_source_by_content_hash(session, run_id, original.content_hash)
                if original is not None and original.content_hash
                else None
            )
            duplicate_id = duplicate.id if duplicate is not None else None
            copied = None if duplicate is not None else copy_source(session, origin.source_id, run_id)
    except (DataError, IntegrityError) as e:
        return _storage_failed(ctx, run_id, result, stats, e, result.url, "")
    except MemoryCopyError as e:  # rolled back: skip this candidate, the run continues
        stats["memory_copy_failed"] += 1
        ctx.event(
            "memory_copy_failed", {"url": result.url, "source_id": str(origin.source_id), "error": str(e)}
        )
        return Outcome.MEMORY_UNAVAILABLE, None
    if duplicate_id is not None:
        stats["duplicate_content"] += 1
        ctx.event("source_duplicate", {"url": result.url, "duplicate_of": str(duplicate_id), "memory": True})
        return Outcome.DUPLICATE, duplicate_id
    if copied is None:
        stats["memory_unavailable"] += 1
        ctx.event("memory_unavailable", {"url": result.url, "source_id": str(origin.source_id)})
        return Outcome.MEMORY_UNAVAILABLE, None
    stats["memory_reused"] += 1
    stats[f"memory_reused:{origin.reason}"] += 1
    ctx.event(
        "memory_reused",
        {
            "url": result.url,
            "reason": origin.reason,
            "source_id": str(copied.source_id),
            "from_source_id": str(copied.from_source_id),
            "from_run_id": str(copied.from_run_id),
            "similarity": origin.similarity,
            "retrieved": origin.fetched_at.isoformat(),
            "chunks": copied.chunks,
        },
    )
    return Outcome.REUSED, copied.source_id


def _search(ctx: RunContext, queries: list[str]) -> list[SearchResult]:
    """Search each query (paced: at least SEARCH_MIN_INTERVAL_S between searches of the run) and log
    every provider call, the fallback included, as search_done."""
    results: list[SearchResult] = []
    for query in queries:
        ctx.pace_search()
        trace = traced_search(ctx.deps.search, query, ctx.settings.max_sources_per_query, ctx.now)
        for call in trace.calls:
            ctx.event(
                "search_done",
                {
                    "query": call.query,
                    "provider": call.provider,
                    "role": call.role,
                    "result_count": call.result_count,
                    "duration_s": call.duration_s,
                    "error": call.error,
                },
            )
            if call.error == FetchErrorReason.REPLAY_MISS.value:  # eval replay: query not recorded
                ctx.event("replay_miss", {"kind": "search", "query": query})
        if trace.error is not None:
            ctx.event("search_failed", {"query": query, "error": str(trace.error)})
        if ctx.deps.search_recorder is not None:
            ctx.deps.search_recorder(query, trace)
        results.extend(trace.results)
    return dedupe_results(results)


def _select(ctx: RunContext, sq: _SubQuestion, candidates: list[SearchResult]) -> list[SearchResult]:
    max_picks = min(ctx.settings.max_sources_per_sub_question, len(candidates))
    aliases = AliasMap([c.url for c in candidates], prefix="R")
    listing = "\n\n".join(
        untrusted(
            f"Title: {c.title}\nSite: {urlsplit(c.url).hostname or ''}\nSnippet: {c.snippet}",
            aliases.alias(c.url),
        )
        for c in candidates
    )
    prompt = load_prompt("select").render(
        question=ctx.question, sub_question=sq.text, max_picks=str(max_picks), candidates=listing
    )
    try:
        result = ctx.llm.generate_structured(
            prompt, selection_schema(aliases.aliases, max_picks), num_predict=ctx.settings.select_num_predict
        )
        urls = aliases.resolve_all(result.value.selected)
    except LLMError as e:
        ctx.event(
            "select_fallback",
            {"sub_question_id": str(sq.id), "error": f"{type(e).__name__}: {e}", "picked": max_picks},
        )
        return candidates[:max_picks]
    by_url = {c.url: c for c in candidates}
    return [by_url[u] for u in urls]


def _process(
    ctx: RunContext, run_id: uuid.UUID, result: SearchResult, stats: Counter[str], blocked: set[str]
) -> tuple[Outcome, uuid.UUID | None]:
    """Fetch, extract, chunk, embed and store one source. Returns the outcome and, for FETCHED or
    DUPLICATE, the id of the source that holds the chunks."""
    s = ctx.settings
    try:
        page = ctx.deps.fetcher.fetch(result.url)
    except FetchError as e:
        stats[f"fetch_failed:{e.reason}"] += 1
        _store_failure(ctx, run_id, result, SourceStatus.FAILED, str(e))  # "<reason>: <message> (<url>)"
        ctx.event("fetch_failed", {"url": result.url, "reason": str(e.reason), "status_code": e.status_code})
        if e.reason is FetchErrorReason.HTTP_ERROR and e.status_code in BLOCKING_STATUSES:
            host = host_of(e.url)
            if host and host not in blocked:
                blocked.add(host)
                ctx.event("host_blocked", {"host": host, "status_code": e.status_code, "url": e.url})
        return Outcome.FETCH_FAILED, None

    extracted = extract(page.body, page.content_type, min_chars=s.extract_min_chars)
    if extracted.status is not ExtractStatus.OK:
        stats[f"extract_skipped:{extracted.reason}"] += 1
        _store_failure(
            ctx,
            run_id,
            result,
            SourceStatus.SKIPPED,
            f"extract: {extracted.reason}",
            page.final_url,
            page.content_type,
        )
        ctx.event("extract_skipped", {"url": result.url, "reason": extracted.reason})
        return Outcome.EXTRACT_SKIPPED, None

    with ctx.db() as session:
        duplicate = find_source_by_content_hash(session, run_id, extracted.content_hash)
        duplicate_id = duplicate.id if duplicate is not None else None
    if duplicate_id is not None:
        try:
            with ctx.db() as session:
                add_source(
                    session,
                    run_id,
                    result.url,
                    status=SourceStatus.SKIPPED,
                    final_url=page.final_url,
                    title=extracted.title or result.title,
                    content_type=page.content_type,
                    content_hash=extracted.content_hash,
                    fetched_at=datetime.now(UTC),
                    error=f"duplicate_content: same text as source {duplicate_id}",
                )
        except (DataError, IntegrityError) as e:
            return _storage_failed(ctx, run_id, result, stats, e, page.final_url, page.content_type)
        stats["duplicate_content"] += 1
        ctx.event("source_duplicate", {"url": result.url, "duplicate_of": str(duplicate_id)})
        return Outcome.DUPLICATE, duplicate_id

    chunks = chunk_text(
        extracted.text,
        target_tokens=s.chunk_target_tokens,
        overlap=s.chunk_overlap,
        chars_per_token=s.chars_per_token,
    )
    if len(chunks) > s.max_chunks_per_source:
        ctx.event(
            "chunks_capped", {"url": result.url, "chunks": len(chunks), "kept": s.max_chunks_per_source}
        )
        chunks = chunks[: s.max_chunks_per_source]
    try:
        vectors = ctx.deps.embedder.embed_documents([c.text for c in chunks])
    except EmbedError as e:
        stats["embed_failed"] += 1
        _store_failure(
            ctx, run_id, result, SourceStatus.FAILED, f"embed_error: {e}", page.final_url, page.content_type
        )
        ctx.event("embed_failed", {"url": result.url, "error": str(e)})
        return Outcome.EMBED_FAILED, None

    try:
        with ctx.db() as session:  # source and its chunks in one transaction
            source = add_source(
                session,
                run_id,
                result.url,
                status=SourceStatus.FETCHED,
                final_url=page.final_url,
                title=extracted.title or result.title,
                content_type=page.content_type,
                content_hash=extracted.content_hash,
                fetched_at=datetime.now(UTC),
                extracted_text=extracted.text,
            )
            add_chunks(
                session,
                source.id,
                [
                    ChunkInput(c.ordinal, c.text, c.char_start, c.char_end, v)
                    for c, v in zip(chunks, vectors, strict=True)
                ],
            )
            source_id = source.id
    except (DataError, IntegrityError) as e:  # that transaction is rolled back; the run goes on
        return _storage_failed(ctx, run_id, result, stats, e, page.final_url, page.content_type)
    stats["fetched"] += 1
    stats["chunks"] += len(chunks)
    ctx.event("source_fetched", {"url": result.url, "source_id": str(source_id), "chunks": len(chunks)})
    return Outcome.FETCHED, source_id


def storage_error_message(e: DataError | IntegrityError) -> str:
    """'storage_error: <driver error type>: <first line>', without the SQL, parameters or control chars."""
    orig = e.orig if e.orig is not None else e
    first_line = (str(orig).strip().splitlines() or [""])[0]
    message = f"storage_error: {type(orig).__name__}: {first_line}"
    return "".join(ch for ch in message if ch >= " ")[:1000]


def _storage_failed(
    ctx: RunContext,
    run_id: uuid.UUID,
    result: SearchResult,
    stats: Counter[str],
    e: DataError | IntegrityError,
    final_url: str,
    content_type: str,
) -> tuple[Outcome, None]:
    """Record a source whose rows the database rejected as failed; the run continues."""
    stats["storage_error"] += 1
    message = storage_error_message(e)
    recorded = True
    try:
        # No title: the title may be what the database rejected.
        with ctx.db() as session:
            add_source(
                session,
                run_id,
                result.url,
                status=SourceStatus.FAILED,
                final_url=final_url,
                content_type=content_type,
                error=message,
            )
    except (DataError, IntegrityError):
        recorded = False
    ctx.event("storage_failed", {"url": result.url, "error": message, "recorded": recorded})
    return Outcome.STORAGE_FAILED, None


def _store_failure(
    ctx: RunContext,
    run_id: uuid.UUID,
    result: SearchResult,
    status: SourceStatus,
    error: str,
    final_url: str | None = None,
    content_type: str | None = None,
) -> None:
    with ctx.db() as session:
        add_source(
            session,
            run_id,
            result.url,
            status=status,
            final_url=final_url,
            title=result.title,
            content_type=content_type,
            error=error[:1000],
        )
