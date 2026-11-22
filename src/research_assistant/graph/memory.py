"""Memory lookup for research: sources from earlier runs offered to the select step.

Per sub-question (first research round only): embed the sub-question, find the nearest fetched
sources of other runs within MEMORY_MAX_AGE_DAYS, keep distinct URLs not already in this run, and
offer at most MEMORY_MAX_CANDIDATES with similarity >= MEMORY_MIN_SIMILARITY as extra select
candidates. The ten best distinct sources are logged (memory_lookup) whether offered or not, so the
threshold can be set from data.
"""

import uuid
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import select

from research_assistant.db.models import Source, SourceStatus
from research_assistant.db.repositories.memory import MemoryHit, nearest_sources
from research_assistant.graph.context import RunContext
from research_assistant.tools.embed import EmbedError
from research_assistant.tools.search import SearchResult
from research_assistant.tools.url import normalize_url

MEMORY_ENGINE = "memory"
LOGGED_TOP = 10


@dataclass(frozen=True)
class MemoryCandidate:
    result: SearchResult  # what select sees (engine "memory")
    source_id: uuid.UUID
    run_id: uuid.UUID
    similarity: float
    fetched_at: datetime
    urls: frozenset[str]  # normalized url and final_url: web results with these are dropped


def _urls(hit: MemoryHit) -> frozenset[str]:
    out = {hit.url}
    if hit.final_url:
        try:
            out.add(normalize_url(hit.final_url))
        except ValueError:
            pass
    return frozenset(out)


EXCERPT_CHARS = 300


def retrieved_note(fetched_at: datetime) -> str:
    return f"from earlier research, retrieved {fetched_at:%Y-%m-%d}"


def memory_snippet(best_chunk: str, fetched_at: datetime) -> str:
    """The start of the source's best-matching chunk (like a search snippet) and where it came from."""
    text = " ".join(best_chunk.split())
    excerpt = text if len(text) <= EXCERPT_CHARS else text[:EXCERPT_CHARS].rsplit(" ", 1)[0] + " …"
    return f"{excerpt} ({retrieved_note(fetched_at)})"


def lookup(
    ctx: RunContext, sub_question_id: uuid.UUID, text: str, seen: Collection[str]
) -> list[MemoryCandidate]:
    s = ctx.settings
    try:
        vector = ctx.deps.embedder.embed_query(text)
    except EmbedError as e:
        ctx.event("memory_failed", {"sub_question_id": str(sub_question_id), "error": str(e)})
        return []
    with ctx.db() as session:
        hits = nearest_sources(
            session,
            ctx.run_id,
            vector,
            max_age_days=s.memory_max_age_days,
            overfetch=s.memory_overfetch_chunks,
        )
        own_hashes = set(
            session.scalars(
                select(Source.content_hash).where(
                    Source.run_id == ctx.run_id, Source.status == SourceStatus.FETCHED
                )
            )
        )
    distinct: list[tuple[MemoryHit, frozenset[str]]] = []
    taken: set[str] = set()
    for hit in sorted(hits, key=lambda h: -h.similarity):
        urls = _urls(hit)
        if urls & taken:  # the same page from another run: keep the most similar
            continue
        taken |= urls
        distinct.append((hit, urls))
    eligible = [
        (h, urls)
        for h, urls in distinct
        if not urls & set(seen) and (h.content_hash is None or h.content_hash not in own_hashes)
    ]
    offered = [(h, u) for h, u in eligible if h.similarity >= s.memory_min_similarity][
        : s.memory_max_candidates
    ]
    ctx.event(
        "memory_lookup",
        {
            "sub_question_id": str(sub_question_id),
            "threshold": s.memory_min_similarity,
            "top": [
                {
                    "similarity": h.similarity,
                    "url": h.url,
                    "run_id": str(h.run_id),
                    "fetched_at": h.fetched_at.isoformat(),
                    "offered": any(h is o for o, _ in offered),
                }
                for h, _ in distinct[:LOGGED_TOP]
            ],
            "offered": len(offered),
        },
    )
    return [
        MemoryCandidate(
            result=SearchResult(
                url=h.url,
                title=h.title or h.url,
                snippet=memory_snippet(h.best_chunk, h.fetched_at),
                engine=MEMORY_ENGINE,
                rank=i,  # rank among memory candidates, by similarity
            ),
            source_id=h.source_id,
            run_id=h.run_id,
            similarity=h.similarity,
            fetched_at=h.fetched_at,
            urls=urls,
        )
        for i, (h, urls) in enumerate(offered, start=1)
    ]


def merge(web: Sequence[SearchResult], memory: Sequence[MemoryCandidate]) -> list[SearchResult]:
    """One candidate list, ordered by each candidate's own rank: a web result's search rank, a memory
    candidate's rank by similarity among the memory candidates (on equal rank the memory candidate
    comes first: it needs no fetch). A web result with a candidate's URL is dropped: the candidate
    stands for that page and it is not fetched again."""
    covered = set().union(*(m.urls for m in memory)) if memory else set()
    kept = [r for r in web if r.url not in covered]
    keyed = [((r.rank, 1, i), r) for i, r in enumerate(kept)] + [
        ((m.result.rank, 0, i), m.result) for i, m in enumerate(memory)
    ]
    merged = [r for _, r in sorted(keyed, key=lambda item: item[0])]
    return [SearchResult(r.url, r.title, r.snippet, r.engine, i) for i, r in enumerate(merged, start=1)]
