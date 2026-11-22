"""Memory: sources of earlier runs offered to select, copied into the run when chosen."""

import json
import math
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import Engine, delete, func, select, text
from sqlalchemy.dialects import postgresql
from sqlalchemy.exc import OperationalError

from research_assistant.db.models import Chunk, ResearchRun, RunStatus, Source, SourceStatus
from research_assistant.db.repositories.chunks import (
    ChunkInput,
    add_chunks,
    nearest_chunks,
    nearest_chunks_statement,
)
from research_assistant.db.repositories.memory import copy_source
from research_assistant.db.repositories.reports import latest_report
from research_assistant.db.repositories.runs import create_run
from research_assistant.db.repositories.sources import add_source
from research_assistant.db.session import session_scope
from research_assistant.evals.citation_check import check_run, connect
from research_assistant.evals.power import PowerState
from research_assistant.evals.questions import EvalQuestion
from research_assistant.evals.runner import load_results, run_eval
from research_assistant.graph.runner import run_research
from research_assistant.tools.fetch import FetchError, FetchErrorReason
from tests.integration.conftest import TrackingSessionFactory
from tests.integration.graph_fakes import OKAPI, PLAN, RFC2606, RFC9309, SECRETS, blocks_in, labels_in
from tests.integration.phase7_support import client_for, make_app
from tests.integration.test_evals import eval_harness
from tests.integration.test_graph import QUESTION, Harness, events_of

MEMORY_URL = "https://memory.example.org/okapi-facts"
SQ0 = PLAN["sub_questions"][0]["question"]  # "What is an okapi and where does it live?"


@pytest.fixture
def h(committing_factory: TrackingSessionFactory) -> Harness:
    return Harness(committing_factory)


def with_similarity(base: list[float], similarity: float) -> list[float]:
    """A unit vector whose cosine similarity to `base` (a unit vector) is exactly `similarity`."""
    pivot = min(range(len(base)), key=lambda i: abs(base[i]))
    other = [0.0] * len(base)
    other[pivot] = 1.0
    dot = sum(a * b for a, b in zip(other, base, strict=True))
    orth = [o - dot * b for o, b in zip(other, base, strict=True)]
    norm = math.sqrt(sum(x * x for x in orth))
    orth = [x / norm for x in orth]
    s = similarity
    return [s * b + math.sqrt(1 - s * s) * o for b, o in zip(base, orth, strict=True)]


def earlier_source(
    h: Harness,
    *,
    similarity: float,
    url: str = MEMORY_URL,
    sub_question: str = SQ0,
    age_days: float = 1.0,
    status: SourceStatus = SourceStatus.FETCHED,
    title: str = "Okapi facts",
) -> tuple[uuid.UUID, uuid.UUID]:
    """An earlier completed run with one source of two chunks; chunk 0 has the given similarity to
    the sub-question's query embedding. Returns (run_id, source_id)."""
    query = h.embedder._vector(sub_question)
    texts = [
        "Okapis live in the Ituri rainforest of the Democratic Republic of the Congo and eat leaves.",
        "The okapi is the only living relative of the giraffe and has striped legs like a zebra.",
    ]
    extracted = "\n".join(texts)
    with session_scope(h.factory) as session:
        run = create_run(session, "an earlier question about okapis", {}, status=RunStatus.COMPLETED)
        src = add_source(
            session,
            run.id,
            url,
            status=status,
            final_url=url + "?from=final",
            title=title,
            content_type="text/html",
            content_hash=f"hash-{uuid.uuid4()}",
            fetched_at=datetime.now(UTC) - timedelta(days=age_days),
            extracted_text=extracted,
        )
        offset = len(texts[0]) + 1
        add_chunks(
            session,
            src.id,
            [
                ChunkInput(0, texts[0], 0, len(texts[0]), with_similarity(query, similarity)),
                ChunkInput(
                    1, texts[1], offset, offset + len(texts[1]), with_similarity(query, similarity / 2)
                ),
            ],
        )
        return run.id, src.id


def lookups(h: Harness, run_id: uuid.UUID) -> list[dict[str, Any]]:
    return events_of(h, run_id, "memory_lookup")


def pick_memory(call: Any) -> str:
    """select: the memory candidate if one is listed, else R1."""
    memory = [label for label, body in blocks_in(call).items() if "from earlier research" in body]
    return json.dumps({"selected": memory[:1] or labels_in(call, "R")[:1]})


# -- lookup and offer ------------------------------------------------------------------------


def test_candidate_offered_copied_and_cited(h: Harness, db_url: str) -> None:
    earlier_run, original = earlier_source(h, similarity=0.9)
    h.model.select = pick_memory
    outcome = run_research(QUESTION, h.deps())
    assert outcome.status is RunStatus.COMPLETED, outcome.reason

    [lookup, *_] = lookups(h, outcome.run_id)
    assert lookup["top"][0]["url"] == MEMORY_URL and lookup["top"][0]["similarity"] == pytest.approx(
        0.9, abs=1e-3
    )
    assert lookup["top"][0]["offered"] is True and lookup["offered"] == 1
    select_prompt = h.model.prompts("select")[0].prompt.user
    retrieved = (datetime.now(UTC) - timedelta(days=1)).strftime("%Y-%m-%d")
    assert (
        f"from earlier research, retrieved {retrieved}" in select_prompt
        and "memory.example.org" in select_prompt
    )
    assert MEMORY_URL not in h.fetcher.calls  # reused, not fetched

    [reused] = events_of(h, outcome.run_id, "memory_reused")
    assert reused["from_source_id"] == str(original) and reused["from_run_id"] == str(earlier_run)
    with session_scope(h.factory) as session:
        orig = session.get_one(Source, original)
        copy = session.get_one(Source, uuid.UUID(reused["source_id"]))
        fields = (
            "url",
            "final_url",
            "title",
            "content_type",
            "content_hash",
            "fetched_at",
            "extracted_text",
            "status",
        )
        assert {f: getattr(copy, f) for f in fields} == {f: getattr(orig, f) for f in fields}
        assert (
            copy.run_id == outcome.run_id and copy.reused_from_source_id == original and copy.id != original
        )

        def chunks_of(source_id: uuid.UUID) -> list[tuple[Any, ...]]:
            rows = session.scalars(select(Chunk).where(Chunk.source_id == source_id).order_by(Chunk.ordinal))
            return [(c.ordinal, c.text, c.char_start, c.char_end, list(c.embedding)) for c in rows]

        assert chunks_of(copy.id) == chunks_of(original) and len(chunks_of(original)) == 2
        assert all(
            c.run_id == outcome.run_id
            for c in session.scalars(select(Chunk).where(Chunk.source_id == copy.id))
        )
        report = latest_report(session, outcome.run_id)
        assert report is not None
        cited = {c["source_id"] for c in report.citations}
    assert str(copy.id) in cited  # the report cites the reused source
    assert f"<{MEMORY_URL}?from=final> (retrieved {retrieved})" in report.markdown

    [research_end, *_] = [e for e in events_of(h, outcome.run_id, "node_end") if "sources_total" in e]
    assert research_end["sources_total"] == research_end["fetch_attempts_total"] + 1
    with connect(db_url) as conn:
        result = check_run(conn, outcome.run_id)
    assert result.failures == [] and result.checks > 0


def test_old_and_failed_sources_excluded(h: Harness) -> None:
    earlier_source(h, similarity=0.95, url="https://old.example.org/okapi", age_days=40)
    earlier_source(h, similarity=0.97, url="https://failed.example.org/okapi", status=SourceStatus.FAILED)
    earlier_source(h, similarity=0.7, url="https://fresh.example.org/okapi")
    outcome = run_research(QUESTION, h.deps())
    top = lookups(h, outcome.run_id)[0]["top"]
    urls = [t["url"] for t in top]
    assert "https://fresh.example.org/okapi" in urls
    assert "https://old.example.org/okapi" not in urls and "https://failed.example.org/okapi" not in urls


def test_similarity_threshold_and_candidate_cap(h: Harness) -> None:
    earlier_source(h, similarity=0.59, url="https://below.example.org/a")
    earlier_source(h, similarity=0.61, url="https://above.example.org/a")
    for i, sim in enumerate((0.8, 0.82, 0.84)):
        earlier_source(h, similarity=sim, url=f"https://many.example.org/{i}")
    earlier_source(h, similarity=0.85, url="https://many.example.org/0")  # same page in another run
    outcome = run_research(QUESTION, h.deps())
    lookup = lookups(h, outcome.run_id)[0]
    by_url = {t["url"]: t for t in lookup["top"]}
    assert len(by_url) == len(lookup["top"])  # distinct pages
    assert by_url["https://many.example.org/0"]["similarity"] == pytest.approx(
        0.85, abs=1e-3
    )  # best copy kept
    assert by_url["https://below.example.org/a"]["offered"] is False
    assert lookup["offered"] == 3  # MEMORY_MAX_CANDIDATES
    offered = sorted(t["similarity"] for t in lookup["top"] if t["offered"])
    assert offered == pytest.approx([0.82, 0.84, 0.85], abs=1e-3)  # 0.61 is above the threshold but not top 3


def test_threshold_boundary(h: Harness) -> None:
    threshold = h.settings.memory_min_similarity
    assert threshold == 0.75  # the provisional default
    earlier_source(h, similarity=threshold - 0.01, url="https://below.example.org/b")
    earlier_source(h, similarity=threshold + 0.01, url="https://above.example.org/b")
    outcome = run_research(QUESTION, h.deps())
    flags = {t["url"]: t["offered"] for t in lookups(h, outcome.run_id)[0]["top"]}
    assert flags == {"https://above.example.org/b": True, "https://below.example.org/b": False}


def test_same_url_web_result_collapses_into_memory(h: Harness) -> None:
    earlier_source(h, similarity=0.9, url=OKAPI)  # the web search for sub-question 1 also returns OKAPI
    outcome = run_research(QUESTION, h.deps())
    select_prompt = h.model.prompts("select")[0].prompt.user
    assert select_prompt.count("simple.wikipedia.org") == 1 and "from earlier research" in select_prompt
    assert OKAPI not in h.fetcher.calls
    assert [e["url"] for e in events_of(h, outcome.run_id, "memory_reused")] == [OKAPI]


def test_memory_off_means_no_lookup(h: Harness, monkeypatch: pytest.MonkeyPatch) -> None:
    earlier_source(h, similarity=0.95)
    import research_assistant.graph.nodes.research as research_module

    def no_lookup(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("memory lookup with memory_enabled=false")

    monkeypatch.setattr(research_module, "lookup", no_lookup)
    off = h.settings.model_copy(update={"memory_enabled": False})
    deps = h.deps()
    deps.settings = off
    outcome = run_research(QUESTION, deps)
    assert outcome.status is RunStatus.COMPLETED
    assert lookups(h, outcome.run_id) == [] and events_of(h, outcome.run_id, "memory_reused") == []
    with session_scope(h.factory) as session:
        assert session.get_one(ResearchRun, outcome.run_id).config["memory_enabled"] is False


def test_deleting_the_original_run_keeps_the_copy(h: Harness, db_url: str) -> None:
    earlier_run, _ = earlier_source(h, similarity=0.9)
    h.model.select = pick_memory
    outcome = run_research(QUESTION, h.deps())
    with session_scope(h.factory) as session:
        session.delete(session.get_one(ResearchRun, earlier_run))
    with session_scope(h.factory) as session:
        copy = session.scalars(
            select(Source).where(Source.run_id == outcome.run_id, Source.reused_from_source_id.is_not(None))
        ).one()
        assert (
            session.scalar(select(text("count(*)")).select_from(Chunk).where(Chunk.source_id == copy.id)) == 2
        )
    with connect(db_url) as conn:
        assert check_run(conn, outcome.run_id).failures == []
    detail = client_for(make_app(h, db_url)).get(f"/runs/{outcome.run_id}").json()
    [reused] = [s for s in detail["sources"] if s["reused"]]
    assert reused["reused_from_source_id"] is not None and reused["reused_from_run_id"] is None


def test_api_lists_reused_sources(h: Harness, db_url: str) -> None:
    earlier_run, original = earlier_source(h, similarity=0.9)
    h.model.select = pick_memory
    outcome = run_research(QUESTION, h.deps())
    detail = client_for(make_app(h, db_url)).get(f"/runs/{outcome.run_id}").json()
    reused = [s for s in detail["sources"] if s["reused"]]
    assert [(s["url"], s["reused_from_source_id"], s["reused_from_run_id"]) for s in reused] == [
        (MEMORY_URL, str(original), str(earlier_run))
    ]
    assert all(s["reused_from_run_id"] is None for s in detail["sources"] if not s["reused"])


def test_tampered_retrieval_date_fails_the_check(h: Harness, db_url: str) -> None:
    earlier_source(h, similarity=0.9)
    h.model.select = pick_memory
    outcome = run_research(QUESTION, h.deps())
    with session_scope(h.factory) as session:
        report = latest_report(session, outcome.run_id)
        assert report is not None and "(retrieved " in report.markdown
        report.markdown = report.markdown.replace("(retrieved 20", "(retrieved 19", 1)
    with connect(db_url) as conn:
        kinds = {f.kind for f in check_run(conn, outcome.run_id).failures}
    assert kinds == {"reference_date_mismatch"}


# -- eval keeps memory off -------------------------------------------------------------------


@pytest.mark.parametrize("with_memory", [False, True])
def test_eval_forces_memory_off_unless_asked(
    h: Harness, tmp_path: Path, db_url: str, with_memory: bool
) -> None:
    earlier_source(h, similarity=0.95)
    question = EvalQuestion("q1", "technical", QUESTION)
    run_eval(
        tmp_path,
        [question],
        h.settings,
        eval_harness(h),
        h.factory,
        with_memory=with_memory,
        check_connection=lambda: connect(db_url),
        power=lambda: PowerState(True, 90),
        log=lambda _line: None,
    )
    results = load_results(tmp_path)
    assert results is not None
    entry = results["runs"]["q1"]
    assert entry["memory_enabled"] is with_memory and results["sessions"][-1]["memory_enabled"] is with_memory
    assert bool(lookups(h, uuid.UUID(entry["run_id"]))) is with_memory
    with session_scope(h.factory) as session:
        config = session.get_one(ResearchRun, uuid.UUID(entry["run_id"])).config
    assert config["memory_enabled"] is with_memory


# -- per-run retrieval stays exact and scoped ------------------------------------------------


def _explain(engine: Engine, stmt: Any) -> str:
    dialect = postgresql.dialect()
    compiled = stmt.compile(dialect=dialect)
    params = {}
    for key, value in compiled.params.items():
        processor = compiled.binds[key].type._cached_bind_processor(dialect)
        params[key] = processor(value) if processor else value
    sql = compiled.string % {k: f"%({k})s" for k in params}
    raw = engine.raw_connection()
    try:
        cur = raw.cursor()
        # Push the planner toward the HNSW index wherever it could be used: with sequential scans and
        # explicit sorts discouraged, an ORDER BY embedding <=> q LIMIT k on chunks takes the index.
        cur.execute("SET enable_seqscan = off")
        cur.execute("SET enable_sort = off")
        cur.execute("EXPLAIN " + sql, params)
        plan = "\n".join(r[0] for r in cur.fetchall())
        cur.execute("RESET ALL")
        return plan
    finally:
        raw.close()


def test_per_run_search_never_uses_hnsw(h: Harness, engine: Engine) -> None:
    run_id, _ = earlier_source(h, similarity=0.9)
    query = h.embedder._vector(SQ0)
    per_run = _explain(engine, nearest_chunks_statement(run_id, query, 5))
    assert "ix_chunks_embedding_hnsw" not in per_run, per_run
    # Positive control: the same search written as a plain run-filtered ORDER BY ... LIMIT does use the
    # approximate index under these settings, so the assertion above can tell the two apart.
    plain = select(Chunk.id).where(Chunk.run_id == run_id).order_by(Chunk.embedding.op("<=>")(query)).limit(5)
    plain_plan = _explain(engine, plain)
    assert "ix_chunks_embedding_hnsw" in plain_plan, plain_plan


def test_per_run_search_is_scoped_to_the_run(h: Harness) -> None:
    query = h.embedder._vector(SQ0)
    other_run, _ = earlier_source(h, similarity=0.99)  # the nearest chunks overall
    this_run, _ = earlier_source(h, similarity=0.5, url="https://this-run.example.org/")
    with session_scope(h.factory) as session:
        hits = nearest_chunks(session, this_run, query, 5)
        sources_of_hits = {session.get_one(Source, hit.source_id).run_id for hit in hits}
    assert sources_of_hits == {this_run} and len(hits) == 2
    assert hits[0].distance == pytest.approx(0.5, abs=1e-3)
    assert other_run not in sources_of_hits


# -- excerpt, ordering and URL-level reuse ----------------------------------------------------


def first_web_label(call: Any) -> str:
    """select: the first candidate that is not from memory."""
    web = [label for label, body in blocks_in(call).items() if "from earlier research" not in body]
    return json.dumps({"selected": web[:1]})


def test_memory_candidate_has_excerpt_in_untrusted_block_and_is_interleaved(h: Harness) -> None:
    earlier_source(h, similarity=0.9)
    run_research(QUESTION, h.deps())
    call = h.model.prompts("select")[0]
    blocks = blocks_in(call)  # only text inside <<<UNTRUSTED_DATA ...>>> blocks
    [(label, body)] = [(k, v) for k, v in blocks.items() if "from earlier research" in v]
    assert "Okapis live in the Ituri rainforest of the Democratic Republic of the Congo" in body
    retrieved = (datetime.now(UTC) - timedelta(days=1)).strftime("%Y-%m-%d")
    assert body.rstrip().endswith(f"(from earlier research, retrieved {retrieved})")
    assert "Title: Okapi facts" in body and "Site: memory.example.org" in body
    # Same layout as a web candidate, and not appended last: memory rank 1 ties web rank 1 and goes first.
    assert label == "R1" and len(blocks) > 1


def test_url_match_in_backfill_reuses_instead_of_fetching(h: Harness) -> None:
    blocked = "https://blocked.example.org/page"
    h.fetcher.pages[blocked] = FetchError(
        FetchErrorReason.HTTP_ERROR, "HTTP 403", url=blocked, status_code=403
    )
    remembered = "https://remembered.example.org/okapi"
    earlier_source(h, similarity=0.1, url=remembered)  # far below the threshold: never offered
    h.search.results["okapi animal"] = [blocked, remembered]
    h.fetcher.pages[remembered] = h.fetcher.pages[OKAPI]
    h.model.select = first_web_label
    outcome = run_research(QUESTION, h.deps())
    [backfill] = events_of(h, outcome.run_id, "backfill")
    assert (backfill["replaces"], backfill["url"]) == (blocked, remembered)
    [reused] = events_of(h, outcome.run_id, "memory_reused")
    assert (reused["url"], reused["reason"], reused["similarity"]) == (remembered, "url_match", None)
    assert remembered not in h.fetcher.calls and blocked in h.fetcher.calls


def test_url_match_in_a_later_sub_question(h: Harness) -> None:
    # Offered (but not chosen) for sub-question 1, then a plain web result for sub-question 2.
    earlier_source(h, similarity=0.9, url=RFC2606)
    h.model.select = first_web_label
    outcome = run_research(QUESTION, h.deps())
    first = lookups(h, outcome.run_id)[0]
    assert [t["url"] for t in first["top"] if t["offered"]] == [RFC2606]
    [reused] = events_of(h, outcome.run_id, "memory_reused")
    assert (reused["url"], reused["reason"]) == (RFC2606, "url_match")
    assert RFC2606 not in h.fetcher.calls
    research_end = next(e for e in events_of(h, outcome.run_id, "node_end") if "sources_total" in e)
    assert research_end["memory_reused:url_match"] == 1
    assert research_end["sources_total"] == research_end["fetch_attempts_total"] + 1


def test_stale_and_failed_sources_are_fetched(h: Harness) -> None:
    earlier_source(h, similarity=0.1, url=OKAPI, age_days=40)
    earlier_source(h, similarity=0.1, url=RFC2606, status=SourceStatus.FAILED)
    outcome = run_research(QUESTION, h.deps())
    assert OKAPI in h.fetcher.calls and RFC2606 in h.fetcher.calls
    assert events_of(h, outcome.run_id, "memory_reused") == []


def test_copy_locks_the_original_against_a_concurrent_delete(h: Harness, engine: Engine) -> None:
    # Without the lock, deleting the earlier run between the copy's read and its chunk INSERT ... SELECT
    # left a "reused" source with no chunks.
    earlier_run, original = earlier_source(h, similarity=0.9)
    with session_scope(h.factory) as session:
        target = create_run(session, "a later question", {}).id
    session = h.factory()
    try:
        copied = copy_source(session, original, target)
        assert copied is not None and copied.chunks == 2
        with engine.connect() as other:
            other.execute(text("SET lock_timeout = '300ms'"))
            with pytest.raises(OperationalError, match="lock timeout"):
                other.execute(delete(ResearchRun).where(ResearchRun.id == earlier_run))
            other.rollback()
        session.commit()
    finally:
        session.close()
    with session_scope(h.factory) as session:  # once the copy is committed, the delete goes through
        session.execute(delete(ResearchRun).where(ResearchRun.id == earlier_run))
    with session_scope(h.factory) as session:
        assert session.scalar(select(func.count()).select_from(Chunk).where(Chunk.run_id == target)) == 2


def test_copy_without_chunks_is_skipped_and_logged(h: Harness) -> None:
    remembered = "https://remembered.example.org/okapi"
    _, original = earlier_source(h, similarity=0.1, url=remembered)
    with session_scope(h.factory) as session:  # what a delete racing the copy used to leave
        session.execute(delete(Chunk).where(Chunk.source_id == original))
    h.search.results["okapi animal"] = [remembered, OKAPI]
    h.model.select = first_web_label
    outcome = run_research(QUESTION, h.deps())
    [failed] = events_of(h, outcome.run_id, "memory_copy_failed")
    assert (failed["url"], failed["source_id"]) == (remembered, str(original))
    assert events_of(h, outcome.run_id, "memory_reused") == []
    with session_scope(h.factory) as session:  # rolled back: no copy row
        copies = (
            select(func.count())
            .select_from(Source)
            .where(Source.run_id == outcome.run_id, Source.reused_from_source_id.is_not(None))
        )
        assert session.scalar(copies) == 0
    [backfill] = events_of(h, outcome.run_id, "backfill")  # the next result takes its place
    assert (backfill["replaces"], backfill["url"]) == (remembered, OKAPI)
    assert remembered not in h.fetcher.calls


def test_memory_off_fetches_even_with_a_url_match(h: Harness) -> None:
    earlier_source(h, similarity=0.1, url=OKAPI)
    deps = h.deps()
    deps.settings = h.settings.model_copy(update={"memory_enabled": False})
    outcome = run_research(QUESTION, deps)
    assert OKAPI in h.fetcher.calls and events_of(h, outcome.run_id, "memory_reused") == []


# -- reuse budget ----------------------------------------------------------------------------


def test_reused_sources_do_not_use_the_fetch_budget(committing_factory: TrackingSessionFactory) -> None:
    h = Harness(committing_factory, max_sources_per_run=2)
    earlier_source(h, similarity=0.1, url=OKAPI)
    earlier_source(h, similarity=0.1, url=RFC2606)
    h.search.results["okapi animal"] = [OKAPI, SECRETS]  # two picks: one reused, one fetched
    outcome = run_research(QUESTION, h.deps())
    assert {e["url"] for e in events_of(h, outcome.run_id, "memory_reused")} == {OKAPI, RFC2606}
    assert SECRETS in h.fetcher.calls and RFC9309 in h.fetcher.calls  # both web fetches still happen
    research_end = next(e for e in events_of(h, outcome.run_id, "node_end") if "sources_total" in e)
    assert (research_end["fetch_attempts_total"], research_end["reused_total"]) == (2, 2)
    problems = [p["message"] for p in events_of(h, outcome.run_id, "problem")]
    assert not any("not researched" in m for m in problems), problems  # every sub-question searched


def test_reuse_budget_falls_back_to_fetching(committing_factory: TrackingSessionFactory) -> None:
    h = Harness(committing_factory, max_reused_sources_per_run=1)
    earlier_source(h, similarity=0.1, url=OKAPI)
    earlier_source(h, similarity=0.1, url=RFC2606)
    outcome = run_research(QUESTION, h.deps())
    [reused] = events_of(h, outcome.run_id, "memory_reused")
    assert reused["url"] == OKAPI  # sub-question 1 comes first
    assert RFC2606 in h.fetcher.calls and OKAPI not in h.fetcher.calls  # the second match is fetched
    assert events_of(h, outcome.run_id, "reuse_budget_spent")  # later lookups are skipped
