"""The research graph end to end against the test database, with fakes for search, fetch, embed and LLM."""

import json
import re
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest
from psycopg.errors import UniqueViolation
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from research_assistant import cli
from research_assistant.config import Settings
from research_assistant.db.models import (
    Chunk,
    Claim,
    ClaimEvidence,
    ResearchRun,
    RunStatus,
    Source,
    SourceStatus,
    SubQuestion,
    Verdict,
)
from research_assistant.db.repositories.events import list_events
from research_assistant.db.repositories.reports import latest_report
from research_assistant.evals.metrics import run_metrics
from research_assistant.graph.context import Deps
from research_assistant.graph.nodes import research as research_node_module
from research_assistant.graph.nodes.write import (
    COVERED_NOTE,
    INSUFFICIENT_NOTE,
    NO_OWN_SOURCES_NOTE,
    PREMISE_NOTE,
)
from research_assistant.graph.runner import RunOutcome, run_research
from research_assistant.llm.errors import LLMOutputInvalid
from research_assistant.llm.fake import FakeCall, FakeLLM
from research_assistant.tools.fetch import FetchError, FetchErrorReason
from tests.integration.conftest import TrackingSessionFactory
from tests.integration.graph_fakes import (
    EXAMPLE,
    OKAPI,
    PLAN,
    PREMISE_PLAN,
    PREMISE_QUESTION,
    PREMISE_SEARCH,
    RFC2606,
    RFC9309,
    SEARCH,
    SECRETS,
    WORDS,
    FakeEmbedder,
    FakeFetcher,
    FakeSearch,
    ScriptedModel,
    blocks_in,
    claim_of,
    default_section,
    labels_in,
    sub_question_of,
    supported,
)

QUESTION = "What do okapis, reserved DNS names and robots.txt have in common?"
SQ = [s["question"] for s in PLAN["sub_questions"]]
GAP_QUERY_SQ2 = f"another angle on {SQ[1][:30]}"  # what ScriptedModel answers for sub-question 2


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class Harness:
    def __init__(self, factory: TrackingSessionFactory, **settings: Any) -> None:
        self.factory = factory
        values: dict[str, Any] = {"max_sub_questions": 3, "max_sources_per_sub_question": 2}
        values.update(settings)
        self.settings = Settings(_env_file=None, **values)
        self.model = ScriptedModel(factory)
        self.search = FakeSearch(dict(SEARCH))
        self.fetcher = FakeFetcher(factory)
        self.embedder = FakeEmbedder(factory)
        self.clock = FakeClock()
        self.sleeps: list[float] = []  # search pacing waits, served by advancing the fake clock
        self.events: list[tuple[str, str]] = []

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(round(seconds, 6))
        self.clock.now += seconds

    def deps(self) -> Deps:
        return Deps(
            settings=self.settings,
            session_factory=self.factory,
            make_llm=lambda on_call: FakeLLM(responder=self.model, on_call=on_call),
            search=self.search,
            fetcher=self.fetcher,
            embedder=self.embedder,
            clock=self.clock,
            on_event=lambda node, event, _payload: self.events.append((node, event)),
            sleep=self.sleep,
        )

    def run(self) -> RunOutcome:
        return run_research(QUESTION, self.deps())

    def session(self) -> Session:
        return self.factory()


@pytest.fixture
def h(committing_factory: TrackingSessionFactory) -> Harness:
    return Harness(committing_factory)


def events_of(h: Harness, run_id: uuid.UUID, event: str) -> list[dict[str, Any]]:
    with h.session() as s:
        return [e.payload for e in list_events(s, run_id) if e.event == event]


def body_of(markdown: str) -> str:
    return markdown.split("## References")[0]


def section_text(markdown: str, n: int) -> str:
    return re.split(r"^## ", body_of(markdown), flags=re.MULTILINE)[n]


def claims_of(h: Harness, run_id: uuid.UUID) -> list[Claim]:
    with h.session() as s:
        return list(s.scalars(select(Claim).where(Claim.run_id == run_id).order_by(Claim.ordinal)))


def verdicts_for(h: Harness, run_id: uuid.UUID, sub_question: str) -> list[Verdict | None]:
    with h.session() as s:
        sq = s.scalar(
            select(SubQuestion).where(SubQuestion.run_id == run_id, SubQuestion.text == sub_question)
        )
        assert sq is not None
        return [c.verdict for c in s.scalars(select(Claim).where(Claim.sub_question_id == sq.id))]


def verify_by(rule: Any) -> Any:
    """verify override: rule(claim_text) -> verdict; quotes the start of C1 unless unsupported."""

    def respond(call: FakeCall) -> str:
        verdict = rule(claim_of(call))
        return supported(call, quote="" if verdict == "unsupported" else None, verdict=verdict)

    return respond


def pick_first(_call: FakeCall) -> str:
    return json.dumps({"selected": ["R1"]})


# -- happy path ------------------------------------------------------------------------------


def test_happy_path(h: Harness) -> None:
    outcome = h.run()
    assert (outcome.status, outcome.reason) == (RunStatus.COMPLETED, None)
    claims = claims_of(h, outcome.run_id)
    assert [c.ordinal for c in claims] == list(range(6))
    assert all(c.verdict is Verdict.SUPPORTED for c in claims)
    with h.session() as s:
        run = s.get(ResearchRun, outcome.run_id)
        assert run is not None and run.finished_at and run.error is None
        for chunk in s.scalars(select(Chunk).where(Chunk.run_id == run.id)):
            source = s.get(Source, chunk.source_id)
            assert source is not None and source.extracted_text is not None
            assert source.extracted_text[chunk.char_start : chunk.char_end] == chunk.text
        for ev in s.scalars(select(ClaimEvidence).join(Claim).where(Claim.run_id == run.id)):
            chunk = s.get(Chunk, ev.chunk_id)
            assert chunk is not None and chunk.text[ev.quote_start : ev.quote_end] == ev.quote
        events = list_events(s, run.id)
    assert [e.node for e in events if e.event == "node_start"] == [
        "plan",
        "research",
        "analyze",
        "verify",
        "write",
    ]
    calls = [e.payload["prompt_name"] for e in events if e.event == "llm_call"]
    assert {n: calls.count(n) for n in set(calls)} == {
        "warm_up": 1,
        "plan": 1,
        "select": 3,
        "analyze": 3,
        "verify": 6,
        "write_section": 3,
    }
    finished = [e.payload for e in events if e.event == "run_finished"][0]
    assert finished["claims"]["overall"] == {"supported": 6}
    assert finished["claims"]["per_sub_question"] == {str(i): {"supported": 2} for i in (1, 2, 3)}
    assert outcome.report_markdown is not None
    assert outcome.report_markdown.count("\n## ") == 5  # 3 sections, references, evidence


def test_every_citation_resolves_through_a_supported_claim(h: Harness) -> None:
    outcome = h.run()
    with h.session() as s:
        report = latest_report(s, outcome.run_id)
        assert report is not None
        markers = [int(n) for n in re.findall(r"\[(\d+)\]", body_of(report.markdown))]
        assert markers and sorted(set(markers)) == [c["n"] for c in report.citations]
        first_seen = list(dict.fromkeys(markers))
        assert first_seen == list(range(1, len(first_seen) + 1))
        references = report.markdown.split("## References")[1].split("## Evidence")[0]
        for c in report.citations:
            source = s.get(Source, uuid.UUID(c["source_id"]))
            assert source is not None and source.run_id == outcome.run_id
            assert source.status is SourceStatus.FETCHED
            assert f"- [{c['n']}] {source.title} — <{source.final_url}>" in references
            assert c["claim_ids"]
            for claim_id in c["claim_ids"]:
                claim = s.get(Claim, uuid.UUID(claim_id))
                assert claim is not None and claim.verdict is Verdict.SUPPORTED
                [ev] = s.scalars(select(ClaimEvidence).where(ClaimEvidence.claim_id == claim.id)).all()
                assert str(ev.chunk_id) in c["chunk_ids"]
                chunk = s.get(Chunk, ev.chunk_id)
                assert chunk is not None and chunk.source_id == source.id


def test_evidence_appendix_quotes_are_exact_substrings_of_stored_chunks(h: Harness) -> None:
    outcome = h.run()
    assert outcome.report_markdown is not None
    appendix = outcome.report_markdown.split("## Evidence")[1]
    entries = re.findall(
        r"^- \*\*Claim (\d+)\*\* \[\d+\]: .*\n((?:  > .*\n?)+)", appendix, flags=re.MULTILINE
    )
    assert len(entries) == 6
    with h.session() as s:
        for ordinal, quote_block in entries:
            quote = "\n".join(line[4:] for line in quote_block.rstrip("\n").split("\n"))
            claim = s.scalar(
                select(Claim).where(Claim.run_id == outcome.run_id, Claim.ordinal == int(ordinal))
            )
            assert claim is not None
            [ev] = s.scalars(select(ClaimEvidence).where(ClaimEvidence.claim_id == claim.id)).all()
            chunk = s.get(Chunk, ev.chunk_id)
            assert chunk is not None and quote == ev.quote and quote in chunk.text


# -- verification ----------------------------------------------------------------------------


def test_quote_found_by_whitespace_normalization(h: Harness) -> None:
    def respond(call: FakeCall) -> str:
        window = blocks_in(call)["C1"][:120]
        assert re.search(r"\s", window)
        return supported(call, quote=re.sub(r"\s+", "  ", window).strip())  # every whitespace run changed

    h.model.verify = respond
    outcome = h.run()
    assert outcome.status is RunStatus.COMPLETED
    assert {v["quote_match"] for v in events_of(h, outcome.run_id, "claim_verified")} == {"whitespace"}
    with h.session() as s:
        for ev in s.scalars(select(ClaimEvidence).join(Claim).where(Claim.run_id == outcome.run_id)):
            chunk = s.get(Chunk, ev.chunk_id)
            assert chunk is not None and chunk.text[ev.quote_start : ev.quote_end] == ev.quote


def test_quote_not_found_makes_claim_unsupported(committing_factory: TrackingSessionFactory) -> None:
    h = Harness(committing_factory, max_gap_loops=0)
    h.model.verify = lambda call: supported(call, quote="These words are not in the chunk at all.")
    outcome = h.run()
    claims = claims_of(h, outcome.run_id)
    assert all(c.verdict is Verdict.UNSUPPORTED and c.rationale == "quote_not_found" for c in claims)
    assert len(events_of(h, outcome.run_id, "quote_not_found")) == 6
    with h.session() as s:
        evidence = s.scalar(
            select(func.count()).select_from(ClaimEvidence).join(Claim).where(Claim.run_id == outcome.run_id)
        )
    assert evidence == 0
    assert outcome.status is RunStatus.PARTIAL


def test_supported_with_empty_quote_is_unsupported(committing_factory: TrackingSessionFactory) -> None:
    h = Harness(committing_factory, max_gap_loops=0)
    h.model.verify = lambda call: supported(call, quote="")
    outcome = h.run()
    assert {c.rationale for c in claims_of(h, outcome.run_id)} == {"quote_not_found"}


def test_only_supported_claims_reach_the_writer(committing_factory: TrackingSessionFactory) -> None:
    h = Harness(committing_factory, max_gap_loops=0)
    verdict_by_number = {"one": "supported", "two": "contradicted", "three": "unsupported", "four": "partial"}

    def analyze(call: FakeCall) -> str:
        label = labels_in(call, "C")[0]
        if "okapi" in sub_question_of(call):
            texts = [f"Okapi finding {WORDS[i]} as stated by the passage." for i in range(1, 5)]
        else:
            texts = [f"Other finding {WORDS[i]} about {sub_question_of(call)[:25]}." for i in range(1, 3)]
        return json.dumps({"claims": [{"text": t, "evidence": [label]} for t in texts]})

    def rule(claim: str) -> str:
        n = re.search(r"Okapi finding (\w+) ", claim)
        return verdict_by_number[n.group(1)] if n else "supported"

    h.model.analyze = analyze
    h.model.verify = verify_by(rule)
    outcome = h.run()
    by_text = {c.text: c.verdict for c in claims_of(h, outcome.run_id)}
    okapi = sorted(str(v) for t, v in by_text.items() if t.startswith("Okapi"))
    assert okapi == ["contradicted", "partial", "supported", "unsupported"]
    writer_prompts = "\n".join(c.prompt.user for c in h.model.prompts("write_section"))
    for text, verdict in by_text.items():
        assert (text in writer_prompts) == (verdict is Verdict.SUPPORTED), (text, verdict)


def test_verify_failure_is_a_problem_and_the_run_continues(
    committing_factory: TrackingSessionFactory,
) -> None:
    h = Harness(committing_factory, max_gap_loops=0, min_supported_claims=1)

    def respond(call: FakeCall) -> str:
        if claim_of(call).startswith("Round one finding one") and "does it live" in claim_of(call):
            raise LLMOutputInvalid("bad verify", raw="{}", errors="bad")
        return supported(call)

    h.model.verify = respond
    outcome = h.run()
    assert outcome.status is RunStatus.PARTIAL
    assert outcome.reason is not None and "verification failed (LLMOutputInvalid)" in outcome.reason
    failed = [c for c in claims_of(h, outcome.run_id) if c.rationale == "verify_failed: LLMOutputInvalid"]
    assert len(failed) == 1 and failed[0].verdict is Verdict.UNSUPPORTED
    assert h.model.count("write_section") == 3  # the run went on to write every section


# -- writing ---------------------------------------------------------------------------------


def test_uncited_sentence_is_deleted_and_logged(h: Harness) -> None:
    def write(call: FakeCall) -> str:
        k = labels_in(call, "K")
        return f"First fact [{k[0]}]. This filler sentence has no label. Second fact [{k[1]}]."

    h.model.write = write
    outcome = h.run()
    assert outcome.status is RunStatus.COMPLETED
    assert outcome.report_markdown is not None
    assert "This filler sentence has no label" not in outcome.report_markdown
    deleted = events_of(h, outcome.run_id, "uncited_sentences_deleted")
    assert len(deleted) == 3
    assert all(d["count"] == 1 and d["sentences"] == ["This filler sentence has no label."] for d in deleted)


def test_unknown_k_marker_regenerates_once_then_fails_the_section(h: Harness) -> None:
    def write(call: FakeCall) -> str:
        if sub_question_of(call) == SQ[1]:
            return "This cites a claim that was never shown [K99]."
        return default_section(labels_in(call, "K"))

    h.model.write = write
    outcome = h.run()
    assert outcome.status is RunStatus.PARTIAL
    assert outcome.reason is not None and "section 2 could not be written" in outcome.reason
    assert sum(1 for c in h.model.prompts("write_section") if sub_question_of(c) == SQ[1]) == 2
    assert [e["attempt"] for e in events_of(h, outcome.run_id, "citation_error")] == [1, 2]
    assert outcome.report_markdown is not None
    assert "This section could not be written" in section_text(outcome.report_markdown, 2)


def test_regeneration_can_succeed(h: Harness) -> None:
    attempts: dict[str, int] = {}

    def write(call: FakeCall) -> str:
        key = sub_question_of(call)
        attempts[key] = attempts.get(key, 0) + 1
        if attempts[key] == 1:
            return "See https://example.org for details [K1]."
        assert "rejected" in call.prompt.user
        return default_section(labels_in(call, "K"))

    h.model.write = write
    outcome = h.run()
    assert outcome.status is RunStatus.COMPLETED
    assert len(events_of(h, outcome.run_id, "citation_error")) == 3


def test_claim_dedupe_drops_later_duplicates_and_is_not_a_problem(h: Harness) -> None:
    def analyze(call: FakeCall) -> str:  # every sub-question yields the same two claims
        label = labels_in(call, "C")[0]
        texts = ["Okapis live in the Ituri rainforest.", "Okapis are related to giraffes."]
        return json.dumps({"claims": [{"text": t, "evidence": [label]} for t in texts]})

    h.model.analyze = analyze
    outcome = h.run()
    assert outcome.status is RunStatus.COMPLETED
    [dedupe] = events_of(h, outcome.run_id, "claims_deduplicated")
    assert [(d["claim"], d["duplicate_of"]) for d in dedupe["dropped"]] == [(2, 0), (3, 1), (4, 0), (5, 1)]
    assert all(d["similarity"] >= 0.9 for d in dedupe["dropped"])
    assert h.model.count("write_section") == 1
    assert outcome.report_markdown is not None
    for n in (2, 3):
        assert COVERED_NOTE in section_text(outcome.report_markdown, n)
    assert h.embedder.clustering_calls == 1
    assert len(events_of(h, outcome.run_id, "claim_similarity_top")[0]["pairs"]) == 10


def test_zero_supported_claims_gets_fixed_section_and_partial(
    committing_factory: TrackingSessionFactory,
) -> None:
    h = Harness(committing_factory, max_gap_loops=0)
    h.model.verify = verify_by(lambda claim: "unsupported" if "protocol" in claim else "supported")
    outcome = h.run()
    assert outcome.status is RunStatus.PARTIAL
    assert outcome.reason == "sub-question 3: no supported claims"
    assert outcome.report_markdown is not None
    assert INSUFFICIENT_NOTE in section_text(outcome.report_markdown, 3)
    assert h.model.count("write_section") == 2


# -- gap loop --------------------------------------------------------------------------------


def test_gap_loop_runs_once_for_weak_sub_questions_only(h: Harness) -> None:
    h.search.results[GAP_QUERY_SQ2] = [SECRETS]
    # sub-question 2's first-round claims fail; its gap-round claims pass
    h.model.verify = verify_by(
        lambda claim: "unsupported" if claim.startswith("Round one") and "testing" in claim else "supported"
    )
    outcome = h.run()
    assert outcome.status is RunStatus.COMPLETED
    gap_calls = h.model.prompts("gap")
    assert [sub_question_of(c) for c in gap_calls] == [SQ[1]]
    assert "reserved dns names" in gap_calls[0].prompt.user  # tried queries are shown
    with h.session() as s:
        events = list_events(s, outcome.run_id)
        sq2 = s.scalar(
            select(SubQuestion).where(SubQuestion.run_id == outcome.run_id, SubQuestion.ordinal == 1)
        )
        assert sq2 is not None and sq2.search_queries == ["reserved dns names", GAP_QUERY_SQ2]
    nodes = [e.node for e in events if e.event == "node_start"]
    assert nodes == ["plan", "research", "analyze", "verify", "gap", "research", "analyze", "verify", "write"]
    assert h.fetcher.calls.count(SECRETS) == 1 and len(h.fetcher.calls) == 4
    analyze_calls = h.model.prompts("analyze")
    assert [sub_question_of(c) for c in analyze_calls[3:]] == [SQ[1]]
    first = next(c for c in analyze_calls if sub_question_of(c) == SQ[1])
    assert not set(blocks_in(analyze_calls[3]).values()) & set(blocks_in(first).values())  # new chunks only
    assert verdicts_for(h, outcome.run_id, SQ[1]).count(Verdict.SUPPORTED) == 2


def test_gap_loop_respects_max_gap_loops(h: Harness) -> None:
    h.search.results[GAP_QUERY_SQ2] = [SECRETS]
    h.model.verify = verify_by(lambda claim: "unsupported" if "testing" in claim else "supported")
    outcome = h.run()
    assert h.model.count("gap") == 1  # max_gap_loops = 1
    assert events_of(h, outcome.run_id, "gap_not_run")[0]["reason"] == "max_gap_loops reached"
    assert outcome.status is RunStatus.PARTIAL
    assert outcome.reason == "sub-question 2: no supported claims"


def test_gap_loop_skipped_for_budget_is_a_problem(committing_factory: TrackingSessionFactory) -> None:
    h = Harness(committing_factory, max_sources_per_run=3)
    h.model.verify = verify_by(lambda claim: "unsupported" if "testing" in claim else "supported")
    outcome = h.run()
    assert h.model.count("gap") == 0
    assert outcome.status is RunStatus.PARTIAL
    assert outcome.reason is not None
    assert "gap loop needed for sub-question(s) 2 but skipped: source budget spent" in outcome.reason


def test_gap_loop_not_needed_when_every_sub_question_has_enough(h: Harness) -> None:
    outcome = h.run()
    assert h.model.count("gap") == 0 and outcome.status is RunStatus.COMPLETED


# -- research --------------------------------------------------------------------------------


def test_backfill_after_fetch_failure(h: Harness) -> None:
    blocked = "https://blocked.example.org/page"
    h.search.results["okapi animal"] = [blocked, OKAPI]
    h.fetcher.pages[blocked] = FetchError(
        FetchErrorReason.HTTP_ERROR, "HTTP 403", url=blocked, status_code=403
    )
    h.model.select = pick_first
    outcome = h.run()
    assert outcome.status is RunStatus.COMPLETED
    [backfill] = events_of(h, outcome.run_id, "backfill")
    assert (backfill["replaces"], backfill["because"], backfill["url"]) == (blocked, "fetch_failed", OKAPI)
    assert h.fetcher.calls[:2] == [blocked, OKAPI]


def test_backfill_after_extract_skip_and_duplicate(h: Harness) -> None:
    mirror = "https://mirror.example.org/rfc2606.txt"
    h.fetcher.pages[mirror] = h.fetcher.pages[RFC2606]
    h.search.results["okapi animal"] = [EXAMPLE, OKAPI]
    h.search.results["robots exclusion"] = [mirror, RFC9309]
    h.model.select = pick_first
    outcome = h.run()
    backfills = {(b["replaces"], b["because"], b["url"]) for b in events_of(h, outcome.run_id, "backfill")}
    assert backfills == {(EXAMPLE, "extract_skipped", OKAPI), (mirror, "duplicate", RFC9309)}
    assert outcome.status is RunStatus.COMPLETED


def test_no_backfill_after_embed_failure(h: Harness) -> None:
    h.embedder.fail_when = "Okapi"
    h.search.results["okapi animal"] = [OKAPI, SECRETS]
    h.model.select = pick_first
    outcome = h.run()
    assert events_of(h, outcome.run_id, "backfill") == []
    assert SECRETS not in h.fetcher.calls
    assert [e["url"] for e in events_of(h, outcome.run_id, "embed_failed")] == [OKAPI]


def test_storage_error_marks_source_failed_and_run_continues(h: Harness) -> None:
    # Real DataError from PostgreSQL: a plain-text page has no title of its own, so the search
    # result's title is stored, and this one contains a NUL byte.
    mirror = "https://mirror.example.org/rfc2606.txt"
    h.fetcher.pages[mirror] = h.fetcher.pages[RFC2606]
    h.search.results["reserved dns names"] = [mirror, RFC2606]
    h.search.titles[mirror] = "RFC 2606\x00 mirror"
    h.model.select = pick_first
    outcome = h.run()
    assert outcome.status is RunStatus.COMPLETED
    [failed] = events_of(h, outcome.run_id, "storage_failed")
    assert failed["url"] == mirror and failed["recorded"] is True
    assert failed["error"].startswith("storage_error: DataError: ") and "NUL" in failed["error"]
    [backfill] = events_of(h, outcome.run_id, "backfill")
    assert (backfill["replaces"], backfill["because"], backfill["url"]) == (mirror, "storage_failed", RFC2606)
    with h.session() as s:
        rows = {src.url: src for src in s.scalars(select(Source).where(Source.run_id == outcome.run_id))}
        assert rows[mirror].status is SourceStatus.FAILED and rows[mirror].title is None
        assert rows[mirror].error == failed["error"] and "INSERT" not in failed["error"]
        mirror_chunks = select(func.count()).select_from(Chunk).where(Chunk.source_id == rows[mirror].id)
        assert s.scalar(mirror_chunks) == 0
        assert rows[RFC2606].status is SourceStatus.FETCHED  # same text, stored after the rollback
    with h.session() as s:
        assert run_metrics(s, outcome.run_id)["sources_failed_by_reason"] == {"storage_error": 1}


def test_storage_error_rolls_back_only_that_source(h: Harness, monkeypatch: pytest.MonkeyPatch) -> None:
    original = research_node_module.add_chunks

    def add_chunks(session: Session, source_id: uuid.UUID, chunks: Any) -> Any:
        if any("Okapia" in c.text for c in chunks):
            raise IntegrityError("INSERT INTO chunks ...", {}, UniqueViolation("duplicate key value"))
        return original(session, source_id, chunks)

    monkeypatch.setattr(research_node_module, "add_chunks", add_chunks)
    outcome = h.run()
    assert outcome.status is not RunStatus.FAILED
    [failed] = events_of(h, outcome.run_id, "storage_failed")
    assert failed["url"] == OKAPI and failed["error"] == "storage_error: UniqueViolation: duplicate key value"
    with h.session() as s:
        rows = {src.url: src for src in s.scalars(select(Source).where(Source.run_id == outcome.run_id))}
        # The FETCHED row inserted in the same transaction as the chunks was rolled back.
        assert rows[OKAPI].status is SourceStatus.FAILED and rows[OKAPI].extracted_text is None
        assert s.scalar(select(func.count()).select_from(Chunk).where(Chunk.source_id == rows[OKAPI].id)) == 0
        assert rows[RFC2606].status is SourceStatus.FETCHED and rows[RFC9309].status is SourceStatus.FETCHED
    assert outcome.run_id is not None and len(claims_of(h, outcome.run_id)) > 0


def test_backfill_limit(h: Harness) -> None:
    bad = [f"https://bad{i}.example.org/" for i in range(3)]
    for url in bad:
        h.fetcher.pages[url] = FetchError(FetchErrorReason.HTTP_ERROR, "HTTP 403", url=url, status_code=403)
    h.search.results["okapi animal"] = [*bad, OKAPI]
    h.model.select = pick_first
    outcome = h.run()
    assert len(events_of(h, outcome.run_id, "backfill")) == 2  # max_backfills_per_sub_question
    assert h.fetcher.calls[:3] == bad and OKAPI not in h.fetcher.calls[:4]


def test_plan_queries_are_deduplicated_by_jaccard(h: Harness) -> None:
    plan = json.loads(json.dumps(PLAN))
    plan["sub_questions"][0]["search_queries"] = ["okapi animal", "Animal okapi", "okapi giraffe relative"]
    h.model.plan = plan
    outcome = h.run()
    assert "Animal okapi" not in h.search.queries and "okapi giraffe relative" in h.search.queries
    [dedupe] = events_of(h, outcome.run_id, "queries_deduplicated")
    assert dedupe["dropped"] == ["Animal okapi"]


def test_sub_question_without_sources_is_written_from_run_wide_evidence(
    committing_factory: TrackingSessionFactory,
) -> None:
    h = Harness(committing_factory, max_gap_loops=0)
    h.search.results["robots exclusion"] = [SECRETS]
    h.fetcher.pages[SECRETS] = FetchError(
        FetchErrorReason.HTTP_ERROR, "HTTP 503", url=SECRETS, status_code=503
    )
    outcome = h.run()
    assert outcome.status is RunStatus.PARTIAL
    assert outcome.reason is not None and "sub-question 3" in outcome.reason
    assert outcome.report_markdown is not None
    section = section_text(outcome.report_markdown, 3)
    assert NO_OWN_SOURCES_NOTE in section and re.search(r"\[\d+\]", section)


def test_no_sources_at_all_fails(h: Harness) -> None:
    h.search.results = {}
    outcome = h.run()
    assert outcome.status is RunStatus.FAILED
    assert outcome.reason is not None and outcome.reason.startswith("no chunks were stored")
    assert ("analyze", "node_start") not in h.events


def test_plan_failure_fails(h: Harness) -> None:
    h.model.plan = LLMOutputInvalid("still invalid", raw="{}", errors="sub_questions: missing")
    outcome = h.run()
    assert outcome.status is RunStatus.FAILED
    assert outcome.reason is not None and outcome.reason.startswith("plan failed: LLMOutputInvalid")


def test_fetch_failure_record(h: Harness) -> None:
    blocked = "https://blocked.example.org/page"
    h.search.results["okapi animal"] = [blocked, OKAPI]
    h.fetcher.pages[blocked] = FetchError(FetchErrorReason.ROBOTS_DISALLOWED, "disallowed", url=blocked)
    outcome = h.run()
    assert outcome.status is RunStatus.COMPLETED
    with h.session() as s:
        row = s.scalar(select(Source).where(Source.run_id == outcome.run_id, Source.url == blocked))
        assert row is not None and row.status is SourceStatus.FAILED
        assert row.error is not None and row.error.startswith("robots_disallowed: disallowed")


def test_same_url_is_fetched_once_per_run(h: Harness) -> None:
    h.search.results["robots exclusion"] = [OKAPI]
    outcome = h.run()
    assert h.fetcher.calls.count(OKAPI) == 1 and outcome.status is RunStatus.COMPLETED


def test_research_deadline_is_partial_with_reason(committing_factory: TrackingSessionFactory) -> None:
    h = Harness(committing_factory, run_deadline_s=1000, research_budget_share=0.7)
    h.fetcher.on_fetch = lambda _url: setattr(h.clock, "now", h.clock.now + 400)
    outcome = h.run()
    assert outcome.status is RunStatus.PARTIAL
    assert outcome.reason is not None and "deadline reached during research" in outcome.reason
    assert len(h.fetcher.calls) == 2


def test_spent_source_budget_skips_search_and_select(committing_factory: TrackingSessionFactory) -> None:
    h = Harness(committing_factory, max_sources_per_run=2, max_gap_loops=0)
    outcome = h.run()
    assert outcome.status is RunStatus.PARTIAL
    assert h.search.queries == ["okapi animal", "reserved dns names"]
    assert h.model.count("select") == 2
    assert outcome.reason is not None and "not researched, source budget of 2 reached" in outcome.reason


def test_search_failure_is_logged(h: Harness) -> None:
    h.search.failing = {"reserved dns names"}
    outcome = h.run()
    assert [e["query"] for e in events_of(h, outcome.run_id, "search_failed")][0] == "reserved dns names"


def test_write_deadline_marks_remaining_sections(committing_factory: TrackingSessionFactory) -> None:
    h = Harness(committing_factory, run_deadline_s=1000, node_timeout_s=5000)

    def slow_write(call: FakeCall) -> str:
        h.clock.now += 600
        return default_section(labels_in(call, "K"))

    h.model.write = slow_write
    outcome = h.run()
    assert outcome.status is RunStatus.PARTIAL
    assert outcome.reason == "section 3 not written: deadline reached"


def test_run_without_report_fails_even_with_chunks(h: Harness, monkeypatch: pytest.MonkeyPatch) -> None:
    from research_assistant.graph import build
    from research_assistant.graph.context import NodeResult

    monkeypatch.setattr(build, "write_node", lambda _ctx, _state: NodeResult({}))
    outcome = h.run()
    assert (outcome.status, outcome.reason) == (RunStatus.FAILED, "no report was written")


# -- transactions ----------------------------------------------------------------------------


def test_fakes_detect_an_open_session(h: Harness) -> None:
    session = h.factory()
    try:
        with pytest.raises(AssertionError, match="database session"):
            h.fetcher.fetch(OKAPI)
        with pytest.raises(AssertionError, match="database session"):
            h.embedder.embed_for_clustering(["x"])
    finally:
        session.close()


def test_no_session_is_open_during_llm_fetch_or_embed(h: Harness) -> None:
    outcome = h.run()
    assert outcome.status is RunStatus.COMPLETED
    assert h.factory.checks >= len(h.model.calls) + len(h.fetcher.calls)
    assert h.factory.open == 0


# -- CLI -------------------------------------------------------------------------------------


def test_cli_research_writes_report_file(h: Harness, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "REPORTS_DIR", tmp_path)

    @contextmanager
    def deps_factory(on_event: Any) -> Iterator[Deps]:
        deps = h.deps()
        deps.on_event = on_event
        yield deps

    assert cli.cmd_research(QUESTION, deps_factory) == 0
    [report_file] = list(tmp_path.glob("*.md"))
    assert report_file.read_text(encoding="utf-8").startswith("# Research report")


def test_cli_research_failed_exits_non_zero(
    h: Harness, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli, "REPORTS_DIR", tmp_path)
    h.model.plan = LLMOutputInvalid("bad", raw="{}", errors="bad")

    @contextmanager
    def deps_factory(on_event: Any) -> Iterator[Deps]:
        yield h.deps()

    assert cli.cmd_research(QUESTION, deps_factory) == 1
    assert list(tmp_path.glob("*.md")) == []


def test_cli_show(h: Harness, capsys: pytest.CaptureFixture[str]) -> None:
    outcome = h.run()
    capsys.readouterr()
    assert cli.cmd_show(outcome.run_id, h.factory) == 0
    out = capsys.readouterr().out
    assert "status:   completed" in out
    for node in ("plan", "research", "analyze", "verify", "write"):
        assert re.search(rf"^\s+{node}\s+node_end\s+[\d.]+s$", out, flags=re.MULTILINE)
    assert "llm calls: 17" in out
    assert re.search(r"^\s+0  supported\s+Round one finding one", out, flags=re.MULTILINE)
    assert "## Evidence" in out
    assert cli.cmd_show(uuid.uuid4(), h.factory) == 2


def test_claims_left_unverified_at_the_deadline(committing_factory: TrackingSessionFactory) -> None:
    h = Harness(committing_factory, run_deadline_s=1000, verify_budget_share=0.5, max_gap_loops=0)

    def slow_verify(call: FakeCall) -> str:
        h.clock.now += 100  # five verifications use the 500 s verify budget
        return supported(call)

    h.model.verify = slow_verify
    outcome = h.run()
    assert h.model.count("verify") == 5
    late = [c for c in claims_of(h, outcome.run_id) if c.rationale == "not_verified: deadline"]
    assert [(c.ordinal, c.verdict) for c in late] == [(5, Verdict.UNSUPPORTED)]
    assert outcome.status is RunStatus.PARTIAL
    assert outcome.reason is not None and "1 claim(s) not verified: deadline reached" in outcome.reason


# -- review fixes ----------------------------------------------------------------------------


def test_text_after_the_last_marker_is_removed_and_logged(h: Harness) -> None:
    first_claims: list[str] = []

    def write(call: FakeCall) -> str:
        k = labels_in(call, "K")
        texts = [blocks_in(call)[label].strip().rstrip(".") for label in k[:2]]
        first_claims.append(texts[0])
        return f"{texts[0]} [{k[0]}], which may have affected everything. {texts[1]} [{k[1]}]."

    h.model.write = write
    outcome = h.run()
    assert outcome.status is RunStatus.COMPLETED
    assert outcome.report_markdown is not None
    assert "which may have affected everything" not in outcome.report_markdown
    for text in first_claims:  # the trimmed sentence keeps its claim and marker
        assert re.search(re.escape(text) + r" \[\d+\]\.", outcome.report_markdown)
    assert events_of(h, outcome.run_id, "sentence_replaced") == []
    removed = events_of(h, outcome.run_id, "trailing_text_removed")
    assert len(removed) == 3
    assert all(r["fragments"] == [", which may have affected everything"] for r in removed)


@pytest.mark.parametrize("status", [403, 429])
def test_host_is_blocked_after_403_or_429(h: Harness, status: int) -> None:
    first, second = "https://blocked.example.org/a", "https://blocked.example.org/b"
    h.fetcher.pages[first] = FetchError(
        FetchErrorReason.HTTP_ERROR, f"HTTP {status}", url=first, status_code=status
    )
    h.fetcher.pages[second] = h.fetcher.pages[RFC2606]  # would succeed if it were fetched
    h.search.results["okapi animal"] = [first, OKAPI]
    h.search.results["reserved dns names"] = [second, RFC2606]
    h.model.select = pick_first
    outcome = h.run()
    assert outcome.status is RunStatus.COMPLETED
    assert second not in h.fetcher.calls  # same host, never fetched
    assert events_of(h, outcome.run_id, "host_blocked") == [
        {"host": "blocked.example.org", "status_code": status, "url": first}
    ]
    backfills = [(b["replaces"], b["because"], b["url"]) for b in events_of(h, outcome.run_id, "backfill")]
    assert backfills == [(first, "fetch_failed", OKAPI), (second, "host_blocked", RFC2606)]
    with h.session() as s:
        assert s.scalar(select(Source).where(Source.run_id == outcome.run_id, Source.url == second)) is None


@pytest.mark.parametrize("status", [404, 500, 503])
def test_other_statuses_do_not_block_the_host(h: Harness, status: int) -> None:
    first, second = "https://flaky.example.org/a", "https://flaky.example.org/b"
    h.fetcher.pages[first] = FetchError(
        FetchErrorReason.HTTP_ERROR, f"HTTP {status}", url=first, status_code=status
    )
    h.fetcher.pages[second] = h.fetcher.pages[RFC2606]
    h.search.results["okapi animal"] = [first, OKAPI]
    h.search.results["reserved dns names"] = [second, RFC2606]
    h.model.select = pick_first
    outcome = h.run()
    assert second in h.fetcher.calls
    assert events_of(h, outcome.run_id, "host_blocked") == []


def test_host_block_lasts_across_gap_rounds(h: Harness) -> None:
    blocked_url = "https://blocked.example.org/a"
    later = "https://blocked.example.org/later"
    h.fetcher.pages[blocked_url] = FetchError(
        FetchErrorReason.HTTP_ERROR, "HTTP 403", url=blocked_url, status_code=403
    )
    h.fetcher.pages[later] = h.fetcher.pages[SECRETS]
    h.search.results["reserved dns names"] = [blocked_url, RFC2606]
    h.search.results[GAP_QUERY_SQ2] = [later, SECRETS]
    h.model.select = pick_first
    h.model.verify = verify_by(
        lambda claim: "unsupported" if claim.startswith("Round one") and "testing" in claim else "supported"
    )
    outcome = h.run()
    assert h.model.count("gap") == 1
    assert later not in h.fetcher.calls and SECRETS in h.fetcher.calls
    assert len(events_of(h, outcome.run_id, "host_blocked")) == 1


def test_speculative_sentence_is_replaced_by_its_claims(h: Harness) -> None:
    def write(call: FakeCall) -> str:
        k = labels_in(call, "K")
        return f"First fact [{k[0]}]. This change likely helped the landing [{k[1]}]."

    h.model.write = write
    outcome = h.run()
    assert outcome.status is RunStatus.COMPLETED
    assert outcome.report_markdown is not None
    assert "likely helped" not in outcome.report_markdown
    replaced = events_of(h, outcome.run_id, "speculation_replaced")
    assert len(replaced) == 3
    for r in replaced:
        assert r["original"].startswith("This change likely helped the landing [K")
        assert r["phrases"] == ["likely"]
        assert r["replacement"].startswith("Round one finding two about")
    claims = {c.text for c in claims_of(h, outcome.run_id)}
    second_claims = [t for t in claims if t.startswith("Round one finding two")]
    assert all(t.rstrip(".") in outcome.report_markdown for t in second_claims)


def test_number_not_in_quote_makes_a_claim_unsupported(committing_factory: TrackingSessionFactory) -> None:
    h = Harness(committing_factory, max_gap_loops=0)

    def analyze(call: FakeCall) -> str:
        label = labels_in(call, "C")[0]
        quoted = blocks_in(call)[label][:80]  # what verify will quote (supported() uses C1[:80])
        texts = [
            f"The passage weighs 999 kilograms according to {sub_question_of(call)[:20]}.",
            f"The passage opens with: {quoted}",  # every number here is in the quote
        ]
        return json.dumps({"claims": [{"text": t, "evidence": [label]} for t in texts]})

    h.model.analyze = analyze
    outcome = h.run()
    claims = claims_of(h, outcome.run_id)
    wrong = [c for c in claims if "999 kilograms" in c.text]
    right = [c for c in claims if c.text.startswith("The passage opens with")]
    assert wrong and all(c.verdict is Verdict.UNSUPPORTED for c in wrong)
    assert all(c.rationale == "number_not_in_quote: 999" for c in wrong)
    assert right and all(c.verdict is Verdict.SUPPORTED for c in right)
    mismatches = events_of(h, outcome.run_id, "number_mismatch")
    assert len(mismatches) == len(wrong) and all(m["missing"] == ["999"] for m in mismatches)
    assert all(m["model_verdict"] == "supported" for m in mismatches)


# -- false premise ---------------------------------------------------------------------------


def premise_harness(committing_factory: TrackingSessionFactory, premise_verdict: str) -> Harness:
    h = Harness(committing_factory, max_gap_loops=0)
    h.model.plan = PREMISE_PLAN
    h.search.results = dict(PREMISE_SEARCH)
    h.model.verify = verify_by(lambda claim: premise_verdict if "lyon" in claim else "supported")
    return h


def test_unsupported_premise_is_reported_not_a_problem(committing_factory: TrackingSessionFactory) -> None:
    h = premise_harness(committing_factory, "unsupported")
    outcome = run_research(PREMISE_QUESTION, h.deps())
    assert outcome.status is RunStatus.COMPLETED, outcome.reason  # not partial because of the premise
    assert outcome.report_markdown is not None
    lines = outcome.report_markdown.split("\n")
    question_line = lines.index(f"**Question:** {PREMISE_QUESTION}")
    assert lines[question_line + 2] == f"_{PREMISE_NOTE}_"  # directly under the question
    first_section = section_text(outcome.report_markdown, 1)
    assert first_section.startswith("1. Was the Eiffel Tower moved to Lyon in 1999?")
    assert f"_{PREMISE_NOTE}_" in first_section and "[" not in first_section
    assert "[1]" in section_text(outcome.report_markdown, 2)  # the other sections are written normally
    assert events_of(h, outcome.run_id, "problem") == []
    assert len(events_of(h, outcome.run_id, "premise_unsupported")) == 1
    with h.session() as s:
        flags = [
            sq.premise_check
            for sq in s.scalars(
                select(SubQuestion).where(SubQuestion.run_id == outcome.run_id).order_by(SubQuestion.ordinal)
            )
        ]
    assert flags == [True, False, False]


def test_supported_premise_gets_no_note(committing_factory: TrackingSessionFactory) -> None:
    h = premise_harness(committing_factory, "supported")
    outcome = run_research(PREMISE_QUESTION, h.deps())
    assert outcome.report_markdown is not None and PREMISE_NOTE not in outcome.report_markdown
    assert events_of(h, outcome.run_id, "premise_unsupported") == []


def test_fuzzy_located_quote_still_gets_the_number_check(committing_factory: TrackingSessionFactory) -> None:
    h = Harness(committing_factory, max_gap_loops=0)

    def analyze(call: FakeCall) -> str:
        label = labels_in(call, "C")[0]
        return json.dumps(
            {"claims": [{"text": "The passage reports 999 units in total.", "evidence": [label]}]}
        )

    def verify(call: FakeCall) -> str:
        chunk = blocks_in(call)["C1"]
        original = chunk[:80]
        i = next(k for k in range(10, 80) if original[k].isalpha())  # change one letter: fuzzy, 1 edit
        typo = original[:i] + ("x" if original[i] != "x" else "y") + original[i + 1 :]
        return json.dumps({"verdict": "supported", "evidence_alias": "C1", "quote": typo, "rationale": "s"})

    h.model.analyze = analyze
    h.model.verify = verify
    outcome = h.run()
    verified = events_of(h, outcome.run_id, "claim_verified")
    assert verified and all(v["match_method"] == "fuzzy" for v in verified)
    claims = claims_of(h, outcome.run_id)
    assert claims and all(c.verdict is Verdict.UNSUPPORTED for c in claims)
    assert all(c.rationale == "number_not_in_quote: 999" for c in claims)
    with h.session() as s:
        stored = [e.quote for e in s.scalars(select(ClaimEvidence))]
        chunks = [ch.text for ch in s.scalars(select(Chunk).where(Chunk.run_id == outcome.run_id))]
    assert stored and all(any(q in text for text in chunks) for q in stored)  # the chunk's own text


def test_claim_verified_carries_the_claim_text(h: Harness) -> None:
    outcome = h.run()
    verified = events_of(h, outcome.run_id, "claim_verified")
    texts = {str(c.id): c.text for c in claims_of(h, outcome.run_id)}
    assert verified and all(v["claim"] == texts[v["claim_id"]] for v in verified)


def test_unfaithful_sentence_is_replaced_by_its_claims(h: Harness) -> None:
    def write(call: FakeCall) -> str:
        k = labels_in(call, "K")
        first = blocks_in(call)[k[0]].strip().rstrip(".")
        return f"{first} [{k[0]}]. This remarkable discovery changed science forever [{k[1]}]."

    h.model.write = write
    outcome = h.run()
    assert outcome.status is RunStatus.COMPLETED and outcome.report_markdown is not None
    assert "remarkable discovery" not in outcome.report_markdown
    replaced = events_of(h, outcome.run_id, "sentence_replaced")
    assert len(replaced) == 3  # one per section
    assert all(
        set(r["unmatched_words"]) >= {"remarkabl", "discovery", "chang", "scienc", "forever"}
        for r in replaced
    )
    assert all(r["unmatched_numbers"] == [] for r in replaced)
    write_end = next(e for e in events_of(h, outcome.run_id, "node_end") if "sentences_checked" in e)
    assert (write_end["sentences_checked"], write_end["sentences_replaced"]) == (6, 3)
    with h.session() as s:
        texts = {c.text.rstrip(".") for c in s.scalars(select(Claim).where(Claim.run_id == outcome.run_id))}
    for r in replaced:  # the replacement is the cited claim's own text with its marker
        assert any(r["replacement"].startswith(t) for t in texts), r["replacement"]


def test_searches_are_paced_and_logged(committing_factory: TrackingSessionFactory) -> None:
    h = Harness(committing_factory, search_min_interval_s=1.5)
    outcome = h.run()
    done = events_of(h, outcome.run_id, "search_done")
    assert [d["query"] for d in done] == h.search.queries  # one event per provider call
    assert all((d["provider"], d["role"], d["error"]) == ("fake", "only", None) for d in done)
    assert [d["result_count"] for d in done] == [len(h.search.results.get(q, [])) for q in h.search.queries]
    # The fake clock only moves when pacing sleeps, so every search after the first waits the full interval.
    assert h.sleeps == [1.5] * (len(done) - 1) and len(done) >= 3


def test_no_pacing_when_interval_is_zero(committing_factory: TrackingSessionFactory) -> None:
    h = Harness(committing_factory, search_min_interval_s=0)
    h.run()
    assert h.sleeps == []
