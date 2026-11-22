"""Phase 8: reports.structure, GET /runs/{id}/report and GET /runs/{id}/sources, source_fetched."""

import uuid
from datetime import UTC, datetime

import pytest
from sqlalchemy import select

from research_assistant.db.models import Claim, ClaimEvidence, RunStatus, Source, SourceStatus
from research_assistant.db.repositories.events import list_events
from research_assistant.db.repositories.reports import latest_report, save_report
from research_assistant.db.repositories.runs import create_run
from research_assistant.db.repositories.sources import add_source
from research_assistant.db.session import session_scope
from research_assistant.graph.citations import render_markdown
from tests.event_names import frontend_event_names
from tests.integration.conftest import TrackingSessionFactory
from tests.integration.phase7_support import client_for, make_app
from tests.integration.test_graph import QUESTION, Harness, events_of


@pytest.fixture
def h(committing_factory: TrackingSessionFactory) -> Harness:
    return Harness(committing_factory)


def test_stored_structure_renders_to_the_stored_markdown(h: Harness) -> None:
    outcome = h.run()
    with session_scope(h.factory) as s:
        report = latest_report(s, outcome.run_id)
        assert report is not None and report.structure is not None
        assert render_markdown(QUESTION, report.structure) == report.markdown  # read back from the DB
        quotes: dict[str, set[str]] = {}
        stmt = (
            select(ClaimEvidence.claim_id, ClaimEvidence.quote)
            .join(Claim)
            .where(Claim.run_id == outcome.run_id)
        )
        for claim_id, quote in s.execute(stmt):
            quotes.setdefault(str(claim_id), set()).add(quote)
        structure = report.structure
    by_n = structure["claims_by_n"]
    markers = 0
    for section in structure["sections"]:
        for sentence in section["sentences"]:
            for part in sentence["parts"]:
                for cite in part.get("cite", []):
                    markers += 1
                    assert set(cite["claim_ids"]) <= {c["claim_id"] for c in by_n[str(cite["n"])]}
    assert markers > 0
    assert all(c["quote"] in quotes[c["claim_id"]] for items in by_n.values() for c in items)  # verbatim
    assert all(c["verdict"] == "supported" for items in by_n.values() for c in items)
    assert {s["kind"] for s in structure["sections"]} == {"normal"}


def test_every_stored_event_name_is_one_the_ui_listens_to(h: Harness) -> None:
    outcome = h.run()
    with session_scope(h.factory) as s:
        names = {e.event for e in list_events(s, outcome.run_id)}
    assert names - frontend_event_names() == set()


def test_report_endpoint_serves_the_structure(h: Harness, db_url: str) -> None:
    outcome = h.run()
    body = client_for(make_app(h, db_url)).get(f"/runs/{outcome.run_id}/report").json()
    with session_scope(h.factory) as s:
        report = latest_report(s, outcome.run_id)
        assert report is not None
        report_id, structure = report.id, report.structure
    assert body["report_id"] == str(report_id) and "legacy_markdown" not in body
    assert {k: body[k] for k in ("premise_note", "sections", "references", "claims_by_n")} == structure


def test_legacy_report_and_missing_reports(h: Harness, db_url: str) -> None:
    markdown = "# Research report\n\n**Question:** old\n\nText [1].\n"
    with session_scope(h.factory) as s:
        legacy = create_run(s, "an old question", {}, status=RunStatus.COMPLETED).id
        save_report(s, legacy, markdown)  # no structure: written before 2026-11-22
        empty = create_run(s, "no report yet", {}, status=RunStatus.RUNNING).id
    client = client_for(make_app(h, db_url))
    response = client.get(f"/runs/{legacy}/report")
    assert response.status_code == 200
    body = response.json()
    assert body["legacy_markdown"] == markdown and "sections" not in body
    for run_id in (empty, uuid.uuid4()):
        missing = client.get(f"/runs/{run_id}/report")
        assert (missing.status_code, missing.json()["error"]["code"]) == (404, "not_found")


def test_sources_endpoint_lists_every_source_with_flags(h: Harness, db_url: str) -> None:
    now = datetime.now(UTC)
    memory_url = "https://memory.example.org/a"
    with session_scope(h.factory) as s:
        earlier = create_run(s, "an earlier question", {}, status=RunStatus.COMPLETED).id
        original = add_source(
            s, earlier, memory_url, status=SourceStatus.FETCHED, title="Old", fetched_at=now
        )
        run_id = create_run(s, "this question", {}, status=RunStatus.COMPLETED).id
        cited = add_source(
            s, run_id, "https://a.example.org/", status=SourceStatus.FETCHED, title="A", fetched_at=now
        )
        reused = add_source(s, run_id, memory_url, status=SourceStatus.FETCHED, title="Old", fetched_at=now)
        reused.reused_from_source_id = original.id
        add_source(
            s, run_id, "https://b.example.org/", status=SourceStatus.FAILED, error="http_error: HTTP 403"
        )
        add_source(
            s, run_id, "https://c.example.org/", status=SourceStatus.SKIPPED, error="duplicate content"
        )
        citation = {"n": 1, "source_id": str(cited.id), "chunk_ids": [], "claim_ids": []}
        save_report(s, run_id, "# Research report\n", [citation])
    client = client_for(make_app(h, db_url))
    rows = {r["url"]: r for r in client.get(f"/runs/{run_id}/sources").json()}
    assert len(rows) == 4
    a = rows["https://a.example.org/"]
    assert (a["status"], a["cited"], a["reused"], a["title"]) == ("fetched", True, False, "A")
    assert a["retrieved_at"] is not None
    m = rows[memory_url]
    assert (m["reused"], m["reused_from_run_id"], m["cited"]) == (True, str(earlier), False)
    b = rows["https://b.example.org/"]
    assert (b["status"], b["reason"], b["retrieved_at"], b["cited"]) == (
        "failed",
        "http_error: HTTP 403",
        None,
        False,
    )
    assert rows["https://c.example.org/"]["status"] == "skipped"
    missing = client.get(f"/runs/{uuid.uuid4()}/sources")
    assert missing.status_code == 404


def test_source_fetched_is_logged_once_per_fetched_source(h: Harness) -> None:
    outcome = h.run()
    fetched = events_of(h, outcome.run_id, "source_fetched")
    with session_scope(h.factory) as s:
        stmt = select(Source.id).where(
            Source.run_id == outcome.run_id,
            Source.status == SourceStatus.FETCHED,
            Source.reused_from_source_id.is_(None),
        )
        ids = sorted(str(i) for i in s.scalars(stmt))
    assert ids and sorted(e["source_id"] for e in fetched) == ids
    assert all(e["chunks"] > 0 and e["url"].startswith("https://") for e in fetched)
