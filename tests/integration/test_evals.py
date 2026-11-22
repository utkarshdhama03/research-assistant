"""Evaluation harness against the test database: metrics, independent citation check, labels, resume."""

import csv
import json
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import psycopg
import pytest

from research_assistant import cli
from research_assistant.db.models import ResearchRun, RunStatus
from research_assistant.db.repositories.runs import create_run, update_run_status
from research_assistant.evals.citation_check import check_run, connect
from research_assistant.evals.labels import CLAIM_COLUMNS, PAIR_COLUMNS, export_labels
from research_assistant.evals.metrics import run_metrics
from research_assistant.evals.power import PowerState
from research_assistant.evals.questions import EvalQuestion
from research_assistant.evals.runner import ORPHAN_REASON, load_results, run_eval, save_results
from research_assistant.graph.context import Deps
from research_assistant.graph.runner import RunOutcome
from research_assistant.llm.errors import LLMOutputInvalid
from research_assistant.tools.fetch import FetchError, FetchErrorReason
from tests.integration.conftest import TrackingSessionFactory
from tests.integration.graph_fakes import OKAPI, PLAN, RFC2606
from tests.integration.test_graph import Harness, pick_first


@pytest.fixture
def h(committing_factory: TrackingSessionFactory) -> Harness:
    return Harness(committing_factory)


@pytest.fixture
def conn(db_url: str) -> Iterator[psycopg.Connection[Any]]:
    with connect(db_url) as c:
        yield c


def sql(conn: psycopg.Connection[Any], statement: str, *params: Any) -> None:
    with conn.cursor() as cur:
        cur.execute(statement, params)
    conn.commit()


def fetch_one(conn: psycopg.Connection[Any], statement: str, *params: Any) -> Any:
    with conn.cursor() as cur:
        cur.execute(statement, params)
        row = cur.fetchone()
    assert row is not None
    return row


# -- metrics ---------------------------------------------------------------------------------


def test_metrics_of_a_fake_run(h: Harness) -> None:
    blocked = "https://blocked.example.org/a"
    h.search.results["okapi animal"] = [blocked, OKAPI]
    h.fetcher.pages[blocked] = FetchError(
        FetchErrorReason.HTTP_ERROR, "HTTP 403", url=blocked, status_code=403
    )
    h.model.select = pick_first
    outcome = h.run()
    with h.session() as s:
        m = run_metrics(s, outcome.run_id)
    assert m["status"] == "completed" and m["problems"] == [] and m["reason"] is None
    assert set(m["node_seconds"]) == {"plan", "research", "analyze", "verify", "write"}
    assert m["llm_calls"] == 17 and m["llm_failed_calls"] == 0
    assert m["llm_by_prompt"]["verify"]["calls"] == 6
    assert m["sources_fetched"] == 3 and m["sources_failed_by_reason"] == {"http_error": 1}
    assert m["blocked_hosts"] == ["blocked.example.org"] and m["backfills"] == 1
    assert m["cited_domains"] == ["rfc-editor.org", "simple.wikipedia.org"]  # two rfc-editor.org sources
    assert m["distinct_cited_domains"] == 2
    assert m["claims_by_verdict"] == {
        "supported": 6,
        "partial": 0,
        "unsupported": 0,
        "contradicted": 0,
        "unverified": 0,
    }
    assert (m["quote_not_found"], m["quote_checks"], m["quote_not_found_rate"]) == (0, 6, 0.0)
    assert m["gap_rounds"] == 0 and m["zero_supported_sub_questions"] == []
    assert m["duration_s"] is not None


def test_metrics_count_write_post_processing(h: Harness) -> None:
    def write(call: Any) -> str:
        from tests.integration.graph_fakes import labels_in

        k = labels_in(call, "K")
        return f"First [{k[0]}], which may matter. Filler sentence. Second likely holds [{k[1]}]."

    h.model.write = write
    outcome = h.run()
    with h.session() as s:
        m = run_metrics(s, outcome.run_id)
    assert (m["uncited_sentences_deleted"], m["trailing_text_removed"], m["speculation_replaced"]) == (
        3,
        3,
        3,
    )


# -- independent citation check --------------------------------------------------------------


def test_clean_run_passes_the_citation_check(h: Harness, conn: psycopg.Connection[Any]) -> None:
    outcome = h.run()
    result = check_run(conn, outcome.run_id)
    assert result.checks > 20 and result.failures == [] and result.validity == 1.0


def kinds(conn: psycopg.Connection[Any], run_id: uuid.UUID) -> set[str]:
    return {f.kind for f in check_run(conn, run_id).failures}


def test_wrong_offsets_are_caught(h: Harness, conn: psycopg.Connection[Any]) -> None:
    run_id = h.run().run_id
    sql(
        conn,
        """UPDATE claim_evidence SET quote_start = quote_start + 1, quote_end = quote_end + 1
           WHERE claim_id = (SELECT id FROM claims WHERE run_id = %s ORDER BY ordinal LIMIT 1)""",
        str(run_id),
    )
    assert kinds(conn, run_id) == {"quote_offsets_mismatch"}


def test_foreign_chunk_is_caught(h: Harness, conn: psycopg.Connection[Any]) -> None:
    run_id = h.run().run_id
    other_run = Harness(h.factory).run().run_id
    (foreign,) = fetch_one(conn, "SELECT id FROM chunks WHERE run_id = %s LIMIT 1", str(other_run))
    sql(
        conn,
        """UPDATE reports SET citations = jsonb_set(citations, '{0,chunk_ids}',
               (citations->0->'chunk_ids') || to_jsonb(%s::text))
           WHERE run_id = %s""",
        str(foreign),
        str(run_id),
    )
    assert kinds(conn, run_id) == {"cited_chunk_not_in_run", "cited_chunk_wrong_source"}
    assert kinds(conn, other_run) == set()


def test_missing_citation_entry_is_caught(h: Harness, conn: psycopg.Connection[Any]) -> None:
    run_id = h.run().run_id
    sql(conn, "UPDATE reports SET citations = citations - 0 WHERE run_id = %s", str(run_id))
    assert {"marker_without_citation", "reference_without_citation"} <= kinds(conn, run_id)


def test_mismatched_reference_url_is_caught(h: Harness, conn: psycopg.Connection[Any]) -> None:
    run_id = h.run().run_id
    sql(
        conn,
        """UPDATE sources SET final_url = 'https://changed.example.org/'
           WHERE id = (SELECT (citations->0->>'source_id')::uuid FROM reports WHERE run_id = %s)""",
        str(run_id),
    )
    assert kinds(conn, run_id) == {"reference_url_mismatch"}


def test_unused_citation_entry_is_caught(h: Harness, conn: psycopg.Connection[Any]) -> None:
    run_id = h.run().run_id
    sql(
        conn,
        """UPDATE reports SET citations = citations || jsonb_build_array(jsonb_build_object(
               'n', 99, 'source_id', citations->0->>'source_id', 'chunk_ids', '[]'::jsonb,
               'claim_ids', '[]'::jsonb))
           WHERE run_id = %s""",
        str(run_id),
    )
    assert kinds(conn, run_id) == {"unused_citation_entry", "citation_without_reference"}


def test_appendix_quote_mismatch_is_caught(h: Harness, conn: psycopg.Connection[Any]) -> None:
    run_id = h.run().run_id
    (markdown,) = fetch_one(conn, "SELECT markdown FROM reports WHERE run_id = %s", str(run_id))
    body, appendix = markdown.split("## Evidence")
    lines = appendix.split("\n")
    first_quote = next(i for i, line in enumerate(lines) if line.startswith("  > "))
    lines[first_quote] = "  > tampered quote"
    sql(
        conn,
        "UPDATE reports SET markdown = %s WHERE run_id = %s",
        body + "## Evidence" + "\n".join(lines),
        str(run_id),
    )
    assert kinds(conn, run_id) == {"appendix_quote_mismatch"}


# -- label export ----------------------------------------------------------------------------


def eval_dir_with(tmp: Path, runs: dict[str, RunOutcome], question: str = "Q?") -> Path:
    tmp.mkdir(parents=True, exist_ok=True)
    save_results(
        tmp,
        {
            "created_at": "x",
            "settings": {},
            "sessions": [],
            "runs": {
                qid: {
                    "state": "finished",
                    "run_id": str(o.run_id),
                    "question": question,
                    "category": "technical",
                }
                for qid, o in runs.items()
            },
        },
    )
    return tmp


def test_label_export_is_deterministic_and_blind(h: Harness, tmp_path: Path) -> None:
    runs = {"q01-a": h.run(), "q02-b": Harness(h.factory).run()}
    first = eval_dir_with(tmp_path / "one", runs)
    second = eval_dir_with(tmp_path / "two", runs)
    quotas = {"supported": 5, "partial": 2, "unsupported_or_contradicted": 2}
    m1 = export_labels(
        first, h.factory, h.embedder.embed_for_clustering, seed=7, claim_quotas=quotas, sentence_quota=4
    )
    export_labels(
        second, h.factory, h.embedder.embed_for_clustering, seed=7, claim_quotas=quotas, sentence_quota=4
    )
    for name in ("claims.csv", "sentences.csv", "dedupe_pairs.csv"):
        assert (first / name).read_bytes() == (second / name).read_bytes(), name
    assert (first / "claims.csv").read_bytes().startswith(b"\xef\xbb\xbf")  # BOM for Excel
    with (first / "claims.csv").open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
        assert next(csv.reader((first / "claims.csv").open(encoding="utf-8-sig"))) == CLAIM_COLUMNS
    assert CLAIM_COLUMNS[-1] == "verifier_verdict" and PAIR_COLUMNS[-1] == "similarity"
    assert len(rows) == m1["claims"]["supported"] == 5  # 12 supported available, quota 5
    per_run = {qid: sum(1 for r in rows if r["id"].startswith(qid)) for qid in runs}
    assert max(per_run.values()) <= 5 // 2 + 2  # ceil(5/2) + 1
    assert all(r["quote"] and r["quote"] in r["chunk_text"] for r in rows)
    assert m1["sentences"] == 4
    other = export_labels(
        eval_dir_with(tmp_path / "three", runs),
        h.factory,
        h.embedder.embed_for_clustering,
        seed=8,
        claim_quotas=quotas,
        sentence_quota=4,
    )
    assert other["claims"]["supported"] == 5
    assert (tmp_path / "three" / "claims.csv").read_bytes() != (first / "claims.csv").read_bytes()


# -- resume ----------------------------------------------------------------------------------


def eval_harness(h: Harness) -> Any:
    @contextmanager
    def deps_factory(on_event: Any) -> Iterator[Deps]:
        yield h.deps()

    return deps_factory


QUESTIONS = [
    EvalQuestion("q1", "technical", "What do okapis and reserved DNS names have in common?"),
    EvalQuestion("q2", "science", "How does the robots exclusion protocol relate to okapis?"),
]


def test_resume_skips_finished_ids(h: Harness, tmp_path: Path, db_url: str) -> None:
    logs: list[str] = []
    common = {
        "check_connection": lambda: connect(db_url),
        "power": lambda: PowerState(True, 90),
        "log": logs.append,
    }
    run_eval(tmp_path, QUESTIONS, h.settings, eval_harness(h), h.factory, ids=["q1"], **common)
    first = load_results(tmp_path)
    assert first is not None and list(first["runs"]) == ["q1"] and first["runs"]["q1"]["state"] == "finished"
    run_eval(tmp_path, QUESTIONS, h.settings, eval_harness(h), h.factory, **common)
    second = load_results(tmp_path)
    assert second is not None
    assert "skip q1: already finished (completed)" in logs
    assert second["runs"]["q1"]["run_id"] == first["runs"]["q1"]["run_id"]
    assert second["runs"]["q2"]["state"] == "finished"
    assert second["runs"]["q2"]["citation_check"]["validity"] == 1.0
    assert len(second["sessions"]) == 2
    assert (tmp_path / "summary.md").exists() and "q2" in (tmp_path / "summary.md").read_text(
        encoding="utf-8"
    )
    with pytest.raises(ValueError, match="unknown question ids"):
        run_eval(tmp_path, QUESTIONS, h.settings, eval_harness(h), h.factory, ids=["nope"], **common)


def test_retry_failed_reruns_only_failed_questions(h: Harness, tmp_path: Path, db_url: str) -> None:
    logs: list[str] = []
    common = {
        "check_connection": lambda: connect(db_url),
        "power": lambda: PowerState(True, 90),
        "log": logs.append,
    }
    h.model.plan = LLMOutputInvalid("still invalid", raw="{}", errors="sub_questions: missing")
    run_eval(tmp_path, QUESTIONS, h.settings, eval_harness(h), h.factory, ids=["q1"], **common)
    h.model.plan = PLAN
    first = load_results(tmp_path)
    assert first is not None and first["runs"]["q1"]["status"] == "failed"
    failed_run = first["runs"]["q1"]["run_id"]

    run_eval(tmp_path, QUESTIONS, h.settings, eval_harness(h), h.factory, **common)  # plain resume
    second = load_results(tmp_path)
    assert second is not None and "skip q1: already finished (failed)" in logs
    assert second["runs"]["q1"]["run_id"] == failed_run and second["runs"]["q2"]["status"] == "completed"
    q2_run = second["runs"]["q2"]["run_id"]

    run_eval(tmp_path, QUESTIONS, h.settings, eval_harness(h), h.factory, retry_failed=True, **common)
    third = load_results(tmp_path)
    assert third is not None
    assert f"retry q1: previous run {failed_run} failed" in logs
    assert "skip q2: already finished (completed)" in logs
    q1 = third["runs"]["q1"]
    assert q1["status"] == "completed" and q1["run_id"] != failed_run
    assert q1["citation_check"]["validity"] == 1.0
    assert [r["run_id"] for r in q1["replaced_runs"]] == [failed_run]
    assert q1["replaced_runs"][0]["status"] == "failed"
    assert third["runs"]["q2"]["run_id"] == q2_run
    assert third["sessions"][-1]["retry_failed"] is True and len(third["sessions"]) == 3
    summary = (tmp_path / "summary.md").read_text(encoding="utf-8")
    assert "| q1 | technical | completed |" in summary and "Problems per run\n\nNone." in summary


def test_cli_retry_failed_needs_an_existing_eval_dir(tmp_path: Path) -> None:
    assert cli.main(["eval", "run", "--retry-failed"]) == 2
    assert cli.main(["eval", "run", "--retry-failed", "--dir", str(tmp_path / "missing")]) == 2


def test_resume_marks_orphaned_run_failed(h: Harness, tmp_path: Path, db_url: str) -> None:
    started = datetime.now(UTC) - timedelta(minutes=1)
    with h.session() as s:
        orphan = create_run(s, QUESTIONS[1].question)
        update_run_status(s, orphan.id, RunStatus.RUNNING)
        s.commit()
        orphan_id = orphan.id
    save_results(
        tmp_path,
        {
            "created_at": "x",
            "settings": {},
            "sessions": [],
            "runs": {
                "q2": {
                    "state": "started",
                    "question": QUESTIONS[1].question,
                    "started_at": started.isoformat(),
                }
            },
        },
    )
    logs: list[str] = []
    run_eval(
        tmp_path,
        QUESTIONS,
        h.settings,
        eval_harness(h),
        h.factory,
        ids=["q2"],
        check_connection=lambda: connect(db_url),
        power=lambda: PowerState(True, 90),
        log=logs.append,
    )
    with h.session() as s:
        orphan_row = s.get(ResearchRun, orphan_id)
        assert (
            orphan_row is not None
            and orphan_row.status is RunStatus.FAILED
            and orphan_row.error == ORPHAN_REASON
        )
    results = load_results(tmp_path)
    assert results is not None and results["runs"]["q2"]["state"] == "finished"
    assert results["runs"]["q2"]["run_id"] != str(orphan_id)
    assert "resume q2: marked 1 orphaned run(s) failed" in logs
    assert json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))["runs"]["q2"]["power_end"] == {
        "on_ac": True,
        "battery_percent": 90,
    }


def test_rfc_sources_share_a_domain() -> None:
    from research_assistant.evals.metrics import domain

    assert domain(RFC2606) == "rfc-editor.org" == domain("https://www.rfc-editor.org/x")
