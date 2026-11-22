"""Eval record/replay: recorded search and fetch inputs, served back without network."""

import json
import socket
import uuid
from pathlib import Path
from typing import Any

import httpx
import pytest

from research_assistant import cli
from research_assistant.evals.citation_check import connect
from research_assistant.evals.metrics import run_metrics
from research_assistant.evals.power import PowerState
from research_assistant.evals.questions import EvalQuestion
from research_assistant.evals.replay import SEARCH_CACHE
from research_assistant.evals.runner import load_results, run_eval
from research_assistant.tools.fetch import FetchError, FetchErrorReason
from tests.integration.conftest import TrackingSessionFactory
from tests.integration.graph_fakes import OKAPI, SECRETS
from tests.integration.test_evals import eval_harness
from tests.integration.test_graph import QUESTION, Harness, events_of, pick_first

QUESTIONS = [EvalQuestion("q1", "technical", QUESTION)]


@pytest.fixture
def h(committing_factory: TrackingSessionFactory) -> Harness:
    h = Harness(committing_factory)
    h.search.results["okapi animal"] = [OKAPI, SECRETS]  # SECRETS is listed but never fetched (pick_first)
    h.model.select = pick_first
    return h


def evaluate(h: Harness, eval_dir: Path, db_url: str, **kwargs: Any) -> dict[str, Any]:
    run_eval(
        eval_dir,
        QUESTIONS,
        h.settings,
        eval_harness(h),
        h.factory,
        check_connection=lambda: connect(db_url),
        power=lambda: PowerState(True, 90),
        log=lambda _line: None,
        **kwargs,
    )
    results = load_results(eval_dir)
    assert results is not None
    return results


def prompts(h: Harness, name: str, start: int) -> list[str]:
    return [c.prompt.user for c in h.model.prompts(name)[start:]]


def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("replay touched the network")

    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)
    monkeypatch.setattr(httpx.Client, "send", refuse)


def test_record_then_replay_gives_identical_inputs_without_network(
    h: Harness, tmp_path: Path, db_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorded = evaluate(h, tmp_path / "rec", db_url, search_mode="record")
    lines = [
        json.loads(x) for x in (tmp_path / "rec" / SEARCH_CACHE).read_text(encoding="utf-8").splitlines()
    ]
    assert [x["query"] for x in lines] == h.search.queries
    assert all(x["provider"] == "fake" and x["calls"][0]["role"] == "only" for x in lines)
    select_recorded, analyze_recorded = prompts(h, "select", 0), prompts(h, "analyze", 0)
    searched, fetched = list(h.search.queries), list(h.fetcher.calls)

    no_network(monkeypatch)
    n_select, n_analyze = len(select_recorded), len(analyze_recorded)
    replayed = evaluate(h, tmp_path / "rep", db_url, search_mode="replay", replay_dir=tmp_path / "rec")
    assert h.search.queries == searched and h.fetcher.calls == fetched  # neither fake was used again
    assert prompts(h, "select", n_select) == select_recorded  # identical search inputs
    assert prompts(h, "analyze", n_analyze) == analyze_recorded  # identical page texts and chunks
    assert (recorded["search_mode"], recorded["replay_source"]) == ("record", None)
    assert (replayed["search_mode"], replayed["replay_source"]) == ("replay", str(tmp_path / "rec"))
    assert (
        replayed["sessions"][-1]["search_mode"] == "replay"
        and replayed["runs"]["q1"]["search_mode"] == "replay"
    )
    run_id = uuid.UUID(replayed["runs"]["q1"]["run_id"])
    assert all(d["provider"] == "replay" for d in events_of(h, run_id, "search_done"))
    assert replayed["runs"]["q1"]["status"] == recorded["runs"]["q1"]["status"] == "completed"


def test_replay_misses_are_logged(
    h: Harness, tmp_path: Path, db_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    evaluate(h, tmp_path / "rec", db_url, search_mode="record")
    no_network(monkeypatch)
    plan = json.loads(json.dumps(h.model.plan))
    plan["sub_questions"][2]["search_queries"] = ["robots exclusion", "a query nobody recorded"]
    h.model.plan = plan

    def pick_second_if_any(call: Any) -> str:  # SECRETS was listed but never fetched in the recording
        labels = [
            line.split("=")[1].rstrip(">") for line in call.prompt.user.split("\n") if "label=R" in line
        ]
        return json.dumps({"selected": [labels[1] if len(labels) > 1 else labels[0]]})

    h.model.select = pick_second_if_any
    replayed = evaluate(h, tmp_path / "rep", db_url, search_mode="replay", replay_dir=tmp_path / "rec")
    run_id = uuid.UUID(replayed["runs"]["q1"]["run_id"])
    misses = events_of(h, run_id, "replay_miss")
    assert misses == [{"kind": "search", "query": "a query nobody recorded"}]
    failed = events_of(h, run_id, "fetch_failed")
    assert [(f["url"], f["reason"]) for f in failed] == [(SECRETS, FetchErrorReason.REPLAY_MISS.value)]
    metrics = replayed["runs"]["q1"]["metrics"]
    assert (metrics["replay_misses_search"], metrics["replay_misses_fetch"]) == (1, 1)
    with h.session() as s:
        assert run_metrics(s, run_id)["searches"] == len(events_of(h, run_id, "search_done"))


def test_recorded_fetch_failures_are_replayed(
    h: Harness, tmp_path: Path, db_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    blocked = "https://blocked.example.org/page"
    h.fetcher.pages[blocked] = FetchError(
        FetchErrorReason.HTTP_ERROR, "HTTP 403", url=blocked, status_code=403
    )
    h.search.results["okapi animal"] = [blocked, OKAPI]
    evaluate(h, tmp_path / "rec", db_url, search_mode="record")
    no_network(monkeypatch)
    replayed = evaluate(h, tmp_path / "rep", db_url, search_mode="replay", replay_dir=tmp_path / "rec")
    run_id = uuid.UUID(replayed["runs"]["q1"]["run_id"])
    [failed] = events_of(h, run_id, "fetch_failed")
    assert (failed["url"], failed["reason"], failed["status_code"]) == (blocked, "http_error", 403)
    assert [b["host"] for b in events_of(h, run_id, "host_blocked")] == ["blocked.example.org"]


def test_cli_replay_needs_a_recorded_directory(tmp_path: Path) -> None:
    assert cli.main(["eval", "run", "--replay", str(tmp_path)]) == 2
    with pytest.raises(SystemExit):
        cli.main(["eval", "run", "--record", "--replay", str(tmp_path)])  # mutually exclusive
