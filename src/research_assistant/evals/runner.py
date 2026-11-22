"""`ra eval run`: run the question set sequentially; resumable; writes results.json and summary.md."""

import contextlib
import dataclasses
import hashlib
import json
import os
import subprocess
import time
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import psycopg
from sqlalchemy import select
from sqlalchemy.orm import Session

from research_assistant.config import Settings
from research_assistant.db.models import ResearchRun, RunStatus
from research_assistant.db.repositories.events import append_event
from research_assistant.db.repositories.runs import update_run_status
from research_assistant.evals.citation_check import check_run, connect
from research_assistant.evals.metrics import aggregate, run_metrics
from research_assistant.evals.power import PowerState, keep_awake, power_state
from research_assistant.evals.questions import QUESTIONS_FILE, EvalQuestion
from research_assistant.evals.replay import (
    SEARCH_CACHE,
    ReplayFetcher,
    ReplaySearch,
    SearchRecorder,
    load_search_cache,
    recorded_run_ids,
)
from research_assistant.graph.context import Deps, EventSink
from research_assistant.graph.runner import run_research

RESULTS = "results.json"
SUMMARY = "summary.md"
EXCLUDED_SETTINGS = frozenset({"contact_url", "database_url", "test_database_url"})
ORPHAN_REASON = "aborted: the eval process stopped during this run; the question was restarted on resume"

DepsFactory = Callable[[EventSink | None], AbstractContextManager[Deps]]


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def git_info(cwd: Path | None = None) -> dict[str, Any]:
    def git(*args: str) -> str:
        out = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=False)
        return out.stdout.strip() if out.returncode == 0 else ""

    commit = git("rev-parse", "HEAD")
    dirty = bool(git("status", "--porcelain", "--untracked-files=no"))
    return {"commit": commit or None, "dirty": dirty}


def settings_snapshot(settings: Settings) -> dict[str, Any]:
    return settings.model_dump(mode="json", exclude=set(EXCLUDED_SETTINGS))


def load_results(eval_dir: Path) -> dict[str, Any] | None:
    path = eval_dir / RESULTS
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def save_results(eval_dir: Path, results: dict[str, Any]) -> None:
    tmp = eval_dir / (RESULTS + ".tmp")
    tmp.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, eval_dir / RESULTS)  # atomic on the same volume


def timing_ok(entry: dict[str, Any]) -> bool:
    """False if the run started or ended on battery (unknown power state counts as fine)."""
    return (entry.get("power_start") or {}).get("on_ac") is not False and (entry.get("power_end") or {}).get(
        "on_ac"
    ) is not False


def _abort_orphan(
    session_factory: Callable[[], Session], q: EvalQuestion, entry: dict[str, Any]
) -> list[str]:
    """Mark runs of this question left in 'running' by a stopped eval process as failed."""
    started = datetime.fromisoformat(entry["started_at"])
    with session_factory() as session:
        orphans = list(
            session.scalars(
                select(ResearchRun).where(
                    ResearchRun.question == q.question,
                    ResearchRun.status == RunStatus.RUNNING,
                    ResearchRun.created_at >= started,
                )
            )
        )
        for run in orphans:
            update_run_status(session, run.id, RunStatus.FAILED, error=ORPHAN_REASON)
            append_event(session, run.id, "run", "run_aborted", {"reason": ORPHAN_REASON})
        session.commit()
        return [str(r.id) for r in orphans]


def run_eval(
    eval_dir: Path,
    questions: Sequence[EvalQuestion],
    settings: Settings,
    deps_factory: DepsFactory,
    session_factory: Callable[[], Session],
    *,
    ids: Sequence[str] | None = None,
    retry_failed: bool = False,
    with_memory: bool = False,
    search_mode: str = "live",  # live | record | replay
    replay_dir: Path | None = None,
    check_connection: Callable[[], psycopg.Connection[Any]] = connect,
    power: Callable[[], PowerState] = power_state,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    known = {q.id for q in questions}
    unknown = [i for i in ids or [] if i not in known]
    if unknown:
        raise ValueError(f"unknown question ids: {unknown}")
    eval_dir.mkdir(parents=True, exist_ok=True)
    results = load_results(eval_dir) or {
        "created_at": _now(),
        "questions_file_sha256": hashlib.sha256(QUESTIONS_FILE.read_bytes()).hexdigest(),
        "settings": settings_snapshot(settings),
        "sessions": [],
        "search_mode": search_mode,
        "replay_source": str(replay_dir) if replay_dir else None,
        "runs": {},
    }
    results["sessions"].append(
        {
            "started_at": _now(),
            "git": git_info(),
            "ids": list(ids or []),
            "retry_failed": retry_failed,
            "memory_enabled": with_memory,
            "search_mode": search_mode,
            "replay_source": str(replay_dir) if replay_dir else None,
        }
    )
    save_results(eval_dir, results)
    selected = [q for q in questions if not ids or q.id in ids]

    if search_mode not in ("live", "record", "replay"):
        raise ValueError(f"unknown search mode {search_mode!r}")
    recorder = SearchRecorder(eval_dir / SEARCH_CACHE) if search_mode == "record" else None
    replay: tuple[ReplaySearch, ReplayFetcher] | None = None
    if search_mode == "replay":
        if replay_dir is None:
            raise ValueError("replay needs the recorded eval directory")
        replay = (
            ReplaySearch(load_search_cache(replay_dir / SEARCH_CACHE)),
            ReplayFetcher(session_factory, recorded_run_ids(replay_dir)),
        )

    def inputs(deps: Deps) -> Deps:
        """Memory off unless asked (a run must not depend on what earlier runs stored); record or replay."""
        update: dict[str, Any] = {"memory_enabled": with_memory}
        if replay is not None:
            update["search_min_interval_s"] = 0.0  # nothing goes to the network
            return dataclasses.replace(
                deps, settings=deps.settings.model_copy(update=update), search=replay[0], fetcher=replay[1]
            )
        return dataclasses.replace(
            deps, settings=deps.settings.model_copy(update=update), search_recorder=recorder
        )

    with keep_awake():
        for q in selected:
            entry = results["runs"].get(q.id)
            replaced: list[dict[str, Any]] = []
            if entry and entry.get("state") == "finished":
                if not (retry_failed and entry.get("status") == str(RunStatus.FAILED)):
                    log(f"skip {q.id}: already finished ({entry.get('status')})")
                    continue
                replaced = [
                    *entry.get("replaced_runs", []),
                    {k: entry.get(k) for k in ("run_id", "status", "reason", "started_at", "finished_at")},
                ]
                log(f"retry {q.id}: previous run {entry.get('run_id')} failed")
            if entry and entry.get("state") == "started":
                aborted = _abort_orphan(session_factory, q, entry)
                log(f"resume {q.id}: marked {len(aborted)} orphaned run(s) failed")
            entry = {
                "state": "started",
                "question": q.question,
                "category": q.category,
                "expectation": q.expectation,
                "started_at": _now(),
                "power_start": power().as_dict(),
                "memory_enabled": with_memory,
                "search_mode": search_mode,
            }
            if replaced:
                entry["replaced_runs"] = replaced  # earlier failed runs of this question
            results["runs"][q.id] = entry
            save_results(eval_dir, results)
            log(f"start {q.id}: {q.question}")
            t0 = time.monotonic()
            with deps_factory(None) as deps:
                # Memory is off unless asked for: a run must not depend on what earlier runs stored.
                outcome = run_research(q.question, inputs(deps))
            entry.update(
                {
                    "state": "finished",
                    "run_id": str(outcome.run_id),
                    "status": str(outcome.status),
                    "reason": outcome.reason,
                    "wall_s": round(time.monotonic() - t0, 1),
                    "finished_at": _now(),
                    "power_end": power().as_dict(),
                }
            )
            with session_factory() as session:
                entry["metrics"] = run_metrics(session, outcome.run_id)
            with contextlib.closing(check_connection()) as conn:
                entry["citation_check"] = check_run(conn, outcome.run_id).as_dict()
            save_results(eval_dir, results)
            write_summary(eval_dir, results)
            log(f"done  {q.id}: {entry['status']} in {entry['wall_s']} s")
    write_summary(eval_dir, results)
    return results


# -- summary.md ------------------------------------------------------------------------------


def _pct(value: float | None) -> str:
    return "–" if value is None else f"{value:.1%}"


def render_summary(results: dict[str, Any]) -> str:
    finished = {qid: e for qid, e in results["runs"].items() if e.get("state") == "finished"}
    s = results["settings"]
    sessions = results.get("sessions", [])
    commits = sorted(
        {f"{x['git']['commit'] or '?'}{' (dirty)' if x['git']['dirty'] else ''}" for x in sessions}
    )
    lines = [
        "# Evaluation summary",
        "",
        f"Created {results['created_at']}; sessions: {len(sessions)}; code: {', '.join(commits)}.",
        f"Search inputs: {results.get('search_mode', 'live')}"
        + (f" from {results['replay_source']}" if results.get("replay_source") else "")
        + ".",
        f"Model {s.get('chat_model')}, num_ctx {s.get('ollama_num_ctx')}, embed {s.get('embed_model')}; "
        f"max_sub_questions {s.get('max_sub_questions')}, "
        f"max_sources_per_run {s.get('max_sources_per_run')}, "
        f"min_supported_claims {s.get('min_supported_claims')}, claim_dedupe_threshold "
        f"{s.get('claim_dedupe_threshold')}.",
        f"Finished runs: {len(finished)} of {len(results['runs'])} started.",
        "",
        "## Per question",
        "",
        "| id | category | status | wall time | LLM calls | tokens | fetched / failed | cited domains | "
        "claims s/p/u/c | quote not found | gap rounds | SQ without support | citation validity |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    battery = []
    for qid, e in sorted(finished.items()):
        m = e["metrics"]
        v = m["claims_by_verdict"]
        ok = timing_ok(e)
        if not ok:
            battery.append(qid)
        wall = f"{e['wall_s']:.0f} s" + ("" if ok else " (battery)")
        cc = e.get("citation_check") or {}
        validity = cc.get("validity")
        lines.append(
            f"| {qid} | {e['category']} | {e['status']} | {wall} | {m['llm_calls']} | "
            f"{m['prompt_tokens'] + m['output_tokens']} | {m['sources_fetched']} / "
            f"{sum(m['sources_failed_by_reason'].values())} | {m['distinct_cited_domains']} | "
            f"{v['supported']}/{v['partial']}/{v['unsupported']}/{v['contradicted']} | "
            f"{m['quote_not_found']}/{m['quote_checks']} | {m['gap_rounds']} | "
            f"{len(m['zero_supported_sub_questions'])} | {_pct(validity)} ({cc.get('checks', 0)} checks) |"
        )
    ordered = sorted(finished.items())
    agg = (
        aggregate([e["metrics"] for _, e in ordered], [timing_ok(e) for _, e in ordered]) if ordered else None
    )
    lines += ["", "## Aggregate", ""]
    if agg is None:
        lines.append("No finished runs.")
    else:
        checks = sum((e.get("citation_check") or {}).get("checks", 0) for _, e in ordered)
        fails = [f for _, e in ordered for f in (e.get("citation_check") or {}).get("failures", [])]
        nodes = ", ".join(f"{n} {t} s" for n, t in agg["median_node_seconds"].items())
        rows = [
            ("runs (status)", f"{agg['runs']} {agg['status']}"),
            ("timed runs (on AC at start and end)", f"{agg['timed_runs']}"),
            ("total time, timed runs", f"{agg['total_duration_s']} s"),
            ("median time per run, timed runs", f"{agg['median_duration_s']} s"),
            ("median time per node, timed runs", nodes or "–"),
            ("LLM calls (failed)", f"{agg['llm_calls']} ({agg['llm_failed_calls']})"),
            ("tokens", f"{agg['tokens']}"),
            ("sources fetched", f"{agg['sources_fetched']}"),
            ("sources failed by reason", f"{agg['sources_failed_by_reason']}"),
            ("blocked hosts / backfills", f"{agg['blocked_hosts']} / {agg['backfills']}"),
            ("median distinct cited domains per run", f"{agg['median_distinct_cited_domains']}"),
            ("claims by verdict", f"{agg['claims_by_verdict']}"),
            ("supported share of claims", _pct(agg["supported_share"])),
            ("quote_not_found rate (of model supported/partial)", _pct(agg["quote_not_found_rate"])),
            ("claims made unsupported: number not in quote", f"{agg['number_mismatch']}"),
            ("quotes located by method (exact/whitespace/list/fuzzy)", f"{agg['match_methods']}"),
            (
                "searches / zero results after fallback / fallback uses",
                f"{agg['searches']} / {agg['zero_result_searches']} / {agg['fallback_uses']}",
            ),
            ("search calls by provider", f"{agg['search_calls_by_provider']}"),
            (
                "replay misses (search / fetch)",
                f"{agg['replay_misses_search']} / {agg['replay_misses_fetch']}",
            ),
            (
                "sentences replaced by the content check (of checked)",
                f"{agg['sentence_replaced']} of {agg['sentences_checked']}",
            ),
            ("uncited sentences deleted", f"{agg['uncited_sentences_deleted']}"),
            ("trailing text removed", f"{agg['trailing_text_removed']}"),
            ("speculative sentences replaced", f"{agg['speculation_replaced']}"),
            ("claims dropped by dedupe", f"{agg['dedupe_dropped']}"),
            ("gap rounds", f"{agg['gap_rounds']}"),
            ("sub-questions with zero supported claims", f"{agg['zero_supported_sub_questions']}"),
            (
                "citation validity",
                f"{_pct(1 - len(fails) / checks) if checks else '–'} "
                f"({checks} checks, {len(fails)} failures)",
            ),
        ]
        lines += ["| metric | value |", "|---|---|"] + [f"| {k} | {v} |" for k, v in rows]
        if battery:
            lines += ["", f"Started or ended on battery, excluded from timing figures: {', '.join(battery)}."]
        lines += ["", "## Citation check failures", ""]
        lines += [f"- {f['run_id']} {f['kind']}: {f['detail']}" for f in fails] or ["None."]
    lines += ["", "## Problems per run", ""]
    any_problem = False
    for qid, e in sorted(finished.items()):
        if e["metrics"]["problems"] or e.get("reason"):
            any_problem = True
            lines.append(f"- {qid} ({e['status']}): {e.get('reason') or '; '.join(e['metrics']['problems'])}")
    if not any_problem:
        lines.append("None.")
    return "\n".join(lines) + "\n"


def write_summary(eval_dir: Path, results: dict[str, Any]) -> None:
    (eval_dir / SUMMARY).write_text(render_summary(results), encoding="utf-8")
