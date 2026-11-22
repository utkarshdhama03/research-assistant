"""Command line: `ra research`, `ra show`, `ra worker`, `ra serve`, `ra healthcheck`, `ra eval ...`."""

import argparse
import logging
import sys
import time
import uuid
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from research_assistant.config import get_settings
from research_assistant.db.models import RunStatus
from research_assistant.db.repositories.claims import list_claims
from research_assistant.db.repositories.events import list_events
from research_assistant.db.repositories.reports import latest_report
from research_assistant.db.repositories.runs import get_run
from research_assistant.db.session import default_session_factory, session_scope
from research_assistant.deps import production_deps
from research_assistant.graph.runner import run_research
from research_assistant.logs import configure_logging
from research_assistant.tools.embed import Embedder

log = logging.getLogger("research_assistant.cli")

REPORTS_DIR = Path("data") / "reports"
_PROGRESS_EVENTS = {
    "node_start",
    "node_end",
    "node_error",
    "llm_call",
    "fetch_failed",
    "extract_skipped",
    "source_duplicate",
    "select_fallback",
    "search_failed",
    "citation_error",
    "section_failed",
    "claim_verified",
    "quote_not_found",
    "backfill",
    "gap_planned",
    "gap_queries",
    "gap_not_run",
    "claims_deduplicated",
    "uncited_sentences_deleted",
    "trailing_text_removed",
    "speculation_replaced",
    "host_blocked",
    "queries_deduplicated",
    "problem",
    "embedder_warm_up",
    "run_finished",
}


def _format_event(node: str, event: str, payload: dict[str, Any]) -> str:
    if event == "llm_call":
        detail = (
            f"{payload.get('prompt_name')} {payload.get('outcome')} "
            f"tokens={payload.get('prompt_tokens')}+{payload.get('eval_tokens')} "
            f"{payload.get('total_duration_s', 0):.1f}s repairs={payload.get('repairs')}"
        )
    elif event in ("fetch_failed", "extract_skipped"):
        detail = f"{payload.get('reason')} {payload.get('url')}"
    elif event == "problem":
        detail = str(payload.get("message"))
    elif event == "claim_verified":
        detail = (
            f"claim {payload.get('ordinal')}: {payload.get('verdict')} (quote {payload.get('quote_match')})"
        )
    else:
        detail = " ".join(f"{k}={v}" for k, v in payload.items() if k not in ("question",))
    return f"{node:<9} {event:<16} {detail}"


def cmd_research(question: str, deps_factory: Callable[..., Any] = production_deps) -> int:
    started = time.monotonic()

    def progress(node: str, event: str, payload: dict[str, Any]) -> None:
        if event in _PROGRESS_EVENTS:
            print(f"[{time.monotonic() - started:7.1f}s] {_format_event(node, event, payload)}", flush=True)

    with deps_factory(progress) as deps:
        outcome = run_research(question, deps)
    print(f"\nrun {outcome.run_id}: {outcome.status}" + (f" - {outcome.reason}" if outcome.reason else ""))
    if outcome.report_markdown is not None:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        path = REPORTS_DIR / f"{outcome.run_id}.md"
        path.write_text(outcome.report_markdown, encoding="utf-8", newline="\n")
        print(f"report saved to the database and to {path}")
    return 1 if outcome.status is RunStatus.FAILED else 0


def cmd_show(run_id: uuid.UUID, session_factory: Callable[[], Session] | None = None) -> int:
    with session_scope(session_factory) as session:
        run = get_run(session, run_id)
        if run is None:
            print(f"run {run_id} not found", file=sys.stderr)
            return 2
        events = list_events(session, run_id)
        report = latest_report(session, run_id)
        print(f"run:      {run.id}")
        print(f"question: {run.question}")
        print(f"status:   {run.status}")
        print(f"reason:   {run.error or '-'}")
        print(f"created:  {run.created_at:%Y-%m-%d %H:%M:%S}   finished: {run.finished_at or '-'}")
        print("\nnode timings:")
        for e in events:
            if e.event in ("node_end", "node_error"):
                print(f"  {e.node:<9} {e.event:<10} {e.payload.get('duration_s', 0):8.1f}s")
            elif e.event == "run_finished":
                print(f"  {'total':<9} {'':<10} {e.payload.get('duration_s', 0):8.1f}s")
        llm_calls = [e for e in events if e.event == "llm_call"]
        tokens = sum(
            (e.payload.get("prompt_tokens") or 0) + e.payload.get("eval_tokens", 0) for e in llm_calls
        )
        print(f"  llm calls: {len(llm_calls)}, tokens: {tokens}")
        claims = list_claims(session, run_id)
        if claims:
            print("\nclaims (ordinal, verdict, text):")
            for c in claims:
                print(f"  {c.ordinal:>3}  {str(c.verdict or 'unverified'):<12} {c.text}")
        print("\n" + (report.markdown if report is not None else "(no report)"))
    return 0


EVALS_DIR = Path("data") / "evals"


def cmd_eval_run(
    ids: list[str] | None,
    eval_dir: Path | None,
    retry_failed: bool = False,
    with_memory: bool = False,
    record: bool = False,
    replay: Path | None = None,
) -> int:
    from datetime import datetime

    from research_assistant.evals.questions import load_questions
    from research_assistant.evals.runner import RESULTS, run_eval

    if retry_failed and (eval_dir is None or not (eval_dir / RESULTS).exists()):
        print(f"--retry-failed needs --dir pointing at an existing eval directory (with {RESULTS})")
        return 2
    from research_assistant.evals.replay import SEARCH_CACHE

    if replay is not None and not (replay / SEARCH_CACHE).exists():
        print(f"--replay needs a directory recorded with --record (no {replay / SEARCH_CACHE})")
        return 2
    target = eval_dir or EVALS_DIR / datetime.now().strftime("%Y%m%d-%H%M%S")
    print(f"eval directory: {target}")
    results = run_eval(
        target,
        load_questions(),
        get_settings(),
        production_deps,
        default_session_factory(),
        ids=ids,
        retry_failed=retry_failed,
        with_memory=with_memory,
        search_mode="replay" if replay is not None else "record" if record else "live",
        replay_dir=replay,
    )
    failed = [q for q, e in results["runs"].items() if e.get("status") == str(RunStatus.FAILED)]
    print(f"summary: {target / 'summary.md'}")
    return 1 if failed else 0


def _eval_run_ids(target: str) -> list[uuid.UUID]:
    path = Path(target)
    if path.is_dir():
        from research_assistant.evals.labels import finished_runs

        return [r.run_id for r in finished_runs(path)]
    return [uuid.UUID(target)]


def cmd_eval_check(target: str) -> int:
    from research_assistant.evals.citation_check import main as check_main

    return check_main([str(r) for r in _eval_run_ids(target)])


def cmd_eval_export_labels(eval_dir: Path, seed: int) -> int:
    from research_assistant.evals.labels import export_labels

    embedder = Embedder(get_settings())
    try:
        manifest = export_labels(
            eval_dir, default_session_factory(), embedder.embed_for_clustering, seed=seed
        )
    finally:
        embedder.close()
    for name, path in manifest["paths"].items():
        print(f"{name}: {path}")
    print(
        f"claims {manifest['claims']} (available {manifest['claims_available']}); "
        f"sentences {manifest['sentences']} of {manifest['sentences_available']}; "
        f"dedupe pairs {manifest['dedupe_pairs']} of {manifest['dedupe_pairs_available']}"
    )
    return 0


def cmd_eval_compare(baseline_dir: Path, new_dir: Path) -> int:
    from research_assistant.evals.compare import compare

    print(compare(baseline_dir, new_dir, default_session_factory()))
    return 0


def cmd_eval_score_labels(eval_dir: Path) -> int:
    from research_assistant.evals.scoring import score_labels

    score_labels(eval_dir)
    print((eval_dir / "scores.md").read_text(encoding="utf-8"))
    return 0


def cmd_worker() -> int:
    import signal
    from types import FrameType

    from research_assistant.graph.checkpoint import postgres_checkpointer
    from research_assistant.worker import Worker

    configure_logging()
    worker_log = logging.getLogger("research_assistant.worker")
    settings = get_settings()
    worker = Worker(
        settings,
        default_session_factory(),
        lambda stop_requested: production_deps(None, stop_requested),
        log_line=worker_log.info,
    )

    def on_signal(_signum: int, _frame: FrameType | None) -> None:
        if worker.stopping():
            raise KeyboardInterrupt  # second signal: stop now
        worker_log.info("stopping: no new runs; the current run stops at its next check (again: now)")
        worker.request_stop()

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)  # docker stop; as PID 1 the default action would be ignored
    if hasattr(signal, "SIGBREAK"):  # Windows Ctrl+Break, and CTRL_BREAK_EVENT from another process
        signal.signal(signal.SIGBREAK, on_signal)
    return worker.serve(lambda: postgres_checkpointer(settings.database_url))


def cmd_serve() -> int:
    from research_assistant.api.app import bind_error, serve

    configure_logging()
    settings = get_settings()
    error = bind_error(settings)
    if error is not None:
        log.error(error)
        return 2
    auth = "on" if settings.auth_enabled else "off (local development)"
    ui = settings.ui_dist_dir or "not served (UI_DIST_DIR unset)"
    log.info("serving on http://%s:%s; login %s; UI %s", settings.api_host, settings.api_port, auth, ui)
    serve(settings, default_session_factory())
    return 0


def cmd_healthcheck(target: str) -> int:
    """Container healthchecks. api: /health answers (200 or 503). worker: heartbeat fresher than
    WORKER_STALE_S. Exit 0 healthy, 1 not."""
    settings = get_settings()
    if target == "api":
        import urllib.error
        import urllib.request

        request = urllib.request.Request(
            f"http://127.0.0.1:{settings.api_port}/health",
            headers={"Host": f"127.0.0.1:{settings.api_port}"},
        )
        try:
            with urllib.request.urlopen(request, timeout=5) as response:  # noqa: S310 (fixed local URL)
                status = response.status
        except urllib.error.HTTPError as e:
            status = e.code
        except OSError as e:
            print(f"api unreachable: {type(e).__name__}")
            return 1
        print(f"api answered {status}")
        return 0 if status in (200, 503) else 1
    from research_assistant.db.repositories.queue import latest_heartbeat

    try:
        with session_scope() as session:
            beat = latest_heartbeat(session)
    except Exception as e:
        print(f"database unreachable: {type(e).__name__}")
        return 1
    if beat is None:
        print("no worker heartbeat")
        return 1
    age = beat[1]
    print(f"last heartbeat {age:.0f} s ago")
    return 0 if age <= settings.worker_stale_s else 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ra", description="Local research assistant")
    commands = parser.add_subparsers(dest="command", required=True)
    research = commands.add_parser("research", help="research a question and write a cited report")
    research.add_argument("question")
    show = commands.add_parser("show", help="show a run's status, timings and report")
    show.add_argument("run_id", type=uuid.UUID)
    commands.add_parser("worker", help="execute queued runs, one at a time (Ctrl+C to stop)")
    commands.add_parser("serve", help="HTTP API (and the built UI) on API_HOST:API_PORT")
    healthcheck = commands.add_parser("healthcheck", help="exit 0 if the api or worker is healthy")
    healthcheck.add_argument("target", choices=["api", "worker"])
    evaluate = commands.add_parser("eval", help="evaluation harness")
    eval_commands = evaluate.add_subparsers(dest="eval_command", required=True)
    run = eval_commands.add_parser("run", help="run the question set (resumable with --dir)")
    run.add_argument("--ids", nargs="+", help="question ids to run (default: all)")
    run.add_argument("--dir", type=Path, help="eval directory to create or resume")
    run.add_argument(
        "--retry-failed",
        action="store_true",
        help="with --dir: also rerun questions whose last result is failed, replacing their entry",
    )
    run.add_argument(
        "--with-memory",
        action="store_true",
        help="let runs reuse sources from earlier runs (memory is off in evals by default)",
    )
    mode = run.add_mutually_exclusive_group()
    mode.add_argument(
        "--record", action="store_true", help="store every search's results in search_cache.jsonl"
    )
    mode.add_argument(
        "--replay", type=Path, metavar="DIR", help="serve searches and fetches from a recorded eval directory"
    )
    check = eval_commands.add_parser("check", help="independent citation check of a run or eval directory")
    check.add_argument("target", help="run id or eval directory")
    export = eval_commands.add_parser(
        "export-labels", help="write claims.csv, sentences.csv, dedupe_pairs.csv"
    )
    export.add_argument("eval_dir", type=Path)
    export.add_argument("--seed", type=int, default=20261118)
    score = eval_commands.add_parser("score-labels", help="score the filled label CSVs")
    score.add_argument("eval_dir", type=Path)
    comp = eval_commands.add_parser("compare", help="compare two eval directories (writes compare.md)")
    comp.add_argument("baseline_dir", type=Path)
    comp.add_argument("new_dir", type=Path)
    args = parser.parse_args(argv)
    if args.command == "research":
        return cmd_research(args.question)
    if args.command == "show":
        return cmd_show(args.run_id)
    if args.command == "worker":
        return cmd_worker()
    if args.command == "serve":
        return cmd_serve()
    if args.command == "healthcheck":
        return cmd_healthcheck(args.target)
    if args.eval_command == "run":
        return cmd_eval_run(args.ids, args.dir, args.retry_failed, args.with_memory, args.record, args.replay)
    if args.eval_command == "check":
        return cmd_eval_check(args.target)
    if args.eval_command == "export-labels":
        return cmd_eval_export_labels(args.eval_dir, args.seed)
    if args.eval_command == "compare":
        return cmd_eval_compare(args.baseline_dir, args.new_dir)
    return cmd_eval_score_labels(args.eval_dir)


if __name__ == "__main__":
    raise SystemExit(main())
