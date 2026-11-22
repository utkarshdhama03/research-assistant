"""`ra eval compare <baseline_dir> <new_dir>`: aggregate metrics side by side, per-question changes, and
the false-premise question's report from the new run. Written to <new_dir>/compare.md."""

import json
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from research_assistant.db.repositories.reports import latest_report
from research_assistant.db.session import session_scope
from research_assistant.evals.metrics import aggregate
from research_assistant.evals.runner import RESULTS, timing_ok

COMPARE = "compare.md"
FALSE_PREMISE_CATEGORY = "false-premise"
Number = int | float


def _finished(results: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {q: e for q, e in sorted(results["runs"].items()) if e.get("state") == "finished"}


def _aggregate(results: dict[str, Any]) -> dict[str, Any]:
    runs = list(_finished(results).values())
    agg = aggregate([e["metrics"] for e in runs], [timing_ok(e) for e in runs])
    checks = sum((e.get("citation_check") or {}).get("checks", 0) for e in runs)
    failures = sum(len((e.get("citation_check") or {}).get("failures", [])) for e in runs)
    agg["citation_checks"] = checks
    agg["citation_failures"] = failures
    agg["citation_validity"] = round(1 - failures / checks, 4) if checks else None
    agg["sources_failed"] = sum(agg["sources_failed_by_reason"].values())
    for verdict in ("supported", "partial", "unsupported", "contradicted"):
        agg[f"claims_{verdict}"] = agg["claims_by_verdict"].get(verdict, 0)
    methods = agg.get("match_methods")
    for method in ("exact", "whitespace", "list", "fuzzy"):
        agg[f"match_{method}"] = methods.get(method, 0) if methods is not None else None
    searches, zero = agg.get("searches"), agg.get("zero_result_searches")
    agg["zero_result_rate"] = round(zero / searches, 4) if searches and zero is not None else None
    providers = agg.get("search_calls_by_provider") or {}
    agg["searxng_zero_results"] = providers.get("searxng", {}).get("zero_results") if providers else None
    agg["completed"] = agg["status"].get("completed", 0)
    agg["partial_runs"] = agg["status"].get("partial", 0)
    agg["failed_runs"] = agg["status"].get("failed", 0)
    return agg


# (label, aggregate key, kind): kind "pct" shows a share, "num" a count or duration
METRICS: list[tuple[str, str, str]] = [
    ("runs", "runs", "num"),
    ("completed", "completed", "num"),
    ("partial", "partial_runs", "num"),
    ("failed", "failed_runs", "num"),
    ("timed runs", "timed_runs", "num"),
    ("total time, timed runs (s)", "total_duration_s", "num"),
    ("median time per run (s)", "median_duration_s", "num"),
    ("LLM calls", "llm_calls", "num"),
    ("LLM calls failed", "llm_failed_calls", "num"),
    ("tokens", "tokens", "num"),
    ("sources fetched", "sources_fetched", "num"),
    ("sources failed", "sources_failed", "num"),
    ("blocked hosts", "blocked_hosts", "num"),
    ("backfills", "backfills", "num"),
    ("median distinct cited domains", "median_distinct_cited_domains", "num"),
    ("claims", "claims", "num"),
    ("claims supported", "claims_supported", "num"),
    ("claims partial", "claims_partial", "num"),
    ("claims unsupported", "claims_unsupported", "num"),
    ("claims contradicted", "claims_contradicted", "num"),
    ("supported share", "supported_share", "pct"),
    ("quote_not_found rate", "quote_not_found_rate", "pct"),
    ("number_mismatch (made unsupported)", "number_mismatch", "num"),
    ("quotes located: exact", "match_exact", "num"),
    ("quotes located: whitespace", "match_whitespace", "num"),
    ("quotes located: list", "match_list", "num"),
    ("quotes located: fuzzy", "match_fuzzy", "num"),
    ("sentences checked (write)", "sentences_checked", "num"),
    ("sentences replaced (content check)", "sentence_replaced", "num"),
    ("searches", "searches", "num"),
    ("zero-result searches (after fallback)", "zero_result_searches", "num"),
    ("zero-result rate", "zero_result_rate", "pct"),
    ("fallback uses (ddgs)", "fallback_uses", "num"),
    ("SearXNG calls with zero results", "searxng_zero_results", "num"),
    ("replay misses: search", "replay_misses_search", "num"),
    ("replay misses: fetch", "replay_misses_fetch", "num"),
    ("uncited sentences deleted", "uncited_sentences_deleted", "num"),
    ("trailing text removed", "trailing_text_removed", "num"),
    ("speculative sentences replaced", "speculation_replaced", "num"),
    ("claims dropped by dedupe", "dedupe_dropped", "num"),
    ("gap rounds", "gap_rounds", "num"),
    ("sub-questions with zero supported claims", "zero_supported_sub_questions", "num"),
    ("citation checks", "citation_checks", "num"),
    ("citation failures", "citation_failures", "num"),
    ("citation validity", "citation_validity", "pct"),
]


def _show(value: Number | None, kind: str) -> str:
    if value is None:
        return "–"
    return f"{value:.1%}" if kind == "pct" else f"{value:g}"


def _diff(old: Number | None, new: Number | None, kind: str) -> str:
    if old is None or new is None:
        return "–"
    delta = new - old
    if kind == "pct":
        return f"{delta * 100:+.1f} pp"
    return f"{delta:+g}"


def _claims(entry: dict[str, Any]) -> str:
    v = entry["metrics"]["claims_by_verdict"]
    return f"{v['supported']}/{v['partial']}/{v['unsupported']}/{v['contradicted']}"


def _validity(entry: dict[str, Any]) -> str:
    cc = entry.get("citation_check") or {}
    return _show(cc.get("validity"), "pct")


def compare(baseline_dir: Path, new_dir: Path, session_factory: Callable[[], Session] | None = None) -> str:
    base = json.loads((baseline_dir / RESULTS).read_text(encoding="utf-8"))
    new = json.loads((new_dir / RESULTS).read_text(encoding="utf-8"))
    a, b = _aggregate(base), _aggregate(new)
    lines = [
        "# Evaluation comparison",
        "",
        f"Baseline: `{baseline_dir}` (created {base['created_at']}).",
        f"New: `{new_dir}` (created {new['created_at']}).",
        f"Search inputs: baseline {base.get('search_mode', 'live')}, new {new.get('search_mode', 'live')}"
        + (f" (replayed from `{new['replay_source']}`)" if new.get("replay_source") else "")
        + ". Live evals see whatever the search engines return that day:"
        + " different inputs, not only different code.",
        "",
        "| metric | baseline | new | difference |",
        "|---|---|---|---|",
    ]
    for label, key, kind in METRICS:
        old, cur = a.get(key), b.get(key)
        lines.append(f"| {label} | {_show(old, kind)} | {_show(cur, kind)} | {_diff(old, cur, kind)} |")

    lines += [
        "",
        "## Per question",
        "",
        "| id | status baseline → new | claims s/p/u/c baseline → new | citation validity | time (s) |",
        "|---|---|---|---|---|",
    ]
    base_runs, new_runs = _finished(base), _finished(new)
    changed = []
    for qid in sorted(set(base_runs) | set(new_runs)):
        old, cur = base_runs.get(qid), new_runs.get(qid)
        status_old = old["status"] if old else "–"
        status_new = cur["status"] if cur else "–"
        if status_old != status_new:
            changed.append(f"{qid}: {status_old} → {status_new}")
        lines.append(
            f"| {qid} | {status_old} → {status_new}{' **changed**' if status_old != status_new else ''} | "
            f"{_claims(old) if old else '–'} → {_claims(cur) if cur else '–'} | "
            f"{_validity(old) if old else '–'} → {_validity(cur) if cur else '–'} | "
            f"{old['wall_s'] if old else '–'} → {cur['wall_s'] if cur else '–'} |"
        )
    lines += ["", "Status changes: " + ("; ".join(changed) if changed else "none") + "."]

    premise = [(q, e) for q, e in new_runs.items() if e.get("category") == FALSE_PREMISE_CATEGORY]
    for qid, entry in premise:
        lines += ["", f"## {qid} (new run {entry['run_id']}): report", ""]
        with session_scope(session_factory) as session:
            report = latest_report(session, uuid.UUID(entry["run_id"]))
            markdown = report.markdown if report is not None else "(no report)"
        lines += ["````markdown", markdown.rstrip(), "````"]
    text = "\n".join(lines) + "\n"
    (new_dir / COMPARE).write_text(text, encoding="utf-8")
    return text
