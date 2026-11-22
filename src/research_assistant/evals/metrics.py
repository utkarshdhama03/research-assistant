"""Per-run metrics from the database and run events, and their aggregation across runs."""

import statistics
import uuid
from collections import Counter, defaultdict
from collections.abc import Sequence
from typing import Any
from urllib.parse import urlsplit

from sqlalchemy import select
from sqlalchemy.orm import Session

from research_assistant.db.models import Claim, ResearchRun, Source, SourceStatus, SubQuestion, Verdict
from research_assistant.db.repositories.events import list_events
from research_assistant.db.repositories.reports import latest_report
from research_assistant.tools.fetch import FetchErrorReason

VERDICTS = ("supported", "partial", "unsupported", "contradicted")


def domain(url: str) -> str:
    """Host without a leading 'www.' (no public-suffix list: a.example.com and b.example.com differ)."""
    host = (urlsplit(url).hostname or "").lower()
    return host[4:] if host.startswith("www.") else host


def _reason(error: str | None) -> str:
    """'http_error: HTTP 403 (...)' -> 'http_error'; 'extract: too_short' -> 'too_short'."""
    if not error:
        return "unknown"
    head, _, rest = error.partition(":")
    if head == "extract":
        return rest.strip() or "extract"
    return head.strip()


MATCH_METHODS = ("exact", "whitespace", "list", "fuzzy")


def run_metrics(session: Session, run_id: uuid.UUID) -> dict[str, Any]:
    run = session.get(ResearchRun, run_id)
    if run is None:
        raise LookupError(f"run {run_id} not found")
    events = list_events(session, run_id)
    by_event: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for e in events:
        by_event[e.event].append(e.payload)

    node_seconds: dict[str, float] = defaultdict(float)
    node_runs: Counter[str] = Counter()
    for e in events:
        if e.event in ("node_end", "node_error"):
            node_seconds[e.node] += float(e.payload.get("duration_s", 0.0))
            node_runs[e.node] += 1
    finished = by_event.get("run_finished", [{}])[-1]

    calls = by_event.get("llm_call", [])
    by_prompt: dict[str, dict[str, int]] = defaultdict(lambda: {"calls": 0, "tokens": 0, "failed": 0})
    for c in calls:
        entry = by_prompt[str(c.get("prompt_name"))]
        entry["calls"] += 1
        entry["tokens"] += int(c.get("prompt_tokens") or 0) + int(c.get("eval_tokens") or 0)
        entry["failed"] += int(c.get("outcome") != "ok")

    sources = list(session.scalars(select(Source).where(Source.run_id == run_id)))
    failed_by_reason = Counter(_reason(s.error) for s in sources if s.status is SourceStatus.FAILED)
    skipped_by_reason = Counter(_reason(s.error) for s in sources if s.status is SourceStatus.SKIPPED)

    report = latest_report(session, run_id)
    cited_domains: list[str] = []
    if report is not None:
        cited_ids = [uuid.UUID(c["source_id"]) for c in report.citations]
        for sid in cited_ids:
            src = session.get(Source, sid)
            if src is not None:
                cited_domains.append(domain(src.final_url or src.url))

    claims = list(session.scalars(select(Claim).where(Claim.run_id == run_id)))
    verdicts = Counter(str(c.verdict) if c.verdict is not None else "unverified" for c in claims)
    model_found = [
        v for v in by_event.get("claim_verified", []) if v.get("model_verdict") in ("supported", "partial")
    ]
    quote_not_found = len(by_event.get("quote_not_found", []))
    # How located quotes were matched; "match_method" is logged from 2026-11-21 (older runs: quote_match).
    match_methods = Counter(
        str(v.get("match_method") or v.get("quote_match"))
        for v in by_event.get("claim_verified", [])
        if v.get("match_method") or v.get("quote_match")
    )

    sub_questions = list(session.scalars(select(SubQuestion).where(SubQuestion.run_id == run_id)))
    supported_per_sq = Counter(c.sub_question_id for c in claims if c.verdict is Verdict.SUPPORTED)
    zero_supported = sorted(sq.ordinal + 1 for sq in sub_questions if supported_per_sq[sq.id] == 0)

    return {
        "run_id": str(run_id),
        "status": str(run.status),
        "reason": run.error,
        "problems": [p.get("message") for p in by_event.get("problem", [])],
        "duration_s": finished.get("duration_s"),
        "node_seconds": {k: round(v, 3) for k, v in node_seconds.items()},
        "node_runs": dict(node_runs),
        "llm_calls": len(calls),
        "llm_failed_calls": sum(1 for c in calls if c.get("outcome") != "ok"),
        "prompt_tokens": sum(int(c.get("prompt_tokens") or 0) for c in calls),
        "output_tokens": sum(int(c.get("eval_tokens") or 0) for c in calls),
        "llm_by_prompt": dict(by_prompt),
        "sub_questions": len(sub_questions),
        "sources_fetched": sum(
            1 for s in sources if s.status is SourceStatus.FETCHED and s.reused_from_source_id is None
        ),
        "sources_reused": sum(1 for s in sources if s.reused_from_source_id is not None),
        "sources_failed_by_reason": dict(failed_by_reason),
        "sources_skipped_by_reason": dict(skipped_by_reason),
        "blocked_hosts": sorted(p["host"] for p in by_event.get("host_blocked", [])),
        "backfills": len(by_event.get("backfill", [])),
        "cited_sources": len(cited_domains),
        "cited_domains": sorted(set(cited_domains)),
        "distinct_cited_domains": len(set(cited_domains)),
        "claims": len(claims),
        "claims_by_verdict": {v: verdicts.get(v, 0) for v in (*VERDICTS, "unverified")},
        "quote_not_found": quote_not_found,
        "number_mismatch": len(by_event.get("number_mismatch", [])),
        "sentence_replaced": len(by_event.get("sentence_replaced", [])),
        **_search_health(by_event),
        "sentences_checked": sum(
            int(p.get("sentences_checked", 0))
            for p in by_event.get("node_end", [])
            if "sentences_checked" in p
        ),
        "match_methods": {m: match_methods.get(m, 0) for m in MATCH_METHODS},
        "quote_checks": len(model_found),
        "quote_not_found_rate": round(quote_not_found / len(model_found), 4) if model_found else None,
        "uncited_sentences_deleted": sum(
            int(p.get("count", 0)) for p in by_event.get("uncited_sentences_deleted", [])
        ),
        "trailing_text_removed": sum(
            int(p.get("count", 0)) for p in by_event.get("trailing_text_removed", [])
        ),
        "speculation_replaced": len(by_event.get("speculation_replaced", [])),
        "dedupe_dropped": sum(len(p.get("dropped", [])) for p in by_event.get("claims_deduplicated", [])),
        "gap_rounds": node_runs.get("gap", 0),
        "zero_supported_sub_questions": zero_supported,
    }


def _search_health(by_event: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """From search_done events (one per provider call). A search is a primary/only call plus its fallback;
    it has zero results when its last call returned none. Runs from before search_done get no keys."""
    calls = by_event.get("search_done", [])
    if not calls:
        return {}
    searches: list[int] = []  # final result count of each search
    providers: dict[str, dict[str, int]] = {}
    for c in calls:
        p = providers.setdefault(str(c.get("provider")), {"calls": 0, "zero_results": 0, "errors": 0})
        p["calls"] += 1
        p["zero_results"] += int(c.get("result_count", 0) == 0)
        p["errors"] += int(bool(c.get("error")))
        if c.get("role") == "fallback" and searches:
            searches[-1] = int(c.get("result_count", 0))
        else:
            searches.append(int(c.get("result_count", 0)))
    fetch_misses = sum(
        1 for f in by_event.get("fetch_failed", []) if f.get("reason") == FetchErrorReason.REPLAY_MISS.value
    )
    return {
        "searches": len(searches),
        "zero_result_searches": sum(1 for n in searches if n == 0),
        "fallback_uses": sum(1 for c in calls if c.get("role") == "fallback"),
        "search_calls_by_provider": providers,
        "replay_misses_search": sum(1 for m in by_event.get("replay_miss", []) if m.get("kind") == "search"),
        "replay_misses_fetch": fetch_misses,
    }


def _merge_providers(metrics: Sequence[dict[str, Any]]) -> dict[str, dict[str, int]] | None:
    merged: dict[str, dict[str, int]] = {}
    for m in metrics:
        for name, counts in (m.get("search_calls_by_provider") or {}).items():
            total = merged.setdefault(name, {"calls": 0, "zero_results": 0, "errors": 0})
            for k, v in counts.items():
                total[k] = total.get(k, 0) + v
    return merged or None


def _optional_sum(metrics: Sequence[dict[str, Any]], key: str) -> int | None:
    """Sum of a metric, or None when no run recorded it (evals from before it existed)."""
    return sum(m.get(key, 0) for m in metrics) if any(key in m for m in metrics) else None


def _median(values: Sequence[float]) -> float | None:
    return round(statistics.median(values), 1) if values else None


def aggregate(metrics: Sequence[dict[str, Any]], timing_ok: Sequence[bool]) -> dict[str, Any]:
    """Totals and medians across runs. Timing figures use only runs with timing_ok (on AC throughout)."""
    timed = [m for m, ok in zip(metrics, timing_ok, strict=True) if ok]
    nodes = sorted({n for m in timed for n in m["node_seconds"]})
    verdicts = Counter[str]()
    failed = Counter[str]()
    for m in metrics:
        verdicts.update(m["claims_by_verdict"])
        failed.update(m["sources_failed_by_reason"])
    checks = sum(m["quote_checks"] for m in metrics)
    not_found = sum(m["quote_not_found"] for m in metrics)
    claims_total = sum(m["claims"] for m in metrics)
    return {
        "runs": len(metrics),
        "status": dict(Counter(m["status"] for m in metrics)),
        "timed_runs": len(timed),
        "total_duration_s": round(sum(m["duration_s"] or 0 for m in timed), 1),
        "median_duration_s": _median([m["duration_s"] for m in timed if m["duration_s"] is not None]),
        "median_node_seconds": {n: _median([m["node_seconds"].get(n, 0.0) for m in timed]) for n in nodes},
        "llm_calls": sum(m["llm_calls"] for m in metrics),
        "llm_failed_calls": sum(m["llm_failed_calls"] for m in metrics),
        "tokens": sum(m["prompt_tokens"] + m["output_tokens"] for m in metrics),
        "sources_fetched": sum(m["sources_fetched"] for m in metrics),
        "sources_reused": sum(m.get("sources_reused", 0) for m in metrics),
        "sources_failed_by_reason": dict(failed),
        "blocked_hosts": sum(len(m["blocked_hosts"]) for m in metrics),
        "backfills": sum(m["backfills"] for m in metrics),
        "median_distinct_cited_domains": _median([m["distinct_cited_domains"] for m in metrics]),
        "claims": claims_total,
        "claims_by_verdict": dict(verdicts),
        "supported_share": round(verdicts["supported"] / claims_total, 4) if claims_total else None,
        "quote_not_found_rate": round(not_found / checks, 4) if checks else None,
        "uncited_sentences_deleted": sum(m["uncited_sentences_deleted"] for m in metrics),
        "trailing_text_removed": sum(m["trailing_text_removed"] for m in metrics),
        "speculation_replaced": sum(m["speculation_replaced"] for m in metrics),
        "dedupe_dropped": sum(m["dedupe_dropped"] for m in metrics),
        "gap_rounds": sum(m["gap_rounds"] for m in metrics),
        "zero_supported_sub_questions": sum(len(m["zero_supported_sub_questions"]) for m in metrics),
        "match_methods": {
            m: sum((x.get("match_methods") or {}).get(m, 0) for x in metrics) for m in MATCH_METHODS
        }
        if any("match_methods" in x for x in metrics)
        else None,
        "sentence_replaced": _optional_sum(metrics, "sentence_replaced"),
        "searches": _optional_sum(metrics, "searches"),
        "zero_result_searches": _optional_sum(metrics, "zero_result_searches"),
        "fallback_uses": _optional_sum(metrics, "fallback_uses"),
        "search_calls_by_provider": _merge_providers(metrics),
        "replay_misses_search": _optional_sum(metrics, "replay_misses_search"),
        "replay_misses_fetch": _optional_sum(metrics, "replay_misses_fetch"),
        "sentences_checked": _optional_sum(metrics, "sentences_checked"),
        # None when no run recorded it (evals from before the number check)
        "number_mismatch": (
            sum(m.get("number_mismatch", 0) for m in metrics)
            if any("number_mismatch" in m for m in metrics)
            else None
        ),
    }
