"""Export label files for human review (UTF-8 CSV with BOM so Excel and LibreOffice read it).

claims.csv     stratified sample of claims; verifier_verdict is the LAST column (hide it while labelling)
sentences.csv  report sentences with the texts of the claims they cite
dedupe_pairs.csv supported-claim pairs with clustering similarity in [0.85, 0.99]; similarity is LAST

Sampling is seeded and stratified by question: within each stratum, every run (question) first gives
at most ceil(quota / number of runs) + 1 items, round-robin in a seeded order; remaining slots are then
filled round-robin from runs that have more.
"""

import csv
import itertools
import json
import math
import random
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from research_assistant.db.models import Chunk, Claim, Source, Verdict
from research_assistant.db.repositories.claims import evidence_for_claims
from research_assistant.db.repositories.events import list_events
from research_assistant.db.repositories.reports import latest_report
from research_assistant.graph.citations import Cite, split_sentences

DEFAULT_SEED = 20261118
CLAIM_QUOTAS = {"supported": 80, "partial": 20, "unsupported_or_contradicted": 20}
SENTENCE_QUOTA = 40
PAIR_CAP = 60
PAIR_MIN, PAIR_MAX, PAIR_BINS = 0.85, 0.99, 10
PASSAGE_SEPARATOR = "\n\n----- next passage -----\n\n"

CLAIM_COLUMNS = [
    "id",
    "claim_id",
    "question",
    "claim",
    "quote",
    "chunk_text",
    "source_url",
    "human_verdict",
    "note",
    "verifier_verdict",
]
SENTENCE_COLUMNS = ["id", "question", "sentence", "cited_claims", "human_label", "note"]
PAIR_COLUMNS = [
    "id",
    "claim_a_id",
    "claim_b_id",
    "question",
    "claim_a",
    "claim_b",
    "human_label",
    "note",
    "similarity",
]


def per_run_cap(quota: int, n_runs: int) -> int:
    return math.ceil(quota / max(n_runs, 1)) + 1


def stratified_sample[T](
    items: Sequence[T], group: Callable[[T], str], quota: int, n_groups: int, rng: random.Random
) -> list[T]:
    """Round-robin by depth over runs in a seeded order, each run's items in seeded order.

    Depths 0..cap-1 give every run at most per_run_cap(quota, n_groups) items; if slots remain after
    that (some runs had few items), deeper rounds fill them from the runs that have more."""
    by_group: dict[str, list[T]] = {}
    for item in items:
        by_group.setdefault(group(item), []).append(item)
    names = sorted(by_group)
    for name in names:
        rng.shuffle(by_group[name])
    rng.shuffle(names)
    picked: list[T] = []
    depth = 0
    while len(picked) < quota and any(len(by_group[n]) > depth for n in names):
        for name in names:
            if len(picked) < quota and depth < len(by_group[name]):
                picked.append(by_group[name][depth])
        depth += 1
    return picked


@dataclass(frozen=True)
class EvalRun:
    qid: str
    run_id: uuid.UUID
    question: str
    category: str


def finished_runs(eval_dir: Path) -> list[EvalRun]:
    data = json.loads((eval_dir / "results.json").read_text(encoding="utf-8"))
    out = []
    for qid, entry in data["runs"].items():
        if entry.get("state") == "finished" and entry.get("run_id"):
            out.append(EvalRun(qid, uuid.UUID(entry["run_id"]), entry["question"], entry["category"]))
    return sorted(out, key=lambda r: r.qid)


def write_csv(path: Path, columns: Sequence[str], rows: Sequence[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(columns), extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in columns})


def _stratum(verdict: Verdict | None) -> str | None:
    if verdict is Verdict.SUPPORTED:
        return "supported"
    if verdict is Verdict.PARTIAL:
        return "partial"
    if verdict in (Verdict.UNSUPPORTED, Verdict.CONTRADICTED):
        return "unsupported_or_contradicted"
    return None


def _claim_rows(session: Session, run: EvalRun) -> list[dict[str, Any]]:
    claims = list(session.scalars(select(Claim).where(Claim.run_id == run.run_id).order_by(Claim.ordinal)))
    evidence = evidence_for_claims(session, [c.id for c in claims])
    events = list_events(session, run.run_id)
    cited_chunks: dict[str, list[str]] = {}
    model_quote: dict[str, str] = {}
    for e in events:
        if e.event == "claims_created":
            for c in e.payload["claims"]:
                cited_chunks[c["claim_id"]] = c["chunk_ids"]
        elif e.event == "claim_verified":
            model_quote[e.payload["claim_id"]] = e.payload.get("model_quote") or ""
    rows = []
    for claim in claims:
        stratum = _stratum(claim.verdict)
        if stratum is None:
            continue
        ev = evidence.get(claim.id)
        chunk_ids = [ev[0].chunk_id] if ev else [uuid.UUID(c) for c in cited_chunks.get(str(claim.id), [])]
        chunks = [ch for cid in chunk_ids if (ch := session.get(Chunk, cid)) is not None]
        urls = []
        for ch in chunks:
            src = session.get(Source, ch.source_id)
            if src is not None:
                urls.append(src.final_url or src.url)
        rows.append(
            {
                "id": f"{run.qid}-claim{claim.ordinal:03d}",
                "claim_id": str(claim.id),
                "question": run.question,
                "claim": claim.text,
                "quote": ev[0].quote if ev else model_quote.get(str(claim.id), ""),
                "chunk_text": PASSAGE_SEPARATOR.join(ch.text for ch in chunks),
                "source_url": "\n".join(dict.fromkeys(urls)),
                "verifier_verdict": str(claim.verdict),
                "_stratum": stratum,
                "_qid": run.qid,
                "_ordinal": claim.ordinal,
            }
        )
    return rows


def _sentence_rows(session: Session, run: EvalRun) -> list[dict[str, Any]]:
    report = latest_report(session, run.run_id)
    if report is None:
        return []
    number = {cid: entry["n"] for entry in report.citations for cid in entry.get("claim_ids", [])}
    claim_text = {str(c.id): c.text for c in session.scalars(select(Claim).where(Claim.run_id == run.run_id))}
    rows: list[dict[str, Any]] = []
    sections = [e for e in list_events(session, run.run_id) if e.event == "section_citations"]
    for section in sections:
        segments = [
            s if isinstance(s, str) else Cite(tuple(uuid.UUID(c) for c in s["cite"]))
            for s in section.payload["segments"]
        ]
        for sentence in split_sentences(segments):
            cited = list(dict.fromkeys(str(i) for p in sentence if isinstance(p, Cite) for i in p.ids))
            if not cited:
                continue
            text = "".join(
                p if isinstance(p, str) else "".join(f"[{number.get(str(i), '?')}]" for i in p.ids)
                for p in sentence
            ).strip()
            rows.append(
                {
                    "id": f"{run.qid}-s{len(rows) + 1:03d}",
                    "question": run.question,
                    "sentence": text,
                    "cited_claims": "\n".join(f"- {claim_text.get(c, '?')}" for c in cited),
                    "_qid": run.qid,
                    "_index": len(rows),
                }
            )
    return rows


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


@dataclass(frozen=True)
class _SupportedClaim:
    id: uuid.UUID
    ordinal: int
    text: str


def _supported_claims(session: Session, run_id: uuid.UUID) -> list[_SupportedClaim]:
    stmt = (
        select(Claim)
        .where(Claim.run_id == run_id, Claim.verdict == Verdict.SUPPORTED)
        .order_by(Claim.ordinal)
    )
    return [_SupportedClaim(c.id, c.ordinal, c.text) for c in session.scalars(stmt)]


def _pair_rows(
    run: EvalRun, claims: list[_SupportedClaim], vectors: list[list[float]]
) -> list[dict[str, Any]]:
    rows = []
    for (a, va), (b, vb) in itertools.combinations(zip(claims, vectors, strict=True), 2):
        sim = round(_cosine(va, vb), 4)
        if PAIR_MIN <= sim <= PAIR_MAX:
            rows.append(
                {
                    "id": f"{run.qid}-p{a.ordinal:03d}-{b.ordinal:03d}",
                    "claim_a_id": str(a.id),
                    "claim_b_id": str(b.id),
                    "question": run.question,
                    "claim_a": a.text,
                    "claim_b": b.text,
                    "similarity": f"{sim:.4f}",
                    "_sim": sim,
                    "_qid": run.qid,
                    "_order": (a.ordinal, b.ordinal),
                }
            )
    return rows


def _even_across_range(rows: list[dict[str, Any]], cap: int, rng: random.Random) -> list[dict[str, Any]]:
    width = (PAIR_MAX - PAIR_MIN) / PAIR_BINS
    bins: list[list[dict[str, Any]]] = [[] for _ in range(PAIR_BINS)]
    for row in rows:
        bins[min(int((row["_sim"] - PAIR_MIN) / width), PAIR_BINS - 1)].append(row)
    for b in bins:
        rng.shuffle(b)
    picked: list[dict[str, Any]] = []
    for depth in itertools.count():
        if len(picked) >= cap or all(len(b) <= depth for b in bins):
            break
        for b in bins:
            if depth < len(b) and len(picked) < cap:
                picked.append(b[depth])
    return picked


def export_labels(
    eval_dir: Path,
    session_factory: Callable[[], Session],
    embed: Callable[[list[str]], list[list[float]]],
    *,
    seed: int = DEFAULT_SEED,
    claim_quotas: dict[str, int] | None = None,
    sentence_quota: int = SENTENCE_QUOTA,
    pair_cap: int = PAIR_CAP,
) -> dict[str, Any]:
    runs = finished_runs(eval_dir)
    quotas = claim_quotas or CLAIM_QUOTAS
    claim_rows: list[dict[str, Any]] = []
    sentence_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    with session_factory() as session:
        for run in runs:
            claim_rows += _claim_rows(session, run)
            sentence_rows += _sentence_rows(session, run)
        supported = {run.run_id: _supported_claims(session, run.run_id) for run in runs}
    for run in runs:  # embedding happens with no session open
        claims = supported[run.run_id]
        if len(claims) >= 2:
            pair_rows += _pair_rows(run, claims, embed([c.text for c in claims]))

    sampled_claims: list[dict[str, Any]] = []
    for stratum, quota in quotas.items():
        items = [r for r in claim_rows if r["_stratum"] == stratum]
        sampled_claims += stratified_sample(
            items, lambda r: str(r["_qid"]), quota, len(runs), random.Random(f"{seed}-claims-{stratum}")
        )
    sampled_claims.sort(key=lambda r: (r["_qid"], r["_ordinal"]))  # by question, not by stratum (blind)
    sampled_sentences = stratified_sample(
        sentence_rows, lambda r: str(r["_qid"]), sentence_quota, len(runs), random.Random(f"{seed}-sentences")
    )
    sampled_sentences.sort(key=lambda r: (r["_qid"], r["_index"]))
    sampled_pairs = _even_across_range(pair_rows, pair_cap, random.Random(f"{seed}-pairs"))
    sampled_pairs.sort(key=lambda r: (r["_qid"], r["_order"]))  # not by similarity (blind)

    paths = {
        "claims": eval_dir / "claims.csv",
        "sentences": eval_dir / "sentences.csv",
        "dedupe_pairs": eval_dir / "dedupe_pairs.csv",
    }
    write_csv(paths["claims"], CLAIM_COLUMNS, sampled_claims)
    write_csv(paths["sentences"], SENTENCE_COLUMNS, sampled_sentences)
    write_csv(paths["dedupe_pairs"], PAIR_COLUMNS, sampled_pairs)
    manifest = {
        "seed": seed,
        "claims": {s: sum(1 for r in sampled_claims if r["_stratum"] == s) for s in quotas},
        "claims_available": {s: sum(1 for r in claim_rows if r["_stratum"] == s) for s in quotas},
        "sentences": len(sampled_sentences),
        "sentences_available": len(sentence_rows),
        "dedupe_pairs": len(sampled_pairs),
        "dedupe_pairs_available": len(pair_rows),
        "paths": {k: str(v) for k, v in paths.items()},
    }
    (eval_dir / "labels_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
