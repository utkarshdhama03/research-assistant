"""Independent citation checker. Reads the database with plain SQL (psycopg) and re-parses the stored
report markdown. It deliberately imports nothing from research_assistant.graph, the repositories or
the citation code, so a bug there cannot hide itself (a test enforces this).

Checks per run:
  marker_without_citation    every [n] in the report body has a reports.citations entry
  citation_source_not_in_run every citation entry's source belongs to the run
  unused_citation_entry      every citation entry's [n] appears in the body
  quote_offsets_mismatch     every claim_evidence quote equals chunk.text[quote_start:quote_end]
  evidence_chunk_not_in_run  every evidence chunk belongs to the run
  cited_chunk_*              every chunk listed in a citation entry exists, is in the run and in that source
  reference_*                every reference line's URL equals the source's final_url (url if null);
                             a `(retrieved YYYY-MM-DD)` suffix equals the source's fetched_at date (UTC).
                             Reports from before Phase 7b have no suffix; a missing date is not a failure.
  appendix_quote_mismatch    every Evidence-appendix quote equals the stored quote of that claim
"""

import argparse
import re
import uuid
from dataclasses import dataclass, field
from typing import Any

import psycopg

from research_assistant.config import get_settings

_MARKER = re.compile(r"\[(\d+)\]")
_REFERENCE = re.compile(r"^- \[(\d+)\] .* — <(.+?)>(?: \(retrieved (\d{4}-\d{2}-\d{2})\))?$")
_APPENDIX_ENTRY = re.compile(r"^- \*\*Claim (\d+)\*\* \[(\d+)\]: ")


@dataclass(frozen=True)
class Failure:
    run_id: str
    kind: str
    detail: str


@dataclass
class CheckResult:
    run_id: str
    checks: int = 0
    failures: list[Failure] = field(default_factory=list)
    note: str | None = None

    @property
    def validity(self) -> float | None:
        return 1 - len(self.failures) / self.checks if self.checks else None

    def check(self, ok: bool, kind: str, detail: str) -> None:
        self.checks += 1
        if not ok:
            self.failures.append(Failure(self.run_id, kind, detail))

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "checks": self.checks,
            "failures": [f.__dict__ for f in self.failures],
            "validity": self.validity,
            "note": self.note,
        }


def connect(database_url: str | None = None) -> psycopg.Connection[Any]:
    url = database_url or get_settings().database_url
    return psycopg.connect(url.replace("postgresql+psycopg://", "postgresql://", 1))


def _split_report(markdown: str) -> tuple[str, str, str]:
    body, _, rest = markdown.partition("\n## References\n")
    references, _, appendix = rest.partition("\n## Evidence\n")
    return body, references, appendix


def _appendix_quotes(appendix: str) -> dict[int, str]:
    quotes: dict[int, list[str]] = {}
    current: int | None = None
    for line in appendix.split("\n"):
        m = _APPENDIX_ENTRY.match(line)
        if m:
            current = int(m.group(1))
            quotes[current] = []
        elif current is not None and line.startswith("  > "):  # the writer emits "  > " + quote line
            quotes[current].append(line[4:])
        else:
            current = None
    return {k: "\n".join(v) for k, v in quotes.items()}


def check_run(conn: psycopg.Connection[Any], run_id: uuid.UUID | str) -> CheckResult:
    rid = str(run_id)
    result = CheckResult(rid)
    with conn.cursor() as cur:
        cur.execute(
            "SELECT markdown, citations FROM reports WHERE run_id = %s ORDER BY created_at DESC, id LIMIT 1",
            (rid,),
        )
        row = cur.fetchone()
        cur.execute(
            "SELECT id, url, final_url, to_char(fetched_at AT TIME ZONE 'UTC', 'YYYY-MM-DD') "
            "FROM sources WHERE run_id = %s",
            (rid,),
        )
        source_rows = cur.fetchall()
        sources = {str(r[0]): (r[1], r[2]) for r in source_rows}
        fetched_dates = {str(r[0]): r[3] for r in source_rows}
        cur.execute(
            """
            SELECT c.ordinal, ce.claim_id, ce.chunk_id, ce.quote, ce.quote_start, ce.quote_end,
                   ch.text, ch.run_id
            FROM claim_evidence ce
            JOIN claims c ON c.id = ce.claim_id
            JOIN chunks ch ON ch.id = ce.chunk_id
            WHERE c.run_id = %s
            """,
            (rid,),
        )
        evidence = cur.fetchall()

    for ordinal, claim_id, chunk_id, quote, start, end, text, chunk_run in evidence:
        result.check(
            text[start:end] == quote,
            "quote_offsets_mismatch",
            f"claim {ordinal} ({claim_id}): chunk {chunk_id}[{start}:{end}] != stored quote",
        )
        result.check(
            str(chunk_run) == rid,
            "evidence_chunk_not_in_run",
            f"claim {ordinal}: chunk {chunk_id} is in run {chunk_run}",
        )

    if row is None:
        result.note = "no report"
        return result
    markdown, citations = row
    body, references, appendix = _split_report(markdown)
    entries: dict[int, dict[str, Any]] = {int(c["n"]): c for c in citations}

    # report markers -> citation entries
    used: set[int] = set()
    for m in _MARKER.finditer(body):
        n = int(m.group(1))
        used.add(n)
        result.check(n in entries, "marker_without_citation", f"[{n}] has no reports.citations entry")
    for n, entry in sorted(entries.items()):
        result.check(n in used, "unused_citation_entry", f"citation [{n}] never appears in the report body")
        result.check(
            entry["source_id"] in sources,
            "citation_source_not_in_run",
            f"citation [{n}] source {entry['source_id']} is not a source of this run",
        )

    # cited chunks belong to the run and to the entry's source
    chunk_ids = sorted({cid for e in entries.values() for cid in e.get("chunk_ids", [])})
    chunk_rows: dict[str, tuple[str, str]] = {}
    if chunk_ids:
        with conn.cursor() as cur:
            cur.execute("SELECT id, run_id, source_id FROM chunks WHERE id = ANY(%s::uuid[])", (chunk_ids,))
            chunk_rows = {str(r[0]): (str(r[1]), str(r[2])) for r in cur.fetchall()}
    for n, entry in sorted(entries.items()):
        for cid in entry.get("chunk_ids", []):
            found = chunk_rows.get(cid)
            result.check(
                found is not None, "cited_chunk_missing", f"citation [{n}] chunk {cid} does not exist"
            )
            if found is None:
                continue
            result.check(
                found[0] == rid, "cited_chunk_not_in_run", f"citation [{n}] chunk {cid} is in run {found[0]}"
            )
            result.check(
                found[1] == entry["source_id"],
                "cited_chunk_wrong_source",
                f"citation [{n}] chunk {cid} belongs to source {found[1]}, not {entry['source_id']}",
            )

    # reference list URLs
    ref_lines = [m for m in (_REFERENCE.match(line) for line in references.split("\n")) if m]
    seen_refs: set[int] = set()
    for m in ref_lines:
        n, url = int(m.group(1)), m.group(2)
        seen_refs.add(n)
        ref_entry = entries.get(n)
        result.check(
            ref_entry is not None, "reference_without_citation", f"reference [{n}] has no citation entry"
        )
        if ref_entry is None or ref_entry["source_id"] not in sources:
            continue
        src_url, final_url = sources[ref_entry["source_id"]]
        expected = final_url or src_url
        result.check(
            url == expected, "reference_url_mismatch", f"reference [{n}] shows {url}, source has {expected}"
        )
        shown_date = m.group(3)
        if shown_date is not None:
            actual = fetched_dates.get(ref_entry["source_id"])
            result.check(
                shown_date == actual,
                "reference_date_mismatch",
                f"reference [{n}] shows retrieved {shown_date}, source was fetched {actual}",
            )
    for n in sorted(entries):
        result.check(n in seen_refs, "citation_without_reference", f"citation [{n}] has no reference line")

    # evidence appendix quotes
    stored = {int(ordinal): quote for ordinal, _, _, quote, *_ in evidence}
    for ordinal, quote in sorted(_appendix_quotes(appendix).items()):
        result.check(
            stored.get(ordinal) == quote,
            "appendix_quote_mismatch",
            f"appendix quote for claim {ordinal} differs from the stored evidence quote",
        )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check citation validity of research runs.")
    parser.add_argument("run_ids", nargs="+", type=uuid.UUID)
    args = parser.parse_args(argv)
    bad = 0
    with connect() as conn:
        for run_id in args.run_ids:
            r = check_run(conn, run_id)
            v = f"{r.validity:.2%}" if r.validity is not None else "n/a"
            print(f"{run_id}: {r.checks} checks, {len(r.failures)} failures, validity {v} {r.note or ''}")
            for f in r.failures:
                print(f"  {f.kind}: {f.detail}")
            bad += len(r.failures)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
