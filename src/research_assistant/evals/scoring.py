"""Score human labels in an eval directory. Only labelled rows count; every figure reports its n."""

import csv
import json
import math
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

VERDICT_LABELS = ("supported", "partial", "unsupported", "contradicted")
SENTENCE_LABELS = ("faithful", "adds_info", "wrong")
PAIR_LABELS = ("duplicate", "distinct")
THRESHOLDS = [round(0.90 + 0.01 * i, 2) for i in range(10)]  # 0.90 .. 0.99
Z95 = 1.959963984540054


def wilson(successes: int, n: int, z: float = Z95) -> tuple[float, float] | None:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return None
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4)


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _labelled(
    rows: Sequence[dict[str, str]], column: str, valid: Sequence[str]
) -> tuple[list[dict[str, str]], list[str]]:
    good, invalid = [], []
    for row in rows:
        value = (row.get(column) or "").strip().lower()
        if not value:
            continue
        if value not in valid:
            invalid.append(f"{row.get('id')}: {value!r}")
            continue
        good.append({**row, column: value})
    return good, invalid


def _question_of(row_id: str, qids: Sequence[str]) -> str:
    return next((q for q in qids if row_id.startswith(q + "-")), "unknown")


def score_claims(
    rows: Sequence[dict[str, str]], qids: Sequence[str], categories: dict[str, str]
) -> dict[str, Any]:
    labelled, invalid = _labelled(rows, "human_verdict", VERDICT_LABELS)
    matrix = {v: dict.fromkeys(VERDICT_LABELS, 0) for v in VERDICT_LABELS}
    for row in labelled:
        verifier = (row.get("verifier_verdict") or "").strip().lower()
        if verifier in matrix:
            matrix[verifier][row["human_verdict"]] += 1
    supported_rows = [r for r in labelled if (r.get("verifier_verdict") or "").strip().lower() == "supported"]
    correct = sum(1 for r in supported_rows if r["human_verdict"] == "supported")
    per_question = Counter(_question_of(r["id"], qids) for r in labelled)
    per_category = Counter(
        categories.get(q, "unknown") for q in (_question_of(r["id"], qids) for r in labelled)
    )
    return {
        "n_rows": len(rows),
        "n_labelled": len(labelled),
        "invalid_labels": invalid,
        "confusion_matrix": matrix,  # rows: verifier verdict, columns: human verdict
        "supported_precision": {
            "n": len(supported_rows),
            "correct": correct,
            "precision": round(correct / len(supported_rows), 4) if supported_rows else None,
            "wilson95": wilson(correct, len(supported_rows)),
        },
        "n_per_question": dict(sorted(per_question.items())),
        "n_per_category": dict(sorted(per_category.items())),
    }


def score_sentences(
    rows: Sequence[dict[str, str]], qids: Sequence[str], categories: dict[str, str]
) -> dict[str, Any]:
    labelled, invalid = _labelled(rows, "human_label", SENTENCE_LABELS)
    counts = Counter(r["human_label"] for r in labelled)
    n = len(labelled)
    per_question = Counter(_question_of(r["id"], qids) for r in labelled)
    per_category = Counter(
        categories.get(q, "unknown") for q in (_question_of(r["id"], qids) for r in labelled)
    )
    return {
        "n_rows": len(rows),
        "n_labelled": n,
        "invalid_labels": invalid,
        "counts": {label: counts.get(label, 0) for label in SENTENCE_LABELS},
        "faithful_rate": round(counts["faithful"] / n, 4) if n else None,
        "wilson95": wilson(counts["faithful"], n),
        "n_per_question": dict(sorted(per_question.items())),
        "n_per_category": dict(sorted(per_category.items())),
    }


def score_pairs(rows: Sequence[dict[str, str]]) -> dict[str, Any]:
    labelled, invalid = _labelled(rows, "human_label", PAIR_LABELS)
    pairs = [(float(r["similarity"]), r["human_label"] == "duplicate") for r in labelled]
    by_threshold = []
    for t in THRESHOLDS:
        tp = sum(1 for sim, dup in pairs if sim >= t and dup)
        fp = sum(1 for sim, dup in pairs if sim >= t and not dup)
        fn = sum(1 for sim, dup in pairs if sim < t and dup)
        by_threshold.append(
            {
                "threshold": t,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": round(tp / (tp + fp), 4) if tp + fp else None,
                "recall": round(tp / (tp + fn), 4) if tp + fn else None,
            }
        )
    no_false_merge = next((row for row in by_threshold if row["fp"] == 0), None)
    return {
        "n_rows": len(rows),
        "n_labelled": len(labelled),
        "invalid_labels": invalid,
        "duplicates": sum(1 for _, dup in pairs if dup),
        "by_threshold": by_threshold,
        "lowest_threshold_without_false_merges": no_false_merge,
    }


def score_labels(eval_dir: Path) -> dict[str, Any]:
    results = json.loads((eval_dir / "results.json").read_text(encoding="utf-8"))
    qids = sorted(results["runs"], key=len, reverse=True)  # longest first, so prefixes do not collide
    categories = {qid: entry.get("category", "unknown") for qid, entry in results["runs"].items()}
    scores = {
        "claims": score_claims(read_rows(eval_dir / "claims.csv"), qids, categories),
        "sentences": score_sentences(read_rows(eval_dir / "sentences.csv"), qids, categories),
        "dedupe_pairs": score_pairs(read_rows(eval_dir / "dedupe_pairs.csv")),
    }
    (eval_dir / "scores.json").write_text(json.dumps(scores, indent=2), encoding="utf-8")
    (eval_dir / "scores.md").write_text(render_scores(scores), encoding="utf-8")
    return scores


def _fmt(value: float | None) -> str:
    return "–" if value is None else f"{value:.1%}"


def _interval(ci: Sequence[float] | None) -> str:
    return "–" if ci is None else f"{ci[0]:.1%}–{ci[1]:.1%}"


def render_scores(scores: dict[str, Any]) -> str:
    c, s, p = scores["claims"], scores["sentences"], scores["dedupe_pairs"]
    lines = ["# Label scores", ""]
    lines += ["## Verifier (claims.csv)", "", f"Labelled rows: {c['n_labelled']} of {c['n_rows']}.", ""]
    lines += [
        "| verifier \\ human | " + " | ".join(VERDICT_LABELS) + " |",
        "|---" * (len(VERDICT_LABELS) + 1) + "|",
    ]
    for verifier, row in c["confusion_matrix"].items():
        lines.append(f"| {verifier} | " + " | ".join(str(row[h]) for h in VERDICT_LABELS) + " |")
    sp = c["supported_precision"]
    lines += [
        "",
        f'Precision of "supported": {_fmt(sp["precision"])} ({sp["correct"]}/{sp["n"]}), '
        f"95% Wilson interval {_interval(sp['wilson95'])}.",
        "The sample is stratified by verifier verdict, so the matrix proportions are not those of all "
        "claims; "
        'the precision of "supported" is computed within its own stratum and is unbiased.',
        "",
    ]
    lines += [
        "## Report sentences (sentences.csv)",
        "",
        f"Labelled rows: {s['n_labelled']} of {s['n_rows']}.",
    ]
    lines += [
        f"Faithful: {_fmt(s['faithful_rate'])}, 95% Wilson interval {_interval(s['wilson95'])}; "
        f"counts {s['counts']}.",
        "",
    ]
    lines += [
        "## Claim dedupe (dedupe_pairs.csv)",
        "",
        f"Labelled pairs: {p['n_labelled']} of {p['n_rows']}, ",
    ]
    lines[-1] += f"{p['duplicates']} labelled duplicate."
    lines += ["", "| threshold | TP | FP | FN | precision | recall |", "|---|---|---|---|---|---|"]
    for row in p["by_threshold"]:
        lines.append(
            f"| {row['threshold']:.2f} | {row['tp']} | {row['fp']} | {row['fn']} | "
            f"{_fmt(row['precision'])} | {_fmt(row['recall'])} |"
        )
    best = p["lowest_threshold_without_false_merges"]
    lines += [
        "",
        "Lowest threshold with no false merges: "
        + (f"{best['threshold']:.2f} (recall {_fmt(best['recall'])})" if best else "none in 0.90–0.99"),
        "",
        "## Labelled rows per question and category",
        "",
        "| question | claims | sentences |",
        "|---|---|---|",
    ]
    for q in sorted(set(c["n_per_question"]) | set(s["n_per_question"])):
        lines.append(f"| {q} | {c['n_per_question'].get(q, 0)} | {s['n_per_question'].get(q, 0)} |")
    lines += ["", "| category | claims | sentences |", "|---|---|---|"]
    for cat in sorted(set(c["n_per_category"]) | set(s["n_per_category"])):
        lines.append(f"| {cat} | {c['n_per_category'].get(cat, 0)} | {s['n_per_category'].get(cat, 0)} |")
    for name, part in (("claims", c), ("sentences", s), ("dedupe_pairs", p)):
        if part["invalid_labels"]:
            lines += ["", f"Invalid labels in {name}.csv (ignored): " + "; ".join(part["invalid_labels"])]
    return "\n".join(lines) + "\n"
