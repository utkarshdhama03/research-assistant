import ast
import random
from collections import Counter
from pathlib import Path

import pytest

from research_assistant.evals import citation_check
from research_assistant.evals.labels import per_run_cap, stratified_sample
from research_assistant.evals.power import PowerState, keep_awake, power_state
from research_assistant.evals.questions import CATEGORIES, load_questions
from research_assistant.evals.runner import render_summary, timing_ok
from research_assistant.evals.scoring import score_claims, score_pairs, score_sentences, wilson

# -- question set ----------------------------------------------------------------------------


def test_question_set() -> None:
    questions = load_questions()
    assert len(questions) == 16
    assert len({q.id for q in questions}) == 16
    assert {q.category for q in questions} == CATEGORIES
    [false_premise] = [q for q in questions if q.category == "false-premise"]
    assert "Eiffel Tower" in false_premise.question and false_premise.expectation


@pytest.mark.parametrize(
    "text",
    [
        "questions: []",
        "questions:\n  - {id: a, category: nonsense, question: q}",
        "questions:\n  - {id: a, category: science, question: q}\n"
        "  - {id: a, category: science, question: r}",
        "questions:\n  - {id: a, category: science}",
        "just text",
    ],
)
def test_bad_question_files(tmp_path: Path, text: str) -> None:
    path = tmp_path / "q.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError):
        load_questions(path)


# -- Wilson interval -------------------------------------------------------------------------


def test_wilson_hand_computed() -> None:
    # 8/10, z = 1.96: centre (0.8 + 0.19208) / 1.38416 = 0.71674, half 0.313626 / 1.38416 = 0.226582
    low, high = wilson(8, 10)  # type: ignore[misc]
    assert low == pytest.approx(0.4902, abs=2e-4) and high == pytest.approx(0.9433, abs=2e-4)


def test_wilson_edges() -> None:
    assert wilson(0, 0) is None
    low, high = wilson(0, 5)  # type: ignore[misc]
    assert low == 0.0 and high == pytest.approx(0.4345, abs=2e-4)
    low, high = wilson(5, 5)  # type: ignore[misc]
    assert high == 1.0 and low == pytest.approx(0.5655, abs=2e-4)


# -- scoring ---------------------------------------------------------------------------------

QIDS = ["q01-a", "q02-b"]
CATS = {"q01-a": "technical", "q02-b": "science"}


def claim_row(row_id: str, human: str, verifier: str) -> dict[str, str]:
    return {"id": row_id, "human_verdict": human, "verifier_verdict": verifier}


def test_score_claims_confusion_precision_and_partial_files() -> None:
    rows = [
        claim_row("q01-a-claim000", "supported", "supported"),
        claim_row("q01-a-claim001", "Supported ", "supported"),
        claim_row("q01-a-claim002", "partial", "supported"),
        claim_row("q02-b-claim000", "unsupported", "unsupported"),
        claim_row("q02-b-claim001", "", "supported"),  # not labelled yet
        claim_row("q02-b-claim002", "maybe", "partial"),  # invalid
    ]
    scores = score_claims(rows, QIDS, CATS)
    assert (scores["n_rows"], scores["n_labelled"]) == (6, 4)
    assert scores["invalid_labels"] == ["q02-b-claim002: 'maybe'"]
    assert scores["confusion_matrix"]["supported"] == {
        "supported": 2,
        "partial": 1,
        "unsupported": 0,
        "contradicted": 0,
    }
    assert scores["confusion_matrix"]["unsupported"]["unsupported"] == 1
    sp = scores["supported_precision"]
    assert (sp["n"], sp["correct"], sp["precision"]) == (3, 2, 0.6667)
    assert sp["wilson95"] == wilson(2, 3)
    assert scores["n_per_question"] == {"q01-a": 3, "q02-b": 1}
    assert scores["n_per_category"] == {"science": 1, "technical": 3}


def test_score_sentences() -> None:
    rows = [
        {"id": "q01-a-s001", "human_label": "faithful"},
        {"id": "q01-a-s002", "human_label": "adds_info"},
        {"id": "q02-b-s001", "human_label": "faithful"},
        {"id": "q02-b-s002", "human_label": ""},
    ]
    scores = score_sentences(rows, QIDS, CATS)
    assert scores["n_labelled"] == 3 and scores["faithful_rate"] == 0.6667
    assert scores["counts"] == {"faithful": 2, "adds_info": 1, "wrong": 0}
    assert scores["wilson95"] == wilson(2, 3)


def test_score_pairs_hand_computed() -> None:
    labelled = [
        (0.995, "duplicate"),
        (0.97, "duplicate"),
        (0.95, "distinct"),
        (0.93, "duplicate"),
        (0.91, "distinct"),
    ]
    rows = [{"id": f"p{i}", "similarity": f"{s}", "human_label": h} for i, (s, h) in enumerate(labelled)]
    rows.append({"id": "p9", "similarity": "0.96", "human_label": ""})  # unlabelled
    scores = score_pairs(rows)
    by_t = {r["threshold"]: r for r in scores["by_threshold"]}
    assert (by_t[0.90]["tp"], by_t[0.90]["fp"], by_t[0.90]["fn"]) == (3, 2, 0)
    assert (by_t[0.90]["precision"], by_t[0.90]["recall"]) == (0.6, 1.0)
    assert (by_t[0.95]["precision"], by_t[0.95]["recall"]) == (0.6667, 0.6667)
    assert (by_t[0.96]["precision"], by_t[0.96]["recall"]) == (1.0, 0.6667)
    assert by_t[0.99]["precision"] == 1.0 and by_t[0.99]["recall"] == 0.3333
    assert scores["lowest_threshold_without_false_merges"]["threshold"] == 0.96
    assert scores["n_labelled"] == 5


def test_score_pairs_without_safe_threshold() -> None:
    rows = [{"id": "p", "similarity": "0.995", "human_label": "distinct"}]
    assert score_pairs(rows)["lowest_threshold_without_false_merges"] is None


# -- stratified sampling ---------------------------------------------------------------------


def items_for(sizes: dict[str, int]) -> list[tuple[str, int]]:
    return [(g, i) for g, n in sizes.items() for i in range(n)]


def test_sample_caps_each_run_when_there_is_enough() -> None:
    items = items_for({f"q{i}": 10 for i in range(16)})
    picked = stratified_sample(items, lambda x: x[0], 80, 16, random.Random("s"))
    counts = Counter(g for g, _ in picked)
    assert len(picked) == 80 and max(counts.values()) <= per_run_cap(80, 16) == 6


def test_sample_fills_from_runs_with_more() -> None:
    items = items_for({"a": 10, "b": 1, "c": 1, "d": 10})
    picked = stratified_sample(items, lambda x: x[0], 12, 4, random.Random("s"))
    assert Counter(g for g, _ in picked) == {"a": 5, "b": 1, "c": 1, "d": 5}  # cap 4, then filled


def test_sample_is_deterministic_and_bounded() -> None:
    items = items_for({"a": 3, "b": 7, "c": 2})
    first = stratified_sample(items, lambda x: x[0], 20, 3, random.Random("seed"))
    second = stratified_sample(items, lambda x: x[0], 20, 3, random.Random("seed"))
    assert first == second and len(first) == 12  # fewer items than the quota: all of them
    assert stratified_sample(items, lambda x: x[0], 20, 3, random.Random("other")) != first


# -- power and summary -----------------------------------------------------------------------


def test_power_state_and_keep_awake() -> None:
    state = power_state()
    assert isinstance(state, PowerState)
    with keep_awake():
        pass


@pytest.mark.parametrize(
    ("start", "end", "ok"),
    [(True, True, True), (False, True, False), (True, False, False), (None, None, True)],
)
def test_timing_ok(start: bool | None, end: bool | None, ok: bool) -> None:
    assert timing_ok({"power_start": {"on_ac": start}, "power_end": {"on_ac": end}}) is ok


def _metrics(duration: float) -> dict[str, object]:
    return {
        "status": "completed",
        "problems": [],
        "duration_s": duration,
        "node_seconds": {"plan": duration / 2, "write": duration / 2},
        "llm_calls": 3,
        "llm_failed_calls": 0,
        "prompt_tokens": 10,
        "output_tokens": 5,
        "sources_fetched": 2,
        "sources_failed_by_reason": {},
        "blocked_hosts": [],
        "backfills": 0,
        "distinct_cited_domains": 1,
        "claims": 2,
        "claims_by_verdict": {
            "supported": 2,
            "partial": 0,
            "unsupported": 0,
            "contradicted": 0,
            "unverified": 0,
        },
        "quote_not_found": 0,
        "quote_checks": 2,
        "uncited_sentences_deleted": 0,
        "trailing_text_removed": 0,
        "speculation_replaced": 0,
        "dedupe_dropped": 0,
        "gap_rounds": 0,
        "zero_supported_sub_questions": [],
    }


def test_summary_marks_battery_runs_and_excludes_their_timing() -> None:
    def entry(on_ac: bool, duration: float) -> dict[str, object]:
        return {
            "state": "finished",
            "category": "science",
            "status": "completed",
            "reason": None,
            "wall_s": duration,
            "power_start": {"on_ac": on_ac},
            "power_end": {"on_ac": True},
            "metrics": _metrics(duration),
            "citation_check": {"checks": 4, "failures": [], "validity": 1.0},
        }

    results = {
        "created_at": "now",
        "settings": {},
        "sessions": [],
        "runs": {"q01-a": entry(True, 100.0), "q02-b": entry(False, 900.0)},
    }
    text = render_summary(results)
    assert "| q02-b | science | completed | 900 s (battery) |" in text
    assert "| timed runs (on AC at start and end) | 1 |" in text
    assert "| total time, timed runs | 100.0 s |" in text
    assert "excluded from timing figures: q02-b" in text
    assert "| claims by verdict |" in text and "'supported': 4" in text  # quality metrics keep both runs


# -- citation checker independence -----------------------------------------------------------


def test_citation_check_imports_nothing_from_the_pipeline() -> None:
    tree = ast.parse(Path(citation_check.__file__).read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    ours = {m for m in imported if m.startswith("research_assistant")}
    assert ours == {"research_assistant.config"}, ours


# -- compare ---------------------------------------------------------------------------------


def test_compare_two_eval_directories(tmp_path: Path) -> None:
    import json

    from research_assistant.evals.compare import compare

    def entry(status: str, duration: float, number_mismatch: int | None) -> dict[str, object]:
        metrics = _metrics(duration)
        metrics["status"] = status
        if number_mismatch is not None:
            metrics["number_mismatch"] = number_mismatch
        return {
            "state": "finished",
            "category": "science",
            "status": status,
            "wall_s": duration,
            "power_start": {"on_ac": True},
            "power_end": {"on_ac": True},
            "metrics": metrics,
            "citation_check": {"checks": 4, "failures": [], "validity": 1.0},
        }

    for name, runs in {
        "base": {"q01-a": entry("completed", 100.0, None), "q02-b": entry("partial", 200.0, None)},
        "new": {"q01-a": entry("completed", 90.0, 2), "q02-b": entry("completed", 150.0, 1)},
    }.items():
        (tmp_path / name).mkdir()
        results = {"created_at": name, "settings": {}, "sessions": [], "runs": runs}
        (tmp_path / name / "results.json").write_text(json.dumps(results), encoding="utf-8")
    text = compare(tmp_path / "base", tmp_path / "new")
    assert "| number_mismatch (made unsupported) | – | 3 | – |" in text  # not recorded in the baseline
    assert "| completed | 1 | 2 | +1 |" in text and "| partial | 1 | 0 | -1 |" in text
    assert "| total time, timed runs (s) | 300 | 240 | -60 |" in text
    assert "| q02-b | partial → completed **changed** |" in text
    assert "Status changes: q02-b: partial → completed." in text
    assert (tmp_path / "new" / "compare.md").read_text(encoding="utf-8") == text
