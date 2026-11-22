import importlib
import os

import pytest

from research_assistant.db.models import RunStatus
from research_assistant.graph.planning import finalize_plan
from research_assistant.graph.status import decide_status
from research_assistant.llm.schemas import ResearchPlan

# -- status rules ----------------------------------------------------------------------------


def test_completed() -> None:
    assert decide_status(plan_failure=None, total_chunks=10, has_report=True, issues=[]) == (
        RunStatus.COMPLETED,
        None,
    )


def test_partial_lists_every_issue() -> None:
    status, reason = decide_status(plan_failure=None, total_chunks=10, has_report=True, issues=["a", "b"])
    assert (status, reason) == (RunStatus.PARTIAL, "a; b")


def test_plan_failure_wins() -> None:
    status, reason = decide_status(
        plan_failure="plan failed: X", total_chunks=10, has_report=True, issues=["a"]
    )
    assert (status, reason) == (RunStatus.FAILED, "plan failed: X")


@pytest.mark.parametrize("has_report", [True, False])
def test_zero_chunks_fails(has_report: bool) -> None:
    status, reason = decide_status(plan_failure=None, total_chunks=0, has_report=has_report, issues=["x"])
    assert status is RunStatus.FAILED
    assert reason == "no chunks were stored in this run (x)"


def test_missing_report_fails() -> None:
    status, reason = decide_status(plan_failure=None, total_chunks=3, has_report=False, issues=[])
    assert (status, reason) == (RunStatus.FAILED, "no report was written")


# -- plan finishing --------------------------------------------------------------------------


def plan(*subs: tuple[str, list[str]]) -> ResearchPlan:
    return ResearchPlan.model_validate(
        {"sub_questions": [{"question": q, "search_queries": qs} for q, qs in subs]}
    )


def test_dedupes_queries_across_the_plan() -> None:
    p = plan(
        ("What is A?", ["alpha basics", "alpha history"]),
        ("What is B?", ["ALPHA  HISTORY", "beta overview", "Alpha Basics"]),
        ("What is C?", ["gamma", "beta   OVERVIEW"]),
    )
    planned, dropped = finalize_plan(p, 5)
    assert [s.queries for s in planned] == [["alpha basics", "alpha history"], ["beta overview"], ["gamma"]]
    assert dropped == ["ALPHA  HISTORY", "Alpha Basics", "beta   OVERVIEW"]


def test_sub_question_without_queries_falls_back_to_its_text() -> None:
    p = plan(("What is A?", ["shared"]), ("What is B?", ["Shared"]), ("What is C?", ["cc"]))
    planned, _ = finalize_plan(p, 5)
    assert planned[1].queries == ["What is B?"]


def test_trims_to_max_before_deduping() -> None:
    p = plan(*[(f"Question {i}?", [f"q{i}"]) for i in range(5)])
    planned, dropped = finalize_plan(p, 3)
    assert [s.question for s in planned] == ["Question 0?", "Question 1?", "Question 2?"] and dropped == []


# -- tracing ---------------------------------------------------------------------------------


def test_importing_graph_forces_langsmith_tracing_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")
    import research_assistant.graph as graph

    importlib.reload(graph)
    assert os.environ["LANGSMITH_TRACING"] == "false"
    assert os.environ["LANGCHAIN_TRACING_V2"] == "false"
