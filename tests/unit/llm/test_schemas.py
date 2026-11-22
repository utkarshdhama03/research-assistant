from typing import Any

import pytest
from pydantic import ValidationError

from research_assistant.config import Settings
from research_assistant.llm.schemas import MAX_SUB_QUESTIONS, MIN_SUB_QUESTIONS, ResearchPlan


def sub(question: str, *queries: str) -> dict[str, Any]:
    return {"question": question, "search_queries": list(queries)}


def plan(n: int = 3, queries: tuple[str, ...] = ("q one",)) -> dict[str, Any]:
    return {"sub_questions": [sub(f"Sub-question number {i}?", *queries) for i in range(n)]}


@pytest.mark.parametrize("n", [3, 4, 5])
def test_valid_plan_sizes(n: int) -> None:
    assert len(ResearchPlan.model_validate(plan(n)).sub_questions) == n


@pytest.mark.parametrize("n", [0, 2, 6])
def test_sub_question_count_bounds(n: int) -> None:
    with pytest.raises(ValidationError):
        ResearchPlan.model_validate(plan(n))


@pytest.mark.parametrize("queries", [(), ("a b", "c d", "e f", "g h")])
def test_query_count_bounds(queries: tuple[str, ...]) -> None:
    with pytest.raises(ValidationError):
        ResearchPlan.model_validate(plan(3, queries))


@pytest.mark.parametrize(
    "data",
    [
        plan(3, ("okapi habitat", "Okapi  HABITAT")),  # duplicate query, ignoring case/spacing
        {"sub_questions": [sub("Same question?", "a b"), sub("same question?", "c d"), sub("Other?", "e f")]},
        plan(3, ("   ",)),  # blank query
        {"sub_questions": [sub("  ", "a b")] + plan(2)["sub_questions"]},  # blank question
        {**plan(3), "notes": "extra field"},
        {"sub_questions": [{**sub("Question one?", "a b"), "answer": "x"}] + plan(2)["sub_questions"]},
    ],
)
def test_invalid_plans(data: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        ResearchPlan.model_validate(data)


def test_whitespace_is_stripped() -> None:
    p = ResearchPlan.model_validate(plan(3, ("  okapi stripes  ",)))
    assert p.sub_questions[0].search_queries == ["okapi stripes"]


def test_config_max_sub_questions_matches_schema_bounds() -> None:
    for ok in range(MIN_SUB_QUESTIONS, MAX_SUB_QUESTIONS + 1):
        assert Settings(_env_file=None, max_sub_questions=ok).max_sub_questions == ok
    for bad in (MIN_SUB_QUESTIONS - 1, MAX_SUB_QUESTIONS + 1):
        with pytest.raises(ValidationError):
            Settings(_env_file=None, max_sub_questions=bad)


# -- select schema ---------------------------------------------------------------------------


def test_selection_schema_enum_and_bounds() -> None:
    from research_assistant.llm.schemas import selection_schema

    model = selection_schema(["R1", "R2", "R3"], 2)
    items = model.model_json_schema()["properties"]["selected"]
    assert items["items"]["enum"] == ["R1", "R2", "R3"] and (items["minItems"], items["maxItems"]) == (1, 2)
    assert model.model_validate_json('{"selected": ["R3", "R1"]}').selected == ["R3", "R1"]
    for bad in (
        '{"selected": ["R4"]}',
        '{"selected": ["R1", "R1"]}',
        '{"selected": []}',
        '{"selected": ["R1","R2","R3"]}',
    ):
        with pytest.raises(ValidationError):
            model.model_validate_json(bad)


@pytest.mark.parametrize(("aliases", "max_picks"), [([], 1), (["R1"], 0), (["R1"], 2)])
def test_selection_schema_rejects_bad_arguments(aliases: list[str], max_picks: int) -> None:
    from research_assistant.llm.schemas import selection_schema

    with pytest.raises(ValueError):
        selection_schema(aliases, max_picks)


def _plan(*flags: bool) -> dict[str, object]:
    return {
        "sub_questions": [
            {
                "question": f"Distinct sub-question {chr(97 + i)}?",
                "search_queries": [f"query {chr(97 + i)}"],
                "premise_check": f,
            }
            for i, f in enumerate(flags)
        ]
    }


def test_premise_check_only_on_the_first_sub_question() -> None:
    assert ResearchPlan.model_validate(_plan(True, False, False)).sub_questions[0].premise_check is True
    assert not any(
        s.premise_check for s in ResearchPlan.model_validate(_plan(False, False, False)).sub_questions
    )
    for flags in ((False, True, False), (True, True, False), (False, False, True)):
        with pytest.raises(ValidationError, match="premise_check"):
            ResearchPlan.model_validate(_plan(*flags))


def test_premise_check_defaults_to_false() -> None:
    plan = ResearchPlan.model_validate(
        {"sub_questions": [{"question": f"Sub-question {c}?", "search_queries": [f"q {c}"]} for c in "abc"]}
    )
    assert [s.premise_check for s in plan.sub_questions] == [False, False, False]
