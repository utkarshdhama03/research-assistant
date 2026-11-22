from research_assistant.graph.planning import finalize_plan
from research_assistant.llm.schemas import ResearchPlan


def test_finalize_plan_keeps_the_premise_flag() -> None:
    plan = ResearchPlan.model_validate(
        {
            "sub_questions": [
                {"question": "Was X moved to Y?", "search_queries": ["x moved y"], "premise_check": True},
                {"question": "Where is X today?", "search_queries": ["x location"]},
                {"question": "What is X made of?", "search_queries": ["x material"]},
            ]
        }
    )
    planned, _ = finalize_plan(plan, 5, 0.6)
    assert [p.premise_check for p in planned] == [True, False, False]
