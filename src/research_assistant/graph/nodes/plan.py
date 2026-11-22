"""plan: question -> 3-5 stored sub-questions with near-duplicate queries removed."""

from research_assistant.db.repositories.runs import add_sub_questions
from research_assistant.graph.context import NodeResult, RunContext
from research_assistant.graph.planning import finalize_plan
from research_assistant.graph.state import GraphState
from research_assistant.llm.errors import LLMError
from research_assistant.llm.prompt_loader import load_prompt
from research_assistant.llm.schemas import ResearchPlan


def plan_node(ctx: RunContext, state: GraphState) -> NodeResult:
    s = ctx.settings
    prompt = load_prompt("plan").render(question=ctx.question)
    try:
        result = ctx.llm.generate_structured(prompt, ResearchPlan, num_predict=s.plan_num_predict)
    except LLMError as e:
        ctx.plan_failure = f"plan failed: {type(e).__name__}: {e}"
        return NodeResult({}, {"sub_questions": 0, "error": type(e).__name__})

    planned, dropped = finalize_plan(result.value, s.max_sub_questions, s.query_jaccard_threshold)
    if dropped:
        ctx.event("queries_deduplicated", {"dropped": dropped, "threshold": s.query_jaccard_threshold})
    with ctx.db() as session:
        rows = add_sub_questions(
            session,
            state.run_id,
            [p.question for p in planned],
            [p.queries for p in planned],
            [p.premise_check for p in planned],
        )
        ids = [row.id for row in rows]
    return NodeResult(
        {"sub_question_ids": ids, "active_sub_question_ids": ids, "queries_searched": dict.fromkeys(ids, 0)},
        {
            "sub_questions": len(ids),
            "queries": sum(len(p.queries) for p in planned),
            "near_duplicate_queries_dropped": len(dropped),
            "trimmed": max(0, len(result.value.sub_questions) - len(planned)),
        },
    )
