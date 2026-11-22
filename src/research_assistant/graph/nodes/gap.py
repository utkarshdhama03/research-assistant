"""gap: new search queries for sub-questions with too few supported claims; research runs them next."""

from research_assistant.db.repositories.runs import append_search_queries
from research_assistant.graph.context import NodeResult, RunContext
from research_assistant.graph.nodes.research import load_sub_questions
from research_assistant.graph.planning import filter_new_queries
from research_assistant.graph.state import GraphState
from research_assistant.llm.errors import LLMError
from research_assistant.llm.prompt_loader import load_prompt
from research_assistant.llm.schemas import GapQueries


def gap_node(ctx: RunContext, state: GraphState) -> NodeResult:
    s = ctx.settings
    all_queries = [q for sq in load_sub_questions(ctx, state.sub_question_ids) for q in sq.queries]
    active = []
    added = 0
    for sq in load_sub_questions(ctx, state.gap_targets):
        ctx.ensure_not_stopped()
        if ctx.now() >= ctx.verify_deadline_at:
            ctx.issue(f"gap round skipped for sub-question {sq.ordinal + 1}: deadline reached")
            continue
        prompt = load_prompt("gap").render(
            question=ctx.question, sub_question=sq.text, tried="\n".join(f"- {q}" for q in sq.queries)
        )
        try:
            result = ctx.llm.generate_structured(prompt, GapQueries, num_predict=s.gap_num_predict)
        except LLMError as e:
            ctx.event(
                "gap_queries", {"sub_question_id": str(sq.id), "error": type(e).__name__, "queries": []}
            )
            continue
        kept, dropped = filter_new_queries(result.value.queries, all_queries, s.query_jaccard_threshold)
        ctx.event(
            "gap_queries",
            {"sub_question_id": str(sq.id), "ordinal": sq.ordinal + 1, "queries": kept, "dropped": dropped},
        )
        if not kept:
            continue
        with ctx.db() as session:
            append_search_queries(session, sq.id, kept)
        all_queries.extend(kept)
        active.append(sq.id)
        added += len(kept)
    return NodeResult(
        {"active_sub_question_ids": active, "gap_targets": [], "gap_rounds": state.gap_rounds + 1},
        {"round": state.gap_rounds + 1, "sub_questions": len(active), "queries_added": added},
    )
