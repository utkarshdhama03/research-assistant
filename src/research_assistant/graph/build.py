"""Graph: plan -> approve -> research -> analyze -> verify
(-> gap -> research -> analyze -> verify) -> write.

The approve node pauses for plan approval unless the run has auto_approve."""

from collections.abc import Callable
from typing import Any, Protocol

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.errors import GraphBubbleUp
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import interrupt

from research_assistant.db.repositories.runs import list_sub_questions
from research_assistant.graph.context import NodeResult, RunContext, RunStopped
from research_assistant.graph.nodes.analyze import analyze_node
from research_assistant.graph.nodes.gap import gap_node
from research_assistant.graph.nodes.plan import plan_node
from research_assistant.graph.nodes.research import research_node
from research_assistant.graph.nodes.verify import verify_node
from research_assistant.graph.nodes.write import write_node
from research_assistant.graph.state import GraphState

NodeFn = Callable[[RunContext, GraphState], NodeResult]


class GraphNode(Protocol):
    """LangGraph's node protocol names the parameter `state`."""

    def __call__(self, state: GraphState) -> dict[str, Any]: ...


def instrumented(ctx: RunContext, name: str, fn: NodeFn) -> GraphNode:
    """Check for cancel/stop, log start/end (with duration and summary) or error, validate the update.

    LangGraph applies node updates to a Pydantic state without validating them, so the merged state
    is validated here before the update is returned.
    """

    def node(state: GraphState) -> dict[str, Any]:
        ctx.current_node = name
        ctx.ensure_not_stopped()
        started = ctx.now()
        ctx.event("node_start")
        try:
            result = fn(ctx, state)
            merged = GraphState.model_validate({**state.model_dump(), **result.update})
        except GraphBubbleUp:  # an interrupt, not an error
            raise
        except RunStopped as e:
            ctx.event("node_stopped", {"reason": str(e), "duration_s": round(ctx.now() - started, 3)})
            raise
        except Exception as e:
            ctx.event(
                "node_error",
                {"error": f"{type(e).__name__}: {e}", "duration_s": round(ctx.now() - started, 3)},
            )
            raise
        ctx.event("node_end", {"duration_s": round(ctx.now() - started, 3), **result.summary})
        return {key: getattr(merged, key) for key in result.update}

    return node


def approval(ctx: RunContext) -> GraphNode:
    """Pause after planning until the plan is approved (POST /runs/{id}/approve), unless auto_approve.

    On resume LangGraph runs this node again from the start, so nothing with side effects happens
    before interrupt(). The API may have replaced the sub-questions while the run was paused, so they
    are reloaded from the database.
    """

    def node(state: GraphState) -> dict[str, Any]:
        ctx.current_node = "approve"
        if ctx.auto_approve:
            ctx.event("plan_auto_approved", {"sub_questions": len(state.sub_question_ids)})
            return {}
        interrupt({"run_id": str(state.run_id)})
        ctx.ensure_not_stopped()
        with ctx.db() as session:
            ids = [sq.id for sq in list_sub_questions(session, state.run_id)]
        ctx.event("plan_resumed", {"sub_questions": len(ids)})
        return {
            "sub_question_ids": ids,
            "active_sub_question_ids": ids,
            "queries_searched": dict.fromkeys(ids, 0),
        }

    return node


def after_plan(state: GraphState) -> str:
    return "approve" if state.sub_question_ids else "end"


def after_research(state: GraphState) -> str:
    return "analyze" if any(state.sources_by_sub_question.values()) else "end"


def after_verify(state: GraphState) -> str:
    return "gap" if state.gap_targets else "write"


def build_graph(
    ctx: RunContext, checkpointer: BaseCheckpointSaver[Any] | None = None
) -> CompiledStateGraph[Any, Any, Any, Any]:
    graph = StateGraph(GraphState)
    graph.add_node("plan", instrumented(ctx, "plan", plan_node))
    graph.add_node("approve", approval(ctx))
    graph.add_node("research", instrumented(ctx, "research", research_node))
    graph.add_node("analyze", instrumented(ctx, "analyze", analyze_node))
    graph.add_node("verify", instrumented(ctx, "verify", verify_node))
    graph.add_node("gap", instrumented(ctx, "gap", gap_node))
    graph.add_node("write", instrumented(ctx, "write", write_node))
    graph.add_edge(START, "plan")
    graph.add_conditional_edges("plan", after_plan, {"approve": "approve", "end": END})
    graph.add_edge("approve", "research")
    graph.add_conditional_edges("research", after_research, {"analyze": "analyze", "end": END})
    graph.add_edge("analyze", "verify")
    graph.add_conditional_edges("verify", after_verify, {"gap": "gap", "write": "write"})
    graph.add_edge("gap", "research")
    graph.add_edge("write", END)
    return graph.compile(checkpointer=checkpointer)
