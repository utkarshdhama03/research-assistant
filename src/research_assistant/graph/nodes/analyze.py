"""analyze: per active sub-question, run-wide retrieval -> claims citing C1..Cn (stored unverified)."""

import uuid

from research_assistant.db.repositories.chunks import ChunkHit, nearest_chunks
from research_assistant.db.repositories.claims import add_claims
from research_assistant.graph.context import NodeResult, RunContext
from research_assistant.graph.nodes.research import load_sub_questions
from research_assistant.graph.state import GraphState
from research_assistant.llm.aliases import AliasMap
from research_assistant.llm.errors import LLMError
from research_assistant.llm.prompt_loader import load_prompt, untrusted
from research_assistant.llm.schemas import claims_schema
from research_assistant.tools.embed import EmbedError


def evidence_block(hits: list[ChunkHit], aliases: AliasMap[uuid.UUID]) -> str:
    from urllib.parse import urlsplit

    return "\n\n".join(
        untrusted(
            f"Source: {urlsplit(h.final_url or h.url).hostname or ''}\n{h.text}", aliases.alias(h.chunk_id)
        )
        for h in hits
    )


def analyze_node(ctx: RunContext, state: GraphState) -> NodeResult:
    s = ctx.settings
    deadline = min(ctx.verify_deadline_at, ctx.now() + s.node_timeout_s)
    analyzed = {k: list(v) for k, v in state.analyzed_chunk_ids.items()}
    claim_chunks = {k: list(v) for k, v in state.claim_chunks.items()}
    created = 0
    for sq in load_sub_questions(ctx, state.active_sub_question_ids):
        ctx.ensure_not_stopped()
        if ctx.now() >= deadline:
            ctx.issue(
                f"sub-question {sq.ordinal + 1} not analyzed: deadline reached", sub_question_id=str(sq.id)
            )
            continue
        shown_before = set(analyzed.get(sq.id, []))
        try:
            query_vector = ctx.deps.embedder.embed_query(sq.text)
        except EmbedError as e:
            ctx.issue(
                f"sub-question {sq.ordinal + 1} not analyzed: embedding failed ({e})",
                sub_question_id=str(sq.id),
            )
            continue
        with ctx.db() as session:
            hits = nearest_chunks(session, state.run_id, query_vector, k=s.analyze_top_k + len(shown_before))
        hits = [h for h in hits if h.chunk_id not in shown_before][: s.analyze_top_k]
        if not hits:
            ctx.event("analyze_no_new_evidence", {"sub_question_id": str(sq.id)})
            continue
        analyzed.setdefault(sq.id, []).extend(h.chunk_id for h in hits)
        aliases = AliasMap([h.chunk_id for h in hits], prefix="C")
        prompt = load_prompt("analyze").render(
            question=ctx.question,
            sub_question=sq.text,
            max_claims=str(s.max_claims_per_sub_question),
            labels=", ".join(aliases.aliases),
            evidence=evidence_block(hits, aliases),
        )
        try:
            result = ctx.llm.generate_structured(
                prompt,
                claims_schema(aliases.aliases, s.max_claims_per_sub_question),
                num_predict=s.analyze_num_predict,
            )
            cited = [aliases.resolve_all(c.evidence) for c in result.value.claims]
        except LLMError as e:
            ctx.issue(
                f"sub-question {sq.ordinal + 1}: analyze failed ({type(e).__name__})",
                sub_question_id=str(sq.id),
            )
            continue
        with ctx.db() as session:
            rows = add_claims(session, state.run_id, sq.id, [c.text for c in result.value.claims])
            ids = [(r.id, r.ordinal) for r in rows]
        for (claim_id, _), chunk_ids in zip(ids, cited, strict=True):
            claim_chunks[claim_id] = chunk_ids
        created += len(ids)
        ctx.event(
            "claims_created",
            {
                "sub_question_id": str(sq.id),
                "claims": [
                    {"claim_id": str(cid), "ordinal": ordinal, "chunk_ids": [str(c) for c in chunks]}
                    for (cid, ordinal), chunks in zip(ids, cited, strict=True)
                ],
            },
        )
    return NodeResult(
        {"analyzed_chunk_ids": analyzed, "claim_chunks": claim_chunks}, {"claims_created": created}
    )
