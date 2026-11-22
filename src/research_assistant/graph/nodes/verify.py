"""verify: one call per unverified claim, showing only the chunks it cited; then the gap decision."""

import uuid
from collections import Counter

from sqlalchemy import select

from research_assistant.db.models import Chunk, Claim, SubQuestion, Verdict
from research_assistant.db.repositories.claims import add_claim_evidence, update_claim_verdict
from research_assistant.db.repositories.sources import count_sources
from research_assistant.graph.context import NodeResult, RunContext
from research_assistant.graph.nodes.research import load_sub_questions
from research_assistant.graph.numbers import extract_numbers, missing_numbers
from research_assistant.graph.state import GraphState
from research_assistant.graph.verification import (
    NUMBER_NOT_IN_QUOTE,
    QUOTE_NOT_FOUND,
    final_verdict,
    locate_quote,
)
from research_assistant.llm.aliases import AliasMap
from research_assistant.llm.errors import LLMError
from research_assistant.llm.prompt_loader import load_prompt, untrusted
from research_assistant.llm.schemas import verification_schema


def verify_node(ctx: RunContext, state: GraphState) -> NodeResult:
    s = ctx.settings
    deadline = min(ctx.verify_deadline_at, ctx.now() + s.node_timeout_s)
    with ctx.db() as session:
        pending = [
            (c.id, c.ordinal, c.text)
            for c in session.scalars(
                select(Claim)
                .where(Claim.run_id == state.run_id, Claim.verdict.is_(None))
                .order_by(Claim.ordinal)
            )
        ]
    verdicts: Counter[str] = Counter()
    not_verified = 0
    for claim_id, ordinal, text in pending:
        ctx.ensure_not_stopped()
        if ctx.now() >= deadline:
            _set_verdict(ctx, claim_id, Verdict.UNSUPPORTED, "not_verified: deadline")
            not_verified += 1
            verdicts[str(Verdict.UNSUPPORTED)] += 1
            continue
        verdicts[str(_verify_claim(ctx, state, claim_id, ordinal, text))] += 1
    if not_verified:
        ctx.issue(f"{not_verified} claim(s) not verified: deadline reached")

    gap_targets = _gap_decision(ctx, state)
    return NodeResult(
        {"gap_targets": gap_targets}, {"verified": len(pending), **verdicts, "gap_targets": len(gap_targets)}
    )


def _verify_claim(
    ctx: RunContext, state: GraphState, claim_id: uuid.UUID, ordinal: int, text: str
) -> Verdict:
    chunk_ids = state.claim_chunks.get(claim_id, [])
    with ctx.db() as session:
        chunks = [(c.id, c.text) for cid in chunk_ids if (c := session.get(Chunk, cid)) is not None]
        claim_row = session.get(Claim, claim_id)
        sub_question = session.get(SubQuestion, claim_row.sub_question_id) if claim_row is not None else None
        context = " ".join([ctx.question, sub_question.text if sub_question is not None else ""])
    if not chunks:
        _set_verdict(ctx, claim_id, Verdict.UNSUPPORTED, "no_cited_chunks")
        return Verdict.UNSUPPORTED
    aliases = AliasMap([cid for cid, _ in chunks], prefix="C")
    text_by_id = dict(chunks)
    evidence = "\n\n".join(untrusted(chunk_text, aliases.alias(cid)) for cid, chunk_text in chunks)
    prompt = load_prompt("verify").render(claim=text, evidence=evidence)
    try:
        result = ctx.llm.generate_structured(
            prompt, verification_schema(aliases.aliases), num_predict=ctx.settings.verify_num_predict
        )
        chosen = aliases.resolve(result.value.evidence_alias)
    except LLMError as e:
        _set_verdict(ctx, claim_id, Verdict.UNSUPPORTED, f"verify_failed: {type(e).__name__}")
        ctx.issue(f"claim {ordinal}: verification failed ({type(e).__name__})", claim_id=str(claim_id))
        return Verdict.UNSUPPORTED

    v = result.value
    match = locate_quote(text_by_id[chosen], v.quote)
    verdict, override = final_verdict(Verdict(v.verdict), match)
    missing: list[str] = []
    if match is not None and verdict in (Verdict.SUPPORTED, Verdict.PARTIAL):
        # Every number in the claim must occur in the located quote (ranges exactly).
        # Numbers that the question or sub-question already states are context, not new facts.
        exempt = extract_numbers(context).values
        missing = missing_numbers(text, text_by_id[chosen][match.start : match.end], exempt)
        if missing:
            verdict, override = Verdict.UNSUPPORTED, f"{NUMBER_NOT_IN_QUOTE}: {', '.join(missing)}"
    with ctx.db() as session:
        if match is not None:
            chunk_text = text_by_id[chosen]
            add_claim_evidence(
                session, claim_id, chosen, chunk_text[match.start : match.end], match.start, match.end
            )
        update_claim_verdict(session, claim_id, verdict, override or v.rationale)
    payload = {
        "claim_id": str(claim_id),
        "ordinal": ordinal,
        "claim": text,  # the UI's verify log shows each claim with its verdict
        "model_verdict": v.verdict,
        "verdict": str(verdict),
        "quote_match": match.how if match else None,
        "match_method": match.how if match else None,  # exact / whitespace / list / fuzzy
        "model_quote": v.quote,
    }
    ctx.event("claim_verified", payload)
    if override == QUOTE_NOT_FOUND:
        ctx.event("quote_not_found", payload)
    if missing:
        ctx.event("number_mismatch", {**payload, "missing": missing, "claim": text})
    return verdict


def _set_verdict(ctx: RunContext, claim_id: uuid.UUID, verdict: Verdict, rationale: str) -> None:
    with ctx.db() as session:
        update_claim_verdict(session, claim_id, verdict, rationale)


def _gap_decision(ctx: RunContext, state: GraphState) -> list[uuid.UUID]:
    """Weak sub-questions (fewer than MIN_SUPPORTED_CLAIMS supported claims) get a gap round if rounds,
    the fetch budget and the research deadline allow. Needed-but-skipped for budget/deadline is a problem."""
    s = ctx.settings
    with ctx.db() as session:
        supported = Counter(
            session.scalars(
                select(Claim.sub_question_id).where(
                    Claim.run_id == state.run_id, Claim.verdict == Verdict.SUPPORTED
                )
            )
        )
        attempts = count_sources(session, state.run_id, reused=False)  # reused sources: own budget
    weak = [
        sq
        for sq in load_sub_questions(ctx, state.sub_question_ids)
        if supported[sq.id] < s.min_supported_claims
    ]
    if not weak or state.gap_rounds >= s.max_gap_loops:
        if weak:
            ctx.event(
                "gap_not_run", {"reason": "max_gap_loops reached", "weak": [w.ordinal + 1 for w in weak]}
            )
        return []
    blocked = None
    if attempts >= s.max_sources_per_run:
        blocked = "source budget spent"
    elif ctx.now() >= ctx.research_deadline_at:
        blocked = "research deadline reached"
    if blocked:
        ctx.issue(
            f"gap loop needed for sub-question(s) {', '.join(str(w.ordinal + 1) for w in weak)} "
            f"but skipped: {blocked}"
        )
        return []
    ctx.event(
        "gap_planned",
        {
            "round": state.gap_rounds + 1,
            "sub_questions": [w.ordinal + 1 for w in weak],
            "supported": {str(w.ordinal + 1): supported[w.id] for w in weak},
        },
    )
    return [w.id for w in weak]
