"""write: claim dedupe, then one section per sub-question from its supported claims, then the report."""

import uuid
from collections import Counter
from datetime import UTC

from sqlalchemy import select

from research_assistant.db.models import Chunk, Claim, Source, Verdict
from research_assistant.db.repositories.claims import evidence_for_claims
from research_assistant.db.repositories.reports import save_report
from research_assistant.graph.citations import (
    CitationError,
    ClaimRef,
    SectionContent,
    SourceRef,
    assemble_report,
    check_sentences,
    parse_section,
    remove_uncited,
)
from research_assistant.graph.context import NodeResult, RunContext
from research_assistant.graph.nodes.research import load_sub_questions
from research_assistant.graph.state import GraphState
from research_assistant.graph.verification import dedupe_claims, most_similar_pairs
from research_assistant.llm.aliases import AliasMap
from research_assistant.llm.errors import LLMError
from research_assistant.llm.prompt_loader import load_prompt, untrusted
from research_assistant.tools.embed import EmbedError

NO_OWN_SOURCES_NOTE = (
    "The searches for this sub-question found no usable sources; "
    "this section draws on sources found for the other sub-questions."
)
DEADLINE_NOTE = "This section was not written: the run's time limit was reached."
INSUFFICIENT_NOTE = "There is insufficient verified evidence to answer this sub-question."
COVERED_NOTE = "The verified findings for this sub-question are covered in the sections above."
# For the premise-check sub-question with no supported claim: its section, and a note under the question.
PREMISE_NOTE = "No verified source supports the premise of this question."


def write_node(ctx: RunContext, state: GraphState) -> NodeResult:
    deadline = min(ctx.run_deadline_at, ctx.now() + ctx.settings.node_timeout_s)
    claims = _supported_claims(ctx, state)
    kept = _dedupe(ctx, claims)
    sections: list[SectionContent] = []
    failed = deleted_total = 0
    counts: Counter[str] = Counter()  # sentences checked / replaced by the content check
    premise_unsupported = False
    for sq in load_sub_questions(ctx, state.sub_question_ids):
        ctx.ensure_not_stopped()
        mine = [c for c in claims.values() if c.sub_question_id == sq.id]
        usable = [c for c in mine if c.claim_id in kept]
        if not mine and sq.premise_check:  # the premise is not supported: reported, not a problem
            premise_unsupported = True
            ctx.event("premise_unsupported", {"sub_question_id": str(sq.id)})
            sections.append(SectionContent(sq.text, note=PREMISE_NOTE, kind="insufficient"))
            continue
        if not mine:
            ctx.issue(f"sub-question {sq.ordinal + 1}: no supported claims", sub_question_id=str(sq.id))
            sections.append(SectionContent(sq.text, note=INSUFFICIENT_NOTE, kind="insufficient"))
            continue
        if not usable:
            sections.append(SectionContent(sq.text, note=COVERED_NOTE, kind="covered"))
            continue
        if ctx.now() >= deadline:
            sections.append(SectionContent(sq.text, note=DEADLINE_NOTE, kind="failed"))
            ctx.issue(f"section {sq.ordinal + 1} not written: deadline reached", sub_question_id=str(sq.id))
            failed += 1
            continue
        section, deleted = _write_section(ctx, sq.id, sq.ordinal, sq.text, usable, counts)
        deleted_total += deleted
        if not section.segments:
            failed += 1
        elif not state.sources_by_sub_question.get(sq.id):
            section.note = NO_OWN_SOURCES_NOTE  # already recorded as a problem by research
            section.kind = "no_sources"
        sections.append(section)

    refs: dict[uuid.UUID, SourceRef] = {}
    with ctx.db() as session:
        for source_id in {c.ref.source_id for c in claims.values()}:
            src = session.get(Source, source_id)
            if src is not None:
                retrieved = src.fetched_at.astimezone(UTC).date() if src.fetched_at else None
                refs[source_id] = SourceRef(
                    title=src.title or src.url,
                    url=src.final_url or src.url,
                    retrieved=retrieved,
                    reused=src.reused_from_source_id is not None,
                )
    report = assemble_report(
        ctx.question,
        sections,
        {c.claim_id: c.ref for c in claims.values()},
        refs,
        premise_note=PREMISE_NOTE if premise_unsupported else None,
    )
    with ctx.db() as session:
        report_id = save_report(session, state.run_id, report.markdown, report.citations, report.structure).id
    return NodeResult(
        {"report_id": report_id},
        {
            "sections": len(sections),
            "sections_failed": failed,
            "supported_claims": len(claims),
            "claims_after_dedupe": len(kept),
            "claims_cited": len(report.cited_claims),
            "citations": len(report.citations),
            "uncited_sentences_deleted": deleted_total,
            "sentences_checked": counts["sentences_checked"],
            "sentences_replaced": counts["sentences_replaced"],
        },
    )


class _Supported:
    def __init__(self, claim_id: uuid.UUID, sub_question_id: uuid.UUID, ref: ClaimRef) -> None:
        self.claim_id = claim_id
        self.sub_question_id = sub_question_id
        self.ref = ref


def _supported_claims(ctx: RunContext, state: GraphState) -> dict[uuid.UUID, _Supported]:
    """Supported claims of the run in ordinal order, with the chunk and source of their quote."""
    out: dict[uuid.UUID, _Supported] = {}
    with ctx.db() as session:
        rows = list(
            session.scalars(
                select(Claim)
                .where(Claim.run_id == state.run_id, Claim.verdict == Verdict.SUPPORTED)
                .order_by(Claim.ordinal)
            )
        )
        evidence = evidence_for_claims(session, [c.id for c in rows])
        for claim in rows:
            ev = evidence.get(claim.id)
            if not ev:  # cannot happen: supported requires a located quote
                continue
            chunk = session.get(Chunk, ev[0].chunk_id)
            if chunk is None:
                continue
            out[claim.id] = _Supported(
                claim.id,
                claim.sub_question_id,
                ClaimRef(
                    claim.ordinal, claim.text, ev[0].quote, chunk.id, chunk.source_id, str(claim.verdict)
                ),
            )
    return out


def _dedupe(ctx: RunContext, claims: dict[uuid.UUID, _Supported]) -> set[uuid.UUID]:
    ids = list(claims)
    if len(ids) < 2:
        return set(ids)
    try:
        vectors = ctx.deps.embedder.embed_for_clustering([claims[i].ref.text for i in ids])
    except EmbedError as e:
        ctx.event("claims_deduplicated", {"error": str(e), "dropped": []})
        return set(ids)
    kept, dropped = dedupe_claims(ids, vectors, ctx.settings.claim_dedupe_threshold)
    ordinal = {i: claims[i].ref.ordinal for i in ids}
    ctx.event(
        "claims_deduplicated",
        {
            "threshold": ctx.settings.claim_dedupe_threshold,
            "dropped": [
                {
                    "claim": ordinal[d.claim_id],
                    "duplicate_of": ordinal[d.duplicate_of],
                    "similarity": d.similarity,
                    "text": claims[d.claim_id].ref.text,
                }
                for d in dropped
            ],
        },
    )
    ctx.event(
        "claim_similarity_top",
        {
            "pairs": [
                {"a": ordinal[a], "b": ordinal[b], "similarity": round(sim, 4)}
                for a, b, sim in most_similar_pairs(ids, vectors, 10)
            ]
        },
    )
    return set(kept)


def _write_section(
    ctx: RunContext,
    sq_id: uuid.UUID,
    ordinal: int,
    text: str,
    usable: list[_Supported],
    counts: Counter[str],
) -> tuple[SectionContent, int]:
    s = ctx.settings
    aliases = AliasMap([c.claim_id for c in usable], prefix="K")
    claims_block = "\n\n".join(untrusted(c.ref.text, aliases.alias(c.claim_id)) for c in usable)
    labels = ", ".join(aliases.aliases)
    correction = ""
    for attempt in range(2):  # one regeneration on invalid citations
        prompt = load_prompt("write_section").render(
            question=ctx.question,
            sub_question=text,
            labels=labels,
            claims=claims_block,
            correction=correction,
        )
        try:
            result = ctx.llm.generate_text(prompt, num_predict=s.write_num_predict)
        except LLMError as e:
            return _failed(ctx, sq_id, ordinal, text, f"the model call failed ({type(e).__name__})"), 0
        try:
            segments = parse_section(result.value.strip(), aliases)
        except CitationError as e:
            ctx.event(
                "citation_error",
                {"sub_question_id": str(sq_id), "attempt": attempt + 1, "unknown": e.unknown, "urls": e.urls},
            )
            correction = (
                f"\nYour previous answer was rejected because it contained {e}. "
                f"Cite only with these labels: {labels}. Do not write URLs or numbered citations."
            )
            continue
        cleaned = remove_uncited(segments)
        segments, deleted = cleaned.segments, cleaned.deleted
        if deleted:
            ctx.event(
                "uncited_sentences_deleted",
                {
                    "sub_question_id": str(sq_id),
                    "section": ordinal + 1,
                    "count": len(deleted),
                    "sentences": deleted,
                },
            )
        if cleaned.trimmed:
            ctx.event(
                "trailing_text_removed",
                {
                    "sub_question_id": str(sq_id),
                    "section": ordinal + 1,
                    "count": len(cleaned.trimmed),
                    "fragments": cleaned.trimmed,
                },
            )
        if not segments:
            return _failed(ctx, sq_id, ordinal, text, "no sentence carried a claim marker"), len(deleted)
        segments, replaced, checked = check_sentences(
            segments,
            {c.claim_id: c.ref.text for c in usable},
            phrases=s.speculation_phrases,
            ignored=[*s.sentence_stopwords, *s.sentence_connectives],
            label=aliases.alias,
        )
        counts["sentences_checked"] += checked
        for r in replaced:
            payload = {
                "sub_question_id": str(sq_id),
                "section": ordinal + 1,
                "original": r.original,
                "replacement": r.replacement,
                "deduplicated": r.deduplicated,  # True: the replacement repeated an earlier sentence, dropped
            }
            if r.kind == "speculation":
                ctx.event("speculation_replaced", {**payload, "phrases": r.phrases})
            else:
                counts["sentences_replaced"] += 1
                ctx.event(
                    "sentence_replaced",
                    {**payload, "unmatched_words": r.words, "unmatched_numbers": r.numbers},
                )
        ctx.event(
            "section_citations",
            {
                "sub_question_id": str(sq_id),
                "segments": [
                    seg if isinstance(seg, str) else {"cite": [str(c) for c in seg.ids]} for seg in segments
                ],
            },
        )
        return SectionContent(text, segments), len(deleted)
    return _failed(ctx, sq_id, ordinal, text, "the model cited claims that do not exist, twice"), 0


def _failed(ctx: RunContext, sq_id: uuid.UUID, ordinal: int, text: str, reason: str) -> SectionContent:
    ctx.event("section_failed", {"sub_question_id": str(sq_id), "reason": reason})
    ctx.issue(f"section {ordinal + 1} could not be written: {reason}", sub_question_id=str(sq_id))
    return SectionContent(text, note=f"This section could not be written: {reason}.", kind="failed")


def verdict_counts(ctx: RunContext, run_id: uuid.UUID) -> dict[str, object]:
    """Claims by verdict, overall and per sub-question ordinal (for the run_finished event)."""
    from research_assistant.db.models import SubQuestion

    with ctx.db() as session:
        rows = session.execute(
            select(SubQuestion.ordinal, Claim.verdict)
            .join(SubQuestion, SubQuestion.id == Claim.sub_question_id)
            .where(Claim.run_id == run_id)
        ).all()
    overall: Counter[str] = Counter()
    per: dict[str, Counter[str]] = {}
    for sq_ordinal, verdict in rows:
        key = str(verdict) if verdict is not None else "unverified"
        overall[key] += 1
        per.setdefault(str(sq_ordinal + 1), Counter())[key] += 1
    return {"overall": dict(overall), "per_sub_question": {k: dict(v) for k, v in per.items()}}
