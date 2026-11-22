// What a run's events say about its progress, in the terms of the run view: four steps (analyze is
// shown as verify, gap as research), a summary per step, one line per verified claim, and problems.
// Pure functions over the event list from useRunEvents.

import type { RunEvent } from "../api/types";

export type StepName = "plan" | "approve" | "research" | "verify" | "write";
export type StepState = "pending" | "running" | "done" | "failed" | "paused";

export interface Step {
  name: StepName;
  state: StepState;
}

const MAIN_STEPS: StepName[] = ["plan", "research", "verify", "write"];
const NODE_STEP: Record<string, StepName> = {
  plan: "plan",
  research: "research",
  gap: "research",
  analyze: "verify",
  verify: "verify",
  write: "write",
};
const NODE_STATE: Record<string, StepState> = {
  node_start: "running",
  node_end: "done",
  node_error: "failed",
  node_stopped: "failed",
};
const TERMINAL = new Set(["completed", "partial", "failed", "cancelled"]);

export function stepOf(node: string): StepName | null {
  return NODE_STEP[node] ?? null;
}

/** The step line: each step's latest node event decides its state; "approve" appears while paused. */
export function steps(events: RunEvent[], status: string): Step[] {
  const state = new Map<StepName, StepState>();
  for (const e of events) {
    const step = stepOf(e.node);
    const next = NODE_STATE[e.event];
    if (step && next) state.set(step, next);
  }
  const line: Step[] = MAIN_STEPS.map((name) => {
    let s = state.get(name) ?? "pending";
    if (s === "running" && TERMINAL.has(status)) s = status === "failed" || status === "cancelled" ? "failed" : "done";
    return { name, state: s };
  });
  if (status === "awaiting_approval") line.splice(1, 0, { name: "approve", state: "paused" });
  return line;
}

// HTTP statuses after which the server stops fetching from a host (research.py BLOCKING_STATUSES).
const BLOCKING_STATUSES = new Set([403, 429]);

export interface ResearchCounts {
  searches: number;
  read: number;
  reused: number;
  blocked: number;
  failed: number;
}

export function researchCounts(events: RunEvent[]): ResearchCounts {
  const c: ResearchCounts = { searches: 0, read: 0, reused: 0, blocked: 0, failed: 0 };
  for (const e of events) {
    if (e.event === "search_done" && e.payload.role !== "fallback") c.searches += 1;
    else if (e.event === "source_fetched") c.read += 1;
    else if (e.event === "memory_reused") c.reused += 1;
    else if (e.event === "fetch_failed") {
      const code = e.payload.status_code;
      if (typeof code === "number" && BLOCKING_STATUSES.has(code)) c.blocked += 1;
      else c.failed += 1;
    }
  }
  return c;
}

/** "9 searches · 7 pages read · 2 blocked": searches and pages always, the rest when not zero. */
export function researchSummary(events: RunEvent[]): string {
  const c = researchCounts(events);
  const parts = [plural(c.searches, "search", "searches"), `${c.read} ${c.read === 1 ? "page" : "pages"} read`];
  if (c.reused) parts.push(`${c.reused} reused`);
  if (c.blocked) parts.push(`${c.blocked} blocked`);
  if (c.failed) parts.push(`${c.failed} failed`);
  return parts.join(" · ");
}

export interface ClaimLine {
  claimId: string;
  ordinal: number | null;
  ok: boolean; // supported
  text: string; // the claim, or "claim <n>" for runs recorded before claim_verified carried the text
  reason: string | null; // why it was not accepted
}

interface ClaimState {
  ordinal: number | null;
  verdict: string;
  text: string | null;
  missing: string[] | null;
  quoteNotFound: boolean;
}

function str(value: unknown): string | null {
  return typeof value === "string" && value ? value : null;
}

/** One line per verified claim (its latest verification), in the order they were first verified. */
export function claimLines(events: RunEvent[]): ClaimLine[] {
  const claims = new Map<string, ClaimState>();
  for (const e of events) {
    const id = str(e.payload.claim_id);
    if (!id) continue;
    if (e.event === "claim_verified") {
      const previous = claims.get(id);
      claims.set(id, {
        ordinal: typeof e.payload.ordinal === "number" ? e.payload.ordinal : null,
        verdict: str(e.payload.verdict) ?? "unknown",
        text: str(e.payload.claim) ?? previous?.text ?? null,
        missing: null, // a new verification: its own flags follow it
        quoteNotFound: false,
      });
    } else {
      const claim = claims.get(id);
      if (!claim) continue;
      if (e.event === "quote_not_found") claim.quoteNotFound = true;
      else if (e.event === "number_mismatch") {
        const missing = Array.isArray(e.payload.missing) ? e.payload.missing.map(String) : [];
        claim.missing = missing;
        claim.text = claim.text ?? str(e.payload.claim);
      }
    }
  }
  return [...claims.entries()].map(([claimId, c]) => ({
    claimId,
    ordinal: c.ordinal,
    ok: c.verdict === "supported",
    text: c.text ?? `claim ${c.ordinal ?? "?"}`,
    reason: c.verdict === "supported" ? null : rejectionReason(c),
  }));
}

function rejectionReason(c: ClaimState): string {
  if (c.missing) return c.missing.length ? `number not in source (${c.missing.join(", ")})` : "number not in source";
  if (c.quoteNotFound) return "quote not found";
  if (c.verdict === "contradicted") return "contradicted";
  if (c.verdict === "partial") return "only partly supported";
  return "not supported by the source";
}

/** Claims created so far (analyze, including gap rounds). */
export function claimsCreated(events: RunEvent[]): number {
  let n = 0;
  for (const e of events) {
    if (e.event === "claims_created" && Array.isArray(e.payload.claims)) n += e.payload.claims.length;
  }
  return n;
}

/** "… checking claim 12 of 18" while claims are still waiting for a verdict, else null. */
export function checkingLine(events: RunEvent[], verifyRunning: boolean): string | null {
  if (!verifyRunning) return null;
  const total = claimsCreated(events);
  const done = claimLines(events).length;
  if (total === 0 || done >= total) return null;
  return `… checking claim ${done + 1} of ${total}`;
}

/** Problems recorded by the run, by the step whose node reported them. */
export function problems(events: RunEvent[]): { step: StepName | null; message: string; id: string }[] {
  return events
    .filter((e) => e.event === "problem" && str(e.payload.message))
    .map((e) => ({ step: stepOf(e.node), message: str(e.payload.message) ?? "", id: e.id }));
}

export function sectionsWritten(events: RunEvent[]): number {
  return events.filter((e) => e.event === "section_citations").length;
}

export function plural(n: number, one: string, many: string): string {
  return `${n} ${n === 1 ? one : many}`;
}
