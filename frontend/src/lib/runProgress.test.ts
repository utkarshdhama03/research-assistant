import { describe, expect, it } from "vitest";

import type { RunEvent } from "../api/types";
import { checkingLine, claimLines, problems, researchSummary, steps } from "./runProgress";

let seq = 0;
function ev(event: string, node: string, payload: Record<string, unknown> = {}): RunEvent {
  seq += 1;
  return { id: String(seq), node, event, payload, created_at: "2026-11-22T10:00:00Z" };
}

const verified = (id: string, ordinal: number, verdict: string, claim?: string) =>
  ev("claim_verified", "verify", { claim_id: id, ordinal, verdict, ...(claim ? { claim } : {}) });

describe("claim lines", () => {
  it("gives every rejection type its plain reason", () => {
    const lines = claimLines([
      verified("c1", 1, "supported", "Okapis live in the Ituri forest."),
      verified("c2", 2, "unsupported", "It weighs 350 kg and is 2 m tall."),
      ev("number_mismatch", "verify", { claim_id: "c2", ordinal: 2, missing: ["350", "2"] }),
      verified("c3", 3, "unsupported", "It was described in 1901."),
      ev("quote_not_found", "verify", { claim_id: "c3", ordinal: 3 }),
      verified("c4", 4, "contradicted", "Okapis are zebras."),
      verified("c5", 5, "partial", "Okapis live alone and in herds."),
      verified("c6", 6, "unsupported", "Okapis can fly."),
    ]);
    expect(lines.map((l) => [l.ok ? "✓" : "✗", l.text, l.reason])).toEqual([
      ["✓", "Okapis live in the Ituri forest.", null],
      ["✗", "It weighs 350 kg and is 2 m tall.", "number not in source (350, 2)"],
      ["✗", "It was described in 1901.", "quote not found"],
      ["✗", "Okapis are zebras.", "contradicted"],
      ["✗", "Okapis live alone and in herds.", "only partly supported"],
      ["✗", "Okapis can fly.", "not supported by the source"],
    ]);
  });

  it("falls back to the number_mismatch text, then to the claim number (runs before claim text)", () => {
    const lines = claimLines([
      verified("c1", 7, "unsupported"),
      ev("number_mismatch", "verify", { claim_id: "c1", ordinal: 7, missing: ["9"], claim: "Nine of them." }),
      verified("c2", 8, "supported"),
    ]);
    expect(lines.map((l) => l.text)).toEqual(["Nine of them.", "claim 8"]);
  });

  it("uses a claim's latest verification and its own flags", () => {
    const lines = claimLines([
      verified("c1", 1, "unsupported", "A claim."),
      ev("quote_not_found", "verify", { claim_id: "c1" }),
      verified("c1", 1, "supported", "A claim."), // verified again in a gap round
    ]);
    expect(lines).toHaveLength(1);
    expect([lines[0]?.ok, lines[0]?.reason]).toEqual([true, null]);
  });
});

describe("checking line", () => {
  const created = ev("claims_created", "analyze", { claims: [{ claim_id: "c1" }, { claim_id: "c2" }, { claim_id: "c3" }] });

  it("counts the claim being checked while verify runs", () => {
    expect(checkingLine([created, verified("c1", 1, "supported", "x")], true)).toBe("… checking claim 2 of 3");
    expect(checkingLine([created], false)).toBeNull(); // verify not running
    const all = [created, verified("c1", 1, "supported"), verified("c2", 2, "supported"), verified("c3", 3, "supported")];
    expect(checkingLine(all, true)).toBeNull(); // nothing left to check
  });
});

describe("steps", () => {
  const states = (events: RunEvent[], status: string) => steps(events, status).map((s) => `${s.name}:${s.state}`);

  it("folds analyze into verify and gap into research", () => {
    const events = [
      ev("node_start", "plan"),
      ev("node_end", "plan"),
      ev("node_start", "research"),
      ev("node_end", "research"),
      ev("node_start", "analyze"),
    ];
    expect(states(events, "running")).toEqual(["plan:done", "research:done", "verify:running", "write:pending"]);
    events.push(ev("node_end", "analyze"), ev("node_start", "verify"), ev("node_end", "verify"), ev("node_start", "gap"));
    expect(states(events, "running")).toEqual(["plan:done", "research:running", "verify:done", "write:pending"]);
  });

  it("shows approve as the current step while the plan waits", () => {
    const events = [ev("node_start", "plan"), ev("node_end", "plan")];
    expect(states(events, "awaiting_approval")).toEqual([
      "plan:done",
      "approve:paused",
      "research:pending",
      "verify:pending",
      "write:pending",
    ]);
  });

  it("marks a step failed on node_error, and an open step failed when the run failed", () => {
    expect(states([ev("node_start", "research"), ev("node_error", "research")], "running")[1]).toBe("research:failed");
    expect(states([ev("node_start", "write")], "failed")[3]).toBe("write:failed");
  });
});

describe("research summary", () => {
  it("counts searches without fallbacks, pages read, reused, blocked (403/429) and other failures", () => {
    const events = [
      ev("search_done", "research", { role: "primary" }),
      ev("search_done", "research", { role: "fallback" }),
      ev("search_done", "research", { role: "only" }),
      ev("source_fetched", "research"),
      ev("memory_reused", "research"),
      ev("fetch_failed", "research", { status_code: 403 }),
      ev("fetch_failed", "research", { status_code: 429 }),
      ev("fetch_failed", "research", { status_code: 500 }),
      ev("fetch_failed", "research", { reason: "timeout" }),
    ];
    expect(researchSummary(events)).toBe("2 searches · 1 page read · 1 reused · 2 blocked · 2 failed");
    expect(researchSummary([])).toBe("0 searches · 0 pages read");
  });

  it("files problems under the step that reported them", () => {
    const found = problems([ev("problem", "research", { message: "sub-question 1: no source with usable text" })]);
    expect(found.map((p) => [p.step, p.message])).toEqual([["research", "sub-question 1: no source with usable text"]]);
  });
});
