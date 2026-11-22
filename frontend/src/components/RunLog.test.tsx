import { fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { RunEvent } from "../api/types";
import { steps } from "../lib/runProgress";
import { runEvent } from "../test/helpers";
import { RunLog } from "./RunLog";

function events(n: number): RunEvent[] {
  const list: RunEvent[] = [
    { id: "s1", ...runEvent("node_start", "verify") },
    { id: "cc", ...runEvent("claims_created", "analyze", { claims: Array.from({ length: 20 }, (_, i) => ({ claim_id: `c${i}` })) }) },
  ];
  for (let i = 0; i < n; i += 1) {
    list.push({ id: `v${i}`, ...runEvent("claim_verified", "verify", { claim_id: `c${i}`, ordinal: i, verdict: "supported", claim: `Claim ${i}.` }) });
  }
  return list;
}

function renderLog(n: number) {
  const list = events(n);
  return render(<RunLog events={list} steps={steps(list, "running")} subQuestions={3} />);
}

afterEach(() => vi.restoreAllMocks());

describe("RunLog", () => {
  it("shows claim lines and the checking line under verify", () => {
    renderLog(2);
    expect(screen.getByText("Claim 1.")).toBeTruthy();
    expect(screen.getByText("… checking claim 3 of 20")).toBeTruthy();
    expect(screen.getByText("2 ✓ · 0 ✗")).toBeTruthy();
  });

  it("follows new lines at the bottom, and stays put after the reader scrolled up", () => {
    vi.spyOn(HTMLElement.prototype, "scrollHeight", "get").mockReturnValue(1000);
    vi.spyOn(HTMLElement.prototype, "clientHeight", "get").mockReturnValue(200);
    const { rerender } = renderLog(2);
    const log = screen.getByRole("log");
    expect(log.scrollTop).toBe(1000); // followed to the bottom
    log.scrollTop = 300; // the reader scrolls up
    fireEvent.scroll(log);
    const more = events(3);
    rerender(<RunLog events={more} steps={steps(more, "running")} subQuestions={3} />);
    expect(log.scrollTop).toBe(300); // not moved
    log.scrollTop = 790; // back near the bottom (within 24 px)
    fireEvent.scroll(log);
    const evenMore = events(4);
    rerender(<RunLog events={evenMore} steps={steps(evenMore, "running")} subQuestions={3} />);
    expect(log.scrollTop).toBe(1000);
  });
});
