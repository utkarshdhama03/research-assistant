import { act, render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import { beforeEach, describe, expect, it, vi } from "vitest";

import type { RunDetail } from "../api/types";
import { MockEventSource, mockFetch, runEvent } from "../test/helpers";
import { RunView } from "./RunView";

function detail(overrides: Partial<RunDetail> = {}): RunDetail {
  return {
    id: "r1",
    question: "How do okapis live in the Ituri forest?",
    status: "running",
    reason: null,
    auto_approve: false,
    created_at: "2026-11-22T10:00:00Z",
    finished_at: null,
    cancel_requested: false,
    sub_questions: [
      { ordinal: 0, question: "Where do okapis live?", search_queries: ["okapi habitat"], premise_check: false },
      { ordinal: 1, question: "What do okapis eat?", search_queries: ["okapi diet"], premise_check: false },
      { ordinal: 2, question: "How many okapis are left?", search_queries: ["okapi population"], premise_check: false },
    ],
    node_timings: [],
    claims_by_verdict: {},
    report_markdown: null,
    ...overrides,
  };
}

function renderView() {
  return render(
    <MemoryRouter>
      <RunView runId="r1" onStatus={() => undefined} />
    </MemoryRouter>,
  );
}

beforeEach(() => MockEventSource.install());

describe("RunView", () => {
  it("shows the status line and the live log with a reason for a rejected claim", async () => {
    mockFetch({ "GET /runs/r1": { body: detail() } });
    renderView();
    expect(await screen.findByText("running")).toBeTruthy(); // the word, next to the dot
    const source = MockEventSource.latest();
    act(() => {
      source.emit("node_start", runEvent("node_start", "verify"), "1");
      source.emit("claim_verified", runEvent("claim_verified", "verify", { claim_id: "c1", ordinal: 1, verdict: "supported", claim: "Okapis live in forests." }), "2");
      source.emit("claim_verified", runEvent("claim_verified", "verify", { claim_id: "c2", ordinal: 2, verdict: "contradicted", claim: "Okapis are zebras." }), "3");
    });
    const log = screen.getByRole("log");
    expect(within(log).getByText("Okapis are zebras.").parentElement?.textContent).toContain("· contradicted");
    expect(screen.getByText("1 claim verified")).toBeTruthy();
    expect(screen.getByLabelText("verify running")).toBeTruthy();
  });

  it("cancels only after confirmation", async () => {
    const { calls } = mockFetch({
      "GET /runs/r1": { body: detail() },
      "POST /runs/r1/cancel": { status: 202, body: { id: "r1", status: "running", cancel_requested: true } },
    });
    renderView();
    const cancel = await screen.findByRole("button", { name: "cancel" });
    vi.spyOn(window, "confirm").mockReturnValueOnce(false).mockReturnValueOnce(true);
    await userEvent.click(cancel);
    expect(calls.filter((c) => c.method === "POST")).toEqual([]);
    await userEvent.click(cancel);
    expect(calls.filter((c) => c.method === "POST").map((c) => [c.path, c.body])).toEqual([["/runs/r1/cancel", {}]]);
  });

  it("shows the plan for approval while the run waits", async () => {
    mockFetch({ "GET /runs/r1": { body: detail({ status: "awaiting_approval" }) } });
    renderView();
    expect(await screen.findByRole("region", { name: "Plan approval" })).toBeTruthy();
    expect(screen.getByLabelText("approve waiting for you")).toBeTruthy();
    expect(screen.getByRole("button", { name: "approve" })).toBeTruthy();
  });

  it("shows report, sources (n) and log tabs once finished", async () => {
    mockFetch({
      "GET /runs/r1": {
        body: detail({ status: "completed", finished_at: "2026-11-22T10:05:31Z", claims_by_verdict: { supported: 17 }, report_markdown: "# r" }),
      },
      "GET /runs/r1/sources": {
        body: [
          { id: "s1", url: "https://example.org/a", final_url: null, title: "A", status: "fetched", reason: null, retrieved_at: null, reused: false, reused_from_run_id: null, cited: true },
          { id: "s2", url: "https://blocked.example/b", final_url: null, title: null, status: "failed", reason: "http_error: HTTP 403 (https://blocked.example/b)", retrieved_at: null, reused: false, reused_from_run_id: null, cited: false },
        ],
      },
      "GET /runs/r1/report": { body: { report_id: "x", created_at: "2026-11-22T10:05:31Z", legacy_markdown: "# r" } },
    });
    renderView();
    expect(await screen.findByRole("tab", { name: "sources (2)" })).toBeTruthy();
    expect(screen.getByText("5 min 31 s")).toBeTruthy();
    expect(screen.getByText("17 claims verified")).toBeTruthy();
    expect(screen.queryByRole("button", { name: "cancel" })).toBeNull();
    await userEvent.click(screen.getByRole("tab", { name: "sources (2)" }));
    expect(screen.getByRole("link", { name: "example.org" })).toBeTruthy();
    const reason = screen.getByText("http_error: HTTP 403"); // the URL is in the host column, not repeated
    expect(reason.parentElement?.getAttribute("title")).toBe("http_error: HTTP 403 (https://blocked.example/b)");
    await userEvent.click(screen.getByRole("tab", { name: "log" }));
    expect(screen.getByText("No events.")).toBeTruthy();
  });
});
