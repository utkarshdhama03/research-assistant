import { act, render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it } from "vitest";

import { StepLine } from "../components/StepLine";
import { researchSummary, steps } from "../lib/runProgress";
import { MockEventSource, runEvent } from "../test/helpers";
import { useRunEvents } from "./useRunEvents";

function Probe({ runId }: { runId: string }) {
  const { events, endStatus } = useRunEvents(runId);
  return (
    <div>
      <ul>
        {events.map((e) => (
          <li key={e.id} data-testid="event">
            {e.event}
          </li>
        ))}
      </ul>
      <span data-testid="end">{endStatus ?? "open"}</span>
      <span data-testid="summary">{researchSummary(events)}</span>
      <StepLine steps={steps(events, "running")} />
    </div>
  );
}

describe("useRunEvents", () => {
  beforeEach(() => MockEventSource.install());

  it("opens the run's stream and drops events it has already seen", () => {
    render(<Probe runId="r1" />);
    const source = MockEventSource.latest();
    expect(source.url).toBe("http://127.0.0.1:8000/runs/r1/events");
    act(() => {
      source.emit("node_start", runEvent("node_start", "plan"), "e1");
      source.emit("node_end", runEvent("node_end", "plan", { duration_s: 2.5 }), "e2");
      source.emit("node_start", runEvent("node_start", "plan"), "e1"); // replayed after a reconnect
      source.emit("search_done", runEvent("search_done", "research", { role: "primary" }), "e3");
      source.emit("search_done", runEvent("search_done", "research", { role: "fallback" }), "e4");
      source.emit("source_fetched", runEvent("source_fetched", "research"), "e5");
      source.emit("fetch_failed", runEvent("fetch_failed", "research"), "e6");
      source.emit("source_fetched", runEvent("source_fetched", "research"), "e5"); // duplicate id
    });
    expect(screen.getAllByTestId("event").map((e) => e.textContent)).toEqual([
      "node_start",
      "node_end",
      "search_done",
      "search_done",
      "source_fetched",
      "fetch_failed",
    ]);
    // the fallback call is not a search; a failure without a 403/429 is "failed", not "blocked"
    expect(screen.getByTestId("summary").textContent).toBe("1 search · 1 page read · 1 failed");
    expect(screen.getByLabelText("plan done")).toBeTruthy();
    expect(screen.getByLabelText("research not started")).toBeTruthy(); // steps follow node events only
  });

  it("closes the stream on the end event", () => {
    render(<Probe runId="r1" />);
    const source = MockEventSource.latest();
    act(() => source.emit("end", { status: "completed" }));
    expect(source.closed).toBe(true);
    expect(screen.getByTestId("end").textContent).toBe("completed");
  });

  it("closes the stream on unmount", () => {
    const { unmount } = render(<Probe runId="r1" />);
    const source = MockEventSource.latest();
    expect(source.closed).toBe(false);
    unmount();
    expect(source.closed).toBe(true);
  });
});
