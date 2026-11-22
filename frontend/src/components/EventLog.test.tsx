import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { runEvent } from "../test/helpers";
import { EventLog } from "./EventLog";

describe("EventLog", () => {
  it("shows one line per event, with an unknown event type dimmed and its raw payload", () => {
    const events = [
      { id: "1", ...runEvent("search_done", "research", { query: "okapi habitat", provider: "searxng", result_count: 5 }) },
      { id: "2", ...runEvent("brand_new_event", "write", { detail: "from a newer server" }) },
    ];
    render(<EventLog events={events} />);
    const lines = screen.getAllByRole("listitem");
    expect(lines).toHaveLength(2);
    expect(lines[0]?.className).toBe("event");
    expect(lines[0]?.textContent).toContain("okapi habitat · searxng · 5 results");
    expect(lines[1]?.className).toBe("event unknown");
    expect(lines[1]?.getAttribute("title")).toBe("unknown event type");
    expect(lines[1]?.textContent).toContain('brand_new_event {"detail":"from a newer server"}');
  });
});
