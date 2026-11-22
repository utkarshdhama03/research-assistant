import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import { afterEach, describe, expect, it, vi } from "vitest";

import { App, NARROW_QUERY } from "./App";
import { MockEventSource, mockFetch } from "./test/helpers";

function setWidth(narrow: boolean): void {
  vi.stubGlobal("matchMedia", (query: string) => ({
    matches: query === NARROW_QUERY && narrow,
    media: query,
    addEventListener: () => undefined,
    removeEventListener: () => undefined,
  }));
}

const RUNS = {
  items: [{ id: "a", question: "Okapi habitat?", status: "completed", reason: null, auto_approve: false, created_at: "2026-11-22T10:00:00Z", finished_at: null }],
  next_cursor: null,
};

function renderApp() {
  MockEventSource.install(); // picking a run opens its event stream
  mockFetch({
    "GET /auth/session": { body: { auth_enabled: false, authenticated: true } },
    "GET /health": { body: { status: "ok" } },
    "GET /runs?limit=20": { body: RUNS },
    "GET /runs/a*": { status: 404, body: { error: { code: "not_found", message: "run a not found" } } },
  });
  return render(
    <MemoryRouter initialEntries={["/"]}>
      <App />
    </MemoryRouter>,
  );
}

afterEach(() => vi.unstubAllGlobals());

describe("layout", () => {
  it("shows the history pane beside the run on a wide window, without a toggle", async () => {
    setWidth(false);
    renderApp();
    expect(await screen.findByRole("navigation", { name: "history" })).toBeTruthy();
    expect(screen.queryByRole("button", { name: "history" })).toBeNull();
  });

  it("collapses the history below 800 px behind a toggle, and closes it after picking a run", async () => {
    setWidth(true);
    renderApp();
    const toggle = await screen.findByRole("button", { name: "history" });
    expect(toggle.getAttribute("aria-expanded")).toBe("false");
    expect(screen.queryByRole("navigation", { name: "history" })).toBeNull();
    await userEvent.click(toggle);
    expect(toggle.getAttribute("aria-expanded")).toBe("true");
    await userEvent.click(await screen.findByRole("link", { name: /Okapi habitat\?/ }));
    await waitFor(() => expect(screen.queryByRole("navigation", { name: "history" })).toBeNull());
    expect(toggle.getAttribute("aria-expanded")).toBe("false");
  });
});
