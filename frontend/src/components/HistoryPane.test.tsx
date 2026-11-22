import { act, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { RunSummary } from "../api/types";
import { mockFetch } from "../test/helpers";
import { FILTER_DEBOUNCE_MS, HistoryPane } from "./HistoryPane";

function run(id: string, question: string, status = "completed"): RunSummary {
  return { id, question, status, reason: null, auto_approve: false, created_at: "2026-11-22T10:00:00Z", finished_at: null };
}

function renderPane(selectedId: string | null = null) {
  return render(
    <MemoryRouter>
      <HistoryPane selectedId={selectedId} refreshKey={0} onPick={() => undefined} />
    </MemoryRouter>,
  );
}

afterEach(() => vi.useRealTimers());

describe("HistoryPane", () => {
  it("lists runs with a status dot and word, highlights the selected one, and loads more", async () => {
    const { calls } = mockFetch({
      "GET /runs?limit=20": { body: { items: [run("a", "First question?"), run("b", "Second question?", "failed")], next_cursor: "C1" } },
      "GET /runs?limit=20&cursor=C1": { body: { items: [run("c", "Third question?", "running")], next_cursor: null } },
    });
    renderPane("b");
    const links = await screen.findAllByRole("link");
    expect(links.map((l) => l.getAttribute("href"))).toEqual(["/run/a", "/run/b"]);
    expect(within(links[0] as HTMLElement).getByText("completed").className).toBe("sr-only"); // not colour alone
    expect(links[1]?.getAttribute("aria-current")).toBe("page");
    expect(links[0]?.getAttribute("aria-current")).toBeNull();
    await userEvent.click(screen.getByRole("button", { name: "load more" }));
    await waitFor(() => expect(screen.getAllByRole("link")).toHaveLength(3));
    expect(calls.map((c) => c.path)).toEqual(["/runs?limit=20", "/runs?limit=20&cursor=C1"]);
    expect(screen.queryByRole("button", { name: "load more" })).toBeNull();
  });

  it("filters on the server after the typing pauses", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    const { calls } = mockFetch({
      "GET /runs?limit=20": { body: { items: [run("a", "Okapi habitat?"), run("b", "Bees?")], next_cursor: null } },
      "GET /runs?limit=20&q=okapi": { body: { items: [run("a", "Okapi habitat?")], next_cursor: null } },
    });
    renderPane();
    await screen.findByText("Bees?");
    const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });
    await user.type(screen.getByLabelText("filter history"), "okapi");
    expect(calls.filter((c) => c.path.includes("q="))).toEqual([]); // nothing sent while typing
    await act(async () => {
      vi.advanceTimersByTime(FILTER_DEBOUNCE_MS);
    });
    await waitFor(() => expect(screen.queryByText("Bees?")).toBeNull());
    expect(calls.map((c) => c.path).filter((p) => p.includes("q="))).toEqual(["/runs?limit=20&q=okapi"]);
  });
});
