import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { mockFetch } from "../test/helpers";
import { AskBar } from "./AskBar";

describe("AskBar", () => {
  it("starts a run on Enter, waiting for approval by default", async () => {
    const { calls } = mockFetch({ "POST /runs": { status: 202, body: { id: "new-run", status: "queued" } } });
    const onCreated = vi.fn();
    render(<AskBar onCreated={onCreated} />);
    await userEvent.type(screen.getByLabelText("ask a new question"), "  How do okapis live?{Enter}");
    expect(calls.map((c) => c.body)).toEqual([{ question: "How do okapis live?", auto_approve: false }]);
    expect(onCreated).toHaveBeenCalledWith("new-run");
    expect((screen.getByLabelText("ask a new question") as HTMLInputElement).value).toBe("");
  });

  it("sends auto_approve when the toggle is on", async () => {
    const { calls } = mockFetch({ "POST /runs": { status: 202, body: { id: "r2", status: "queued" } } });
    render(<AskBar onCreated={() => undefined} />);
    await userEvent.click(screen.getByLabelText("auto-approve"));
    await userEvent.type(screen.getByLabelText("ask a new question"), "Why is the sky blue?{Enter}");
    expect(calls[0]?.body).toEqual({ question: "Why is the sky blue?", auto_approve: true });
  });

  it("checks the length before sending", async () => {
    const { calls } = mockFetch({});
    render(<AskBar onCreated={() => undefined} />);
    await userEvent.type(screen.getByLabelText("ask a new question"), "Why{Enter}");
    expect(screen.getByRole("alert").textContent).toBe("The question needs 5 to 500 characters (now 3).");
    expect(calls).toEqual([]);
  });
});
