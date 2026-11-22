import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import type { SubQuestion } from "../api/types";
import { mockFetch } from "../test/helpers";
import { PlanEditor } from "./PlanEditor";

const PLAN: SubQuestion[] = [
  { ordinal: 0, question: "What is an okapi?", search_queries: ["okapi animal"], premise_check: false },
  { ordinal: 1, question: "Where do okapis live?", search_queries: ["okapi habitat", "okapi range"], premise_check: false },
  { ordinal: 2, question: "What do okapis eat?", search_queries: ["okapi diet"], premise_check: false },
];

function renderEditor(onDone = vi.fn()) {
  render(<PlanEditor runId="r1" subQuestions={PLAN} onDone={onDone} />);
  return onDone;
}

describe("PlanEditor", () => {
  it("keeps sub-questions within 3-5 and queries within 1-3", async () => {
    mockFetch({});
    renderEditor();
    const removeSq = () => screen.getAllByRole("button", { name: /^Remove sub-question/ });
    expect(removeSq().every((b) => (b as HTMLButtonElement).disabled)).toBe(true); // 3 is the minimum
    const add = screen.getByRole("button", { name: "+ sub-question" }) as HTMLButtonElement;
    await userEvent.click(add);
    await userEvent.click(add);
    expect(add.disabled).toBe(true); // 5 is the maximum
    expect(removeSq().every((b) => !(b as HTMLButtonElement).disabled)).toBe(true);

    const addQuery = screen.getByRole("button", { name: "Add query to sub-question 2" }) as HTMLButtonElement;
    await userEvent.click(addQuery);
    expect(addQuery.disabled).toBe(true); // 3 queries
    const removeQuery = screen.getByRole("button", { name: "Remove query 1 of sub-question 1" }) as HTMLButtonElement;
    expect(removeQuery.disabled).toBe(true); // 1 query is the minimum
  });

  it("validates like the server before sending: duplicate query and short question", async () => {
    const { calls } = mockFetch({});
    renderEditor();
    const query = screen.getByLabelText("Sub-question 2 query 2");
    await userEvent.clear(query);
    await userEvent.type(query, "  OKAPI   habitat ");
    const question = screen.getByLabelText("Sub-question 3");
    await userEvent.clear(question);
    await userEvent.type(question, "Diet");
    await userEvent.click(screen.getByRole("button", { name: "approve edited plan" }));
    expect(screen.getByText("This query repeats another query of this sub-question.")).toBeTruthy();
    expect(screen.getByText("A sub-question needs 5 to 300 characters.")).toBeTruthy();
    expect(calls).toHaveLength(0);
  });

  it("shows the server's 422 details next to their fields", async () => {
    mockFetch({
      "POST /runs/r1/approve": {
        status: 422,
        body: {
          error: {
            code: "validation_error",
            message: "invalid plan",
            details: [
              { loc: ["sub_questions", 1, "search_queries", 0], msg: "String should have at most 200 characters" },
              { loc: ["sub_questions"], msg: "premise_check may be true for the first sub-question only" },
            ],
          },
        },
      },
    });
    renderEditor();
    const question = screen.getByLabelText("Sub-question 1");
    await userEvent.type(question, " exactly");
    await userEvent.click(screen.getByRole("button", { name: "approve edited plan" }));
    expect(await screen.findByText("String should have at most 200 characters")).toBeTruthy();
    expect(screen.getByLabelText("Sub-question 2 query 1").getAttribute("aria-invalid")).toBe("true");
    expect(screen.getByRole("alert").textContent).toContain(
      "sub_questions: premise_check may be true for the first sub-question only",
    );
  });

  it("approves an unchanged plan with {} and an edited one with the full plan", async () => {
    const { calls } = mockFetch({ "POST /runs/r1/approve": { body: { id: "r1", status: "queued", cancel_requested: false } } });
    const onDone = renderEditor();
    await userEvent.click(screen.getByRole("button", { name: "approve" }));
    expect(calls[0]?.body).toEqual({});
    expect(onDone).toHaveBeenCalledTimes(1);

    const question = screen.getByLabelText("Sub-question 1");
    await userEvent.clear(question);
    await userEvent.type(question, "  What kind of animal is an okapi?  ");
    await userEvent.click(screen.getByRole("button", { name: "approve edited plan" }));
    expect(calls[1]?.body).toEqual({
      sub_questions: [
        { question: "What kind of animal is an okapi?", search_queries: ["okapi animal"], premise_check: false },
        { question: "Where do okapis live?", search_queries: ["okapi habitat", "okapi range"], premise_check: false },
        { question: "What do okapis eat?", search_queries: ["okapi diet"], premise_check: false },
      ],
    });
    expect(calls[1]?.headers["Content-Type"]).toBe("application/json");
  });

  it("rejects only after confirmation", async () => {
    const { calls } = mockFetch({ "POST /runs/r1/reject": { body: { id: "r1", status: "cancelled", cancel_requested: false } } });
    const confirm = vi.spyOn(window, "confirm").mockReturnValueOnce(false).mockReturnValueOnce(true);
    const onDone = renderEditor();
    await userEvent.click(screen.getByRole("button", { name: "reject" }));
    expect(calls).toHaveLength(0);
    await userEvent.click(screen.getByRole("button", { name: "reject" }));
    expect(confirm).toHaveBeenCalledTimes(2);
    expect(calls.map((c) => `${c.method} ${c.path}`)).toEqual(["POST /runs/r1/reject"]);
    expect(onDone).toHaveBeenCalledTimes(1);
  });
});
