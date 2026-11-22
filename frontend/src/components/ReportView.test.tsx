import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import type { StructuredReport } from "../api/types";
import { mockFetch } from "../test/helpers";
import { ReportView } from "./ReportView";

const QUOTE = "The okapi (Okapia johnstoni) is\nnative to the Ituri Rainforest.";

const REPORT: StructuredReport = {
  report_id: "rep1",
  created_at: "2026-11-22T10:00:00+00:00",
  premise_note: null,
  sections: [
    {
      ordinal: 1,
      heading: "What is an okapi?",
      kind: "normal",
      note: null,
      sentences: [
        {
          text: "Okapis live in the Ituri Rainforest.",
          citations: [1],
          parts: [{ text: "Okapis live in the Ituri Rainforest " }, { cite: [{ n: 1, claim_ids: ["c1"] }] }, { text: "." }],
        },
        {
          text: "A link test.",
          citations: [2],
          parts: [{ text: " A link test " }, { cite: [{ n: 2, claim_ids: ["c2"] }] }, { text: "." }],
        },
      ],
    },
    { ordinal: 2, heading: "Where else?", kind: "insufficient", note: "There is insufficient verified evidence.", sentences: [] },
  ],
  references: [
    { n: 1, title: "Okapi - Wikipedia", url: "https://en.wikipedia.org/wiki/Okapi", retrieved_at: "2026-11-20", reused: true },
    { n: 2, title: "Bad link", url: "javascript:alert(1)", retrieved_at: null, reused: false },
  ],
  claims_by_n: {
    "1": [{ claim_id: "c1", ordinal: 0, text: "Okapis are native to the Ituri Rainforest.", verdict: "supported", quote: QUOTE }],
    "2": [{ claim_id: "c2", ordinal: 1, text: "A second claim.", verdict: "supported", quote: "second quote" }],
  },
};

describe("ReportView", () => {
  it("opens a popover with the claim, its verdict, the exact quote and the source link", async () => {
    mockFetch({ "GET /runs/r1/report": { body: REPORT } });
    render(<ReportView runId="r1" />);
    await userEvent.click(await screen.findByRole("button", { name: "citation 1" }));
    const popover = screen.getByRole("dialog", { name: "Source 1" });
    const quote = popover.querySelector("blockquote");
    expect(quote?.textContent).toBe(QUOTE); // verbatim, line break included
    expect(within(popover).getByText("Okapis are native to the Ituri Rainforest.")).toBeTruthy();
    expect(within(popover).getByText("supported")).toBeTruthy();
    const link = within(popover).getByRole("link", { name: "Okapi - Wikipedia" });
    expect(link.getAttribute("href")).toBe("https://en.wikipedia.org/wiki/Okapi");
    await userEvent.keyboard("{Escape}");
    expect(screen.queryByRole("dialog")).toBeNull();
  });

  it("renders sections, notes, references with reused tag and the evidence list", async () => {
    mockFetch({ "GET /runs/r1/report": { body: REPORT } });
    render(<ReportView runId="r1" />);
    expect(await screen.findByText("There is insufficient verified evidence.")).toBeTruthy();
    expect(screen.getByText("insufficient evidence")).toBeTruthy();
    expect(screen.getByText("reused")).toBeTruthy();
    expect(screen.getByText("· retrieved 2026-11-20")).toBeTruthy();
    expect(screen.getByText("Claim 1")).toBeTruthy(); // evidence list, by ordinal
  });

  it("does not render a javascript: reference URL as a link", async () => {
    mockFetch({ "GET /runs/r1/report": { body: REPORT } });
    render(<ReportView runId="r1" />);
    await screen.findByText("Bad link");
    expect(screen.queryByRole("link", { name: "Bad link" })).toBeNull();
    for (const link of screen.getAllByRole("link")) {
      expect(link.getAttribute("href")?.startsWith("https://")).toBe(true);
    }
  });

  it("shows a legacy report as plain text", async () => {
    const markdown = "# Research report\n\n**Question:** old\n\nText with <b>tags</b> [1].";
    mockFetch({ "GET /runs/r1/report": { body: { report_id: "old", created_at: "2026-11-01T00:00:00Z", legacy_markdown: markdown } } });
    const { container } = render(<ReportView runId="r1" />);
    const pre = await screen.findByText((_content, el) => el?.tagName === "PRE");
    expect(pre.textContent).toBe(markdown);
    expect(container.querySelector("b")).toBeNull(); // not interpreted as HTML
    expect(container.querySelector("h1")).toBeNull(); // not rendered as Markdown
  });

  describe("popover placement", () => {
    // jsdom has no layout: give the popover and its marker the rectangles a browser would.
    function layout(popover: { left: number; top: number }, viewport: { width: number; height: number }) {
      // the visible area without scrollbars, as the popover measures it
      vi.spyOn(document.documentElement, "clientWidth", "get").mockReturnValue(viewport.width);
      vi.spyOn(document.documentElement, "clientHeight", "get").mockReturnValue(viewport.height);
      vi.spyOn(HTMLElement.prototype, "getBoundingClientRect").mockImplementation(function (this: HTMLElement) {
        const r = this.classList.contains("popover")
          ? { x: popover.left, y: popover.top + 20, width: 420, height: 300 }
          : this.classList.contains("marker-wrap")
            ? { x: popover.left, y: popover.top, width: 16, height: 18 }
            : { x: 0, y: 0, width: 0, height: 0 };
        return { ...r, left: r.x, top: r.y, right: r.x + r.width, bottom: r.y + r.height, toJSON: () => r } as DOMRect;
      });
    }

    it("shifts a popover opened near the right edge back inside the window", async () => {
      layout({ left: 976, top: 340 }, { width: 1040, height: 900 });
      mockFetch({ "GET /runs/r1/report": { body: REPORT } });
      render(<ReportView runId="r1" />);
      await userEvent.setup().click(await screen.findByRole("button", { name: "citation 1" }));
      const popover = screen.getByRole("dialog", { name: "Source 1" });
      expect(popover.style.transform).toBe("translateX(-364px)"); // right edge at 1040 - 8
      expect(popover.style.maxWidth).toBe("1024px"); // never wider than the visible area
      expect(popover.classList.contains("popover-above")).toBe(false);
    });

    it("opens above the marker near the bottom of the window", async () => {
      layout({ left: 100, top: 700 }, { width: 1040, height: 900 });
      mockFetch({ "GET /runs/r1/report": { body: REPORT } });
      render(<ReportView runId="r1" />);
      await userEvent.setup().click(await screen.findByRole("button", { name: "citation 1" }));
      const popover = screen.getByRole("dialog", { name: "Source 1" });
      expect(popover.classList.contains("popover-above")).toBe(true);
      expect(popover.style.transform).toBe("");
    });
  });
});
