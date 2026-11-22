import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { ExternalLink } from "../components/ExternalLink";
import { safeHttpUrl } from "./safeUrl";

describe("safe links", () => {
  it("accepts only absolute http and https URLs", () => {
    expect(safeHttpUrl("https://example.org/a")).toBe("https://example.org/a");
    expect(safeHttpUrl("http://example.org/")).toBe("http://example.org/");
    for (const bad of ["javascript:alert(1)", " JAVASCRIPT:alert(1)", "data:text/html,x", "/relative", "", null]) {
      expect(safeHttpUrl(bad)).toBeNull();
    }
  });

  it("renders a javascript: URL as text, not as a link", () => {
    render(<ExternalLink href="javascript:alert(1)">Click me</ExternalLink>);
    expect(screen.queryByRole("link")).toBeNull();
    expect(screen.getByText("Click me").tagName).toBe("SPAN");
  });

  it("opens http links in a new tab without opener or referrer", () => {
    render(<ExternalLink href="https://example.org/page">Example</ExternalLink>);
    const link = screen.getByRole("link", { name: "Example" });
    expect(link.getAttribute("href")).toBe("https://example.org/page");
    expect(link.getAttribute("target")).toBe("_blank");
    expect(link.getAttribute("rel")).toBe("noopener noreferrer");
  });
});
