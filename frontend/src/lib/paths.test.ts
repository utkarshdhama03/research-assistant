import { describe, expect, it } from "vitest";

import { runPath, safeNext } from "./paths";

describe("paths", () => {
  it("puts a run's page under /run/, away from the API's /runs/", () => {
    expect(runPath("abc")).toBe("/run/abc");
    expect(runPath("a/b")).toBe("/run/a%2Fb");
  });

  it("returns after login only to a path on this site", () => {
    expect(safeNext("/run/abc?tab=sources")).toBe("/run/abc?tab=sources");
    for (const bad of [null, "", "https://evil.example/", "//evil.example/x", String.raw`/\evil.example`, "javascript:alert(1)", "/login?next=/"]) {
      expect(safeNext(bad)).toBe("/");
    }
  });
});
