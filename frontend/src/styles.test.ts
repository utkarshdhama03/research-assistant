import { describe, expect, it } from "vitest";

import css from "./styles.css?raw";

// WCAG 2.x contrast ratio of two #rrggbb colours.
function luminance(hex: string): number {
  const [r, g, b] = [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16) / 255).map((c) =>
    c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4,
  ) as [number, number, number];
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

export function contrast(a: string, b: string): number {
  const [hi, lo] = [luminance(a), luminance(b)].sort((x, y) => y - x) as [number, number];
  return (hi + 0.05) / (lo + 0.05);
}

function rootVariables(source: string): Record<string, string> {
  const root = /:root\s*{([^}]*)}/.exec(source)?.[1] ?? "";
  const vars: Record<string, string> = {};
  for (const m of root.matchAll(/--([\w-]+):\s*(#[0-9a-fA-F]{6})\s*;/g)) vars[m[1] as string] = (m[2] as string).toLowerCase();
  return vars;
}

const TEXT = ["fg", "dim", "accent", "green", "amber", "red"];
const BACKGROUNDS = ["bg", "surface", "selected"];

describe("palette", () => {
  const vars = rootVariables(css);

  it("defines every colour the checks need", () => {
    for (const name of [...TEXT, ...BACKGROUNDS]) expect(vars[name], name).toMatch(/^#[0-9a-f]{6}$/);
  });

  it("gives every text colour at least 4.5:1 on every background (WCAG AA)", () => {
    const ratios: Record<string, number> = {};
    for (const fg of TEXT) {
      for (const bg of BACKGROUNDS) ratios[`${fg} on ${bg}`] = contrast(vars[fg] as string, vars[bg] as string);
    }
    const failing = Object.entries(ratios).filter(([, r]) => r < 4.5);
    expect(failing).toEqual([]);
  });

  it("matches the known ratio for a reference pair", () => {
    expect(contrast("#ffffff", "#000000")).toBeCloseTo(21, 5);
  });
});
