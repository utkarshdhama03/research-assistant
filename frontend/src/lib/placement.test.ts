import { describe, expect, it } from "vitest";

import { placePopover, VIEWPORT_MARGIN } from "./placement";

const box = (left: number, top: number, width: number, height: number) => ({
  left,
  top,
  right: left + width,
  bottom: top + height,
});
const VIEW = { left: 0, top: 0, right: 1040, bottom: 900 };
const anchorAt = (left: number, top: number) => box(left, top, 16, 18);

describe("placePopover", () => {
  it("leaves a popover that fits where it is", () => {
    expect(placePopover(box(100, 120, 420, 300), anchorAt(100, 100), VIEW)).toEqual({ shiftX: 0, above: false });
  });

  it("shifts left so the right edge stays a margin inside the window", () => {
    // the clean-machine case: marker at x=976 in a 1040 px window
    const { shiftX } = placePopover(box(976, 360, 420, 300), anchorAt(976, 340), VIEW);
    expect(976 + 420 + shiftX).toBe(VIEW.right - VIEWPORT_MARGIN);
    expect(shiftX).toBe(-364);
  });

  it("keeps the left edge in view on a window narrower than the popover", () => {
    const narrow = { left: 0, top: 0, right: 360, bottom: 700 };
    const { shiftX } = placePopover(box(200, 120, 400, 300), anchorAt(200, 100), narrow);
    expect(200 + shiftX).toBe(VIEWPORT_MARGIN);
  });

  it("flips above the marker when it does not fit below and there is more room above", () => {
    expect(placePopover(box(100, 720, 420, 300), anchorAt(100, 700), VIEW).above).toBe(true);
  });

  it("stays below when there is more room below, even if it does not fully fit", () => {
    const short = { left: 0, top: 0, right: 1040, bottom: 400 };
    expect(placePopover(box(100, 120, 420, 350), anchorAt(100, 100), short).above).toBe(false);
  });

  it("keeps it inside a scrolling pane that does not start at the window edge", () => {
    // the main pane: x 200-1265 (its scrollbar excluded), y 37-740
    const pane = { left: 200, top: 37, right: 1265, bottom: 740 };
    const { shiftX } = placePopover(box(1000, 300, 420, 200), anchorAt(1000, 280), pane);
    expect(1000 + 420 + shiftX).toBe(1265 - VIEWPORT_MARGIN);
    const near = placePopover(box(150, 300, 420, 200), anchorAt(150, 280), pane); // starts left of the pane
    expect(150 + near.shiftX).toBe(200 + VIEWPORT_MARGIN);
  });
});
