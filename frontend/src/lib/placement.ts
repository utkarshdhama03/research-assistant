// Where to put a popover that opens below its marker: shifted sideways to stay inside the visible area,
// and flipped above the marker when it does not fit below but fits (or fits better) above.

export const VIEWPORT_MARGIN = 8; // px kept free at each edge of the visible area

export interface Box {
  left: number;
  right: number;
  top: number;
  bottom: number;
}

export interface Placement {
  shiftX: number; // px to move the popover right (+) or left (-) from its natural position
  above: boolean;
}

/**
 * popover: its rectangle at the natural position (below the marker, left edges aligned);
 * anchor: the marker's rectangle; bounds: the visible area it must stay in (all in viewport pixels).
 */
export function placePopover(popover: Box, anchor: Box, bounds: Box): Placement {
  const m = VIEWPORT_MARGIN;
  let shiftX = 0;
  if (popover.right > bounds.right - m) shiftX = bounds.right - m - popover.right;
  if (popover.left + shiftX < bounds.left + m) shiftX = bounds.left + m - popover.left; // keep the left edge in view
  const height = popover.bottom - popover.top;
  const spaceBelow = bounds.bottom - m - anchor.bottom;
  const spaceAbove = anchor.top - m - bounds.top;
  const above = height > spaceBelow && spaceAbove > spaceBelow;
  return { shiftX: Math.round(shiftX), above };
}

/**
 * The area an element can be seen in: the inside of its nearest scrolling ancestor (without its
 * scrollbars), clipped to the window (without the page's scrollbars).
 */
export function visibleBounds(el: Element): Box {
  const root = document.documentElement;
  let bounds: Box = {
    left: 0,
    top: 0,
    right: root.clientWidth || window.innerWidth,
    bottom: root.clientHeight || window.innerHeight,
  };
  for (let p = el.parentElement; p && p !== document.body; p = p.parentElement) {
    const style = window.getComputedStyle(p);
    if (/(auto|scroll|hidden)/.test(`${style.overflowX} ${style.overflowY}`)) {
      const r = p.getBoundingClientRect();
      const left = r.left + p.clientLeft;
      const top = r.top + p.clientTop;
      const inner: Box = { left, top, right: left + p.clientWidth, bottom: top + p.clientHeight };
      bounds = {
        left: Math.max(bounds.left, inner.left),
        top: Math.max(bounds.top, inner.top),
        right: Math.min(bounds.right, inner.right),
        bottom: Math.min(bounds.bottom, inner.bottom),
      };
      break;
    }
  }
  return bounds;
}
