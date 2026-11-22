import { useEffect, useLayoutEffect, useRef } from "react";

import type { CitedClaim, Reference } from "../api/types";
import { placePopover, VIEWPORT_MARGIN, visibleBounds } from "../lib/placement";
import { ExternalLink } from "./ExternalLink";

/** The claims cited at one [n] marker: text, verdict, the verbatim quote, and the source. */
export function CitationPopover({
  n,
  claims,
  reference,
  onClose,
}: {
  n: number;
  claims: CitedClaim[];
  reference: Reference | undefined;
  onClose: () => void;
}) {
  const box = useRef<HTMLDivElement>(null);

  // Keep it inside the visible area (the scrolling pane it sits in, within the window): measure at the
  // natural position (below the marker), then shift sideways and flip above when needed. Again on resize.
  // Styles are set through the CSSOM (allowed by the CSP).
  useLayoutEffect(() => {
    const place = (): void => {
      const el = box.current;
      const anchor = el?.closest(".marker-wrap");
      if (!el || !anchor) return;
      const bounds = visibleBounds(anchor);
      el.style.transform = "";
      el.style.maxWidth = `${bounds.right - bounds.left - 2 * VIEWPORT_MARGIN}px`;
      el.classList.remove("popover-above");
      const { shiftX, above } = placePopover(el.getBoundingClientRect(), anchor.getBoundingClientRect(), bounds);
      if (shiftX !== 0) el.style.transform = `translateX(${shiftX}px)`;
      if (above) el.classList.add("popover-above");
    };
    place();
    window.addEventListener("resize", place);
    return () => window.removeEventListener("resize", place);
  }, []);

  useEffect(() => {
    const onKey = (e: KeyboardEvent): void => {
      if (e.key === "Escape") onClose();
    };
    const onPointer = (e: MouseEvent): void => {
      const target = e.target as Node | null;
      if (box.current && target && !box.current.contains(target)) onClose();
    };
    document.addEventListener("keydown", onKey);
    document.addEventListener("mousedown", onPointer);
    return () => {
      document.removeEventListener("keydown", onKey);
      document.removeEventListener("mousedown", onPointer);
    };
  }, [onClose]);

  return (
    <div className="popover" role="dialog" aria-label={`Source ${n}`} ref={box}>
      {claims.map((claim) => (
        <div key={claim.claim_id} className="popover-claim">
          <p>
            <strong>Claim {claim.ordinal}</strong> <span className={`verdict verdict-${claim.verdict}`}>{claim.verdict}</span>
          </p>
          <p>{claim.text}</p>
          <blockquote>{claim.quote}</blockquote>
        </div>
      ))}
      {claims.length === 0 && <p className="dim">No claim details for this marker.</p>}
      <p className="popover-source">
        [{n}] {reference ? <ExternalLink href={reference.url}>{reference.title}</ExternalLink> : "unknown source"}
      </p>
      <button type="button" onClick={onClose}>
        close
      </button>
    </div>
  );
}
