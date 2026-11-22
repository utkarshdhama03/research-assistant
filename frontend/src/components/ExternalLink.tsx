import type { ReactNode } from "react";

import { safeHttpUrl } from "../lib/safeUrl";

/** A link only for http(s) URLs, opened in a new tab without access to this page; anything else is text. */
export function ExternalLink({ href, children }: { href: string | null | undefined; children?: ReactNode }) {
  const safe = safeHttpUrl(href);
  const label = children ?? href ?? "";
  if (safe === null) return <span className="plain-url">{label}</span>;
  return (
    <a href={safe} target="_blank" rel="noopener noreferrer">
      {label}
    </a>
  );
}
