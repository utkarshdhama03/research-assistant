/** The URL if it is absolute http(s), else null; anything else (javascript:, data:, relative) stays text. */
export function safeHttpUrl(value: string | null | undefined): string | null {
  if (!value) return null;
  let url: URL;
  try {
    url = new URL(value);
  } catch {
    return null;
  }
  return url.protocol === "http:" || url.protocol === "https:" ? url.href : null;
}
