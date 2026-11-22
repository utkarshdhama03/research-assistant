# Test fixtures

Real responses saved once on 2026-11-16 and stored byte for byte (`.gitattributes` marks this
folder `-text` so line endings are not rewritten). Unit tests read these files and never go to the
network.

| File | Source | Content type when saved | License |
|---|---|---|---|
| `simplewiki_okapi.html` | https://simple.wikipedia.org/wiki/Okapi | `text/html; charset=UTF-8` | CC BY-SA 4.0 (Wikipedia contributors) |
| `python_docs_secrets.html` | https://docs.python.org/3/library/secrets.html | `text/html` | PSF License (Python documentation) |
| `example_com.html` | https://example.com/ | `text/html; charset=utf-8` | IANA example page, reserved for documentation (RFC 2606) |
| `rfc2606.txt` | https://www.rfc-editor.org/rfc/rfc2606.txt | `text/plain;charset=utf-8` | IETF RFC; may be copied in full (IETF Trust) |
| `rfc9309.pdf` | https://www.rfc-editor.org/rfc/rfc9309.pdf | `application/pdf` | IETF RFC; may be copied in full (IETF Trust Legal Provisions) |
| `searxng_python_asyncio.json` | local SearXNG instance, `GET /search?q=python+asyncio+event+loop&format=json` | `application/json` | SearXNG output (titles and snippets from the result pages) |

`searxng_python_asyncio.json` is the real response trimmed to its first 10 results; nothing else
was changed.

## Synthetic fixtures

Made by hand on 2026-11-18, not saved responses. They reproduce the q12 baseline failure (a PDF
whose extracted text contained NUL bytes, which PostgreSQL text columns reject).

| File | Content |
|---|---|
| `control_chars.pdf` | One-page PDF 1.4 written byte by byte (Helvetica, four `Tj` lines). The content stream contains the octal escapes `\000`, `\001`, `\002`, `\010`, `\037`, and the `/Title` contains `\001`; pypdf extracts them as raw control characters. |
| `control_chars.html` | Short UTF-8 HTML article with literal U+0000, U+0001, U+0007 and U+001F in the title and paragraphs (lxml already drops them; kept as a guard). |

## Quote-location cases

`vacuum_quote_cases.json`: the 13 claims of live runs bf0e907b and dfc3f59e (2026-11-19/21) whose verifier
quotes were not found, with the model's quote, the expected match method, and two short excerpts (exact
substrings of the stored chunks) from https://oneuptime.com/blog/post/2026-01-21-postgresql-vacuum-autovacuum/view
and https://www.geeksforgeeks.org/postgresql/what-is-vacuum-in-postgresql/, quoted only as test data.
