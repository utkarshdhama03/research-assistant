"""Turn fetched bytes into clean text.

Only bytes we fetched ourselves (through the SSRF-safe fetcher) are passed in. Never use the
download helpers of trafilatura, htmldate or courlan, or anything else in them that does network
I/O; scripts/check greps src/ for those helpers and fails if one appears.
"""

import hashlib
import io
import re
from dataclasses import dataclass
from enum import StrEnum

import pypdf
import trafilatura
from trafilatura.settings import Document

HTML_TYPES = frozenset({"text/html", "application/xhtml+xml"})
MAX_PDF_PAGES = 200

_LINE_BREAKS = str.maketrans(
    {"\r": "\n", "\x0b": "\n", "\x0c": "\n", "\x1c": "\n", "\x1d": "\n", "\x1e": "\n", "\x85": "\n"}
    | {" ": "\n", " ": "\n\n", "​": None, "﻿": None}
)
_SPACES = re.compile(r"[^\S\n]+")
_EXTRA_NEWLINES = re.compile(r"\n{3,}")
# Every other C0 control character except \t \n \r is removed (\r and the separators above become
# line breaks first). PDFs can yield NUL bytes, which PostgreSQL text columns reject.
_CONTROL_CHARS = str.maketrans(dict.fromkeys((chr(c) for c in range(0x20) if chr(c) not in "\t\n\r"), None))


class ExtractStatus(StrEnum):
    OK = "ok"
    SKIPPED = "skipped"


@dataclass(frozen=True, slots=True)
class ExtractResult:
    status: ExtractStatus
    title: str | None
    text: str  # normalized
    content_hash: str  # sha256 hex of text
    reason: str | None = None  # why it was skipped: too_short, empty, pdf_error, unsupported_type


def normalize_whitespace(text: str) -> str:
    """One space between words, no spaces at line ends, at most one blank line between paragraphs,
    no C0 control characters other than newline. Runs before the content hash is computed."""
    text = text.replace("\r\n", "\n").translate(_LINE_BREAKS).translate(_CONTROL_CHARS)
    text = _SPACES.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    return _EXTRA_NEWLINES.sub("\n\n", text).strip()


def _charset(content_type: str) -> str | None:
    for param in content_type.split(";")[1:]:
        key, _, value = param.partition("=")
        if key.strip().lower() == "charset" and value.strip():
            return value.strip().strip('"').lower()
    return None


def _decode_text(body: bytes, content_type: str) -> str:
    charset = _charset(content_type)
    for encoding in [charset, "utf-8"] if charset else ["utf-8"]:
        try:
            return body.decode(encoding)
        except (LookupError, UnicodeDecodeError):
            continue
    return body.decode("cp1252", errors="replace")


def _extract_html(body: bytes) -> tuple[str | None, str]:
    doc = trafilatura.bare_extraction(body, include_comments=False, with_metadata=True)
    if not isinstance(doc, Document):
        return None, ""
    return doc.title, doc.text or ""


def _extract_pdf(body: bytes) -> tuple[str | None, str]:
    reader = pypdf.PdfReader(io.BytesIO(body))
    if reader.is_encrypted and not reader.decrypt(""):
        raise ValueError("encrypted PDF")
    title = reader.metadata.title if reader.metadata else None
    pages = [page.extract_text() or "" for page in reader.pages[:MAX_PDF_PAGES]]
    return (str(title) if title else None), "\n\n".join(pages)


def extract(body: bytes, content_type: str, *, min_chars: int = 300) -> ExtractResult:
    """Extract title and main text. Results shorter than min_chars are skipped with reason too_short."""
    media = content_type.split(";", 1)[0].strip().lower()
    title: str | None
    if media in HTML_TYPES:
        title, raw = _extract_html(body)
    elif media == "application/pdf":
        try:
            title, raw = _extract_pdf(body)
        except Exception as e:  # pypdf raises many exception types on malformed files
            return _skipped(None, "", f"pdf_error: {type(e).__name__}: {e}")
    elif media == "text/plain":
        title, raw = None, _decode_text(body, content_type)
    else:
        return _skipped(None, "", "unsupported_type")

    text = normalize_whitespace(raw)
    title = (normalize_whitespace(title) or None) if title else None
    if not text:
        return _skipped(title, text, "empty")
    if len(text) < min_chars:
        return _skipped(title, text, "too_short")
    return ExtractResult(ExtractStatus.OK, title, text, _sha256(text))


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _skipped(title: str | None, text: str, reason: str) -> ExtractResult:
    return ExtractResult(ExtractStatus.SKIPPED, title, text, _sha256(text), reason)
