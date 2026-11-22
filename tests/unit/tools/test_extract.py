import hashlib
import socket
from pathlib import Path

import pytest

from research_assistant.tools.extract import ExtractStatus, extract, normalize_whitespace

FIXTURES = Path(__file__).parents[2] / "fixtures"

# (file, content type as served when it was saved)
DOCUMENTS = [
    ("simplewiki_okapi.html", "text/html; charset=UTF-8"),
    ("python_docs_secrets.html", "text/html"),
    ("example_com.html", "text/html; charset=utf-8"),
    ("rfc2606.txt", "text/plain;charset=utf-8"),
    ("rfc9309.pdf", "application/pdf"),
    ("control_chars.pdf", "application/pdf"),
    ("control_chars.html", "text/html; charset=utf-8"),
]
ALLOWED_CONTROL = frozenset("\n")  # \t and \r are allowed too but normalized away


def control_chars(text: str) -> list[str]:
    return [hex(ord(c)) for c in text if ord(c) < 0x20 and c not in ALLOWED_CONTROL]


def run(name: str, content_type: str, min_chars: int = 300) -> object:
    return extract((FIXTURES / name).read_bytes(), content_type, min_chars=min_chars)


def test_html_wikipedia() -> None:
    r = extract((FIXTURES / "simplewiki_okapi.html").read_bytes(), "text/html; charset=UTF-8")
    assert r.status is ExtractStatus.OK
    assert r.title == "Okapi - Simple English Wikipedia, the free encyclopedia"
    assert "Okapia johnstoni" in r.text
    assert "Retrieved from" not in r.text  # page chrome is not main text


def test_html_python_docs() -> None:
    r = extract((FIXTURES / "python_docs_secrets.html").read_bytes(), "text/html")
    assert r.status is ExtractStatus.OK
    assert r.title is not None and r.title.startswith("secrets")
    assert "cryptographically strong" in r.text


def test_short_page_is_skipped() -> None:
    r = extract((FIXTURES / "example_com.html").read_bytes(), "text/html; charset=utf-8", min_chars=300)
    assert (r.status, r.reason, r.title) == (ExtractStatus.SKIPPED, "too_short", "Example Domain")
    assert r.text.startswith("This domain is for use in documentation examples")
    ok = extract((FIXTURES / "example_com.html").read_bytes(), "text/html", min_chars=0)
    assert ok.status is ExtractStatus.OK


def test_plain_text_passthrough() -> None:
    r = extract((FIXTURES / "rfc2606.txt").read_bytes(), "text/plain;charset=utf-8")
    assert r.status is ExtractStatus.OK
    assert r.title is None
    assert "Reserved Top Level DNS Names" in r.text
    assert "\n\n" in r.text and "\n\n\n" not in r.text


def test_pdf() -> None:
    r = extract((FIXTURES / "rfc9309.pdf").read_bytes(), "application/pdf")
    assert r.status is ExtractStatus.OK
    assert r.title == "RFC 9309: Robots Exclusion Protocol"
    assert "Robots Exclusion Protocol" in r.text
    assert "crawlers" in r.text


@pytest.mark.parametrize(("name", "content_type"), DOCUMENTS)
def test_hash_is_sha256_of_text_and_stable(name: str, content_type: str) -> None:
    body = (FIXTURES / name).read_bytes()
    first, second = extract(body, content_type), extract(body, content_type)
    assert first.content_hash == hashlib.sha256(first.text.encode("utf-8")).hexdigest()
    assert first == second


@pytest.mark.parametrize(("name", "content_type"), DOCUMENTS)
def test_no_control_characters_in_text_or_title(name: str, content_type: str) -> None:
    r = extract((FIXTURES / name).read_bytes(), content_type, min_chars=0)
    assert control_chars(r.text) == [] and control_chars(r.title or "") == []


def test_control_characters_stripped_before_hashing() -> None:
    # pypdf passes NUL and other C0 bytes from the content stream through unchanged (q12 baseline).
    body = (FIXTURES / "control_chars.pdf").read_bytes()
    assert b"\\000" in body
    r = extract(body, "application/pdf", min_chars=0)
    assert r.status is ExtractStatus.OK
    assert r.title == "Ownership Types in Rust"
    assert r.text.splitlines() == [
        "Rustownershiptypes prevent data races at compile time.",
        "A value has either one mutable reference or any number of shared references.",
        "The borrow checker rejects programs that break this rule before they run.",
        "Threads can only share data through types that are Send and Sync.",
    ]
    assert r.content_hash == hashlib.sha256(r.text.encode("utf-8")).hexdigest()
    html = extract((FIXTURES / "control_chars.html").read_bytes(), "text/html", min_chars=0)
    assert "\x00" in (FIXTURES / "control_chars.html").read_text(encoding="utf-8")
    assert "The borrow checker enforces" in html.text and control_chars(html.text) == []


def test_control_characters_stripped_from_plain_text() -> None:
    r = extract(b"a\x00b\x01c\td\x1be\x7ff\r\ng\x0ch", "text/plain", min_chars=0)
    assert r.text == "abc de\x7ff\ng\nh"  # \t -> space, \r\n and form feed -> newline, DEL is not C0


def test_broken_pdf_is_skipped() -> None:
    r = extract(b"%PDF-1.4\nthis is not a real pdf", "application/pdf")
    assert r.status is ExtractStatus.SKIPPED
    assert r.reason is not None and r.reason.startswith("pdf_error")


def test_unsupported_type_is_skipped() -> None:
    r = extract(b"\x89PNG", "image/png")
    assert (r.status, r.reason) == (ExtractStatus.SKIPPED, "unsupported_type")


def test_empty_html_is_skipped() -> None:
    r = extract(b"<html><body></body></html>", "text/html")
    assert (r.status, r.reason) == (ExtractStatus.SKIPPED, "empty")


@pytest.mark.parametrize(
    ("body", "content_type", "expected"),
    [
        ("café".encode(), "text/plain; charset=utf-8", "café"),
        ("café".encode("latin-1"), "text/plain; charset=ISO-8859-1", "café"),
        ("café".encode("cp1252"), "text/plain", "café"),  # not UTF-8, falls back to cp1252
        ("café".encode(), "text/plain; charset=bogus-charset", "café"),
    ],
)
def test_plain_text_decoding(body: bytes, content_type: str, expected: str) -> None:
    assert extract(body, content_type, min_chars=0).text == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("a  b\t\tc", "a b c"),
        ("a\r\nb\rc", "a\nb\nc"),
        ("a  b", "a b"),
        ("zero​width﻿", "zerowidth"),
        ("line  \n  next", "line\nnext"),
        ("p1\n\n\n\n\np2", "p1\n\np2"),
        ("p1\n \n\t\np2", "p1\n\np2"),
        ("a b c", "a\nb\n\nc"),
        ("  \n\n padded \n\n ", "padded"),
    ],
)
def test_normalize_whitespace(raw: str, expected: str) -> None:
    assert normalize_whitespace(raw) == expected


def test_extraction_does_no_network_io(monkeypatch: pytest.MonkeyPatch) -> None:
    def no_network(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("extract() tried to use the network")

    monkeypatch.setattr(socket, "getaddrinfo", no_network)
    monkeypatch.setattr(socket, "create_connection", no_network)
    monkeypatch.setattr(socket.socket, "connect", no_network)
    for name, content_type in DOCUMENTS:
        extract((FIXTURES / name).read_bytes(), content_type)
