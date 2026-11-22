import pytest

from research_assistant.tools.url import normalize_url


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # scheme and host lowercased, path case kept
        ("HTTP://Example.COM/Path/To", "http://example.com/Path/To"),
        # default ports dropped, others kept
        ("http://example.com:80/a", "http://example.com/a"),
        ("https://example.com:443/a", "https://example.com/a"),
        ("http://example.com:443/a", "http://example.com:443/a"),
        ("https://example.com:8443/a", "https://example.com:8443/a"),
        ("http://example.com:/a", "http://example.com/a"),
        # empty path becomes "/", trailing slashes untouched
        ("http://example.com", "http://example.com/"),
        ("http://example.com?x=1", "http://example.com/?x=1"),
        ("http://example.com/dir/", "http://example.com/dir/"),
        ("http://example.com/dir", "http://example.com/dir"),
        ("http://example.com//double//", "http://example.com//double//"),
        # fragment dropped
        ("https://example.com/a#section-2", "https://example.com/a"),
        ("https://example.com/a?b=1#frag", "https://example.com/a?b=1"),
        # tracking params dropped, others keep their exact bytes and order
        ("https://e.com/?utm_source=x&id=7&utm_medium=y", "https://e.com/?id=7"),
        ("https://e.com/?UTM_Campaign=x&id=7", "https://e.com/?id=7"),
        ("https://e.com/?fbclid=abc", "https://e.com/"),
        ("https://e.com/?gclid=1&mc_cid=2&mc_eid=3&ref_src=twsrc", "https://e.com/"),
        ("https://e.com/?b=2&a=1&fbclid=z", "https://e.com/?b=2&a=1"),
        ("https://e.com/?q=a%20b&q=c+d&flag", "https://e.com/?q=a%20b&q=c+d&flag"),
        ("https://e.com/?utm_source", "https://e.com/"),
        # look-alike keys are not tracking params
        ("https://e.com/?notutm_source=1&gclidx=2&ref=3", "https://e.com/?notutm_source=1&gclidx=2&ref=3"),
        # percent-encoding in the path untouched
        ("https://e.com/%7Euser/A%2Fb", "https://e.com/%7Euser/A%2Fb"),
        # IPv6 host lowercased, brackets kept, default port dropped
        ("https://[2001:DB8::1]:443/x", "https://[2001:db8::1]/x"),
        ("http://[::1]:8080/", "http://[::1]:8080/"),
        # surrounding whitespace ignored
        ("  https://e.com/a  ", "https://e.com/a"),
        # userinfo is kept (the fetcher rejects it; normalizing must not hide it)
        ("http://User@Example.com/", "http://User@example.com/"),
    ],
)
def test_normalize_url(raw: str, expected: str) -> None:
    assert normalize_url(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        "ftp://example.com/file",
        "mailto:someone@example.com",
        "javascript:alert(1)",
        "file:///etc/passwd",
        "example.com/no-scheme",
        "http:///path-only",
        "http://example.com:99999/",
        "http://example.com:abc/",
        "",
    ],
)
def test_normalize_url_rejects(raw: str) -> None:
    with pytest.raises(ValueError):
        normalize_url(raw)


@pytest.mark.parametrize(
    "raw",
    ["HTTP://Example.COM:80/A/?utm_source=x&b=1#f", "https://e.com", "https://[::1]:443/?q=a%20b"],
)
def test_normalize_url_is_idempotent(raw: str) -> None:
    once = normalize_url(raw)
    assert normalize_url(once) == once
