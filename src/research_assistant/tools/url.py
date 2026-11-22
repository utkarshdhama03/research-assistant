"""URL normalization. The normalized form is the dedupe key for search results and sources."""

from urllib.parse import urlsplit, urlunsplit

DEFAULT_PORTS = {"http": 80, "https": 443}
TRACKING_PARAMS = frozenset({"fbclid", "gclid", "mc_cid", "mc_eid", "ref_src"})


def _is_tracking_param(part: str) -> bool:
    key = part.split("=", 1)[0].lower()
    return key.startswith("utm_") or key in TRACKING_PARAMS


def normalize_url(url: str) -> str:
    """Return a canonical http(s) URL.

    Lowercases scheme and host, drops the default port, the fragment and tracking parameters
    (utm_*, fbclid, gclid, mc_cid, mc_eid, ref_src), and turns an empty path into "/".
    The path (case, trailing slash, percent-encoding) is left as is. Other query parameters keep
    their exact bytes and order: the query is filtered on "&", never decoded and re-encoded.

    Raises ValueError for anything that is not an http(s) URL with a host.
    """
    parts = urlsplit(url.strip())
    scheme = parts.scheme.lower()
    if scheme not in DEFAULT_PORTS:
        raise ValueError(f"not an http(s) URL: {url!r}")
    host = parts.hostname  # lowercased by urlsplit, IPv6 brackets removed
    if not host:
        raise ValueError(f"URL has no host: {url!r}")
    port = parts.port  # raises ValueError for a non-numeric or out-of-range port

    userinfo, at, _ = parts.netloc.rpartition("@")
    netloc = f"[{host}]" if ":" in host else host
    if port is not None and port != DEFAULT_PORTS[scheme]:
        netloc = f"{netloc}:{port}"
    if at:
        netloc = f"{userinfo}@{netloc}"

    query = "&".join(p for p in parts.query.split("&") if not _is_tracking_param(p)) if parts.query else ""
    return urlunsplit((scheme, netloc, parts.path or "/", query, ""))
