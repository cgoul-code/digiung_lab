"""SSRF-safe outbound HTTP helpers.

Any server-side fetch of a caller-influenced URL must go through here, so it can't
be aimed at internal, loopback, link-local or cloud-metadata addresses. On Azure
the instance metadata service (169.254.169.254) will hand out managed-identity
tokens to anything that can make it issue a request, so an unguarded server-side
fetch is a token-exfiltration primitive.

The guard resolves the host and rejects any non-public IP, and follows redirects
manually so every hop is re-validated (a public URL can 302 to an internal one).

Residual risk: the OS re-resolves the host at connect time, so a DNS-rebinding
attacker who flips the record between our check and the connection could still get
through. Closing that fully means pinning the socket to the validated IP; this
resolve-and-validate guard is a proportionate, large improvement over none for an
admin-facing fetch, and blocks every static internal hostname/IP outright.
"""

import ipaddress
import socket
from urllib.parse import urljoin, urlparse

import requests

# Follow at most this many redirects before giving up.
_MAX_REDIRECTS = 5

# Redirect status codes we follow manually (so each hop is re-validated).
_REDIRECT_CODES = (301, 302, 303, 307, 308)


class SSRFError(ValueError):
    """A URL was rejected because it is not an allowed (public) http(s) target."""


def _assert_host_public(host: str) -> None:
    """Raise SSRFError unless every IP `host` resolves to is a public address."""
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as e:
        raise SSRFError(f"Kunne ikke slå opp vertsnavn: {host}") from e
    ips = {info[4][0] for info in infos}
    if not ips:
        raise SSRFError(f"Ingen IP-adresse for vertsnavn: {host}")
    for ip in ips:
        addr = ipaddress.ip_address(ip)
        if (addr.is_private or addr.is_loopback or addr.is_link_local
                or addr.is_reserved or addr.is_multicast or addr.is_unspecified):
            raise SSRFError(f"Blokkert intern adresse ({ip}) for {host}")


def assert_public_url(url: str) -> None:
    """Raise SSRFError unless `url` is an http(s) URL whose host resolves only to
    public IP addresses. Use this before any server-side fetch of a URL that a
    caller can influence."""
    parsed = urlparse(url)
    if parsed.scheme.lower() not in ("http", "https"):
        raise SSRFError("Bare http- og https-URL-er støttes")
    if not parsed.hostname:
        raise SSRFError("URL mangler vertsnavn")
    _assert_host_public(parsed.hostname)


def safe_get(url, *, headers=None, timeout=30, max_bytes=None):
    """SSRF-guarded GET. Validates the URL and every redirect hop, caps redirects,
    and — when `max_bytes` is set — reads at most that many bytes of the body
    (raising SSRFError if exceeded). Returns the final ``requests.Response`` with
    its body already read, so ``.content`` / ``.text`` are available.

    Raises SSRFError for a disallowed target or oversized body, and propagates the
    usual ``requests`` exceptions for network/HTTP failures.
    """
    current = url
    for _ in range(_MAX_REDIRECTS + 1):
        assert_public_url(current)
        resp = requests.get(current, headers=headers, timeout=timeout,
                            allow_redirects=False, stream=True)
        location = resp.headers.get("Location")
        if resp.status_code in _REDIRECT_CODES and location:
            resp.close()
            current = urljoin(current, location)
            continue
        # Terminal response — read the body (optionally size-capped) and cache it
        # on the Response so callers can use .content/.text as usual.
        if max_bytes is not None:
            chunks, total = [], 0
            for chunk in resp.iter_content(64 * 1024):
                total += len(chunk)
                if total > max_bytes:
                    resp.close()
                    raise SSRFError(f"Svaret er for stort (> {max_bytes} bytes)")
                chunks.append(chunk)
            resp._content = b"".join(chunks)
            resp._content_consumed = True
        else:
            _ = resp.content  # force the body to load while the connection is open
        return resp
    raise SSRFError("For mange omdirigeringer")
