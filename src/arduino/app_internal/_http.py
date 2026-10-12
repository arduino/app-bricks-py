# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""A small HTTP client on the Python standard library.

One request at a time, JSON and multipart bodies, the response whatever its status, bodies read as a stream, and GET
with the retry rules of urllib3. Nothing heavier than ``json`` is imported until the first request: ``requests``
costs about 0.6 s to import on the board, ``urllib.request`` a third of it.
"""

from __future__ import annotations

import json as _json
import os
from collections.abc import Collection, Generator, Mapping
from contextlib import contextmanager
from typing import IO, TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    import http.client
    import urllib.error
    import urllib.request
    from email.message import Message

Files = Mapping[str, tuple[str, bytes | IO[bytes], str]]
"""Multipart file parts as requests takes them: field name -> (file name, content, content type)."""

DEFAULT_STATUS_FORCELIST = (411, 500, 502, 503, 504)
"""Statuses that get_with_retry retries by default."""

RETRY_AFTER_STATUSES = frozenset({413, 429, 503})
"""Statuses also retried when the response carries a Retry-After header."""

BACKOFF_MAX_S = 120.0
RETRY_AFTER_MAX_S = 21600.0


class Response:
    """Status, headers and body of an HTTP response."""

    __slots__ = ("status_code", "headers", "content")

    def __init__(self, status_code: int, headers: Message, content: bytes) -> None:
        self.status_code = status_code
        self.headers = headers
        self.content = content

    @property
    def text(self) -> str:
        """The body decoded as UTF-8, with invalid bytes replaced."""
        return self.content.decode("utf-8", errors="replace")

    def json(self) -> Any:  # noqa: ANN401
        """The body decoded as JSON."""
        return _json.loads(self.content)

    def raise_for_status(self) -> None:
        """Raise HTTPStatusError when the status is 4xx or 5xx."""
        if self.status_code >= 400:
            raise HTTPStatusError(self)


class ConnectionFailed(OSError):
    """No HTTP response arrived: refused or reset connection, unknown host, timeout, malformed or truncated response."""


class HTTPStatusError(Exception):
    """A response has a 4xx or 5xx status where a successful one is required."""

    def __init__(self, response: Response) -> None:
        super().__init__(f"HTTP {response.status_code}: {response.text[:200]}")
        self.response = response


def request(
    method: str,
    url: str,
    *,
    json: Any = None,  # noqa: ANN401
    files: Files | None = None,
    data: bytes | None = None,
    headers: Mapping[str, str] | None = None,
    timeout: float | None = None,
) -> Response:
    """Send one request and return the response, whatever its status. Redirects are followed.

    Args:
        method (str): The HTTP method.
        url (str): The URL, query string included.
        json (Any): A body to send as JSON. NaN and infinities are rejected, as requests does.
        files (Files | None): A body to send as multipart/form-data.
        data (bytes | None): A raw body.
        headers (Mapping[str, str] | None): Headers added to the request.
        timeout (float | None): Seconds allowed to connect and for each read; None waits as long as the socket default.

    Returns:
        Response: The response, 4xx and 5xx included.

    Raises:
        ConnectionFailed: No response arrived.
        ValueError: The json body cannot be serialized.
    """
    import http.client
    import urllib.error

    req = _build_request(method, url, json=json, files=files, data=data, headers=headers)
    try:
        try:
            with _open(req, timeout) as resp:
                return Response(resp.status, resp.headers, resp.read())
        except urllib.error.HTTPError as e:  # A response with a 4xx or 5xx status. Before URLError, its base class
            with e:
                return Response(e.code, e.headers, e.read())
    except (OSError, http.client.HTTPException) as e:  # URLError is an OSError; IncompleteRead is not
        raise ConnectionFailed(f"{method} {url}: {e}") from e


def post(
    url: str,
    *,
    json: Any = None,  # noqa: ANN401
    files: Files | None = None,
    headers: Mapping[str, str] | None = None,
    timeout: float | None = None,
) -> Response:
    """Send a POST request, see request()."""
    return request("POST", url, json=json, files=files, headers=headers, timeout=timeout)


@contextmanager
def stream(
    method: str,
    url: str,
    *,
    json: Any = None,  # noqa: ANN401
    headers: Mapping[str, str] | None = None,
    timeout: float | None = None,
) -> Generator[http.client.HTTPResponse]:
    """Send one request and yield the response with its body still to read, whatever its status.

    The body is read from the yielded response: read1(n) returns what has arrived, at most n bytes, without waiting for
    more. The connection is closed when the block exits.

    Raises:
        ConnectionFailed: No response arrived.
        ValueError: The json body cannot be serialized.
    """
    import http.client
    import urllib.error

    req = _build_request(method, url, json=json, headers=headers)
    error: urllib.error.HTTPError | None = None
    try:
        resp = _open(req, timeout)
    except urllib.error.HTTPError as e:  # A response with a 4xx or 5xx status: its body is read like any other
        error = e  # It owns the response and closes it once collected
        resp = cast("http.client.HTTPResponse", e.fp)
    except (OSError, http.client.HTTPException) as e:
        raise ConnectionFailed(f"{method} {url}: {e}") from e
    with resp:
        yield resp
    if error is not None:  # Held until here, so that the response stays open while its body is read
        error.close()


def get_with_retry(
    url: str,
    *,
    total: int,
    backoff_factor: float = 1.0,
    status_forcelist: Collection[int] = DEFAULT_STATUS_FORCELIST,
    headers: Mapping[str, str] | None = None,
    timeout: float | None = None,
) -> Response:
    """Send a GET request, retrying it as urllib3's Retry does for GET.

    A failed attempt (no response) is retried, as is a response whose status is in status_forcelist, or in
    RETRY_AFTER_STATUSES with a Retry-After header. Before retry n the wait is the response's Retry-After, if any, or
    else 0 for the first retry and backoff_factor * 2 ** (n - 1) seconds for the next ones, at most BACKOFF_MAX_S.

    Args:
        url (str): The URL, query string included.
        total (int): Retries after the first attempt.
        backoff_factor (float): Base of the exponential wait between retries, in seconds.
        status_forcelist (Collection[int]): Statuses to retry.
        headers (Mapping[str, str] | None): Headers added to the request.
        timeout (float | None): Seconds allowed to connect and for each read, per attempt.

    Returns:
        Response: The first response that is not retried, with a status below 400.

    Raises:
        ConnectionFailed: The last attempt got no response.
        HTTPStatusError: The last response has a 4xx or 5xx status, whether it was retried or not.
    """
    retries = 0
    while True:
        try:
            response = request("GET", url, headers=headers, timeout=timeout)
        except ConnectionFailed:
            if retries >= total:
                raise
            retries += 1
            _sleep(_backoff(retries, backoff_factor))
            continue

        retry_after = _retry_after(response)
        retryable = response.status_code in status_forcelist or (retry_after is not None and response.status_code in RETRY_AFTER_STATUSES)
        if retryable and retries < total:
            retries += 1
            _sleep(retry_after if retry_after else _backoff(retries, backoff_factor))
            continue
        response.raise_for_status()
        return response


def _build_request(
    method: str,
    url: str,
    *,
    json: Any = None,  # noqa: ANN401
    files: Files | None = None,
    data: bytes | None = None,
    headers: Mapping[str, str] | None = None,
) -> urllib.request.Request:
    import urllib.request

    all_headers = dict(headers or {})
    body = data
    if json is not None:
        body = _json.dumps(json, allow_nan=False).encode()
        all_headers.setdefault("Content-Type", "application/json")
    elif files is not None:
        body, content_type = _multipart(files)
        all_headers.setdefault("Content-Type", content_type)
    return urllib.request.Request(url, data=body, headers=all_headers, method=method)


def _open(req: urllib.request.Request, timeout: float | None) -> http.client.HTTPResponse:
    import urllib.request

    if timeout is None:  # Leave the socket default in place, as requests does
        return urllib.request.urlopen(req)
    return urllib.request.urlopen(req, timeout=timeout)


def _multipart(files: Files) -> tuple[bytes, str]:
    boundary = os.urandom(16).hex()
    parts: list[bytes] = []
    for field, (filename, content, content_type) in files.items():
        payload = content if isinstance(content, bytes) else content.read()
        head = f'--{boundary}\r\nContent-Disposition: form-data; name="{field}"; filename="{filename}"\r\nContent-Type: {content_type}\r\n\r\n'
        parts.append(head.encode() + payload + b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def _backoff(retry: int, backoff_factor: float) -> float:
    if retry <= 1:
        return 0.0
    return min(BACKOFF_MAX_S, backoff_factor * 2 ** (retry - 1))


def _retry_after(response: Response) -> float | None:
    """Seconds requested by the Retry-After header, at most RETRY_AFTER_MAX_S; None when absent or unreadable."""
    import email.utils
    import re
    import time

    value = response.headers.get("Retry-After")
    if value is None:
        return None
    if re.match(r"^\s*[0-9]+\s*$", value):
        seconds = float(int(value))
    else:
        parsed = email.utils.parsedate_tz(value)
        if parsed is None:
            return None
        seconds = email.utils.mktime_tz(parsed) - time.time()
    return min(max(seconds, 0.0), RETRY_AFTER_MAX_S)


def _sleep(seconds: float) -> None:
    import time

    if seconds > 0:
        time.sleep(seconds)
