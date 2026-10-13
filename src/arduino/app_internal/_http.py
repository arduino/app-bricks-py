# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""A small HTTP client on the Python standard library.

The bricks reach the local inference runners and a few cloud APIs with it: one request at a time, JSON and multipart
bodies, 4xx and 5xx responses returned like any other, a body read as it arrives, and a retry on GET. The stdlib
client is imported with the first request: on the board it costs about a third of ``requests``, which every app with
an AI brick would otherwise load before "App started". ``arduino.app_utils.HttpClient``, the public client on
requests, is unaffected.
"""

from __future__ import annotations

import functools
import json as _json
import os
import time
from collections.abc import Generator, Iterator, Mapping
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, cast
from urllib.parse import urlencode

if TYPE_CHECKING:
    import http.client
    import urllib.request
    from email.message import Message

Files = Mapping[str, tuple[str, bytes, str]]
"""Multipart file parts, as requests takes them: field name -> (file name, content, content type)."""

BACKOFF_MAX_S = 120.0
"""Longest wait between two retries."""

_sleep = time.sleep


class Response:
    """Status, headers and body of an HTTP response."""

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


class Stream:
    """An HTTP response whose body is read as it arrives."""

    def __init__(self, status_code: int, response: http.client.HTTPResponse) -> None:
        self.status_code = status_code
        self._response = response

    def iter_content(self, chunk_size: int) -> Iterator[bytes]:
        """Yield the body as it arrives, in pieces of at most chunk_size bytes, until it ends."""
        while chunk := self._response.read1(chunk_size):
            yield chunk

    def json(self) -> Any:  # noqa: ANN401
        """The rest of the body decoded as JSON."""
        return _json.loads(self._response.read())


class ConnectionFailed(OSError):
    """No HTTP response arrived: refused or reset connection, unknown host, timeout, malformed or truncated response."""


class HTTPStatusError(Exception):
    """A response has a 4xx or 5xx status where a successful one is required."""

    def __init__(self, response: Response) -> None:
        super().__init__(f"HTTP {response.status_code}: {response.text[:200]}")
        self.response = response


def get(
    url: str,
    *,
    params: Mapping[str, str] | None = None,
    headers: Mapping[str, str] | None = None,
    timeout: float | None = None,
    retries: int = 0,
    backoff_factor: float = 1.0,
) -> Response:
    """Send a GET request and return the response, whatever its status.

    With retries, an attempt that gets no response or a 5xx status is repeated: at once the first time, then after
    backoff_factor * 2 ** (n - 1) seconds, at most BACKOFF_MAX_S. The last response is returned as it is.

    Args:
        url (str): The URL.
        params (Mapping[str, str] | None): Query parameters, URL-encoded and appended to the URL.
        headers (Mapping[str, str] | None): Headers added to the request.
        timeout (float | None): Seconds allowed to connect and for each read; None waits as long as the socket default.
        retries (int): Attempts after the first one.
        backoff_factor (float): Base of the exponential wait between retries, in seconds.

    Returns:
        Response: The response, 4xx and 5xx included.

    Raises:
        ConnectionFailed: No response arrived, at the last attempt either.
    """
    if params:
        url = f"{url}?{urlencode(params)}"
    attempt = 0
    while True:
        try:
            response = _request("GET", url, headers=headers, timeout=timeout)
        except ConnectionFailed:
            if attempt >= retries:
                raise
        else:
            if response.status_code < 500 or attempt >= retries:
                return response
        attempt += 1
        _sleep(_backoff(attempt, backoff_factor))


def post(
    url: str,
    *,
    json: Any = None,  # noqa: ANN401
    files: Files | None = None,
    headers: Mapping[str, str] | None = None,
    timeout: float | None = None,
) -> Response:
    """Send a POST request and return the response, whatever its status.

    Args:
        url (str): The URL.
        json (Any): A body to send as JSON. NaN and infinities are rejected, as requests does.
        files (Files | None): A body to send as multipart/form-data.
        headers (Mapping[str, str] | None): Headers added to the request.
        timeout (float | None): Seconds allowed to connect and for each read; None waits as long as the socket default.

    Returns:
        Response: The response, 4xx and 5xx included.

    Raises:
        ConnectionFailed: No response arrived.
        ValueError: The json body cannot be serialized.
    """
    return _request("POST", url, json=json, files=files, headers=headers, timeout=timeout)


@contextmanager
def stream(
    method: str,
    url: str,
    *,
    json: Any = None,  # noqa: ANN401
    headers: Mapping[str, str] | None = None,
    timeout: float | None = None,
) -> Generator[Stream]:
    """Send a request and yield the response with its body still to read, whatever its status.

    The connection is closed when the block exits, whether the body was read to its end or not.

    Raises:
        ConnectionFailed: No response arrived.
        ValueError: The json body cannot be serialized.
    """
    import http.client

    request = _build_request(method, url, json=json, headers=headers)
    try:
        response = _open(request, timeout)
    except (OSError, http.client.HTTPException) as e:
        raise ConnectionFailed(f"{method} {url}: {e}") from e
    with response:
        yield Stream(response.status, response)


def _request(
    method: str,
    url: str,
    *,
    json: Any = None,  # noqa: ANN401
    files: Files | None = None,
    headers: Mapping[str, str] | None = None,
    timeout: float | None = None,
) -> Response:
    import http.client

    request = _build_request(method, url, json=json, files=files, headers=headers)
    try:
        with _open(request, timeout) as response:
            return Response(response.status, response.headers, response.read())
    except (OSError, http.client.HTTPException) as e:  # URLError is an OSError; IncompleteRead is not
        raise ConnectionFailed(f"{method} {url}: {e}") from e


def _build_request(
    method: str,
    url: str,
    *,
    json: Any = None,  # noqa: ANN401
    files: Files | None = None,
    headers: Mapping[str, str] | None = None,
) -> urllib.request.Request:
    import urllib.request  # Imported here, with http.client: slow imports at app start, needed by the first request only

    all_headers = dict(headers or {})
    body = None
    if json is not None:
        body = _json.dumps(json, allow_nan=False).encode()
        all_headers.setdefault("Content-Type", "application/json")
    elif files is not None:
        body, all_headers["Content-Type"] = _multipart(files)
    return urllib.request.Request(url, data=body, headers=all_headers, method=method)


def _open(request: urllib.request.Request, timeout: float | None) -> http.client.HTTPResponse:
    if timeout is None:  # Leave the socket default in place, as requests does
        return cast("http.client.HTTPResponse", _opener().open(request))
    return cast("http.client.HTTPResponse", _opener().open(request, timeout=timeout))


@functools.cache
def _opener() -> urllib.request.OpenerDirector:
    import urllib.request

    class ReturnErrorResponses(urllib.request.HTTPErrorProcessor):
        """Return 4xx and 5xx responses like the others, instead of raising HTTPError; redirects are still followed."""

        def http_response(self, request: urllib.request.Request, response: http.client.HTTPResponse) -> http.client.HTTPResponse:
            if 300 <= response.status < 400:
                return super().http_response(request, response)
            return response

        https_response = http_response

    return urllib.request.build_opener(ReturnErrorResponses)


def _multipart(files: Files) -> tuple[bytes, str]:
    boundary = os.urandom(16).hex()
    parts: list[bytes] = []
    for field, (filename, content, content_type) in files.items():
        head = f'--{boundary}\r\nContent-Disposition: form-data; name="{field}"; filename="{filename}"\r\nContent-Type: {content_type}\r\n\r\n'
        parts.append(head.encode() + content + b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def _backoff(attempt: int, backoff_factor: float) -> float:
    if attempt <= 1:
        return 0.0
    return min(BACKOFF_MAX_S, backoff_factor * 2 ** (attempt - 1))
