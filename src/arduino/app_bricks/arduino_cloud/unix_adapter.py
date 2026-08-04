# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""A ``requests`` transport adapter that speaks HTTP over an AF_UNIX socket.

The arduino-cloud-connector daemon serves its REST/SSE API on a UNIX-domain socket
(bind-mounted into the app container) so the API is never exposed on a network
interface. ``requests`` has no built-in UNIX-socket support, so this adapter
adds it without pulling an extra dependency: URLs use the ``http+unix://`` scheme
with the socket path percent-encoded in the host component
(e.g. ``http+unix://%2Frun%2Farduino-cloud-connector%2Fdaemon.sock/v1/variables/led``).

It works for plain requests and for streaming (SSE), because only the underlying
connection transport is swapped — everything else is standard urllib3.
"""

import socket
from collections.abc import Mapping
from typing import Any

from requests import PreparedRequest
from requests.adapters import HTTPAdapter
from urllib3.connection import HTTPConnection
from urllib3.connectionpool import HTTPConnectionPool


class _UnixHTTPConnection(HTTPConnection):
    def __init__(self, socket_path: str, **kwargs: Any) -> None:
        # The host is irrelevant for an AF_UNIX socket, but urllib3 requires one.
        super().__init__("localhost", **kwargs)
        self._unix_socket_path = socket_path

    def connect(self) -> None:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        # urllib3 sets self.timeout to the per-request connect timeout before
        # calling connect(); honour it when it is a concrete value.
        if isinstance(self.timeout, (int, float)):
            sock.settimeout(self.timeout)
        sock.connect(self._unix_socket_path)
        self.sock = sock


class _UnixHTTPConnectionPool(HTTPConnectionPool):
    def __init__(self, socket_path: str, **kwargs: Any) -> None:
        super().__init__("localhost", **kwargs)
        self._unix_socket_path = socket_path

    def _new_conn(self) -> _UnixHTTPConnection:
        return _UnixHTTPConnection(self._unix_socket_path, timeout=self.timeout)


class UnixHTTPAdapter(HTTPAdapter):
    """Routes ``http+unix://`` requests over a fixed AF_UNIX socket path."""

    def __init__(self, socket_path: str, **kwargs: Any) -> None:
        self._unix_socket_path = socket_path
        self._pool = None
        super().__init__(**kwargs)

    def _pool_for(self) -> _UnixHTTPConnectionPool:
        if self._pool is None:
            self._pool = _UnixHTTPConnectionPool(self._unix_socket_path)
        return self._pool

    # requests >= 2.32 calls this; older versions call get_connection.
    def get_connection_with_tls_context(
        self,
        request: PreparedRequest,
        verify: bool | str | None,
        proxies: Mapping[str, str] | None = None,
        cert: tuple[str, str] | str | None = None,
    ) -> _UnixHTTPConnectionPool:
        return self._pool_for()

    def get_connection(self, url: str | bytes, proxies: Mapping[str, str] | None = None) -> _UnixHTTPConnectionPool:
        return self._pool_for()

    def request_url(self, request: PreparedRequest, proxies: Mapping[str, str] | None) -> str:
        # Use the path (and query) only, so the request line is
        # "GET /v1/variables/led HTTP/1.1" rather than the http+unix:// URL.
        return request.path_url

    def close(self) -> None:
        if self._pool is not None:
            self._pool.close()
            self._pool = None
        super().close()
