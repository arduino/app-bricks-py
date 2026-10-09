# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""``http.client`` connections to an HTTP server given by URL, AF_UNIX sockets included.

The URL is ``http://`` or ``https://``, or ``http+unix://`` with the socket path
percent-encoded as host (e.g.
``http+unix://%2Frun%2Farduino-cloud-connector%2Fdaemon.sock``), which keeps an
API off every network interface.
"""

import http.client
import socket
from urllib.parse import unquote, urlparse


class UnixHTTPConnection(http.client.HTTPConnection):
    """HTTP connection over an AF_UNIX socket."""

    def __init__(self, socket_path: str, timeout: float) -> None:
        # The host only fills the Host header
        super().__init__("localhost", timeout=timeout)
        self._socket_path = socket_path

    def connect(self) -> None:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            sock.settimeout(self.timeout)
            sock.connect(self._socket_path)
        except OSError:
            sock.close()
            raise
        self.sock = sock


class HTTPEndpoint:
    """An HTTP server address parsed from its URL."""

    def __init__(self, url: str) -> None:
        parsed = urlparse(url.rstrip("/"))
        if parsed.scheme not in ("http", "https", "http+unix"):
            raise ValueError(f"unsupported URL scheme: {url}")
        self._scheme = parsed.scheme
        self._netloc = parsed.netloc
        self.path_prefix = parsed.path
        self.socket_path = unquote(parsed.netloc) if parsed.scheme == "http+unix" else None

    def connection(self, timeout: float) -> http.client.HTTPConnection:
        """Create an unconnected connection to the server."""
        if self.socket_path is not None:
            return UnixHTTPConnection(self.socket_path, timeout)
        if self._scheme == "https":
            return http.client.HTTPSConnection(self._netloc, timeout=timeout)
        return http.client.HTTPConnection(self._netloc, timeout=timeout)
