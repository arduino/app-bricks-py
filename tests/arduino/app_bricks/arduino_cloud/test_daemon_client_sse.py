# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import http.server
import io
import socket
import socketserver
import tempfile
import threading
from collections.abc import Iterator
from urllib.parse import quote

import pytest

from arduino.app_bricks.arduino_cloud.daemon_client import DaemonClient

STREAM = b': heartbeat\nevent: lastvalue\ndata: {"value": 1}\n\nevent: update\ndata: {"value": 2}\n\n'


def test_sse_parser_yields_complete_events():
    events = list(DaemonClient._iter_events(io.BytesIO(STREAM), threading.Event()))

    assert events == [("lastvalue", {"value": 1}), ("update", {"value": 2})]


class _UnixHTTPServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True

    def get_request(self):
        # BaseHTTPRequestHandler expects a (host, port) client address
        return self.socket.accept()[0], ("localhost", 0)


@pytest.fixture(params=["tcp", "unix"])
def serve(request: pytest.FixtureRequest) -> Iterator:
    """Start a server for a handler class and return the daemon URL to reach it."""
    servers: list[socketserver.BaseServer] = []

    def start(handler: type[http.server.BaseHTTPRequestHandler], sock_dir: str) -> str:
        if request.param == "unix":
            if not hasattr(socket, "AF_UNIX"):
                pytest.skip("AF_UNIX not available on this platform")
            path = f"{sock_dir}/daemon.sock"
            server: socketserver.BaseServer = _UnixHTTPServer(path, handler)
            url = "http+unix://" + quote(path, safe="")
        else:
            server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
            server.daemon_threads = True
            url = f"http://127.0.0.1:{server.server_address[1]}"
        servers.append(server)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        return url

    with tempfile.TemporaryDirectory(prefix="cloud") as sock_dir:
        yield lambda handler: start(handler, sock_dir)
        for server in servers:
            server.shutdown()
            server.server_close()


def test_close_unblocks_an_idle_stream(serve):
    # Like the daemon: one chunked frame, then silence with the stream held open
    release = threading.Event()

    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            frame = b'event: lastvalue_missing\ndata: {"name": "temp"}\n\n'
            self.wfile.write(b"%x\r\n%s\r\n" % (len(frame), frame))
            self.wfile.flush()
            release.wait(5)

        def log_message(self, *args):
            pass

    client = DaemonClient(serve(Handler))
    stop = threading.Event()
    ready = threading.Event()
    t = threading.Thread(target=client.stream_events, args=("temp", lambda event, payload: None, stop, ready), daemon=True)
    t.start()
    try:
        assert ready.wait(timeout=1), "the first frame was not delivered while the stream stayed open"
        stop.set()
        client.close()
        t.join(timeout=1)
        assert not t.is_alive(), "close() did not unblock the idle stream"
    finally:
        release.set()
