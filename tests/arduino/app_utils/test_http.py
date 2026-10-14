# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""arduino.app_utils.http against a local HTTP server: encoding, bodies, statuses, streaming, retries, Unix sockets and its imports."""

import json
import socket
import socketserver
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from email.parser import BytesParser
from email.policy import default as default_policy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from arduino.app_utils import http


@dataclass
class Server:
    url: str
    received: list[tuple[str, str, dict[str, str], bytes]] = field(default_factory=list)
    """(method, path, headers, body) of each request."""
    flaky_statuses: list[int] = field(default_factory=lambda: [503, 503, 200])
    """Statuses answered to /flaky, in order; the last one is repeated."""
    release_stream: threading.Event = field(default_factory=threading.Event)
    """Lets /stream send its second chunk."""


@pytest.fixture
def server() -> Iterator[Server]:
    state = Server(url="")

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _handle(self) -> None:
            body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
            state.received.append((self.command, self.path, dict(self.headers), body))
            path = self.path.split("?")[0]
            if path == "/stream":
                self.send_response(200)
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                for index, chunk in enumerate([b"first", b"second"]):
                    if index == 1:
                        state.release_stream.wait(5)
                    self.wfile.write(f"{len(chunk):x}\r\n".encode() + chunk + b"\r\n")
                    self.wfile.flush()
                self.wfile.write(b"0\r\n\r\n")
                return
            if path == "/slow":
                time.sleep(0.5)  # The client gives up meanwhile: the answer below lands on a closed socket
            if path.startswith("/status/"):
                status = int(path.rsplit("/", 1)[1])
            elif path == "/flaky":
                status = state.flaky_statuses.pop(0) if len(state.flaky_statuses) > 1 else state.flaky_statuses[0]
            else:
                status = 200
            payload = json.dumps({"status": status, "method": self.command, "body": body.decode(errors="replace")}).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(payload)

        def handle(self) -> None:
            try:
                super().handle()
            except (BrokenPipeError, ConnectionResetError):  # The client went away: /slow
                pass

        do_GET = do_POST = do_HEAD = _handle

        def log_message(self, format: str, *args: object) -> None:
            pass

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()
    state.url = f"http://127.0.0.1:{httpd.server_address[1]}"
    yield state
    state.release_stream.set()
    httpd.shutdown()
    httpd.server_close()


@pytest.fixture
def sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """The waits between retries, recorded instead of slept."""
    recorded: list[float] = []
    monkeypatch.setattr(http, "_sleep", recorded.append)
    return recorded


@pytest.fixture
def closed_port() -> str:
    """A URL on a port nothing listens to."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return f"http://127.0.0.1:{sock.getsockname()[1]}"


def test_get_encodes_the_query_parameters_and_sends_the_headers(server: Server):
    response = http.get(f"{server.url}/items", params={"q": "a b", "t": "x&y"}, headers={"X-Token": "secret"})

    assert response.status_code == 200
    assert response.json()["method"] == "GET"
    method, path, headers, _ = server.received[0]
    assert (method, path, headers["X-Token"]) == ("GET", "/items?q=a+b&t=x%26y", "secret")
    assert response.headers.get("Content-Type") == "application/json"


def test_the_url_path_is_percent_encoded_as_requests_does(server: Server):
    """Spaces and non-ASCII letters get encoded; RFC 3986 path characters and what is already encoded stay as they are."""
    http.get(f"{server.url}/feed/São Paulo/geo:45.07;7.68/already%20done/", params={"token": "t k"})

    assert server.received[0][1] == "/feed/S%C3%A3o%20Paulo/geo:45.07;7.68/already%20done/?token=t+k"


def test_post_sends_json_with_its_content_type(server: Server):
    response = http.post(f"{server.url}/items", json={"features": [1.0, 2.5]})

    assert response.status_code == 200
    _, _, headers, body = server.received[0]
    assert headers["Content-Type"] == "application/json"
    assert json.loads(body) == {"features": [1.0, 2.5]}


def test_head_sends_no_body_and_gets_none_back(server: Server):
    response = http.head(f"{server.url}/items", timeout=5)

    assert response.status_code == 200
    assert response.content == b""
    assert response.headers.get("Content-Type") == "application/json"
    method, path, _, body = server.received[0]
    assert (method, path, body) == ("HEAD", "/items", b"")


def test_basic_authentication_goes_in_the_authorization_header(server: Server):
    http.head(f"{server.url}/items", auth=("admin", "s:cret"))
    http.get(f"{server.url}/items", auth=("admin", "s:cret"))
    http.post(f"{server.url}/items", json={}, auth=("admin", "s:cret"))

    assert [headers["Authorization"] for _, _, headers, _ in server.received] == ["Basic YWRtaW46czpjcmV0"] * 3


def test_post_rejects_nan_in_json_as_requests_does(server: Server):
    with pytest.raises(ValueError):
        http.post(f"{server.url}/items", json={"value": float("nan")})
    assert server.received == []


def test_post_sends_files_as_multipart_form_data(server: Server):
    http.post(f"{server.url}/image", files={"file": ("image.jpeg", b"\xff\xd8 jpeg bytes", "image/jpeg")})

    _, _, headers, body = server.received[0]
    message = BytesParser(policy=default_policy).parsebytes(f"Content-Type: {headers['Content-Type']}\r\n\r\n".encode() + body)
    (part,) = list(message.iter_parts())
    assert part.get_param("name", header="content-disposition") == "file"
    assert part.get_filename() == "image.jpeg"
    assert part.get_content_type() == "image/jpeg"
    assert part.get_payload(decode=True) == b"\xff\xd8 jpeg bytes"


@pytest.mark.parametrize("status", [404, 409, 500])
def test_error_statuses_are_returned_with_their_body(server: Server, status: int):
    response = http.get(f"{server.url}/status/{status}")

    assert response.status_code == status
    assert response.json()["status"] == status
    assert response.text.startswith('{"status"')
    with pytest.raises(http.HTTPStatusError, match=f"HTTP {status}: ") as info:
        response.raise_for_status()
    assert info.value.response is response


def test_a_successful_response_passes_raise_for_status(server: Server):
    http.get(f"{server.url}/ok").raise_for_status()


def test_no_response_raises_connection_failed(closed_port: str):
    with pytest.raises(http.ConnectionFailed, match="GET") as info:
        http.get(f"{closed_port}/items")
    assert isinstance(info.value, OSError)
    with pytest.raises(http.ConnectionFailed, match="POST"):
        http.post(f"{closed_port}/items", json={})


def test_a_timeout_raises_connection_failed(server: Server):
    with pytest.raises(http.ConnectionFailed, match="timed out"):
        http.get(f"{server.url}/slow", timeout=0.1)


def test_get_retries_5xx_responses_with_an_exponential_backoff(server: Server, sleeps: list[float]):
    response = http.get(f"{server.url}/flaky", retries=5)

    assert response.status_code == 200
    assert [path for _, path, _, _ in server.received] == ["/flaky"] * 3
    assert sleeps == [0.0, 2.0]


def test_get_returns_the_last_5xx_response_after_the_retries(server: Server, sleeps: list[float]):
    response = http.get(f"{server.url}/status/503", retries=3, backoff_factor=0.5)

    assert response.status_code == 503
    assert len(server.received) == 4
    assert sleeps == [0.0, 1.0, 2.0]


def test_get_does_not_retry_4xx_responses(server: Server, sleeps: list[float]):
    assert http.get(f"{server.url}/status/404", retries=3).status_code == 404
    assert len(server.received) == 1
    assert sleeps == []


def test_get_retries_connection_failures_then_raises(closed_port: str, sleeps: list[float]):
    with pytest.raises(http.ConnectionFailed):
        http.get(f"{closed_port}/items", retries=2)
    assert sleeps == [0.0, 2.0]


def test_the_backoff_is_capped():
    assert [http._backoff(attempt, 1.0) for attempt in range(1, 8)] == [0.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]  # pyright: ignore[reportPrivateUsage]
    assert http._backoff(20, 1.0) == http.BACKOFF_MAX_S  # pyright: ignore[reportPrivateUsage]


def test_stream_yields_the_body_as_it_arrives(server: Server):
    with http.stream("POST", f"{server.url}/stream", json={"text": "hello"}) as response:
        assert response.status_code == 200
        chunks = response.iter_content(1024)
        assert next(chunks) == b"first"  # Before the server sends the rest
        server.release_stream.set()
        assert list(chunks) == [b"second"]


def test_stream_returns_error_statuses_with_their_body(server: Server):
    with http.stream("POST", f"{server.url}/status/500", json={}) as response:
        assert response.status_code == 500
        assert response.json()["status"] == 500


def test_stream_raises_connection_failed_without_a_response(closed_port: str):
    with pytest.raises(http.ConnectionFailed):
        with http.stream("POST", f"{closed_port}/stream", json={}):
            pass


@pytest.fixture
def unix_server(tmp_path) -> Iterator[tuple[str, list[tuple[str, str, dict[str, str]]]]]:
    """An HTTP server on a Unix domain socket, answering GET /scan with JSON; yields (socket path, received requests)."""
    if not hasattr(socket, "AF_UNIX"):
        pytest.skip("AF_UNIX not available on this platform")
    received: list[tuple[str, str, dict[str, str]]] = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            received.append((self.command, self.path, dict(self.headers)))
            body = json.dumps({"access_points": []} if self.path == "/scan" else {"error": "not found"}).encode()
            self.send_response(200 if self.path == "/scan" else 404)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            pass

    class UnixHTTPServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
        pass

    path = str(tmp_path / "scanner with space.sock")
    httpd = UnixHTTPServer(path, Handler)
    threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()
    yield path, received
    httpd.shutdown()
    httpd.server_close()


def test_a_service_on_a_unix_socket_is_reached_through_its_http_unix_url(unix_server):
    path, received = unix_server
    url = http.unix_socket_url(path)

    assert url == "http+unix://" + path.replace("/", "%2F").replace(" ", "%20")
    response = http.get(f"{url}/scan", timeout=5)
    assert response.status_code == 200
    assert response.json() == {"access_points": []}
    assert http.get(f"{url}/missing").status_code == 404
    method, request_path, headers = received[0]
    assert (method, request_path, headers["Host"]) == ("GET", "/scan", "localhost")


def test_a_missing_unix_socket_raises_connection_failed(tmp_path):
    if not hasattr(socket, "AF_UNIX"):
        pytest.skip("AF_UNIX not available on this platform")
    missing = http.unix_socket_url(str(tmp_path / "missing.sock"))
    with pytest.raises(http.ConnectionFailed, match="scan"):
        http.get(f"{missing}/scan", timeout=1)


def test_importing_the_client_loads_no_http_library():
    """App start time: the stdlib client, like requests, loads with the first request."""
    heavy = ["requests", "urllib3", "urllib.request", "http.client", "ssl"]
    code = f"import json, sys\nimport arduino.app_utils.http\nprint(json.dumps([m for m in {heavy!r} if m in sys.modules]))"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120, check=True)

    assert json.loads(result.stdout.strip().splitlines()[-1]) == []
