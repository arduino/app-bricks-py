# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""The standard-library HTTP client: requests against a local server, and urllib3's retry rules for GET."""

import json
import socket
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

from arduino.app_internal import _http


@dataclass
class Reply:
    status: int = 200
    body: bytes = b"{}"
    headers: dict[str, str] = field(default_factory=dict)
    delay: float = 0.0


@dataclass
class Received:
    method: str
    path: str
    headers: dict[str, str]
    body: bytes


@dataclass
class Server:
    url: str
    replies: list[Reply]
    """Answered in order; the last one is repeated once the others are used up."""
    received: list[Received] = field(default_factory=list)


@pytest.fixture
def server() -> Iterator[Server]:
    state = Server(url="", replies=[Reply()])

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _answer(self) -> None:
            length = int(self.headers.get("Content-Length") or 0)
            state.received.append(Received(self.command, self.path, dict(self.headers), self.rfile.read(length)))
            reply = state.replies.pop(0) if len(state.replies) > 1 else state.replies[0]
            time.sleep(reply.delay)
            self.send_response(reply.status)
            for name, value in reply.headers.items():
                self.send_header(name, value)
            self.send_header("Content-Length", str(len(reply.body)))
            self.end_headers()
            self.wfile.write(reply.body)

        do_GET = do_POST = _answer

        def log_message(self, format: str, *args: object) -> None:
            pass

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()
    state.url = f"http://127.0.0.1:{httpd.server_address[1]}"
    yield state
    httpd.shutdown()
    httpd.server_close()


@pytest.fixture
def sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """The waits between retries, recorded instead of slept."""
    recorded: list[float] = []
    monkeypatch.setattr(_http, "_sleep", recorded.append)
    return recorded


def closed_port_url() -> str:
    with socket.socket() as s:  # A port the OS just handed out and nobody listens on
        s.bind(("127.0.0.1", 0))
        return f"http://127.0.0.1:{s.getsockname()[1]}/"


def test_import_loads_no_http_stack():
    """App start time: the HTTP modules load with the first request."""
    heavy = ["urllib.request", "http.client", "ssl", "email.utils"]
    code = f"import json, sys\nimport arduino.app_internal._http\nprint(json.dumps([m for m in {heavy!r} if m in sys.modules]))"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120, check=True)
    assert json.loads(result.stdout.strip().splitlines()[-1]) == []


def test_request_returns_error_statuses_with_their_body(server: Server):
    server.replies = [Reply(status=409, body=b'{"error": "busy"}')]

    response = _http.request("GET", f"{server.url}/x")

    assert response.status_code == 409
    assert response.json() == {"error": "busy"}
    assert response.text == '{"error": "busy"}'


def test_post_sends_json(server: Server):
    _http.post(f"{server.url}/api/features", json={"features": [1.5, 2]})

    received = server.received[0]
    assert (received.method, received.path) == ("POST", "/api/features")
    assert received.headers["Content-Type"] == "application/json"
    assert json.loads(received.body) == {"features": [1.5, 2]}


def test_post_rejects_nan_as_requests_does():
    with pytest.raises(ValueError):
        _http.post("http://127.0.0.1:9/", json={"features": [float("nan")]})


@pytest.mark.parametrize("as_file", [False, True], ids=["bytes", "file object"])
def test_post_sends_a_multipart_file_intact(server: Server, as_file: bool):
    import io

    content = b"\x89PNG\r\n\x00\xff binary\r\n--not-the-boundary"
    _http.post(f"{server.url}/api/image", files={"file": ("image.png", io.BytesIO(content) if as_file else content, "image/png")})

    received = server.received[0]
    message = BytesParser(policy=default_policy).parsebytes(f"Content-Type: {received.headers['Content-Type']}\r\n\r\n".encode() + received.body)
    (part,) = list(message.iter_parts())
    assert part.get_param("name", header="content-disposition") == "file"
    assert part.get_filename() == "image.png"
    assert part.get_content_type() == "image/png"
    assert part.get_payload(decode=True) == content


def test_no_response_raises_connection_failed():
    with pytest.raises(_http.ConnectionFailed):
        _http.request("GET", closed_port_url())


def test_timeout_raises_connection_failed(server: Server):
    server.replies = [Reply(delay=1.0)]

    with pytest.raises(_http.ConnectionFailed):
        _http.request("GET", f"{server.url}/slow", timeout=0.2)


def test_get_with_retry_waits_like_urllib3_and_gives_up_after_total_retries(server: Server, sleeps: list[float]):
    server.replies = [Reply(status=503)]

    with pytest.raises(_http.HTTPStatusError) as exc_info:
        _http.get_with_retry(f"{server.url}/api/info", total=6)

    assert exc_info.value.response.status_code == 503
    assert len(server.received) == 7
    assert sleeps == [0.0, 2.0, 4.0, 8.0, 16.0, 32.0]


def test_get_with_retry_returns_the_first_good_response(server: Server, sleeps: list[float]):
    server.replies = [Reply(status=502), Reply(status=504), Reply(status=200, body=b'{"ok": true}')]

    response = _http.get_with_retry(f"{server.url}/api/info", total=6)

    assert response.json() == {"ok": True}
    assert sleeps == [0.0, 2.0]


def test_retry_after_replaces_the_backoff(server: Server, sleeps: list[float]):
    server.replies = [Reply(status=503), Reply(status=503, headers={"Retry-After": "7"}), Reply(status=200)]

    _http.get_with_retry(f"{server.url}/api/info", total=6)

    assert sleeps == [0.0, 7.0]


def test_status_with_retry_after_is_retried_only_with_the_header(server: Server, sleeps: list[float]):
    server.replies = [Reply(status=429, headers={"Retry-After": "1"}), Reply(status=200)]
    assert _http.get_with_retry(f"{server.url}/a", total=6).status_code == 200

    server.replies = [Reply(status=429)]
    with pytest.raises(_http.HTTPStatusError):
        _http.get_with_retry(f"{server.url}/b", total=6)
    assert sleeps == [1.0]


def test_other_errors_are_not_retried(server: Server, sleeps: list[float]):
    server.replies = [Reply(status=404)]

    with pytest.raises(_http.HTTPStatusError):
        _http.get_with_retry(f"{server.url}/missing", total=6)

    assert len(server.received) == 1
    assert sleeps == []


def test_no_response_is_retried(sleeps: list[float]):
    with pytest.raises(_http.ConnectionFailed):
        _http.get_with_retry(closed_port_url(), total=3)

    assert sleeps == [0.0, 2.0, 4.0]


def test_stream_reads_the_body_as_it_arrives(server: Server):
    server.replies = [Reply(body=b"x" * 10_000)]

    with _http.stream("POST", f"{server.url}/tts", json={"text": "hi"}) as response:
        assert response.status == 200
        chunks = []
        while chunk := response.read1(4096):
            assert len(chunk) <= 4096
            chunks.append(chunk)

    assert b"".join(chunks) == b"x" * 10_000
    assert json.loads(server.received[0].body) == {"text": "hi"}


def test_stream_yields_error_responses_with_their_body(server: Server):
    server.replies = [Reply(status=500, body=b'{"detail": "boom"}')]

    with _http.stream("POST", f"{server.url}/tts") as response:
        assert response.status == 500
        assert json.loads(response.read()) == {"detail": "boom"}
