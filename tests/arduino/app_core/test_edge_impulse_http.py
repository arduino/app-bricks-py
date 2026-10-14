# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""The Edge Impulse facade against a fake runner on a local HTTP server, and the model information cache."""

import json
import threading
from collections.abc import Iterator
from dataclasses import dataclass, field
from email.parser import BytesParser
from email.policy import default as default_policy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from arduino.app_utils import http
from arduino.app_internal.core.ei import EdgeImpulseRunnerFacade

MODEL_INFO = {
    "project": {"name": "test", "id": 1},
    "modelParameters": {"model_type": "classification", "input_features_count": 3, "thresholds": [], "labels": ["a", "b"]},
}


@dataclass
class Runner:
    url: str
    info_statuses: list[int] = field(default_factory=lambda: [200])
    """Statuses of /api/info, in order; the last one is repeated."""
    received: list[tuple[str, str, dict[str, str], bytes]] = field(default_factory=list)


@pytest.fixture
def runner(monkeypatch: pytest.MonkeyPatch) -> Iterator[Runner]:
    """A fake Edge Impulse runner; the facade class resolves its address to it."""
    state = Runner(url="")

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _answer(self) -> None:
            body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
            state.received.append((self.command, self.path, dict(self.headers), body))
            if self.path == "/api/info":
                status = state.info_statuses.pop(0) if len(state.info_statuses) > 1 else state.info_statuses[0]
                payload = MODEL_INFO if status == 200 else {"error": "loading"}
            else:
                status, payload = 200, {"result": {"classification": {"a": 0.9, "b": 0.1}}}
            data = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        do_GET = do_POST = _answer

        def log_message(self, format: str, *args: object) -> None:
            pass

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()
    state.url = f"http://127.0.0.1:{httpd.server_address[1]}"
    monkeypatch.setattr(EdgeImpulseRunnerFacade, "_get_ei_url", classmethod(lambda cls: state.url))
    yield state
    httpd.shutdown()
    httpd.server_close()


@pytest.fixture
def sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    recorded: list[float] = []
    monkeypatch.setattr(http, "_sleep", recorded.append)
    return recorded


def test_the_image_reaches_the_runner_as_a_multipart_file(runner: Runner):
    result = EdgeImpulseRunnerFacade().infer_from_image(b"\xff\xd8 jpeg bytes", "jpg")

    assert result == {"result": {"classification": {"a": 0.9, "b": 0.1}}}
    method, path, headers, body = runner.received[0]
    assert (method, path) == ("POST", "/api/image")
    message = BytesParser(policy=default_policy).parsebytes(f"Content-Type: {headers['Content-Type']}\r\n\r\n".encode() + body)
    (part,) = list(message.iter_parts())
    assert part.get_param("name", header="content-disposition") == "file"
    assert part.get_filename() == "image.jpeg"
    assert part.get_content_type() == "image/jpeg"
    assert part.get_payload(decode=True) == b"\xff\xd8 jpeg bytes"


def test_model_info_waits_for_a_runner_still_loading(runner: Runner, sleeps: list[float]):
    runner.info_statuses = [503, 503, 200]

    info = EdgeImpulseRunnerFacade.get_model_info()

    assert info is not None and info.input_features_count == 3
    assert [path for _, path, _, _ in runner.received] == ["/api/info"] * 3
    assert sleeps == [0.0, 2.0]


def test_model_info_gives_up_after_the_retries(runner: Runner, sleeps: list[float]):
    runner.info_statuses = [503]

    assert EdgeImpulseRunnerFacade.get_model_info() is None
    assert len(runner.received) == 7
    assert sleeps == [0.0, 2.0, 4.0, 8.0, 16.0, 32.0]


def test_features_are_sent_as_json_trimmed_to_the_model_input(runner: Runner):
    result = EdgeImpulseRunnerFacade.infer_from_features([1.0, 2.0, 3.0, 4.0, 5.0])

    assert result == {"result": {"classification": {"a": 0.9, "b": 0.1}}}
    method, path, headers, body = runner.received[-1]
    assert (method, path, headers["Content-Type"]) == ("POST", "/api/features", "application/json")
    assert json.loads(body) == {"features": [1.0, 2.0, 3.0]}


def test_the_model_info_is_fetched_once_per_class(runner: Runner):
    for _ in range(3):
        assert EdgeImpulseRunnerFacade.infer_from_features([1.0, 2.0, 3.0]) is not None

    assert [path for _, path, _, _ in runner.received] == ["/api/info", "/api/features", "/api/features", "/api/features"]


def test_a_failed_model_info_fetch_is_not_kept(runner: Runner, sleeps: list[float]):
    runner.info_statuses = [503] * 7 + [200]

    assert EdgeImpulseRunnerFacade.infer_from_features([1.0, 2.0, 3.0]) is None  # The runner never answered
    assert [path for _, path, _, _ in runner.received].count("/api/info") == 7  # 1 attempt + 6 retries, then gave up
    assert EdgeImpulseRunnerFacade.infer_from_features([1.0, 2.0, 3.0]) is not None  # Asked again, and now it does
    assert [path for _, path, _, _ in runner.received].count("/api/info") == 8
