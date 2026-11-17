# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import asyncio
import importlib.util
import os
import socket
import threading
import time
from collections.abc import Iterator
from unittest.mock import patch

import numpy as np
import pytest
import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient

from arduino.app_peripherals.camera.base_camera import BaseCamera
from arduino.app_bricks.web_ui import web_ui
from arduino.app_bricks.web_ui.web_ui import WebUI


def test_webui_init_defaults():
    ui = WebUI()
    assert ui._addr == "0.0.0.0"
    assert ui._port == 7000
    assert ui._ui_path_prefix == ""
    assert ui._api_path_prefix == ""
    assert ui._assets_dir_path.endswith(os.path.join("app", "assets"))
    assert ui._certs_dir_path.endswith(os.path.join("app", "certs"))
    assert ui._use_tls is False
    assert ui._protocol == "http"
    assert ui._server is None
    assert ui._server_loop is None


def test_webui_init_use_ssl_deprecated():
    webui = WebUI(use_ssl=True)
    assert webui._use_tls is True


def test_expose_api_route():
    ui = WebUI()

    def dummy():
        return {"ok": True}

    ui.expose_api("GET", "/dummy", dummy)
    client = TestClient(ui.app)
    response = client.get("/dummy")
    assert response.status_code == 200
    assert response.json() == {"ok": True}


def test_on_connect_and_disconnect():
    ui = WebUI()
    called = {"connect": False, "disconnect": False}

    def connect_cb(sid):
        called["connect"] = True

    def disconnect_cb(sid):
        called["disconnect"] = True

    ui.on_connect(connect_cb)
    ui.on_disconnect(disconnect_cb)
    assert ui._on_connect_cb == connect_cb
    assert ui._on_disconnect_cb == disconnect_cb


def test_on_message_registration():
    ui = WebUI()

    def msg_cb(sid, data):
        return "pong"

    ui.on_message("ping", msg_cb)
    assert "ping" in ui._on_message_cbs
    assert ui._on_message_cbs["ping"] == msg_cb


def test_send_message_no_loop():
    ui = WebUI()
    ui.send_message("test", {"msg": "hi"})  # Should not raise


def test_stop_sets_should_exit():
    import unittest.mock

    ui = WebUI()
    dummy_server = unittest.mock.Mock()
    dummy_server.should_exit = False
    ui._server = dummy_server
    ui.stop()
    assert dummy_server.should_exit is True


def test_cors_default():
    """Test that CORS defaults to wildcard allowing any origin."""
    ui = WebUI()
    client = TestClient(ui.app)

    def dummy():
        return {"ok": True}

    ui.expose_api("GET", "/dummy", dummy)
    response = client.get("/dummy", headers={"Origin": "http://example.com"})
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == "*"


def test_cors_single_origin():
    """Test that CORS works with a single specific origin."""
    ui = WebUI(cors_origins="http://localhost:3000")
    client = TestClient(ui.app)

    def dummy():
        return {"ok": True}

    ui.expose_api("GET", "/dummy", dummy)
    response = client.get("/dummy", headers={"Origin": "http://localhost:3000"})
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == "http://localhost:3000"


def test_cors_multiple_origins():
    """Test that CORS works with multiple comma-separated origins."""
    ui = WebUI(cors_origins="http://localhost:3000,https://example.com")
    client = TestClient(ui.app)

    def dummy():
        return {"ok": True}

    ui.expose_api("GET", "/dummy", dummy)
    response = client.get("/dummy", headers={"Origin": "http://localhost:3000"})
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == "http://localhost:3000"

    response = client.get("/dummy", headers={"Origin": "https://example.com"})
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == "https://example.com"


class _FakeCamera(BaseCamera):
    """Camera yielding the given frames, then failing to end the stream."""

    def __init__(self, *frames: np.ndarray) -> None:
        super().__init__(fps=1000)
        self._frames = iter(frames)

    def _open_camera(self) -> None:
        pass

    def _close_camera(self) -> None:
        pass

    def _read_frame(self) -> np.ndarray | None:
        frame = next(self._frames, None)
        if frame is None:
            raise RuntimeError("end")
        return frame


def test_expose_camera_starts_camera_if_not_started():
    ui = WebUI()
    camera = _FakeCamera(np.zeros((2, 2, 3), dtype=np.uint8))

    ui.expose_camera("/stream", camera)
    response = TestClient(ui.app).get("/stream")

    assert camera.is_started()
    assert response.content.startswith(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n\xff\xd8")
    camera.stop()


def test_expose_camera_streams_mjpeg_response():
    ui = WebUI()
    camera = _FakeCamera(np.zeros((2, 2, 3), dtype=np.uint8))
    camera.start()

    fake_jpeg = b"\xff\xd8jpeg"

    with patch("arduino.app_utils.image.compress_to_jpeg", return_value=np.frombuffer(fake_jpeg, dtype=np.uint8)):
        ui.expose_camera("/stream", camera)
        response = TestClient(ui.app).get("/stream")

    assert response.status_code == 200
    assert "multipart/x-mixed-replace" in response.headers["content-type"]
    assert fake_jpeg in response.content
    camera.stop()


def test_expose_camera_passes_quality_to_compress():
    ui = WebUI()
    fake_frame = np.zeros((2, 2, 3), dtype=np.uint8)
    camera = _FakeCamera(fake_frame)
    camera.start()

    with patch("arduino.app_utils.image.compress_to_jpeg", return_value=np.array([0], dtype=np.uint8)) as mock_compress:
        ui.expose_camera("/stream", camera, jpeg_quality=95)
        TestClient(ui.app).get("/stream")

    mock_compress.assert_called_with(fake_frame, quality=95)
    camera.stop()


class _FakeServer:
    """A stand-in for the server exposing the two stop flags and the wake-up."""

    def __init__(self):
        self.should_exit = False
        self.force_exit = False
        self.woken = False

    def wake(self):
        self.woken = True


def test_start_bounds_the_graceful_shutdown():
    """Uvicorn waits forever by default, which never terminates while a client is connected."""
    ui = WebUI()
    ui.start()

    assert ui._server is not None
    assert ui._server.config.timeout_graceful_shutdown == web_ui.GRACEFUL_SHUTDOWN_TIMEOUT_S
    assert ui._server.config.timeout_graceful_shutdown > 0


@pytest.fixture
def server_loop() -> Iterator[asyncio.AbstractEventLoop]:
    """An event loop running in its own thread, as uvicorn's does while the server serves."""
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    while not loop.is_running():
        time.sleep(0.001)
    yield loop
    if not loop.is_closed():
        _close(loop)
    thread.join(2)


def _close(loop: asyncio.AbstractEventLoop) -> None:
    """Stop and close the loop of the server_loop fixture, as asyncio.run() does when the server has stopped."""
    loop.call_soon_threadsafe(loop.stop)
    while loop.is_running():
        time.sleep(0.001)
    loop.close()


def _wait_for(condition, timeout: float = 3) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and not condition():
        time.sleep(0.01)
    return condition()


def _ui_serving_on(loop: asyncio.AbstractEventLoop, monkeypatch) -> tuple[WebUI, _FakeServer]:
    """A WebUI whose fake server runs on loop. The Socket.IO layer, which stop() also shuts down, is left out."""
    ui = WebUI()
    monkeypatch.setattr(ui.app, "sio", None)
    server = _FakeServer()
    ui._server = server
    ui._server_loop = loop
    return ui, server


def test_stop_forces_the_server_to_exit_when_the_graceful_path_stalls(monkeypatch, server_loop):
    """A stalled graceful shutdown must not outlive the app's shutdown budget."""
    monkeypatch.setattr(web_ui, "FORCE_SHUTDOWN_TIMEOUT_S", 0.05)
    ui, server = _ui_serving_on(server_loop, monkeypatch)

    ui.stop()

    assert server.should_exit is True
    assert _wait_for(lambda: server.woken), "the server loop was not woken up"
    assert _wait_for(lambda: server.force_exit), "the server was never forced to stop"


def test_the_escalation_ends_with_the_server_loop(monkeypatch, server_loop):
    """A server that stops in time leaves nothing behind: no thread, and no late force_exit."""
    monkeypatch.setattr(web_ui, "FORCE_SHUTDOWN_TIMEOUT_S", 0.2)
    ui, server = _ui_serving_on(server_loop, monkeypatch)
    threads = threading.active_count()

    ui.stop()
    assert threading.active_count() == threads, "stop() started a thread"
    _close(server_loop)  # The server stopped in time
    time.sleep(0.4)

    assert server.force_exit is False


def test_the_escalation_reaches_only_the_server_being_stopped(monkeypatch, server_loop):
    """After a stop and a start in the same process, the escalation of the first stop must spare the new server."""
    monkeypatch.setattr(web_ui, "FORCE_SHUTDOWN_TIMEOUT_S", 0.1)
    ui, stopped = _ui_serving_on(server_loop, monkeypatch)

    ui.stop()
    started = _FakeServer()
    ui._server = started  # As start() does

    assert _wait_for(lambda: stopped.force_exit)
    assert started.force_exit is False


def test_stop_without_a_running_server_loop_is_safe():
    """stop() may run before the server loop ever started, e.g. if start() failed."""
    ui = WebUI()
    server = _FakeServer()
    ui._server = server
    ui._server_loop = None

    ui.stop()

    assert server.should_exit is True
    assert server.force_exit is False


def _serve_a_stream(ui: WebUI) -> tuple[threading.Thread, socket.socket]:
    """Run the real server and keep a never-ending streaming response in flight, as expose_camera() does."""

    def endless():
        while True:
            yield b"frame\n"
            time.sleep(0.02)

    ui.app.add_api_route("/stream", lambda: StreamingResponse(endless()), methods=["GET"])
    ui.start()
    serving = threading.Thread(target=ui.execute, daemon=True)
    serving.start()
    assert _wait_for(lambda: ui._server_loop is not None and ui._server is not None and ui._server.started), "the server did not start"
    client = socket.create_connection(("127.0.0.1", ui._port), timeout=5)
    client.sendall(b"GET /stream HTTP/1.1\r\nHost: localhost\r\n\r\n")
    assert client.recv(64).startswith(b"HTTP/1.1 200")
    return serving, client


def test_stop_ends_a_real_server_with_a_stream_in_flight(monkeypatch):
    """The graceful shutdown waits for the stream at most GRACEFUL_SHUTDOWN_TIMEOUT_S, then execute() returns."""
    monkeypatch.setattr(web_ui, "GRACEFUL_SHUTDOWN_TIMEOUT_S", 0.5)
    ui = WebUI(addr="127.0.0.1", port=0)
    serving, client = _serve_a_stream(ui)

    started = time.monotonic()
    ui.stop()
    serving.join(5)
    elapsed = time.monotonic() - started
    client.close()

    assert not serving.is_alive(), "execute() did not return"
    assert elapsed < 0.5 + 1.0, f"the stop took {elapsed:.2f} s"


@pytest.mark.skipif(importlib.util.find_spec("uvloop") is None, reason="uvloop, which the image runs, lets force_exit end the wait")
def test_the_escalation_ends_a_real_server_whose_graceful_shutdown_stalls(monkeypatch):
    """With the graceful wait out of the way, force_exit alone must end the server.

    On the asyncio loop, Server.wait_closed() waits for every connection since Python 3.12, and force_exit cannot cut
    that wait short: only the graceful timeout can, which is why it comes first.
    """
    monkeypatch.setattr(web_ui, "GRACEFUL_SHUTDOWN_TIMEOUT_S", 30)
    monkeypatch.setattr(web_ui, "FORCE_SHUTDOWN_TIMEOUT_S", 0.3)
    ui = WebUI(addr="127.0.0.1", port=0)
    serving, client = _serve_a_stream(ui)

    started = time.monotonic()
    ui.stop()
    serving.join(5)
    elapsed = time.monotonic() - started
    client.close()

    assert not serving.is_alive(), "execute() did not return"
    assert ui._server is not None and ui._server.force_exit is True
    assert elapsed < 0.3 + 1.0, f"the stop took {elapsed:.2f} s"


def test_stop_without_a_server_is_a_noop():
    ui = WebUI()

    ui.stop()  # must not raise

    assert ui._server is None


def test_the_server_loop_ends_as_soon_as_it_is_asked_to():
    """Uvicorn's own loop looks at should_exit every 100 ms: ours is woken up."""
    server = web_ui._Server(uvicorn.Config(FastAPI(), port=0, log_level="warning"))

    async def stop_and_time():
        loop = asyncio.get_running_loop()
        loop.call_later(0.02, lambda: (setattr(server, "should_exit", True), server.wake()))
        started = time.monotonic()
        await server.main_loop()
        return time.monotonic() - started

    elapsed = asyncio.run(stop_and_time())
    assert 0.02 <= elapsed < 0.09, f"the loop ended after {elapsed:.3f}s"


def test_the_server_stops_within_a_few_milliseconds_once_asked():
    """Uvicorn sleeps 100 ms in its shutdown and polls every 100 ms: ours waits only for what is open."""
    server = web_ui._Server(uvicorn.Config(FastAPI(), port=0, log_level="warning"))

    async def serve_then_stop():
        serving = asyncio.create_task(server.serve())
        while not server.started:
            await asyncio.sleep(0.005)
        started = time.monotonic()
        server.should_exit = True
        server.wake()
        await serving
        return time.monotonic() - started

    assert asyncio.run(serve_then_stop()) < 0.06
