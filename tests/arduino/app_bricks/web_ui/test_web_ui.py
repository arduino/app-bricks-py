# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import os
import threading
import time
from unittest.mock import patch

import numpy as np
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
    """A stand-in for uvicorn.Server exposing just the two stop flags."""

    def __init__(self):
        self.should_exit = False
        self.force_exit = False


def test_start_bounds_the_graceful_shutdown():
    """Uvicorn waits forever by default, which never terminates while a client is connected."""
    ui = WebUI()
    ui.start()

    assert ui._server is not None
    assert ui._server.config.timeout_graceful_shutdown == web_ui.GRACEFUL_SHUTDOWN_TIMEOUT_S
    assert ui._server.config.timeout_graceful_shutdown > 0


def test_stop_forces_the_server_to_exit_when_the_graceful_path_stalls(monkeypatch):
    """A stalled graceful shutdown must not outlive the app's shutdown budget."""
    monkeypatch.setattr(web_ui, "FORCE_SHUTDOWN_TIMEOUT_S", 0.05)

    ui = WebUI()
    server = _FakeServer()
    ui._server = server

    ui.stop()

    assert server.should_exit is True

    deadline = time.monotonic() + 3
    while time.monotonic() < deadline and not server.force_exit:
        time.sleep(0.01)

    assert server.force_exit is True, "the server was never forced to stop"


class _StoppingServer(_FakeServer):
    """Runs until asked to exit, as uvicorn.Server.run() does when the graceful shutdown completes."""

    def run(self):
        while not self.should_exit:
            time.sleep(0.01)


def test_a_server_stopping_in_time_leaves_no_timer_behind():
    """The force-stop timer must end with the server, not linger for its whole timeout."""
    ui = WebUI()
    server = _StoppingServer()
    ui._server = server
    serving = threading.Thread(target=ui.execute)
    serving.start()

    ui.stop()
    serving.join(2)
    timer = ui._force_stop_timer

    assert not serving.is_alive()
    assert timer is not None
    timer.join(1)
    assert not timer.is_alive(), "the timer outlived the server"
    assert server.force_exit is False


def test_stop_without_a_running_server_loop_is_safe():
    """stop() may run before the server loop ever started, e.g. if start() failed."""
    ui = WebUI()
    server = _FakeServer()
    ui._server = server
    ui._server_loop = None

    ui.stop()

    assert server.should_exit is True


def test_stop_without_a_server_is_a_noop():
    ui = WebUI()

    ui.stop()  # must not raise

    assert ui._server is None
