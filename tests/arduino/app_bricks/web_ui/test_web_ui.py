# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

import os

from fastapi.testclient import TestClient
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


def test_expose_camera_starts_camera_if_not_started():
    from unittest.mock import Mock, patch
    import numpy as np

    ui = WebUI()
    mock_camera = Mock()
    mock_camera.is_started = False
    mock_camera.capture = Mock(side_effect=[np.zeros((2, 2, 3), dtype=np.uint8), RuntimeError("end")])

    with patch("arduino.app_utils.image.compress_to_jpeg", return_value=np.array([0], dtype=np.uint8)):
        ui.expose_camera("/stream", mock_camera)
        TestClient(ui.app, raise_server_exceptions=False).get("/stream")

    mock_camera.start.assert_called_once()


def test_expose_camera_streams_mjpeg_response():
    from unittest.mock import Mock, patch
    import numpy as np

    ui = WebUI()
    mock_camera = Mock()
    mock_camera.is_started = True

    fake_frame = np.zeros((2, 2, 3), dtype=np.uint8)
    mock_camera.capture = Mock(side_effect=[fake_frame, RuntimeError("end")])

    fake_jpeg = b"\xff\xd8jpeg"

    with patch("arduino.app_utils.image.compress_to_jpeg", return_value=np.frombuffer(fake_jpeg, dtype=np.uint8)):
        ui.expose_camera("/stream", mock_camera)
        response = TestClient(ui.app).get("/stream")

    assert response.status_code == 200
    assert "multipart/x-mixed-replace" in response.headers["content-type"]
    assert fake_jpeg in response.content


def test_expose_camera_passes_quality_to_compress():
    from unittest.mock import Mock, patch
    import numpy as np

    ui = WebUI()
    mock_camera = Mock()
    mock_camera.is_started = True

    fake_frame = np.zeros((2, 2, 3), dtype=np.uint8)
    mock_camera.capture = Mock(side_effect=[fake_frame, RuntimeError("end")])

    with patch("arduino.app_utils.image.compress_to_jpeg", return_value=np.array([0], dtype=np.uint8)) as mock_compress:
        ui.expose_camera("/stream", mock_camera, jpeg_quality=95)
        TestClient(ui.app).get("/stream")

    mock_compress.assert_called_with(fake_frame, quality=95)
