# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

from fastapi.testclient import TestClient
from arduino.app_bricks.web_ui.web_ui import WebUI


def test_webui_init_defaults():
    ui = WebUI()
    assert ui._addr == "0.0.0.0"
    assert ui._port == 7000
    assert ui._ui_path_prefix == ""
    assert ui._api_path_prefix == ""
    assert ui._assets_dir_path.endswith("/app/assets")
    assert ui._certs_dir_path.endswith("/app/certs")
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
