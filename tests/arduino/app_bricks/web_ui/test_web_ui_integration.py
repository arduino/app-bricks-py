# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

import threading
import time
import pytest
import requests
import socketio
from arduino.app_bricks.web_ui.web_ui import WebUI


@pytest.fixture(scope="module")
def webui_server():
    import tempfile

    with tempfile.TemporaryDirectory() as tmp_assets:
        import os

        os.makedirs(tmp_assets, exist_ok=True)
        with open(os.path.join(tmp_assets, "index.html"), "w") as f:
            f.write("<html><body>Hello</body></html>")

        ui = WebUI(port=0, assets_dir_path=tmp_assets)

        def get_hello():
            return {"msg": "hello"}

        ui.expose_api("GET", "/api/hello", get_hello)

        def post_echo(data: dict):
            return {"echo": data.get("value")}

        ui.expose_api("POST", "/api/echo", post_echo)
        ui.start()

        thread = threading.Thread(target=ui.execute, daemon=True)
        thread.start()

        time.sleep(1)  # Wait for server to start

        yield ui

        ui.stop()
        thread.join(timeout=2)


def test_http_index(webui_server):
    resp = requests.get(f"{webui_server.url}/")
    assert resp.status_code == 200
    assert "Hello" in resp.text


def test_expose_api_rest(webui_server):
    resp = requests.get(f"{webui_server.url}/api/hello")
    assert resp.status_code == 200
    assert resp.json() == {"msg": "hello"}

    resp = requests.post(f"{webui_server.url}/api/echo", json={"value": "test123"})
    assert resp.status_code == 200
    assert resp.json() == {"echo": "test123"}


def test_websocket_exchange(webui_server):
    sio = socketio.Client()
    received = {}
    test_done = threading.Event()

    @sio.event
    def connect():
        received["connect"] = True

    @sio.event
    def disconnect():
        received["disconnect"] = True

    def on_ping_response(data):
        received["ping_response"] = data
        test_done.set()

    sio.on("ping_response", on_ping_response)

    # Register a ping handler on server
    def ping_cb(sid, data):
        return "pong"

    webui_server.on_message("ping", ping_cb)

    sio.connect(f"{webui_server.url}", socketio_path="/socket.io")
    sio.emit("ping", {"msg": "hi"})
    test_done.wait(timeout=2)
    sio.disconnect()

    assert received.get("connect") is True
    assert received.get("ping_response") == "pong"
    assert received.get("disconnect") is True
