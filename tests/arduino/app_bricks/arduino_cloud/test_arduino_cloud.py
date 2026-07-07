# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import http.server
import json
import os
import socket
import socketserver
import threading
import time
import warnings
from urllib.parse import quote

import pytest

from arduino.app_bricks.arduino_cloud import (
    ArduinoCloud,
    ColoredLight,
    DEVICE_WINS,
    CLOUD_WINS,
    MOST_RECENT_WINS,
)
from arduino.app_bricks.arduino_cloud import arduino_cloud as ac_module
from arduino.app_bricks.arduino_cloud.daemon_client import DaemonClient, parse_timestamp
from arduino.app_bricks.arduino_cloud.objects import CloudObject

_HAS_AF_UNIX = os.name == "posix" and hasattr(socket, "AF_UNIX") and hasattr(socketserver, "UnixStreamServer")


class FakeDaemonClient:
    """In-memory stand-in for DaemonClient: records PUTs and lets tests feed SSE events."""

    def __init__(self, base_url=None):
        self.puts = []
        self.handlers = {}
        self._ready = threading.Event()

    def put_value(self, name, value):
        self.puts.append((name, value))

    def stream_events(self, name, handler, stop_event):
        self.handlers[name] = handler
        self._ready.set()
        stop_event.wait()  # mimic the blocking listener thread

    def close(self):
        pass

    def feed(self, name, value, timestamp="2026-06-22T10:00:00Z", last_value=False):
        # Wait for the listener thread to register its handler.
        for _ in range(200):
            if name in self.handlers:
                break
            time.sleep(0.005)
        self.handlers[name](
            "lastvalue" if last_value else "update",
            {"name": name, "value": value, "timestamp": timestamp, "last_value": last_value},
        )


@pytest.fixture
def fake_client(monkeypatch):
    created = {}

    def factory(base_url):
        client = FakeDaemonClient(base_url)
        created["client"] = client
        return client

    monkeypatch.setattr(ac_module, "DaemonClient", factory)
    return created


def _make_cloud(fake_client):
    cloud = ArduinoCloud()
    return cloud, fake_client["client"]


# ── parse_timestamp ─────────────────────────────────────────────────────────


def test_parse_timestamp():
    assert parse_timestamp("2026-06-22T10:00:00Z") == pytest.approx(parse_timestamp("2026-06-22T10:00:00+00:00"))
    # Over-long (nanosecond) fractional part is tolerated.
    assert parse_timestamp("2026-06-22T10:00:00.123456789Z") is not None
    assert parse_timestamp(None) is None
    assert parse_timestamp("not-a-date") is None


# ── CloudObject sync policies (unit) ─────────────────────────────────────────


def test_cloud_wins_always_applies():
    obj = CloudObject("v", value=1, sync=CLOUD_WINS)
    assert obj.apply_cloud(2, cloud_ts=100.0) is True
    assert obj.value == 2
    # Same value → no change reported.
    assert obj.apply_cloud(2, cloud_ts=200.0) is False


def test_most_recent_wins_respects_local_timestamp():
    pushes = []
    obj = CloudObject("v", value=1, sync=MOST_RECENT_WINS)
    obj.bind(lambda n, val: pushes.append((n, val)))
    obj.set_local(5)  # stamps a local timestamp = now
    assert pushes == [("v", 5)]
    # An older cloud value is ignored.
    assert obj.apply_cloud(9, cloud_ts=time.time() - 100) is False
    assert obj.value == 5
    # A newer cloud value wins.
    assert obj.apply_cloud(9, cloud_ts=time.time() + 100) is True
    assert obj.value == 9


def test_device_wins_repushes_local_and_ignores_cloud():
    pushes = []
    obj = CloudObject("v", value=7, sync=DEVICE_WINS)
    obj.bind(lambda n, val: pushes.append((n, val)))
    # A diverging cloud value is ignored and the local value is re-pushed.
    assert obj.apply_cloud(3, cloud_ts=time.time()) is False
    assert obj.value == 7
    assert pushes == [("v", 7)]
    # A cloud value equal to local does not trigger a re-push (no echo loop).
    pushes.clear()
    assert obj.apply_cloud(7, cloud_ts=time.time()) is False
    assert pushes == []


def test_invalid_sync_policy_rejected():
    with pytest.raises(ValueError):
        CloudObject("v", sync="bogus")


# ── Brick behaviour with a fake daemon ───────────────────────────────────────


def test_setattr_pushes_value(fake_client):
    cloud, client = _make_cloud(fake_client)
    cloud.register("led", value=False)
    cloud.led = True
    assert client.puts == [("led", True)]
    assert cloud.led is True


def test_sse_update_fires_on_write(fake_client):
    cloud, client = _make_cloud(fake_client)
    received = []
    cloud.register("led", value=False, on_write=lambda c, v: received.append(v))
    cloud.start()
    try:
        client.feed("led", True)
        cloud.loop()  # on_write is scheduled then fired in the loop pass
        assert received == [True]
        assert cloud.led is True
    finally:
        cloud.stop()


def test_complex_object_subscribes_and_pushes_each_leaf(fake_client):
    cloud, client = _make_cloud(fake_client)
    writes = []
    cloud.register(ColoredLight("clight", swi=True, on_write=lambda c, v: writes.append(v.swi)))
    cloud.start()
    try:
        # Setting a sub-attribute pushes the namespaced leaf variable.
        cloud.clight.hue = 120
        assert ("clight:hue", 120) in client.puts
        # A cloud update on a leaf schedules the parent object's on_write.
        client.feed("clight:swi", False)
        cloud.loop()
        assert writes == [False]
        assert cloud.clight.swi is False
    finally:
        cloud.stop()


def test_get_returns_default_for_unknown(fake_client):
    cloud, _ = _make_cloud(fake_client)
    assert cloud.get("missing", default=42) == 42
    cloud.register("known", value=7)
    assert cloud.get("known") == 7


def test_legacy_args_emit_deprecation_warning(fake_client):
    with pytest.warns(DeprecationWarning):
        ArduinoCloud(device_id="x", secret="y")


def test_no_legacy_args_is_silent(fake_client):
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any DeprecationWarning would fail the test
        ArduinoCloud()


# ── UNIX-socket transport ────────────────────────────────────────────────────


def test_default_daemon_url_is_unix_socket(monkeypatch):
    monkeypatch.delenv("ARDUINO_CLOUD_CONNECTOR_URL", raising=False)
    monkeypatch.delenv("ARDUINO_CLOUD_CONNECTOR_SOCKET", raising=False)
    url = ArduinoCloud._default_daemon_url()
    assert url.startswith("http+unix://")
    assert "daemon.sock" in url


def test_daemon_client_mounts_unix_adapter():
    client = DaemonClient("http+unix://%2Frun%2Farduino-cloud-connector%2Fdaemon.sock")
    assert client._socket_path == "/run/arduino-cloud-connector/daemon.sock"
    assert "http+unix://" in client._session.adapters


def test_daemon_client_plain_http_has_no_socket():
    client = DaemonClient("http://127.0.0.1:5683")
    assert client._socket_path is None


@pytest.mark.skipif(not _HAS_AF_UNIX, reason="AF_UNIX not available on this platform")
def test_put_and_sse_over_unix_socket(tmp_path):
    sock_path = str(tmp_path / "daemon.sock")
    received = {}

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_PUT(self):
            length = int(self.headers.get("Content-Length", 0))
            received["path"] = self.path
            received["body"] = self.rfile.read(length)
            self.send_response(204)
            self.end_headers()

        def do_GET(self):
            received["sse_path"] = self.path
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            payload = json.dumps({"name": "led", "value": True, "timestamp": "2026-06-22T10:00:00Z", "last_value": True})
            self.wfile.write(f"event: lastvalue\ndata: {payload}\n\n".encode())
            self.wfile.flush()

        def log_message(self, *args):
            pass

    class UnixHTTPServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
        daemon_threads = True

        def get_request(self):
            # BaseHTTPRequestHandler expects a (host, port) client address.
            return self.socket.accept()[0], ("localhost", 0)

    server = UnixHTTPServer(sock_path, Handler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    client = DaemonClient("http+unix://" + quote(sock_path, safe=""))
    try:
        # PUT over the socket.
        client.put_value("led", True)
        for _ in range(200):
            if "body" in received:
                break
            time.sleep(0.005)
        assert received.get("path") == "/v1/variables/led"
        assert json.loads(received["body"]) == {"value": True}

        # SSE over the socket: the first event is delivered to the handler.
        events = []
        stop = threading.Event()
        sse_thread = threading.Thread(
            target=client.stream_events,
            args=("led", lambda evt, data: (events.append((evt, data)), stop.set()), stop),
            daemon=True,
        )
        sse_thread.start()
        for _ in range(200):
            if events:
                break
            time.sleep(0.005)
        stop.set()
        assert events and events[0][0] == "lastvalue"
        assert events[0][1]["value"] is True
    finally:
        stop.set()
        client.close()
        server.shutdown()
        server.server_close()
