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
from arduino.app_bricks.arduino_cloud import objects as ac_objects
from arduino.app_bricks.arduino_cloud.daemon_client import (
    DaemonClient,
    parse_timestamp,
    EVENT_LASTVALUE,
    EVENT_LASTVALUE_MISSING,
    EVENT_THING_UNAVAILABLE,
)
from arduino.app_bricks.arduino_cloud.objects import CloudObject, ON_CHANGE

_HAS_AF_UNIX = os.name == "posix" and hasattr(socket, "AF_UNIX") and hasattr(socketserver, "UnixStreamServer")


class FakeDaemonClient:
    """In-memory stand-in for DaemonClient: records PUTs, serves a configurable
    initial sync frame (fetch_initial) and lets tests feed live SSE events."""

    def __init__(self, base_url=None):
        self.puts = []
        self.handlers = {}
        self._ready = threading.Event()
        # Per-name first frame returned by fetch_initial; the default is
        # "thing assigned, no cloud value" (steady) so local values are pushed.
        self.initial = {}
        self.default_initial = (EVENT_LASTVALUE_MISSING, {})

    def fetch_initial(self, name, stop_event):
        event, payload = self.initial.get(name, self.default_initial)
        frame = {"name": name}
        frame.update(payload)
        return (event, frame)

    def put_value(self, name, value):
        self.puts.append((name, value))

    def stream_events(self, name, handler, stop_event):
        self.handlers[name] = handler
        self._ready.set()
        stop_event.wait()  # mimic the blocking listener thread

    def close(self):
        pass

    def feed(self, name, value, timestamp="2026-06-22T10:00:00Z", last_value=False):
        self.feed_event(
            name,
            EVENT_LASTVALUE if last_value else "update",
            {"name": name, "value": value, "timestamp": timestamp, "last_value": last_value},
        )

    def feed_event(self, name, event, payload=None):
        # Wait for the listener thread to register its handler, then dispatch.
        for _ in range(200):
            if name in self.handlers:
                break
            time.sleep(0.005)
        frame = {"name": name}
        if payload:
            frame.update(payload)
        self.handlers[name](event, frame)


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
    obj.set_local(5)  # stamps a local timestamp = now; marks dirty (not sent yet)
    assert pushes == []  # set_local no longer publishes directly
    obj.pump(time.time(), ON_CHANGE)  # the loop's pump publishes the change
    assert pushes == [("v", 5)]
    # An older cloud value is ignored; the newer local value is re-pushed
    # immediately (convergence) so the cloud converges to the device.
    assert obj.apply_cloud(9, cloud_ts=time.time() - 100) is False
    assert obj.value == 5
    assert pushes == [("v", 5), ("v", 5)]
    # A newer cloud value wins (and is not re-pushed).
    assert obj.apply_cloud(9, cloud_ts=time.time() + 100) is True
    assert obj.value == 9
    assert pushes == [("v", 5), ("v", 5)]


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


# ── Update policy: ON_CHANGE vs timed (pump) ─────────────────────────────────


def test_on_change_publishes_changes_throttled():
    # ON_CHANGE mirrors the C++ publishOnChange: publish a changed value, but no
    # more than once per _MIN_PUBLISH_INTERVAL (first publish immediate).
    pushes = []
    obj = CloudObject("v", value=None, interval=ON_CHANGE)
    obj.bind(lambda n, val: pushes.append((n, val)))
    t = 1000.0
    obj.set_local(1)
    obj.pump(t, ON_CHANGE)  # first change → immediate
    assert pushes == [("v", 1)]
    # A change within the throttle window is held back.
    obj.set_local(2)
    obj.pump(t + 0.1, ON_CHANGE)
    assert pushes == [("v", 1)]
    # Once the window elapses, the latest value is published.
    obj.pump(t + ac_objects._MIN_PUBLISH_INTERVAL, ON_CHANGE)
    assert pushes == [("v", 1), ("v", 2)]
    # No further change → no publish, however long we wait.
    obj.pump(t + 100, ON_CHANGE)
    assert pushes == [("v", 1), ("v", 2)]


def test_timed_republishes_latest_value_every_interval():
    # Timed mode mirrors the C++ publishEvery: republish the current value every
    # `interval` seconds regardless of change; values changing within a window
    # are not individually sent — only the latest at the tick.
    pushes = []
    obj = CloudObject("v", value=None, interval=5)
    obj.bind(lambda n, val: pushes.append((n, val)))
    t = 1000.0
    # A purely cloud-controlled value (device never set it) is not republished.
    obj.pump(t, 5)
    assert pushes == []
    obj.set_local(10)
    obj.pump(t, 5)  # first tick → immediate
    assert pushes == [("v", 10)]
    # A change within the window is not sent yet.
    obj.set_local(11)
    obj.pump(t + 2, 5)
    assert pushes == [("v", 10)]
    # At the tick only the latest value is sent; the intermediate 11 is dropped.
    obj.set_local(12)
    obj.pump(t + 5, 5)
    assert pushes == [("v", 10), ("v", 12)]
    # Republishes the current value even without any change.
    obj.pump(t + 10, 5)
    assert pushes == [("v", 10), ("v", 12), ("v", 12)]


# ── Brick behaviour with a fake daemon ───────────────────────────────────────


def test_setattr_pushes_value(fake_client, monkeypatch):
    monkeypatch.setattr(ac_objects, "_MIN_PUBLISH_INTERVAL", 0.0)  # no throttle delay in the test
    cloud, client = _make_cloud(fake_client)
    # Default seed is lastvalue_missing (thing assigned, no cloud value), so the
    # registered default is pushed up first (immediate convergence). The explicit
    # set is published by the loop's pump (ON_CHANGE), not synchronously.
    cloud.register("led", value=False)
    assert client.puts == [("led", False)]
    cloud.led = True
    cloud.loop()
    assert client.puts == [("led", False), ("led", True)]
    assert cloud.led is True


def test_sse_update_fires_on_write(fake_client):
    cloud, client = _make_cloud(fake_client)
    received = []
    cloud.register("led", value=False, on_write=lambda c, v: received.append(v))
    cloud.start()
    try:
        # on_write fires immediately when the update arrives, not on a later,
        # interval-gated loop pass (C++ parity).
        client.feed("led", True)
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
        # Setting a sub-attribute publishes the namespaced leaf variable on the
        # next loop pass (first publish of that leaf is immediate, not throttled).
        cloud.clight.hue = 120
        cloud.loop()
        assert ("clight:hue", 120) in client.puts
        # A cloud update on a leaf fires the parent object's on_write immediately.
        client.feed("clight:swi", False)
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


# ── Synchronous register + sync-frame resolution ─────────────────────────────


def test_lastvalue_sync_fires_on_write_when_cloud_wins(fake_client):
    # Mirrors the C++ ArduinoIoTCloud library: the last-value sync (initial or on
    # reconnect) fires on_write whenever the synced cloud value wins per policy
    # and actually differs from the local value, so an actuator is restored to a
    # cloud state changed while the board was offline.
    cloud, client = _make_cloud(fake_client)
    received = []
    # Cloud has a last value at register time → seeded synchronously as a
    # 'lastvalue' frame that differs from the register default.
    client.initial["led"] = (EVENT_LASTVALUE, {"value": True, "timestamp": "2026-07-07T10:00:00Z", "last_value": True})
    # on_write fires immediately during the synchronous initial sync in register().
    cloud.register("led", value=False, on_write=lambda c, v: received.append(v))
    assert cloud.led is True  # cloud value applied
    assert received == [True]  # on_write fired for the sync (C++ parity)
    cloud.start()
    try:
        # A subsequent live update still fires on_write immediately.
        client.feed("led", False)  # 'update' event
        assert received == [True, False]
    finally:
        cloud.stop()


def test_lastvalue_sync_no_on_write_when_value_unchanged(fake_client):
    # The C++ guard (isDifferentFromCloud): a synced cloud value equal to the
    # local value does not fire on_write.
    cloud, client = _make_cloud(fake_client)
    received = []
    client.initial["led"] = (EVENT_LASTVALUE, {"value": True, "timestamp": "2026-07-07T10:00:00Z", "last_value": True})
    cloud.register("led", value=True, on_write=lambda c, v: received.append(v))
    cloud.start()
    try:
        assert cloud.led is True
        assert received == []  # value unchanged → no on_write
    finally:
        cloud.stop()


def test_lastvalue_sync_no_on_write_device_wins(fake_client):
    # DEVICE_WINS (onForceDeviceSync) never applies the cloud value, so the sync
    # does not fire on_write; the local value is pushed up instead.
    cloud, client = _make_cloud(fake_client)
    received = []
    client.initial["temp"] = (EVENT_LASTVALUE, {"value": 100, "timestamp": "2026-07-07T10:00:00Z", "last_value": True})
    cloud.register("temp", value=7, sync=DEVICE_WINS, on_write=lambda c, v: received.append(v))
    cloud.start()
    try:
        assert cloud.temp == 7  # local wins
        assert received == []  # on_write not fired on sync
        assert ("temp", 7) in client.puts  # local pushed up
    finally:
        cloud.stop()


def test_register_seeds_cloud_value_cloud_wins(fake_client):
    cloud, client = _make_cloud(fake_client)
    client.initial["temp"] = (EVENT_LASTVALUE, {"value": 100, "timestamp": "2026-07-07T10:00:00Z", "last_value": True})
    cloud.register("temp", value=0)  # CLOUD_WINS (default)
    # The cloud last value wins over the register default, synchronously.
    assert cloud.temp == 100
    # The stale default is not pushed up under CLOUD_WINS.
    assert client.puts == []


def test_register_seeds_lastvalue_missing_pushes_local(fake_client):
    cloud, client = _make_cloud(fake_client)
    client.initial["temp"] = (EVENT_LASTVALUE_MISSING, {})
    cloud.register("temp", value=7)
    # No cloud value: the local default wins and is pushed up.
    assert cloud.temp == 7
    assert client.puts == [("temp", 7)]


def test_register_without_value_missing_pushes_nothing(fake_client):
    cloud, client = _make_cloud(fake_client)
    client.initial["temp"] = (EVENT_LASTVALUE_MISSING, {})
    cloud.register("temp")  # no default value
    assert cloud.get("temp") is None
    assert client.puts == []  # nothing local to assert yet


def test_thing_unavailable_defers_writes_until_sync(fake_client):
    cloud, client = _make_cloud(fake_client)
    client.initial["temp"] = (EVENT_THING_UNAVAILABLE, {})
    cloud.register("temp", value=0)
    cloud.start()
    try:
        # No thing assigned yet: writes are kept locally, not pushed.
        cloud.temp = 42
        assert client.puts == []
        assert cloud.temp == 42
        # Thing arrives with no cloud value → resync lastvalue_missing: the
        # deferred local value now wins and is pushed up.
        client.feed_event("temp", EVENT_LASTVALUE_MISSING)
        assert ("temp", 42) in client.puts
        assert cloud.temp == 42
    finally:
        cloud.stop()


def test_cloud_wins_discards_prewrite_on_sync(fake_client):
    cloud, client = _make_cloud(fake_client)
    client.initial["temp"] = (EVENT_THING_UNAVAILABLE, {})
    cloud.register("temp", value=0)  # CLOUD_WINS
    cloud.start()
    try:
        cloud.temp = 42  # pending: kept local, not pushed
        assert client.puts == []
        # Thing arrives with a cloud value → CLOUD_WINS: cloud wins, prewrite dropped.
        client.feed("temp", 100, last_value=True)
        assert cloud.temp == 100
        assert client.puts == []  # the stale 42 was never sent
    finally:
        cloud.stop()


def test_device_wins_sends_local_on_sync(fake_client):
    cloud, client = _make_cloud(fake_client)
    client.initial["temp"] = (EVENT_THING_UNAVAILABLE, {})
    cloud.register("temp", value=7, sync=DEVICE_WINS)
    cloud.start()
    try:
        # Thing arrives with a diverging cloud value → DEVICE_WINS: local wins, pushed.
        client.feed("temp", 100, last_value=True)
        assert cloud.temp == 7
        assert ("temp", 7) in client.puts
    finally:
        cloud.stop()


def test_fetch_initial_and_put_409_over_tcp():
    """Exercise the real DaemonClient: fetch_initial parses the first frame and
    a 409 PUT (thing_unavailable) is handled without raising."""
    frame = {"name": "temp", "value": 21.5, "timestamp": "2026-07-07T10:00:00Z", "last_value": True}

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_PUT(self):
            length = int(self.headers.get("Content-Length", 0))
            self.rfile.read(length)
            self.send_response(409)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"error":"thing_unavailable"}')

        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            self.wfile.write(f"event: lastvalue\ndata: {json.dumps(frame)}\n\n".encode())
            self.wfile.flush()

        def log_message(self, *args):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    port = server.server_address[1]

    client = DaemonClient(f"http://127.0.0.1:{port}")
    try:
        event, payload = client.fetch_initial("temp", threading.Event())
        assert event == EVENT_LASTVALUE
        assert payload["value"] == 21.5
        # A 409 PUT must not raise (it is logged and swallowed).
        client.put_value("temp", 5)
    finally:
        client.close()
        server.shutdown()
        server.server_close()


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
