# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import threading
import time
import warnings

import pytest

from arduino.app_bricks.arduino_cloud import (
    ArduinoCloud,
    ColoredLight,
    DEVICE_WINS,
    CLOUD_WINS,
    MOST_RECENT_WINS,
)
from arduino.app_bricks.arduino_cloud import arduino_cloud as ac_module
from arduino.app_bricks.arduino_cloud.daemon_client import parse_timestamp
from arduino.app_bricks.arduino_cloud.objects import CloudObject


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
