# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import json
import socket
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest.mock import MagicMock

import pytest

import arduino.app_bricks.tps as tps_module
from arduino.app_bricks.tps import TPS

SCAN_RESULT = {
    "timestamp_ms": 1000,
    "age_ms": 500,
    "cached": False,
    "count": 2,
    "access_points": [
        {"bssid": "fc:40:09:d8:e6:06", "ssid": "net", "signal": -47.0, "channel": 104, "last_seen_ms": 20, "connected": True},
        {"bssid": "10:e1:8e:00:00:01", "ssid": "", "signal": None, "channel": None, "last_seen_ms": 0, "connected": False},
    ],
}

LOCATION_RESPONSE = {
    "location": {"lat": 45.07, "lng": 7.68},
    "accuracy": 25.0,
    "nap": 2,
    "streetAddress": {"city": "Turin", "countryCode": "IT"},
}


@pytest.fixture
def client(monkeypatch):
    """A client with fake credentials whose scanner returns SCAN_RESULT."""
    monkeypatch.setattr(TPS, "_scan", lambda self: json.loads(json.dumps(SCAN_RESULT)))
    api = TPS(auth_key="key", auth_user="user")
    yield api
    api.stop()


@pytest.fixture
def post(monkeypatch):
    """Capture the request to the location API and answer with LOCATION_RESPONSE."""
    response = MagicMock()
    response.json.return_value = LOCATION_RESPONSE
    response.headers = {"Skyhook-Request-Token": "token-from-server"}
    mock = MagicMock(return_value=response)
    monkeypatch.setattr(tps_module.requests, "post", mock)
    return mock


def test_requires_credentials(monkeypatch):
    monkeypatch.delenv("AUTH_KEY", raising=False)
    monkeypatch.delenv("AUTH_USER", raising=False)
    with pytest.raises(ValueError, match="AUTH_KEY"):
        TPS()


def test_credentials_from_environment(monkeypatch):
    monkeypatch.setenv("AUTH_KEY", "env-key")
    monkeypatch.setenv("AUTH_USER", "env-user")
    api = TPS()
    assert (api.auth_key, api.auth_user) == ("env-key", "env-user")
    api.stop()


def test_rejects_plain_http_location_api(monkeypatch):
    monkeypatch.setattr(tps_module, "TPS_LOC_API_URL", "http://example.com/location")
    with pytest.raises(ValueError, match="https"):
        TPS(auth_key="key", auth_user="user")


def test_locate_builds_request_and_parses_response(client, post):
    result = client.locate(street_address=True, device_id="dev-1", opt_in=True)

    kwargs = post.call_args.kwargs
    assert post.call_args.args[0] == tps_module.TPS_LOC_API_URL
    assert kwargs["timeout"] == tps_module.HTTP_REQ_TIMEOUT_SEC
    assert kwargs["headers"]["Skyhook-Auth-Key"] == "key"
    assert kwargs["headers"]["Skyhook-Auth-User"] == "user"
    assert kwargs["headers"]["Skyhook-PID"] == "dev-1"
    assert kwargs["headers"]["Skyhook-Opt-In"] == "true"
    assert kwargs["headers"]["Skyhook-Request-Token"]

    payload = kwargs["json"]
    assert payload["streetAddressLookupType"] == "full"
    assert payload["wifiAccessPoints"] == [
        {"macAddress": "FC:40:09:D8:E6:06", "signalStrength": -47, "age": 520, "channel": 104, "ssid": "net", "connected": True},
        {"macAddress": "10:E1:8E:00:00:01", "signalStrength": -100, "age": 500},
    ]

    assert result["location"] == {"lat": 45.07, "lng": 7.68}
    assert result["accuracy"] == 25.0
    assert result["nap"] == 2
    assert result["request_token"] == "token-from-server"
    assert result["street_address"]["city"] == "Turin"
    assert result["street_address"]["country_code"] == "IT"


def test_locate_without_device_id_omits_pid_headers(client, post):
    client.locate()
    headers = post.call_args.kwargs["headers"]
    assert "Skyhook-PID" not in headers
    assert "Skyhook-Opt-In" not in headers
    assert "streetAddressLookupType" not in post.call_args.kwargs["json"]


def test_locate_fails_without_access_points(client, post, monkeypatch):
    monkeypatch.setattr(TPS, "_scan", lambda self: {"access_points": []})
    with pytest.raises(RuntimeError, match="No access points"):
        client.locate()
    post.assert_not_called()


def test_locate_wraps_location_api_errors(client, post):
    post.return_value.raise_for_status.side_effect = tps_module.requests.HTTPError("401 Unauthorized")
    with pytest.raises(RuntimeError, match="Location request failed"):
        client.locate()


class _UnixHTTPServer(HTTPServer):
    address_family = socket.AF_UNIX

    def server_bind(self):
        self.socket.bind(self.server_address)


@pytest.fixture
def socket_dir():
    """A short-lived directory whose path fits the Unix socket path limit."""
    with tempfile.TemporaryDirectory(prefix="tps") as path:
        yield path


@pytest.fixture
def scanner_socket(socket_dir):
    """A fake scanner listening on a Unix socket, answering /scan with SCAN_RESULT."""

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            body = json.dumps(SCAN_RESULT if self.path == "/scan" else {"detail": "not found"}).encode()
            self.send_response(200 if self.path == "/scan" else 404)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    path = f"{socket_dir}/scanner.sock"
    server = _UnixHTTPServer(path, Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield path
    server.shutdown()
    server.server_close()


def test_scan_talks_to_the_scanner_over_a_unix_socket(scanner_socket):
    api = TPS(auth_key="key", auth_user="user")
    api.scanner_socket_path = scanner_socket
    assert api._scan() == SCAN_RESULT
    api.stop()


def test_scan_reports_unreachable_scanner(socket_dir):
    api = TPS(auth_key="key", auth_user="user")
    api.scanner_socket_path = f"{socket_dir}/missing.sock"
    with pytest.raises(RuntimeError, match="unreachable"):
        api._scan()
    api.stop()


def test_async_locate_delivers_result(client, post):
    done = threading.Event()
    outcome = {}

    def on_location(result, error):
        outcome["result"], outcome["error"] = result, error
        done.set()

    client.async_locate(on_location)
    assert done.wait(timeout=5)
    assert outcome["error"] is None
    assert outcome["result"]["location"] == {"lat": 45.07, "lng": 7.68}
    assert outcome["result"]["elapsed_ms"] >= 0


def test_async_locate_delivers_error(client, post):
    post.side_effect = tps_module.requests.ConnectionError("down")
    done = threading.Event()
    outcome = {}

    def on_location(result, error):
        outcome["result"], outcome["error"] = result, error
        done.set()

    client.async_locate(on_location)
    assert done.wait(timeout=5)
    assert outcome["result"] is None
    assert isinstance(outcome["error"], RuntimeError)


def test_periodic_locate_runs_until_stopped(client, post):
    first = threading.Event()

    def on_location(result, error):
        first.set()

    stop = client.periodic_locate(on_location, period_sec=3600)
    assert first.wait(timeout=5)
    stop()
    calls = post.call_count
    assert calls == 1
    # Stopping is idempotent and stop() on the brick tolerates already stopped loops
    stop()
    client.stop()
    assert post.call_count == calls
