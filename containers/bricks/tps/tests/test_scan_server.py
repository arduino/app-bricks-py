# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Unit tests for scan_server.py."""

import pytest
from fastapi.testclient import TestClient

import scan_server
from iw_scanner import AccessPoint, ScanError, ScanResult


@pytest.fixture
def client(monkeypatch):
    """Serve the API with a scripted scanner and a fresh cache."""
    scans = []

    def fake_scan():
        scans.append(True)
        return ScanResult([AccessPoint("aa:bb:cc:dd:ee:ff", ssid="net", signal=-50.0, channel=6, last_seen_ms=10)], 1_000)

    monkeypatch.setattr(scan_server, "scan", fake_scan)
    monkeypatch.setattr(scan_server, "cache", scan_server.ScanCache(ttl_seconds=10))
    monkeypatch.setattr(scan_server.time, "time", lambda: 1.5)
    test_client = TestClient(scan_server.app)
    test_client.scans = scans
    return test_client


def test_scan_returns_access_points(client):
    body = client.get("/scan").json()
    assert body["cached"] is False
    assert body["timestamp_ms"] == 1_000
    assert body["age_ms"] == 500
    assert body["count"] == 1
    assert body["access_points"][0] == {
        "bssid": "aa:bb:cc:dd:ee:ff",
        "ssid": "net",
        "signal": -50.0,
        "channel": 6,
        "last_seen_ms": 10,
        "connected": False,
    }


def test_scan_is_cached_within_ttl(client):
    client.get("/scan")
    body = client.get("/scan").json()
    assert body["cached"] is True
    assert client.scans == [True]


def test_scan_expires_after_ttl(client, monkeypatch):
    client.get("/scan")
    monkeypatch.setattr(scan_server.time, "time", lambda: 12.0)
    body = client.get("/scan").json()
    assert body["cached"] is False
    assert len(client.scans) == 2


def test_scan_ignores_request_parameters(client):
    client.get("/scan", params={"interface": "eth0", "force_refresh": "true"})
    assert client.scans == [True]


def test_scan_failure_is_a_503_without_details(client, monkeypatch):
    def failing_scan():
        raise ScanError("scan failed on wlan0: secret details")

    monkeypatch.setattr(scan_server, "scan", failing_scan)
    response = client.get("/scan")
    assert response.status_code == 503
    assert "secret" not in response.text


def test_health_reports_interfaces(client, monkeypatch):
    monkeypatch.setattr(scan_server, "list_interfaces", lambda: ["wlan0"])
    assert client.get("/health").json() == {"status": "ok", "interfaces": ["wlan0"]}


def test_docs_are_disabled(client):
    assert client.get("/docs").status_code == 404
    assert client.get("/openapi.json").status_code == 404
