# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Unit tests for iw_scanner.py."""

import subprocess

import pytest

import iw_scanner
from iw_scanner import ScanError, parse_scan_output, scan

IW_DEV = """phy#0
\tInterface wlp3s0
\t\tifindex 3
\t\ttype managed
"""

# Trimmed output of 'iw dev wlp3s0 scan' captured on a VENTUNO Q board
IW_SCAN = """BSS fc:40:09:d8:e6:06(on wlp3s0) -- associated
\tTSF: 1771447705669 usec (20d, 12:04:07)
\tfreq: 5520.0
\tsignal: -47.00 dBm
\tlast seen: 7397352 ms ago
\tSSID: JustSpeed-d8e605
\tDS Parameter set: channel 104
\tHT operation:
\t\t * primary channel: 104
BSS 10:e1:8e:00:00:01(on wlp3s0)
\tfreq: 2437.0
\tsignal: -71.50 dBm
\tlast seen: 120 ms ago
\tSSID:\x20
\tHT operation:
\t\t * primary channel: 6
BSS fc:40:09:d8:e6:06(on wlp3s0)
\tsignal: -60.00 dBm
\tlast seen: 5 ms ago
\tSSID: JustSpeed-d8e605
"""


def completed(args, returncode=0, stdout="", stderr=""):
    return subprocess.CompletedProcess(args, returncode, stdout, stderr)


class FakeIw:
    """Scripted replacement of the iw command, keyed by the first three arguments after iw."""

    def __init__(self):
        self.calls = []
        self.responses = {("dev",): completed(["iw", "dev"], stdout=IW_DEV)}

    def on(self, *key, outcomes):
        self.responses[key] = outcomes
        return self

    def run(self, args, **kwargs):
        self.calls.append(args)
        handler = self.responses.get(tuple(args[1:4]))
        if handler is None:
            raise AssertionError(f"unexpected iw invocation: {args}")
        outcome = handler.pop(0) if isinstance(handler, list) else handler
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


@pytest.fixture
def iw(monkeypatch):
    """Replace the iw command with a scripted fake, recording the invocations."""
    fake = FakeIw()
    monkeypatch.setattr(iw_scanner.subprocess, "run", fake.run)
    monkeypatch.setattr(iw_scanner.time, "sleep", lambda s: None)
    return fake


def test_parse_scan_output_extracts_fields():
    aps = parse_scan_output(IW_SCAN)
    assert [ap.bssid for ap in aps] == ["fc:40:09:d8:e6:06", "10:e1:8e:00:00:01", "fc:40:09:d8:e6:06"]
    first = aps[0]
    assert first.ssid == "JustSpeed-d8e605"
    assert first.signal == -47.0
    assert first.channel == 104
    assert first.last_seen_ms == 7397352
    assert first.connected is True
    hidden = aps[1]
    assert hidden.ssid == ""
    assert hidden.channel == 6
    assert hidden.connected is False


def test_parse_scan_output_tolerates_missing_signal():
    aps = parse_scan_output("BSS aa:bb:cc:dd:ee:ff(on wlan0)\n\tSSID: x\n")
    assert aps[0].signal is None
    assert aps[0].channel is None


def test_scan_sorts_and_deduplicates(iw):
    iw.on("dev", "wlp3s0", "scan", outcomes=completed([], stdout=IW_SCAN))
    result = scan()
    assert [ap.bssid for ap in result.access_points] == ["fc:40:09:d8:e6:06", "10:e1:8e:00:00:01"]
    assert result.access_points[0].signal == -47.0
    assert iw.calls[-1] == ["iw", "dev", "wlp3s0", "scan", "duration", str(iw_scanner.SCAN_CHANNEL_DWELL_TU)]


def test_scan_retries_when_busy(iw):
    busy = completed([], returncode=240, stderr="command failed: Device or resource busy (-16)")
    iw.on("dev", "wlp3s0", "scan", outcomes=[busy, completed([], stdout=IW_SCAN)])
    scan()
    assert sum(1 for c in iw.calls if c[:4] == ["iw", "dev", "wlp3s0", "scan"]) == 2


def test_scan_fails_when_not_permitted(iw):
    iw.on("dev", "wlp3s0", "scan", outcomes=completed([], returncode=255, stderr="command failed: Operation not permitted (-1)"))
    with pytest.raises(ScanError, match="not permitted"):
        scan()
    assert not any(c[-1] == "dump" for c in iw.calls)


def test_scan_fails_when_iw_cannot_run(iw):
    iw.on("dev", "wlp3s0", "scan", outcomes=PermissionError(1, "Operation not permitted"))
    with pytest.raises(ScanError, match="cannot run iw"):
        scan()


def test_scan_fails_when_radio_stays_busy(iw, monkeypatch):
    monkeypatch.setattr(iw_scanner, "SCAN_RETRIES", 2)
    busy = completed([], returncode=240, stderr="command failed: Device or resource busy (-16)")
    iw.on("dev", "wlp3s0", "scan", outcomes=[busy, busy])
    with pytest.raises(ScanError, match="busy"):
        scan()


def test_scan_fails_on_unexpected_error(iw, monkeypatch):
    monkeypatch.setattr(iw_scanner, "SCAN_CHANNEL_DWELL_TU", 0)
    iw.on("dev", "wlp3s0", "scan", outcomes=completed([], returncode=1, stderr="command failed: No such device (-19)"))
    with pytest.raises(ScanError, match="No such device"):
        scan()


def test_scan_rejects_unknown_interface(iw):
    with pytest.raises(ScanError, match="unknown wireless interface"):
        scan("eth0; reboot")
    assert iw.calls == [["iw", "dev"]]


def test_scan_fails_without_interfaces(iw):
    iw.on("dev", outcomes=completed(["iw", "dev"], stdout=""))
    with pytest.raises(ScanError, match="no wireless interface"):
        scan()
