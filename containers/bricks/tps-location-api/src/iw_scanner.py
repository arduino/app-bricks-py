# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Wi-Fi access point scanning through the iw command line tool."""

import logging
import os
import re
import subprocess
import time
from dataclasses import dataclass

logger = logging.getLogger("iw_scanner")

SCAN_TIMEOUT_SECONDS = int(os.getenv("SCAN_TIMEOUT_SECONDS", "10"))
SCAN_RETRIES = int(os.getenv("SCAN_RETRIES", "3"))
# Per-channel dwell time in TUs (1 TU is 1024 us), 0 leaves the driver default
SCAN_CHANNEL_DWELL_TU = int(os.getenv("SCAN_CHANNEL_DWELL_TU", "60"))

_BSS_RE = re.compile(r"^BSS\s+([0-9a-f]{2}(?::[0-9a-f]{2}){5})", re.IGNORECASE)
_SSID_RE = re.compile(r"^SSID:\s?(.*)$")
_SIGNAL_RE = re.compile(r"^signal:\s*(-?\d+(?:\.\d+)?)\s*dBm")
_LAST_SEEN_RE = re.compile(r"^last seen:\s*(\d+)\s*ms ago")
_CHANNEL_RE = re.compile(r"^(?:DS Parameter set: channel|\* primary channel:)\s*(\d+)$")


class ScanError(RuntimeError):
    """Raised when no scan result can be obtained."""


@dataclass
class AccessPoint:
    """A Wi-Fi access point seen by the wireless interface."""

    bssid: str
    ssid: str = ""
    signal: float | None = None
    channel: int | None = None
    last_seen_ms: int = 0
    connected: bool = False


@dataclass(frozen=True)
class ScanResult:
    """Access points collected at timestamp_ms."""

    access_points: list[AccessPoint]
    timestamp_ms: int


def _iw(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["iw", *args], capture_output=True, text=True, timeout=SCAN_TIMEOUT_SECONDS)


def list_interfaces() -> list[str]:
    """Return the wireless interface names known to the kernel."""
    try:
        result = _iw("dev")
    except (subprocess.TimeoutExpired, OSError) as e:
        raise ScanError(f"cannot list wireless interfaces: {e}") from e
    if result.returncode != 0:
        raise ScanError(f"cannot list wireless interfaces: {result.stderr.strip()}")
    return [line.split()[1] for line in result.stdout.splitlines() if line.strip().startswith("Interface ")]


def first_interface() -> str:
    """Return the first wireless interface known to the kernel."""
    interfaces = list_interfaces()
    if not interfaces:
        raise ScanError("no wireless interface found")
    return interfaces[0]


def _scan_output(interface: str) -> str:
    """Run an active scan, retrying while the radio is busy, and return its output."""
    args = ["dev", interface, "scan"]
    if SCAN_CHANNEL_DWELL_TU > 0:
        args += ["duration", str(SCAN_CHANNEL_DWELL_TU)]

    for attempt in range(1, SCAN_RETRIES + 1):
        try:
            result = _iw(*args)
        except subprocess.TimeoutExpired:
            logger.warning("scan on %s timed out (attempt %d/%d)", interface, attempt, SCAN_RETRIES)
            continue
        except OSError as e:
            raise ScanError(f"cannot run iw: {e}") from e
        if result.returncode == 0:
            return result.stdout
        stderr = result.stderr.strip()
        if "Device or resource busy" not in stderr:
            raise ScanError(f"scan failed on {interface}: {stderr}")
        time.sleep(2 * attempt)

    raise ScanError(f"scan failed on {interface}: radio busy after {SCAN_RETRIES} attempts")


def scan() -> ScanResult:
    """Scan for nearby access points on the first wireless interface, strongest first and deduplicated by BSSID."""
    interface = first_interface()
    access_points = parse_scan_output(_scan_output(interface))
    access_points.sort(key=lambda ap: ap.signal if ap.signal is not None else float("-inf"), reverse=True)
    unique: dict[str, AccessPoint] = {}
    for ap in access_points:
        unique.setdefault(ap.bssid, ap)
    return ScanResult(list(unique.values()), int(time.time() * 1000))


def parse_scan_output(output: str) -> list[AccessPoint]:
    """Parse the output of 'iw dev <interface> scan'."""
    access_points: list[AccessPoint] = []
    current: AccessPoint | None = None

    for raw_line in output.splitlines():
        bss_match = _BSS_RE.match(raw_line)
        if bss_match:
            current = AccessPoint(bssid=bss_match.group(1).lower(), connected="associated" in raw_line)
            access_points.append(current)
            continue
        if current is None:
            continue

        line = raw_line.strip()
        if match := _SSID_RE.match(line):
            current.ssid = match.group(1)
        elif match := _SIGNAL_RE.match(line):
            current.signal = float(match.group(1))
        elif match := _LAST_SEEN_RE.match(line):
            current.last_seen_ms = int(match.group(1))
        elif (match := _CHANNEL_RE.match(line)) and current.channel is None:
            current.channel = int(match.group(1))

    return access_points
