import os
import subprocess
import re
import time
from typing import List, Dict, Optional, Tuple

SCAN_TIMEOUT_SECONDS = int(os.getenv("SCAN_TIMEOUT_SECONDS", "10"))
SCAN_RETRIES = int(os.getenv("SCAN_RETRIES", "3"))
SCAN_CHANNEL_DWELL_TIME_MS = int(os.getenv("SCAN_CHANNEL_DWELL_TIME_MS", "60"))


def get_wireless_interface() -> Optional[str]:
    """Detect the first available wireless interface."""
    try:
        result = subprocess.run(
            ["iw", "dev"],
            capture_output=True,
            text=True,
            timeout=SCAN_TIMEOUT_SECONDS
        )
        if result.returncode != 0:
            return None

        for line in result.stdout.splitlines():
            line = line.strip()
            if line.startswith("Interface"):
                return line.split()[1]
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None

    return None


def scan_access_points(interface: Optional[str] = None) -> Optional[Tuple[List[Dict[str, str]], int]]:
    """
    Scan for nearby WiFi Access Points using 'iw' command.

    Args:
        interface: Wireless interface name. If None, auto-detects.

    Returns:
        Tuple of (list of AP dicts, timestamp_ms) where timestamp_ms is
        the collection time in milliseconds since epoch, or None on failure.
    """
    if interface is None:
        interface = get_wireless_interface()
        if interface is None:
            return None

    result = None
    for attempt in range(SCAN_RETRIES):
        try:
            result = subprocess.run(
                ["iw", "dev", interface, "scan", "duration", str(SCAN_CHANNEL_DWELL_TIME_MS)],
                capture_output=True,
                text=True,
                timeout=SCAN_TIMEOUT_SECONDS
            )
            if result.returncode == 0:
                break
            # Handle "resource busy" (-16) by waiting with exponential backoff
            # With 3 retries: waits 2s, 4s (6s total covers ~5s scan duration)
            if "Device or resource busy" in result.stderr:
                time.sleep(2 * (attempt + 1))
                continue
        except (subprocess.TimeoutExpired, FileNotFoundError):
            continue
    else:
        return None

    if result is None or result.returncode != 0:
        return None

    timestamp_ms = int(time.time() * 1000)
    access_points = parse_iw_scan_output(result.stdout)

    # Sort by signal strength (strongest first)
    access_points.sort(key=lambda ap: ap.get("signal", -999.0), reverse=True)

    # Remove duplicates by BSSID, keeping the strongest signal
    seen_bssids = set()
    unique_aps = []
    for ap in access_points:
        bssid = ap["bssid"]
        if bssid not in seen_bssids:
            seen_bssids.add(bssid)
            unique_aps.append(ap)

    return unique_aps, timestamp_ms


def parse_iw_scan_output(output: str) -> List[Dict[str, str]]:
    """Parse the output of 'iw dev <interface> scan' command."""
    access_points = []
    current_ap = None

    for line in output.splitlines():
        bss_match = re.match(r"^BSS\s+([0-9a-f:]{17})", line, re.IGNORECASE)
        if bss_match:
            if current_ap:
                access_points.append(current_ap)
            current_ap = {
                "bssid": bss_match.group(1),
                "ssid": "",
                "signal": "",
                "channel": "",
                "connected": "associated" in line.lower()
            }
            continue

        if current_ap is None:
            continue

        line = line.strip()

        ssid_match = re.match(r"^SSID:\s*(.*)", line)
        if ssid_match:
            current_ap["ssid"] = ssid_match.group(1)
            continue

        signal_match = re.match(r"^signal:\s*([-\d.]+)\s*dBm", line)
        if signal_match:
            current_ap["signal"] = float(signal_match.group(1))
            continue

        channel_match = re.match(r"^\* primary channel:\s*(\d+)", line)
        if channel_match:
            current_ap["channel"] = channel_match.group(1)
            continue

    if current_ap:
        access_points.append(current_ap)

    return access_points
