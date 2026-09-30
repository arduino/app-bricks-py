# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Wi-Fi based geolocation through the TPS Location API cloud service."""

import http.client
import json
import os
import re
import socket
import threading
import time
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from urllib.parse import urlsplit

import requests

from arduino.app_utils import Logger, brick

logger = Logger("TPS")

SCANNER_SOCKET_PATH = os.getenv("SCANNER_SOCKET_PATH", "/app/.cache/tps_location_api/scanner.sock")
TPS_LOC_API_URL = "https://global.skyhook.com/wps2/json/location"
TPS_AUTH_VERSION = "2.3"
TPS_PROTO_VERSION = "2.41"
HTTP_REQ_TIMEOUT_SEC = 15

STREET_ADDRESS_FIELDS = (
    "distanceToPoint",
    "streetNumber",
    "addressLine",
    "neighborhood",
    "city",
    "metro1",
    "metro2",
    "postalCode",
    "county",
    "province",
    "region",
    "stateCode",
    "stateName",
    "countryCode",
    "countryName",
)

LocationCallback = Callable[[dict[str, Any] | None, Exception | None], None]


def _snake_case(name: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def _build_payload(scan_result: dict[str, Any], street_address: bool) -> dict[str, Any]:
    """Turn a scanner response into a TPS Location API request body.

    Each access point becomes a wifiAccessPoints entry with its MAC address, signal strength in dBm
    and age in milliseconds, which is the scan age plus the time since the access point was last seen.
    Channel, SSID and connected flag are included when known.

    Args:
        scan_result (dict): Scanner response, see TPS._scan().
        street_address (bool): Ask for a full street address lookup.

    Returns:
        dict: The request body.

    Raises:
        RuntimeError: If the scan found no access points.
    """
    access_points = scan_result.get("access_points", [])
    if not access_points:
        raise RuntimeError("No access points found for location request.")

    scan_age_ms = int(scan_result.get("age_ms", 0))
    wifi_aps = []
    for ap in access_points:
        signal = ap.get("signal")
        entry = {
            "macAddress": ap["bssid"].upper(),
            "signalStrength": int(signal) if signal is not None else -100,
            "age": scan_age_ms + int(ap.get("last_seen_ms") or 0),
        }
        if ap.get("channel"):
            entry["channel"] = int(ap["channel"])
        if ap.get("ssid"):
            entry["ssid"] = ap["ssid"]
        if ap.get("connected"):
            entry["connected"] = True
        wifi_aps.append(entry)

    payload = {"considerIp": "false", "includeBeaconCounts": "true", "wifiAccessPoints": wifi_aps}
    if street_address:
        payload["streetAddressLookupType"] = "full"
    return payload


class _UnixSocketConnection(http.client.HTTPConnection):
    """HTTP connection over a Unix domain socket."""

    def __init__(self, path: str, timeout: float) -> None:
        super().__init__("localhost", timeout=timeout)
        self._path = path

    def connect(self) -> None:
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect(self._path)


@brick
class TPS:
    """Client for the TPS Location API cloud service.

    Nearby Wi-Fi access points are collected by the scanner container and resolved to a
    geographic position, optionally with a street address, by the TPS Location API.
    """

    def __init__(self, auth_key: str | None = None, auth_user: str | None = None) -> None:
        """Initialize the TPS Location API client.

        Args:
            auth_key (str | None): TPS authentication key. Defaults to the AUTH_KEY brick variable.
            auth_user (str | None): TPS authentication user. Defaults to the AUTH_USER brick variable.

        Raises:
            ValueError: If credentials are missing or the location API URL does not use https.
        """
        self.auth_key = auth_key or os.getenv("AUTH_KEY", "")
        self.auth_user = auth_user or os.getenv("AUTH_USER", "")
        if not self.auth_key or not self.auth_user:
            raise ValueError("TPS credentials missing: set the AUTH_KEY and AUTH_USER brick variables")

        self.loc_api_url = TPS_LOC_API_URL
        if urlsplit(self.loc_api_url).scheme != "https":
            raise ValueError("TPS_LOC_API_URL must use https: credentials are sent with every request")

        self.scanner_socket_path = SCANNER_SOCKET_PATH
        self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="tps-locate")
        self._lock = threading.Lock()
        self._periodic_stops: set[threading.Event] = set()

    def stop(self) -> None:
        """Stop periodic updates and pending background lookups."""
        with self._lock:
            stops = list(self._periodic_stops)
            self._periodic_stops.clear()
        for stop_event in stops:
            stop_event.set()
        self._executor.shutdown(wait=False, cancel_futures=True)

    def locate(
        self,
        request_token: str | None = None,
        street_address: bool = False,
        device_id: str | None = None,
        opt_in: bool = False,
    ) -> dict[str, Any]:
        """Get the device location by scanning Wi-Fi access points and querying the TPS Location API.

        Args:
            request_token (str | None): Request token, a UUID is generated when None.
            street_address (bool): Include the reverse geocoded street address in the response.
            device_id (str | None): Device identifier, sent in the Skyhook-PID header when provided.
            opt_in (bool): Allow the TPS Location API to persist device_id. Only meaningful with device_id.

        Returns:
            dict: location (lat, lng), accuracy in meters, nap (access points used), request_token and,
                when requested and available, street_address.

        Raises:
            RuntimeError: If the scan or the location request fails.
        """
        headers = self._build_headers(request_token, device_id, opt_in)
        payload = _build_payload(self._scan(), street_address)
        try:
            response = requests.post(self.loc_api_url, json=payload, headers=headers, timeout=HTTP_REQ_TIMEOUT_SEC)
            response.raise_for_status()
            data = response.json()
        except (requests.RequestException, ValueError) as e:
            raise RuntimeError(f"Location request failed: {e}") from e

        location = data.get("location", {})
        result: dict[str, Any] = {
            "location": {"lat": location.get("lat"), "lng": location.get("lng")},
            "accuracy": data.get("accuracy"),
            "nap": data.get("nap"),
        }
        if response_token := response.headers.get("Skyhook-Request-Token"):
            result["request_token"] = response_token
        if street_address and "streetAddress" in data:
            result["street_address"] = {_snake_case(field): data["streetAddress"].get(field) for field in STREET_ADDRESS_FIELDS}
        return result

    def async_locate(
        self,
        callback: LocationCallback,
        request_token: str | None = None,
        street_address: bool = False,
        device_id: str | None = None,
        opt_in: bool = False,
    ) -> None:
        """Run locate() in a background thread and deliver the outcome to callback.

        Args:
            callback (Callable): Called with (result, error). On success result is the location dict,
                extended with elapsed_ms, and error is None. On failure result is None and error is the exception.
            request_token (str | None): Request token, a UUID is generated when None.
            street_address (bool): Include the reverse geocoded street address in the response.
            device_id (str | None): Device identifier, sent in the Skyhook-PID header when provided.
            opt_in (bool): Allow the TPS Location API to persist device_id. Only meaningful with device_id.
        """

        def _worker() -> None:
            start = time.monotonic()
            try:
                result = self.locate(request_token=request_token, street_address=street_address, device_id=device_id, opt_in=opt_in)
            except Exception as e:
                logger.error(f"Location lookup failed: {e}")
                callback(None, e)
                return
            result["elapsed_ms"] = int((time.monotonic() - start) * 1000)
            callback(result, None)

        self._executor.submit(_worker)

    def periodic_locate(
        self,
        callback: LocationCallback,
        period_sec: int = 30,
        street_address: bool = False,
        device_id: str | None = None,
        opt_in: bool = False,
    ) -> Callable[[], None]:
        """Call locate() every period_sec seconds in the background and deliver each outcome to callback.

        Args:
            callback (Callable): Called with (result, error) after each lookup, see async_locate().
            period_sec (int): Interval in seconds between lookups. Defaults to 30.
            street_address (bool): Include the reverse geocoded street address in each response.
            device_id (str | None): Device identifier, sent in the Skyhook-PID header when provided.
            opt_in (bool): Allow the TPS Location API to persist device_id. Only meaningful with device_id.

        Returns:
            Callable[[], None]: A function that stops the periodic updates.
        """
        stop_event = threading.Event()
        with self._lock:
            self._periodic_stops.add(stop_event)

        def _scheduler() -> None:
            while not stop_event.is_set():
                self.async_locate(callback=callback, street_address=street_address, device_id=device_id, opt_in=opt_in)
                stop_event.wait(timeout=period_sec)

        def stop() -> None:
            stop_event.set()
            with self._lock:
                self._periodic_stops.discard(stop_event)

        threading.Thread(target=_scheduler, daemon=True, name="tps-periodic-locate").start()
        return stop

    def _scan(self) -> dict[str, Any]:
        """Fetch the nearby access points from the scanner container.

        Returns:
            dict: Scanner response with access_points, age_ms, timestamp_ms and cached keys.

        Raises:
            RuntimeError: If the scanner is unreachable or the scan failed.
        """
        connection = _UnixSocketConnection(self.scanner_socket_path, HTTP_REQ_TIMEOUT_SEC)
        try:
            connection.request("GET", "/scan")
            response = connection.getresponse()
            body = response.read()
        except OSError as e:
            raise RuntimeError(f"Wi-Fi scanner unreachable at {self.scanner_socket_path}: {e}") from e
        finally:
            connection.close()

        if response.status != 200:
            raise RuntimeError(f"Wi-Fi scan failed: scanner returned HTTP {response.status}")
        return json.loads(body)

    def _build_headers(self, request_token: str | None, device_id: str | None, opt_in: bool) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Skyhook-Auth-Ver": TPS_AUTH_VERSION,
            "Skyhook-Proto-Ver": TPS_PROTO_VERSION,
            "Skyhook-Request-Token": request_token or str(uuid.uuid4()),
            "Skyhook-Auth-Key": self.auth_key,
            "Skyhook-Auth-User": self.auth_user,
        }
        if device_id:
            headers["Skyhook-PID"] = device_id
            headers["Skyhook-Opt-In"] = str(opt_in).lower()
        return headers
