import os
import time
import uuid
import threading
import requests
from typing import List, Dict, Optional, Callable
from concurrent.futures import ThreadPoolExecutor


SCANNER_HOST = os.getenv("SCANNER_HOST", "172.17.0.1")
SCANNER_PORT = os.getenv("SCANNER_PORT", "8089")
SCANNER_BASE_URL = f"http://{SCANNER_HOST}:{SCANNER_PORT}"

TPS_LOC_API_URL = os.getenv("TPS_LOC_API_URL", "https://global.skyhook.com/wps2/json/location")
AUTH_KEY = os.getenv("AUTH_KEY", "")
AUTH_USER = os.getenv("AUTH_USER", "")
TPS_AUTH_VERSION = os.getenv("TPS_AUTH_VERSION", "2.3")
TPS_PROTO_VERSION = os.getenv("TPS_PROTO_VERSION", "2.41")
HTTP_REQ_TIMEOUT_SEC = int(os.getenv("HTTP_REQ_TIMEOUT_SEC", "15"))


class TPSLocationAPI:
    """Client for the TPS Location API cloud service."""

    def __init__(self):
        """Initialize the TPS Location API client."""
        self.scanner_base_url = SCANNER_BASE_URL
        self.loc_api_url = TPS_LOC_API_URL
        self.auth_key = AUTH_KEY
        self.auth_user = AUTH_USER
        self._executor = ThreadPoolExecutor(max_workers=4)

    def _scan(self, interface: Optional[str] = None, force_refresh: bool = False) -> Dict:
        """
        Internal: Scan nearby WiFi Access Points by sending a GET request to the client scan container.

        Args:
            interface: Wireless interface to use (auto-detects if not provided).
            force_refresh: Force a new scan, bypassing cache.

        Returns:
            Dict with keys:
                - access_points: List of AP dicts (bssid, ssid, signal, channel, connected)
                - age_ms: Age of the scan data in milliseconds
                - timestamp_ms: Collection time in milliseconds since epoch
                - cached: Whether the result was served from cache

        Raises:
            RuntimeError: If the scan request fails.
        """
        params = {}
        if interface:
            params["interface"] = interface
        if force_refresh:
            params["force_refresh"] = "true"

        try:
            response = requests.get(f"{self.scanner_base_url}/scan", params=params, timeout=HTTP_REQ_TIMEOUT_SEC)
            response.raise_for_status()
        except requests.RequestException as e:
            raise RuntimeError(f"WiFi scan request failed: {e}") from e

        return response.json()

    def locate(self, request_token: Optional[str] = None, street_address: bool = False, device_id: Optional[str] = None, opt_in: bool = False) -> Dict:
        """
        Get location by scanning WiFi APs and sending them to the TPS Location API.

        Args:
            request_token: Optional request token. If None, a UUID is generated.
            street_address: If True, include street address lookup in the response.
            device_id: Optional device identifier. When provided, sent via Skyhook-PID header.
            opt_in: Controls whether the TPS Location API persists the device_id (only meaningful when device_id is provided).

        Returns:
            Dict with keys:
                - location: Dict with 'lat' and 'lng' (floats)
                - accuracy: Accuracy in meters (float)
                - nap: Number of access points used (int)
                - street_address: (only if street_address=True) Dict with street address fields

        Raises:
            RuntimeError: If the scan or location request fails.
        """
        # Get scan results
        scan_result = self._scan(force_refresh=False)
        access_points = scan_result.get("access_points", [])
        age_ms = scan_result.get("age_ms", 0)

        if not access_points:
            raise RuntimeError("No access points found for location request.")

        # Build Skyhook-compatible request payload
        wifi_aps = []
        for ap in access_points:
            entry = {
                "macAddress": ap.get("bssid", "").upper(),
                "signalStrength": int(ap.get("signal", -100)),
                "age": age_ms,
            }
            if ap.get("channel"):
                entry["channel"] = ap["channel"]
            if ap.get("ssid"):
                entry["ssid"] = ap["ssid"]
            if ap.get("connected"):
                entry["connected"] = True
            wifi_aps.append(entry)

        payload = {
            "considerIp": "false",
            "includeBeaconCounts": "true",
            "wifiAccessPoints": wifi_aps
        }

        if street_address:
            payload["streetAddressLookupType"] = "full"

        if request_token is None:
            request_token = str(uuid.uuid4())

        # Build headers
        headers = {
            "Content-Type": "application/json",
            "Skyhook-Auth-Ver": TPS_AUTH_VERSION,
            "Skyhook-Proto-Ver": TPS_PROTO_VERSION,
            "Skyhook-Request-Token": request_token,
            "Skyhook-Auth-Key": self.auth_key,
            "Skyhook-Auth-User": self.auth_user,
        }

        # Send device ID if provided; opt_in controls whether the TPS Location API persists it
        if device_id:
            headers["Skyhook-PID"] = device_id
            headers["Skyhook-Opt-In"] = str(opt_in).lower()


        # Send request to TPS Location cloud service
        try:
            response = requests.post(
                self.loc_api_url,
                json=payload,
                headers=headers,
                timeout=HTTP_REQ_TIMEOUT_SEC
            )
            response.raise_for_status()
        except requests.RequestException as e:
            raise RuntimeError(f"Location request failed: {e}") from e

        response_token = response.headers.get("Skyhook-Request-Token")

        data = response.json()

        # Parse into structured response
        location_data = data.get("location", {})
        result = {
            "location": {
                "lat": location_data.get("lat"),
                "lng": location_data.get("lng")
            },
            "accuracy": data.get("accuracy"),
            "nap": data.get("nap")
        }
        if response_token:
            result["request_token"] = response_token
            
        if street_address and "streetAddress" in data:
            sa = data["streetAddress"]
            result["street_address"] = {
                "distance_to_point": sa.get("distanceToPoint"),
                "street_number": sa.get("streetNumber"),
                "address_line": sa.get("addressLine"),
                "neighborhood": sa.get("neighborhood"),
                "city": sa.get("city"),
                "metro1": sa.get("metro1"),
                "metro2": sa.get("metro2"),
                "postal_code": sa.get("postalCode"),
                "county": sa.get("county"),
                "province": sa.get("province"),
                "region": sa.get("region"),
                "state_code": sa.get("stateCode"),
                "state_name": sa.get("stateName"),
                "country_code": sa.get("countryCode"),
                "country_name": sa.get("countryName"),
            }

        return result

    def async_locate(
        self,
        callback: Callable[[Optional[Dict], Optional[Exception]], None],
        request_token: Optional[str] = None,
        street_address: bool = False,
        device_id: Optional[str] = None,
        opt_in: bool = False
    ) -> None:
        """
        Non-blocking version of locate(). Performs the location lookup in a
        background thread and invokes the callback when the result is ready.

        Args:
            callback: A function called with (result, error). On success,
                      result is the location dict and error is None. On failure,
                      result is None and error is the raised exception.
            request_token: Optional request token. If None, a UUID is generated.
            street_address: If True, include street address lookup in the response.
            device_id: Optional device identifier. When provided, sent via Skyhook-PID header.
            opt_in: Controls whether the TPS Location API persists the device_id (only meaningful when device_id is provided).

        Returns:
            None. The result is delivered via the callback.

        Example:
            def on_location(result, error):
                if error:
                    logger.error(f"Locate failed: {error}")
                else:
                    logger.info(f"Location: {result}")

            loc_api.async_locate(callback=on_location)
        """
        def _worker():
            try:
                start_ms = time.time() * 1000
                result = self.locate(request_token=request_token, street_address=street_address, device_id=device_id, opt_in=opt_in)
                result["elapsed_ms"] = int(time.time() * 1000 - start_ms)
                callback(result, None)
            except Exception as e:
                callback(None, e)

        self._executor.submit(_worker)

    def periodic_locate(
        self,
        callback: Callable[[Optional[Dict], Optional[Exception]], None],
        period_sec: int = 30,
        street_address: bool = False,
        device_id: Optional[str] = None,
        opt_in: bool = False
    ) -> Callable[[], None]:
        """
        Periodically call locate() in a background thread and deliver results
        via the callback. The caller does not need to implement a loop.

        Args:
            callback: A function called with (result, error) after each location fix.
                      On success, result is the location dict and error is None.
                      On failure, result is None and error is the raised exception.
            period_sec: Interval in seconds between locate() calls (default: 30).
            street_address: If True, include street address lookup in each response.
            device_id: Optional device identifier. When provided, sent via Skyhook-PID header.
            opt_in: Controls whether the TPS Location API persists the device_id (only meaningful when device_id is provided).

        Returns:
            A stop function. Call it to stop the periodic location updates.

        Example:
            def on_location(result, error):
                if error:
                    logger.error(f"Locate failed: {error}")
                else:
                    logger.info(f"Location: {result}")

            stop = loc_api.periodic_locate(callback=on_location, period_sec=10)
            # ... later ...
            stop()  # stops the periodic updates
        """
        stop_event = threading.Event()

        def _scheduler():
            """Schedule locate calls at fixed intervals using async_locate."""
            while not stop_event.is_set():
                request_token = str(uuid.uuid4())
                self.async_locate(callback=callback, request_token=request_token, street_address=street_address, device_id=device_id, opt_in=opt_in)
                stop_event.wait(timeout=period_sec)

        thread = threading.Thread(target=_scheduler, daemon=True, name="periodic-locate")
        thread.start()

        def stop():
            """Stop the periodic locate loop."""
            stop_event.set()

        return stop
