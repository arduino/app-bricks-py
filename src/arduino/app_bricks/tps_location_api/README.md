# 📍 TPS Location API

A custom Arduino APP Lab Brick that provides WiFi-based geolocation using the TPS Location API cloud service.

## Overview

**TPS Location API** scans nearby Wi-Fi Access Points and resolves the device's geographic position (latitude, longitude, accuracy) using the TPS Location API cloud service. It can also resolve the device's civic/street address (city, state, country, postal code, etc.) via reverse geocoding. It supports synchronous, asynchronous (callback-based), and periodic location modes.

## How to get TPS Auth Key

To obtain your TPS authentication credentials, access the [TPS Portal](https://www.my.skyhook.com/). If you do not have an account, register for access and create a new project. Upon project creation, you will receive your Auth Key, which include a 60-day evaluation period.

## How to use it

Import and use the `TPSLocationAPI` client in your main application code:

### Synchronous (blocking)

```python
from arduino.app_utils import App
from arduino.app_bricks.tps_location_api import TPSLocationAPI

client = TPSLocationAPI()
location = client.locate()

print(f"Lat: {location['location']['lat']}")
print(f"Lng: {location['location']['lng']}")
print(f"Accuracy: {location['accuracy']}m")
print(f"APs used: {location['nap']}")
# Output: Lat: 40.123456, Lng: -74.654321, Accuracy: 25.0m, APs used: 17
```

### Asynchronous (non-blocking with callback)

```python
from arduino.app_utils import App
from arduino.app_bricks.tps_location_api import TPSLocationAPI

client = TPSLocationAPI()

def on_location(result, error):
    if error:
        print(f"Error: {error}")
        return
    loc = result["location"]
    print(f"Location: lat={loc['lat']}, lng={loc['lng']} | "
          f"Accuracy: {result['accuracy']}m | "
          f"Elapsed: {result['elapsed_ms']}ms")

client.async_locate(callback=on_location)
```

### Periodic (automatic fixed-interval updates)

```python
from arduino.app_utils import App
from arduino.app_bricks.tps_location_api import TPSLocationAPI

client = TPSLocationAPI()

def on_location(result, error):
    if error:
        print(f"Error: {error}")
        return
    loc = result["location"]
    print(f"Location: lat={loc['lat']}, lng={loc['lng']} | "
          f"Accuracy: {result['accuracy']}m | "
          f"Elapsed: {result['elapsed_ms']}ms")

# Start periodic location updates every 30 seconds
stop = client.periodic_locate(callback=on_location, period_sec=30)

# Later, to stop:
stop()
```

### Street Address (street address lookup)

Pass `street_address=True` to any locate method to include a reverse-geocoded street address in the response:

```python
from arduino.app_utils import App
from arduino.app_bricks.tps_location_api import TPSLocationAPI

client = TPSLocationAPI()

# Synchronous with street address
location = client.locate(street_address=True)
addr = location["street_address"]
print(f"Address: {addr['address_line']}, {addr['city']}, {addr['state_name']} {addr['postal_code']}")
# Output: Address: 123 Main St, Springfield, New Jersey 07081

# Async with street address
client.async_locate(callback=on_location, street_address=True)

# Periodic with street address
stop = client.periodic_locate(callback=on_location, period_sec=30, street_address=True)
```

### Full application example

```python
# python/main.py
from arduino.app_utils import App
from arduino.app_bricks.tps_location_api import TPSLocationAPI

# Assuming (e.g.) MAC address as device ID
device_id = "14:b5:cd:e8:7d:43"

client = TPSLocationAPI()

def on_location(result, error):
    if error:
        print(f"Error: {error}")
        return
    loc = result["location"]
    print(f"Location: lat={loc['lat']}, lng={loc['lng']} | "
          f"Accuracy: {result['accuracy']}m | "
          f"Elapsed: {result['elapsed_ms']}ms")

# Start periodic location updates every 30 seconds
stop = client.periodic_locate(callback=on_location, period_sec=30, device_id=device_id, opt_in=True)

App.run()
```

## Public API

### `TPSLocationAPI.locate(request_token=None, street_address=False, device_id=None, opt_in=False) -> Dict`

**Blocking** — Scans WiFi APs and resolves device location. Blocks until result is available.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `request_token` | `str` or `None` | `None` | Request token (UUID generated if None) |
| `street_address` | `bool` | `False` | If True, include street address in response |
| `device_id` | `str` or `None` | `None` | Device identifier. When provided, sent via Skyhook-PID header |
| `opt_in` | `bool` | `False` | Controls whether the TPS Location API persists the device_id (only meaningful when device_id is provided) |

**Returns:**
```python
{
    "location": {"lat": 40.123456, "lng": -74.654321},
    "accuracy": 25.0,
    "nap": 17,
    "request_token": "uuid-string",
    # Only if street_address=True (some fields may be None):
    "street_address": {
        "distance_to_point": 12.5,
        "street_number": "123",
        "address_line": "123 Main St",
        "neighborhood": "Downtown",
        "city": "Springfield",
        "metro1": "New York",
        "metro2": "Newark",
        "postal_code": "07081",
        "county": "Union",
        "province": None,
        "region": None,
        "state_code": "NJ",
        "state_name": "New Jersey",
        "country_code": "US",
        "country_name": "United States",
    }
}
```

**Raises:** `RuntimeError` if the scan or location request fails.

---

### `TPSLocationAPI.async_locate(callback, request_token=None, street_address=False, device_id=None, opt_in=False) -> None`

**Non-blocking** — Performs the location lookup in a background thread and invokes the callback when ready.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `callback` | `Callable[[Dict, Exception], None]` | *(required)* | Called with `(result, error)` |
| `request_token` | `str` or `None` | `None` | Request token (UUID generated if None) |
| `street_address` | `bool` | `False` | If True, include street address in response |
| `device_id` | `str` or `None` | `None` | Device identifier. When provided, sent via Skyhook-PID header |
| `opt_in` | `bool` | `False` | Controls whether the TPS Location API persists the device_id |

**Callback signature:**
```python
def on_location(result: Optional[Dict], error: Optional[Exception]) -> None:
    # On success: result is the location dict, error is None
    # On failure: result is None, error is the exception
```

The result dict includes `elapsed_ms` (time taken for the locate call in milliseconds).

---

### `TPSLocationAPI.periodic_locate(callback, period_sec=30, street_address=False, device_id=None, opt_in=False) -> Callable`

**Periodic** — Schedules locate calls at fixed intervals. A new call is dispatched every `period_sec` seconds regardless of how long the previous call takes (calls run in a thread pool).

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `callback` | `Callable[[Dict, Exception], None]` | *(required)* | Called with `(result, error)` after each fix |
| `period_sec` | `int` | `30` | Interval between calls in seconds |
| `street_address` | `bool` | `False` | If True, include street address in each response |
| `device_id` | `str` or `None` | `None` | Device identifier. When provided, sent via Skyhook-PID header |
| `opt_in` | `bool` | `False` | Controls whether the TPS Location API persists the device_id |

**Returns:** A `stop()` function. Call it to cancel periodic updates.

```python
stop = client.periodic_locate(callback=on_location, period_sec=10)
# ... later ...
stop()  # stops periodic updates
```

---

## Configuration

The following variables are defined in `brick_config.yaml` and exposed as environment variables:

### User-facing Variables

| Variable | Description | Type | Default |
|----------|-------------|------|---------|
| `AUTH_KEY` | TPS authentication key | string (secret) | *(required)* |
| `AUTH_USER` | TPS authentication user | string | *(required)* |

### Hidden (Advanced) Variables

| Variable | Description | Type | Default |
|----------|-------------|------|---------|
| `TPS_LOC_API_URL` | TPS Location API cloud service endpoint URL | string | `https://global.skyhook.com/wps2/json/location` |
| `TPS_AUTH_VERSION` | TPS Location API authentication version header | string | `2.3` |
| `TPS_PROTO_VERSION` | TPS Location API protocol version header | string | `2.41` |
| `HTTP_REQ_TIMEOUT_SEC` | Timeout for outgoing HTTP requests | int | `15` |
| `SCANNER_HOST` | Host/IP of the scanner container. The scanner uses `network_mode: host`, so it is reachable from the app container via the Docker bridge gateway (`172.17.0.1`). Override if your Docker bridge uses a different subnet. | string | `172.17.0.1` |
| `SCANNER_PORT` | Scanner service port | int | `8089` |
| `SCAN_CHANNEL_DWELL_TIME_MS` | Per-channel dwell time during WiFi scan (ms) | int | `60` |
| `SCAN_TIMEOUT_SECONDS` | Timeout for iw scan command | int | `10` |
| `SCAN_RETRIES` | Retry attempts on scan failure | int | `3` |
| `SCAN_CACHE_SECONDS` | Duration to cache scan results | int | `10` |

## What's in the brick folder

| File | Description |
|------|-------------|
| `__init__.py` | Public client class (`TPSLocationAPI`) with `locate()`, `async_locate()`, `periodic_locate()` |
| `brick_config.yaml` | Brick identity and configurable variables |
| `brick_compose.yaml` | Docker Compose service definition for the scanner container |

The scanner server code (`scan_server.py`, `iw_scanner.py`) and its dependencies are baked into the `tps-location-api` container image under `containers/bricks/tps-location-api/`.

## Architecture

```
┌─────────────────────┐      ┌──────────────────────┐      ┌─────────────────┐
│   Main App          │      │  Scanner Container   │      │  TPS Loc API    │
│   (TPSLocationAPI)  │─────▶│  (FastAPI + iw)      │      │  Cloud Service  │
│                     │      │  172.17.0.1:8089     │      │                 │
│   locate()          │──────────────────────────────────▶ │  /wps2/json/    │
│   async_locate()    │      │  network_mode: host  │      │  location       │
│   periodic_locate() │      │  cap_add: NET_ADMIN  │      │                 │
└─────────────────────┘      └──────────────────────┘      └─────────────────┘
```

1. `locate()` / `async_locate()` / `periodic_locate()` calls the scanner container to get WiFi AP data
2. Builds a location-compatible JSON payload with AP details
3. Sends the payload to the TPS Location API cloud service with authentication headers
4. Returns parsed latitude, longitude, accuracy, number of APs used, and optionally street address