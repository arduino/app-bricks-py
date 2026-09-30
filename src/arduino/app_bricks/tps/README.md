# TPS Brick

This Brick provides Wi-Fi® based geolocation for your Arduino® App Lab applications. It scans the nearby Wi-Fi access points and resolves the device position, and optionally its street address, through the Qualcomm® TPS Location API cloud service.

## Overview

The TPS Brick allows you to:

- Get the device latitude, longitude and accuracy from the surrounding Wi-Fi access points.
- Get the reverse geocoded street address of the device.
- Locate once, in the background with a callback, or periodically at a fixed interval.

## Features

- Scans nearby Wi-Fi access points with the board's wireless interface, no GPS required
- Returns latitude, longitude, accuracy in meters and the number of access points used
- Optional reverse geocoding to a full street address
- Blocking, background and periodic lookups with a single `(result, error)` callback signature
- Credentials read from **Brick Configuration**, never hardcoded

## Prerequisites

- **Supported boards**: Arduino UNO™ Q and Arduino VENTUNO™ Q.
- **Internet connection**: The board must reach the TPS Location API cloud service.
- **Wi-Fi interface**: The board's wireless interface must be available for scanning, even when the board is connected to the internet another way. The location is computed from visible access points only, with no IP-based fallback.
- **TPS credentials**: Register on the [TPS Portal](https://my.skyhook.com/), create a project and copy its Auth Key. The key comes with a 60-day evaluation period. Set the key and your authentication user in **Brick Configuration** in App Lab, as described in [Configuration](#configuration).

## Code example and usage

### Locate once

`locate()` blocks until the lookup completes and returns the result:

```python
from arduino.app_bricks.tps import TPS

client = TPS()

location = client.locate()
print(f"Lat: {location['location']['lat']}, Lng: {location['location']['lng']}, Accuracy: {location['accuracy']}m")
```

### Locate in the background

`async_locate()` returns immediately and delivers the outcome to a callback, as a result on success or an exception on failure:

```python
from arduino.app_bricks.tps import TPS
from arduino.app_utils import App

client = TPS()

def on_location(result, error):
    if error:
        print(f"Error: {error}")
        return
    print(f"Location: {result['location']} in {result['elapsed_ms']}ms")

client.async_locate(on_location)

App.run()
```

### Locate periodically

`periodic_locate()` repeats the lookup at a fixed interval, here every 30 seconds, and returns a function that stops it:

```python
from arduino.app_bricks.tps import TPS
from arduino.app_utils import App

client = TPS()

def on_location(result, error):
    if error:
        print(f"Error: {error}")
        return
    print(f"Location: {result['location']}, accuracy {result['accuracy']}m")

stop = client.periodic_locate(on_location, period_sec=30)

App.run()
```

Call `stop()` to end the periodic updates, or `client.stop()` to end all periodic updates and pending background lookups at once.

**Note:** Background and periodic lookups run on separate threads. Keep the application running, for example with `App.run()`, or the script exits before any callback fires.

### Get the street address

Pass `street_address=True` to any of the methods above to add the street address to the result:

```python
location = client.locate(street_address=True)
address = location.get("street_address")
if address:
    # Unknown fields are None, so print only the ones with a value
    fields = (address["address_line"], address["city"], address["country_name"])
    print(", ".join(value for value in fields if value))
```

## Understanding the Result

`locate()` returns a dictionary, which `async_locate()` and `periodic_locate()` deliver as the first callback argument:

| Key | Description |
|-----|-------------|
| `location` | Dictionary with `lat` and `lng`. Values are `None` if the service returns no position. |
| `accuracy` | Accuracy in meters |
| `nap` | Number of access points used for the fix |
| `request_token` | Token identifying the request, when returned by the service |
| `elapsed_ms` | Duration of the lookup, only for `async_locate()` and `periodic_locate()` |
| `street_address` | Only with `street_address=True`, when the service returns an address. Dictionary with `address_line`, `street_number`, `neighborhood`, `city`, `postal_code`, `county`, `province`, `region`, `state_code`, `state_name`, `country_code`, `country_name`, `metro1`, `metro2` and `distance_to_point`. Unknown fields are `None`. |

## Configuration

Set these variables in **Brick Configuration** in App Lab:

| Variable | Description | Default |
|----------|-------------|---------|
| `AUTH_KEY` | TPS authentication key, stored as a secret | *(required)* |
| `AUTH_USER` | TPS authentication user | *(required)* |

Both can also be passed to the constructor: `TPS(auth_key=..., auth_user=...)`.

## Methods

- **`locate(request_token=None, street_address=False, device_id=None, opt_in=False)`**: Scans the access points, queries the TPS Location API and returns the result dictionary. Blocks until the lookup completes.
- **`async_locate(callback, request_token=None, street_address=False, device_id=None, opt_in=False)`**: Runs `locate()` in a background thread and calls `callback(result, error)` when done.
- **`periodic_locate(callback, period_sec=30, street_address=False, device_id=None, opt_in=False)`**: Runs a background lookup every `period_sec` seconds and calls `callback(result, error)` after each one. Returns a function that stops the updates.
- **`stop()`**: Stops all periodic updates and cancels pending background lookups.

The optional parameters work the same way for every method:

- `request_token`: Custom token identifying the request. A UUID is generated when omitted.
- `street_address`: Adds the reverse geocoded street address to the result.
- `device_id`: Identifier of your device, sent along with the request.
- `opt_in`: Allows the service to persist `device_id`. Only meaningful with `device_id`.

## Errors

- The constructor raises `ValueError` if `AUTH_KEY` or `AUTH_USER` is missing.
- `locate()` raises `RuntimeError` if the scanner is unreachable, the scan finds no access points, or the location request fails.
- `async_locate()` and `periodic_locate()` never raise for a failed lookup: they deliver the exception as the second callback argument.

## Working Principle

The Brick runs a companion scanner container alongside your app. The container uses the board's wireless interface to scan for nearby access points and shares the results with the Brick. To limit the scan rate, results are reused for 10 seconds, so lookups closer together than that share the same scan.

For each lookup, the Brick sends the MAC address, signal strength and age of every visible access point to the TPS Location API, which returns the estimated position. If no access points are visible, for example in a shielded room, the lookup fails because the service does not fall back to IP-based location.
