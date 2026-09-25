# TPS

Wi-Fi based geolocation for your Arduino app. The brick scans the nearby Wi-Fi access points and resolves the device position, and optionally its street address, through the [TPS Location API](https://www.my.skyhook.com/) cloud service.

## Overview

The TPS brick allows you to:

- Get the device latitude, longitude and accuracy from the surrounding Wi-Fi access points
- Get the reverse geocoded street address of the device
- Locate once, in the background with a callback, or periodically at a fixed interval

An internet connection and TPS credentials are required. Register on the [TPS Portal](https://my.skyhook.com/), create a project and copy its Auth Key: it comes with a 60-day evaluation period.

## Code example and usage

Locate the device once:

```python
from arduino.app_bricks.tps import TPS

client = TPS()

location = client.locate()
print(f"Lat: {location['location']['lat']}, Lng: {location['location']['lng']}, Accuracy: {location['accuracy']}m")
```

Locate in the background and receive the result through a callback:

```python
def on_location(result, error):
    if error:
        print(f"Error: {error}")
        return
    print(f"Location: {result['location']} in {result['elapsed_ms']}ms")

client.async_locate(on_location)
```

Locate periodically, here every 30 seconds, and stop whenever needed:

```python
stop = client.periodic_locate(on_location, period_sec=30)
# ... later ...
stop()
```

Pass `street_address=True` to any of the methods above to add the street address to the result:

```python
location = client.locate(street_address=True)
address = location["street_address"]
print(f"{address['address_line']}, {address['city']}, {address['country_name']}")
```

## Understanding the result

`locate()` returns a dictionary, delivered as the first callback argument by `async_locate()` and `periodic_locate()`:

| Key | Description |
|-----|-------------|
| `location` | Dictionary with `lat` and `lng` |
| `accuracy` | Accuracy in meters |
| `nap` | Number of access points used for the fix |
| `request_token` | Token identifying the request, as returned by the service |
| `elapsed_ms` | Duration of the lookup, only for `async_locate()` and `periodic_locate()` |
| `street_address` | Only with `street_address=True`. Dictionary with `address_line`, `street_number`, `neighborhood`, `city`, `postal_code`, `county`, `province`, `region`, `state_code`, `state_name`, `country_code`, `country_name`, `metro1`, `metro2` and `distance_to_point`. Unknown fields are `None`. |

All methods also accept `device_id`, an identifier of your device sent along with the request, and `opt_in`, which allows the service to persist it. `locate()` and `async_locate()` accept a custom `request_token` too.

A failed lookup raises `RuntimeError` from `locate()`, and is delivered as the second callback argument by the other methods.

## Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `AUTH_KEY` | TPS authentication key | *(required)* |
| `AUTH_USER` | TPS authentication user | *(required)* |

Both can also be passed to the constructor: `TPS(auth_key=..., auth_user=...)`.
