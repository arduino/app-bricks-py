# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import json
import threading
from collections.abc import Iterator
from dataclasses import dataclass, field
from email.message import Message
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

import pytest

import arduino.app_bricks.weather_forecast as weather_forecast
from arduino.app_utils import http
from arduino.app_bricks.weather_forecast import WeatherData, WeatherForecast

CITY_API_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_API_URL = "https://api.open-meteo.com/v1/forecast"
TURIN = {"name": "Turin", "latitude": 45.07, "longitude": 7.68}
FORECAST = {"daily": {"time": ["2026-10-05"], "weather_code": [80]}}
RAIN_SHOWERS = WeatherData(code=80, description="Rain shower(s), slight", category="rainy")


@pytest.fixture
def get():
    """Replace the HTTP GET with a mock answering the geocoding and forecast APIs.

    `get.calls` lists the requests made, as (URL, query parameters).
    """
    payloads = {CITY_API_URL: {"results": [TURIN]}, FORECAST_API_URL: FORECAST}
    calls: list[tuple[str, dict[str, str]]] = []

    def respond(url: str, *, params: dict[str, str] | None = None, **kwargs: object) -> http.Response:
        calls.append((url, {key: str(value) for key, value in (params or {}).items()}))  # As they go on the wire
        return http.Response(200, Message(), json.dumps(payloads[url]).encode())

    with patch.object(http, "get", side_effect=respond) as mock_get:
        mock_get.payloads = payloads
        mock_get.calls = calls
        yield mock_get


def test_get_forecast_by_coords_maps_the_weather_code(get):
    assert WeatherForecast().get_forecast_by_coords("45.07", "7.68") == RAIN_SHOWERS


def test_get_forecast_by_coords_sends_the_forecast_parameters(get):
    WeatherForecast().get_forecast_by_coords("45.07", "7.68", timezone="Europe/Rome", forecast_days=1)

    assert get.calls == [
        (
            FORECAST_API_URL,
            {"latitude": "45.07", "longitude": "7.68", "timezone": "Europe/Rome", "daily": "weather_code", "forecast_days": "1", "format": "json"},
        )
    ]


def test_get_forecast_by_city_url_encodes_the_city(get):
    WeatherForecast().get_forecast_by_city("New York")

    assert get.calls[0] == (CITY_API_URL, {"name": "New York"})


def test_get_forecast_by_city_raises_when_the_city_is_not_found(get):
    get.payloads[CITY_API_URL] = {}

    with pytest.raises(RuntimeError, match="City not found"):
        WeatherForecast().get_forecast_by_city("Nowhere")


def test_http_errors_become_runtime_errors(monkeypatch: pytest.MonkeyPatch):
    unavailable = http.Response(503, Message(), b"Service Unavailable")
    monkeypatch.setattr(http, "get", lambda url, **kwargs: unavailable)

    with pytest.raises(RuntimeError, match="Failed to look city up: HTTP 503: Service Unavailable"):
        WeatherForecast().get_forecast_by_city("Turin")
    with pytest.raises(RuntimeError, match="Failed to get weather data: HTTP 503: Service Unavailable"):
        WeatherForecast().get_forecast_by_coords("45.07", "7.68")


@pytest.mark.parametrize(
    "item, latitude, longitude",
    [
        ({"latitude": "45.07", "longitude": "7.68"}, "45.07", "7.68"),
        ({"latitude": "45.07", "longitude": "7.68", "city": "Milan"}, "45.07", "7.68"),
        ({"city": "Turin"}, "45.07", "7.68"),
    ],
)
def test_process_dispatches_on_the_item_keys(get, item, latitude, longitude):
    assert WeatherForecast().process(item) == RAIN_SHOWERS
    url, params = get.calls[-1]
    assert url == FORECAST_API_URL
    assert (params["latitude"], params["longitude"]) == (latitude, longitude)


@pytest.mark.parametrize("item", [{}, {"latitude": "45.07"}, {"longitude": "7.68"}])
def test_process_rejects_a_dict_without_coordinates_or_city(get, item):
    with pytest.raises(ValueError, match="Input dict must contain 'latitude' and 'longitude', or 'city'"):
        WeatherForecast().process(item)
    get.assert_not_called()


@pytest.mark.parametrize("item", ["Turin", None, 42])
def test_process_rejects_a_non_dict_item(get, item):
    with pytest.raises(ValueError, match="Input must be a dict"):
        WeatherForecast().process(item)
    get.assert_not_called()


@dataclass
class LocalApi:
    base: str
    paths: list[str] = field(default_factory=list)
    """Path and query of each request received."""


@pytest.fixture
def local_api(monkeypatch: pytest.MonkeyPatch) -> Iterator[LocalApi]:
    """Point the brick to a local HTTP server answering both APIs; any other path answers 404."""
    paths: list[str] = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            paths.append(self.path)
            payload = {"/search": {"results": [TURIN]}, "/forecast": FORECAST}.get(self.path.partition("?")[0])
            body = json.dumps(payload).encode()
            self.send_response(200 if payload is not None else 404)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    monkeypatch.setattr(weather_forecast, "city_api_url", f"{base}/search")
    monkeypatch.setattr(weather_forecast, "forecast_api_url", f"{base}/forecast")
    yield LocalApi(base, paths)
    server.shutdown()
    server.server_close()


def test_the_brick_talks_http_to_a_real_server(local_api: LocalApi, monkeypatch: pytest.MonkeyPatch):
    """The real HTTP client: query encoding, the JSON response and HTTP errors, without mocks."""
    assert WeatherForecast().get_forecast_by_city("Turin") == RAIN_SHOWERS
    assert local_api.paths[0] == "/search?name=Turin"
    assert local_api.paths[1].startswith("/forecast?latitude=45.07&longitude=7.68&")

    monkeypatch.setattr(weather_forecast, "forecast_api_url", f"{local_api.base}/missing")
    with pytest.raises(RuntimeError, match="Failed to get weather data: HTTP 404"):
        WeatherForecast().get_forecast_by_coords("45.07", "7.68")
