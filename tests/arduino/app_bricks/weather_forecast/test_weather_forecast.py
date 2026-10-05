# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

from unittest.mock import MagicMock, patch

import pytest

from arduino.app_bricks.weather_forecast import WeatherData, WeatherForecast

CITY_API_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_API_URL = "https://api.open-meteo.com/v1/forecast"
TURIN = {"name": "Turin", "latitude": 45.07, "longitude": 7.68}
RAIN_SHOWERS = WeatherData(code=80, description="Rain shower(s), slight", category="rainy")


@pytest.fixture
def get():
    """Replace requests.get with a mock answering the geocoding and forecast APIs."""
    payloads = {CITY_API_URL: {"results": [TURIN]}, FORECAST_API_URL: {"daily": {"time": ["2026-10-05"], "weather_code": [80]}}}

    def respond(url, params):
        return MagicMock(json=MagicMock(return_value=payloads[url]))

    with patch("arduino.app_bricks.weather_forecast.requests.get", side_effect=respond) as mock_get:
        mock_get.payloads = payloads
        yield mock_get


def test_get_forecast_by_coords_maps_the_weather_code(get):
    assert WeatherForecast().get_forecast_by_coords("45.07", "7.68") == RAIN_SHOWERS


def test_get_forecast_by_city_raises_when_the_city_is_not_found(get):
    get.payloads[CITY_API_URL] = {}

    with pytest.raises(RuntimeError, match="City not found"):
        WeatherForecast().get_forecast_by_city("Nowhere")


@pytest.mark.parametrize(
    "item, latitude, longitude",
    [
        ({"latitude": "45.07", "longitude": "7.68"}, "45.07", "7.68"),
        ({"latitude": "45.07", "longitude": "7.68", "city": "Milan"}, "45.07", "7.68"),
        ({"city": "Turin"}, 45.07, 7.68),
    ],
)
def test_process_dispatches_on_the_item_keys(get, item, latitude, longitude):
    assert WeatherForecast().process(item) == RAIN_SHOWERS
    assert get.call_args.args == (FORECAST_API_URL,)
    assert get.call_args.kwargs["params"]["latitude"] == latitude
    assert get.call_args.kwargs["params"]["longitude"] == longitude


@pytest.mark.parametrize("item", [{}, {"latitude": "45.07"}, "Turin", None])
def test_process_returns_an_empty_dict_on_an_unsupported_item(get, item):
    assert WeatherForecast().process(item) == {}
    get.assert_not_called()
