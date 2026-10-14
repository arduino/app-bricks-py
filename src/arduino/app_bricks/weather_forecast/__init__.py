# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import json
import os
from dataclasses import dataclass
from typing import Any

from arduino.app_utils import http
from arduino.app_utils import brick

city_api_url = "https://geocoding-api.open-meteo.com/v1/search"
forecast_api_url = "https://api.open-meteo.com/v1/forecast"


def _get_json(url: str, params: dict[str, str]) -> Any:  # noqa: ANN401
    """GET url with the query parameters and decode the JSON response. HTTP errors (4xx, 5xx) raise."""
    response = http.get(url, params=params)
    response.raise_for_status()
    return response.json()


@dataclass(frozen=True)
class WeatherData:
    """Weather forecast data with standardized codes and categories.

    Attributes:
        code (int): WMO weather code representing specific weather conditions.
        description (str): Human-readable weather description (e.g., "Partly cloudy", "Heavy rain").
        category (str): Simplified weather category: "sunny", "cloudy", "rainy", "snowy", or "foggy".
    """

    code: int
    description: str
    category: str


# The weather codes have been taken from here: https://www.nodc.noaa.gov/archive/arc0021/0002199/1.1/data/0-data/HTML/WMO-CODE/WMO4677.HTM
with open(os.path.join(os.path.dirname(__file__), "weather_data.json"), encoding="utf-8") as file:
    weather_data = json.load(file)


@brick
class WeatherForecast:
    """Weather forecast service using the open-meteo.com API.

    Provides weather forecasts by city name or geographic coordinates with no API key required.
    Returns structured weather data with WMO codes, descriptions, and simplified categories.
    """

    def get_forecast_by_city(self, city: str, timezone: str = "GMT", forecast_days: int = 1) -> WeatherData:
        """Get weather forecast for a specified city.

        Args:
            city (str): City name (e.g., "London", "New York").
            timezone (str): Timezone identifier. Defaults to "GMT".
            forecast_days (int): Number of days to forecast. Defaults to 1.

        Returns:
            WeatherData: Weather forecast with code, description, and category.

        Raises:
            RuntimeError: If city lookup or weather data retrieval fails.
        """
        try:
            data = _get_json(city_api_url, {"name": city})
        except Exception as e:
            raise RuntimeError(f"Failed to look city up: {e}")
        results = data.get("results", [])
        if results:
            result = results[0]
        else:
            raise RuntimeError("City not found")

        return self.get_forecast_by_coords(result["latitude"], result["longitude"], timezone=timezone, forecast_days=forecast_days)

    def get_forecast_by_coords(self, latitude: str, longitude: str, timezone: str = "GMT", forecast_days: int = 1) -> WeatherData:
        """Get weather forecast for specific coordinates.

        Args:
            latitude (str): Latitude coordinate (e.g., "45.0703").
            longitude (str): Longitude coordinate (e.g., "7.6869").
            timezone (str): Timezone identifier. Defaults to "GMT".
            forecast_days (int): Number of days to forecast. Defaults to 1.

        Returns:
            WeatherData: Weather forecast with code, description, and category.

        Raises:
            RuntimeError: If weather data retrieval fails.
        """
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "timezone": timezone,
            "daily": "weather_code",
            "forecast_days": f"{forecast_days}",
            "format": "json",
        }
        try:
            data = _get_json(forecast_api_url, params)
        except Exception as e:
            raise RuntimeError(f"Failed to get weather data: {e}")

        if "daily" not in data or "weather_code" not in data["daily"]:
            raise RuntimeError("Invalid response format")

        # This is the exact format of the response:
        # {
        #   "latitude":45.08,
        #   "longitude":7.68,
        #   "generationtime_ms":0.014185905456542969,
        #   "utc_offset_seconds":0,
        #   "timezone":"GMT",
        #   "timezone_abbreviation":"GMT",
        #   "elevation":239.0,
        #   "daily_units":{
        #       "time":"iso8601","weather_code":"wmo code"
        #   },
        #   "daily":{
        #       "time":["2025-05-23"],"weather_code":[80]
        #   }
        # }
        weather_code = data["daily"]["weather_code"][forecast_days - 1]

        return WeatherData(
            code=weather_code,
            description=weather_data[weather_code]["description"],
            category=weather_data[weather_code]["category"],
        )

    def process(self, item: dict[str, Any]) -> WeatherData:
        """Process dictionary input to get weather forecast.

        If the dictionary has latitude and longitude, it retrieves the weather forecast by coordinates.
        Otherwise, if it has a city name, it retrieves the weather forecast by city.

        Args:
            item (dict[str, Any]): Dictionary with either "latitude"/"longitude" keys or "city" key.

        Returns:
            WeatherData: Weather forecast with code, description, and category.

        Raises:
            ValueError: If the input is not a dictionary or has neither "latitude"/"longitude" nor "city".
            RuntimeError: If city lookup or weather data retrieval fails.
        """
        match item:
            case {"latitude": latitude, "longitude": longitude}:
                return self.get_forecast_by_coords(latitude, longitude)
            case {"city": city}:
                return self.get_forecast_by_city(city)
            case dict():
                raise ValueError("Input dict must contain 'latitude' and 'longitude', or 'city'")
            case _:
                raise ValueError("Input must be a dict")


class CityLookupError(Exception):
    """Exception raised when the city lookup (geocoding) fails."""

    pass


class WeatherForecastLookupError(Exception):
    """Exception raised when the weather forecast lookup fails."""

    pass
