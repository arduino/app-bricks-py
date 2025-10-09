# SPDX-FileCopyrightText: Copyright (C) 2025 ARDUINO SA <http://www.arduino.cc>
#
# SPDX-License-Identifier: MPL-2.0

import requests
import json
from dataclasses import dataclass
import importlib.resources

from arduino.app_utils import brick

city_api_url = "https://geocoding-api.open-meteo.com/v1/search"
forecast_api_url = "https://api.open-meteo.com/v1/forecast"


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
with importlib.resources.open_text(__package__, "weather_data.json") as file:
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
            response = requests.get(city_api_url, params={"name": city})
        except:
            raise RuntimeError("Failed to look city up")

        data = response.json()
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
            response = requests.get(forecast_api_url, params=params)
        except:
            raise RuntimeError("Failed to get weather data")

        data = response.json()
        if response.status_code != 200:
            raise RuntimeError(f"Failed to get weather data: {data.get('reason', 'Unknown error')}")

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

    def process(self, item):
        """Process dictionary input to get weather forecast.

        This method checks if the item is a dictionary with latitude and longitude or city name.
        If it is a dictionary with latitude and longitude, it retrieves the weather forecast by coordinates.
        If it is a dictionary with city name, it retrieves the weather forecast by city.

        Args:
            item (dict): Dictionary with either "city" key or "latitude"/"longitude" keys.

        Returns:
            WeatherData | dict: WeatherData object if valid input provided, empty dict if input format is invalid.

        Raises:
            CityLookupError: If the city is not found.
            WeatherForecastLookupError: If the weather forecast cannot be retrieved.
        """
        output = {}
        if isinstance(item, dict):
            if "latitude" in item and "longitude" in item:
                return self.get_forecast_by_coords(item["latitude"], item["longitude"])
            elif "city" in item:
                return self.get_forecast_by_city(item["city"])

        return output


class CityLookupError(Exception):
    """Exception raised when the city lookup (geocoding) fails."""

    pass


class WeatherForecastLookupError(Exception):
    """Exception raised when the weather forecast lookup fails."""

    pass
