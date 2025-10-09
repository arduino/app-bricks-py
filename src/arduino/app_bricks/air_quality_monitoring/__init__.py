# SPDX-FileCopyrightText: Copyright (C) 2025 ARDUINO SA <http://www.arduino.cc>
#
# SPDX-License-Identifier: MPL-2.0

import requests
from dataclasses import dataclass

from arduino.app_utils import brick


@dataclass(frozen=True)
class AQILevel:
    """Data class to represent AQI levels.

    Attributes:
        min_value (int): Minimum AQI value for the level.
        max_value (int): Maximum AQI value for the level.
        description (str): Description of the AQI level.
        color (str): Color associated with the AQI level in hex.
    """

    min_value: int
    max_value: int
    description: str
    color: str  # color in hex e.g. '#00e400'


# Define AQI levels
AQI_LEVELS: list[AQILevel] = [
    AQILevel(0, 50, "Good", "#00e400"),
    AQILevel(51, 100, "Moderate", "#ffff00"),
    AQILevel(101, 150, "Unhealthy for Sensitive Groups", "#ff7e00"),
    AQILevel(151, 200, "Unhealthy", "#ff0000"),
    AQILevel(201, 300, "Very Unhealthy", "#8f3f97"),
    AQILevel(301, 500, "Hazardous", "#7e0023"),
]


@dataclass(frozen=True)
class AirQualityData:
    """Data class to represent air quality data.

    Attributes:
        city (str): Name of the city.
        lat (float): Latitude of the city.
        lon (float): Longitude of the city.
        url (str): URL for more information about the air quality data.
        last_update (str): Last update timestamp of the air quality data.
        aqi (int): Air Quality Index value.
        dominantpol (str): Dominant pollutant in the air.
        iaqi (dict): Individual AQI values for various pollutants.
    """

    city: str
    lat: float
    lon: float
    url: str
    last_update: str
    aqi: int
    dominantpol: str
    iaqi: dict

    # Properties for easy access to IAQI values
    @property
    def pandas_dict(self) -> dict:
        """Return the data as a dictionary suitable for pandas DataFrame."""
        return {
            "city": [self.city],
            "lat": [self.lat],
            "lon": [self.lon],
            "url": [self.url],
            "last_update": [self.last_update],
            "aqi": [self.aqi],
            "dominantpol": [self.dominantpol],
            "iaqi": [self.iaqi],
        }


@brick
class AirQualityMonitoring:
    """Class to get air quality data from AQICN API."""

    def __init__(self, token: str):
        """Initialize the AirQualityMonitoring class with the API token.

        Args:
            token (str): API token for AQICN service.

        Raises:
            ValueError: If the token is not provided.
        """
        self.__token = token
        self.city_api_url = "https://api.waqi.info/feed/"
        self.geo_api_url = "https://api.waqi.info/feed/geo:{lat};{lng}/"
        self.ip_api_url = "https://api.waqi.info/feed/here/"

    def get_air_quality_by_city(self, city: str) -> AirQualityData:
        """Get air quality data by city name.

        Args:
            city (str): Name of the city.

        Returns:
            AirQualityData: Air quality assembled data.

        Raises:
            AirQualityLookupError: If the API request fails.
        """
        url = f"{self.city_api_url}{city}/"
        params = {"token": self.__token}
        response = requests.get(url, params=params)
        data = response.json()
        if response.status_code != 200 or data.get("status") != "ok":
            raise AirQualityLookupError.from_api_response(data)
        return self.assemble_data(data["data"])

    def get_air_quality_by_coords(self, latitude: float, longitude: float) -> AirQualityData:
        """Get air quality data by coordinates.

        Args:
            latitude (float): Latitude.
            longitude (float): Longitude.

        Returns:
            AirQualityData: Air quality assembled data.

        Raises:
            AirQualityLookupError: If the API request fails.
        """
        url = self.geo_api_url.format(lat=latitude, lng=longitude)
        params = {"token": self.__token}
        response = requests.get(url, params=params)
        data = response.json()
        if response.status_code != 200 or data.get("status") != "ok":
            raise AirQualityLookupError.from_api_response(data)
        return self.assemble_data(data["data"])

    def get_air_quality_by_ip(self) -> AirQualityData:
        """Get air quality data by IP address.

        Returns:
            AirQualityData: Air quality assembled data.

        Raises:
            AirQualityLookupError: If the API request fails.
        """
        url = self.ip_api_url
        params = {"token": self.__token}
        response = requests.get(url, params=params)
        data = response.json()
        if response.status_code != 200 or data.get("status") != "ok":
            raise AirQualityLookupError.from_api_response(data)
        return self.assemble_data(data["data"])

    def process(self, item: dict) -> dict:
        """Process the input dictionary to get air quality data.

        Args:
            item (dict): Input dictionary containing either 'city', 'latitude' and 'longitude', or 'ip'.

        Returns:
            dict: Air quality data.

        Raises:
            ValueError: If the input dictionary is not valid.
        """
        if not isinstance(item, dict):
            raise ValueError("Input must be a dict")
        # method selection
        if "city" in item:
            return self.get_air_quality_by_city(item["city"])
        elif "latitude" in item and "longitude" in item:
            return self.get_air_quality_by_coords(item["latitude"], item["longitude"])
        elif "ip" in item and item["ip"]:
            return self.get_air_quality_by_ip()
        else:
            raise ValueError("Input dict must contain 'city', 'latitude' and 'longitude', or 'ip': True")

    def assemble_data(self, data: dict) -> AirQualityData:
        """Create a payload for the air quality data.

        Args:
            data (dict): Air quality data.

        Returns:
            dict: Payload with relevant air quality information.
        """
        aqi_data = AirQualityData(
            city=data.get("city", {}).get("name", "N/A"),
            lat=data.get("city", {}).get("geo", [None, None])[0],
            lon=data.get("city", {}).get("geo", [None, None])[1],
            url=data.get("city", {}).get("url", "N/A"),
            last_update=data.get("time", {}).get("s", "N/A"),
            aqi=data.get("aqi", "N/A"),
            dominantpol=data.get("dominantpol", "N/A"),
            iaqi=data.get("iaqi", {}),
        )
        return aqi_data

    @staticmethod
    def map_aqi_level(aqi: int) -> AQILevel | None:
        """Returns AQILevel class matching provided AQI."""
        for level in AQI_LEVELS:
            if level.min_value <= aqi <= level.max_value:
                return level
        return None


class AirQualityLookupError(Exception):
    """Custom exception for air quality lookup errors."""

    def __init__(self, message: str, status: str = None):
        """Initialize the AirQualityLookupError with a message and status.

        Args:
            message (str): Error message.
            status (str): Status of the error, defaults to None.
        """
        super().__init__(message)
        self.status = status
        self.message = message

    @classmethod
    def from_api_response(cls, data: dict):
        """AirQualityLookupError error handling based on response provided by AQI API.

        Documented errors:
        - {"status": "error", "data": "Invalid key"}
        - {"status": "error", "data": "Unknown station"}
        - {"status": "error", "data": "Over quota"}
        - {"status": "error", "data": "Invalid query"}
        - {"status": "error", "data": "Too Many Requests"}
        - {"status": "error", "data": "IP not allowed"}
        - {"status": "error", "data": "Unknown error"}
        - {"status": "error", "data": {"message": "..."}}

        Args:
            data (dict): Response data from the AQI API.

        Returns:
            AirQualityLookupError: An instance of AirQualityLookupError with the error message and status.
        """
        status = data.get("status")
        # 'data' field can be a string or a dict with 'message' attribute
        if status != "error":
            raise ValueError("Status must be 'error'")
        if status is None:
            raise ValueError("Status cannot be None")

        raw_data = data.get("data")
        if isinstance(raw_data, dict) and "message" in raw_data:
            message = raw_data["message"]
        elif isinstance(raw_data, str):
            message = raw_data
        else:
            message = str(raw_data)
        return cls(message=message, status=status)
