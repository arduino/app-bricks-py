# SPDX-FileCopyrightText: Copyright (C) 2025 ARDUINO SA <http://www.arduino.cc>
#
# SPDX-License-Identifier: MPL-2.0

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from arduino.app_utils import Logger

logger = Logger("HttpClient")


class HttpClient:
    """A simple HTTP client that uses requests.

    Session and urllib3's Retry utility to perform GET and POST requests with built-in retry logic.
    """

    def __init__(
        self,
        total_retries: int = 5,
        backoff_factor: int = 1,
        status_forcelist: frozenset = (411, 500, 502, 503, 504),
        allowed_methods: frozenset = frozenset(["GET", "POST", "PUT", "DELETE"]),
    ):
        self.__total_retries = total_retries
        self.__backoff_factor = backoff_factor
        self.__status_forcelist = status_forcelist

        # Configure the Retry strategy
        retries_strategy = Retry(
            total=self.__total_retries,
            read=self.__total_retries,
            connect=self.__total_retries,
            backoff_factor=self.__backoff_factor,
            status_forcelist=list(self.__status_forcelist),
            allowed_methods=allowed_methods,
        )

        adapter = HTTPAdapter(max_retries=retries_strategy)

        self.__http_session = requests.Session()

        self.__http_session.mount("http://", adapter)
        self.__http_session.mount("https://", adapter)

    def request_with_retry(
        self,
        url: str,
        method: str = "GET",
        data: dict | str = None,
        json: dict = None,
        headers: dict = None,
        timeout: int = 5,
    ):
        """Performs a GET or POST request to a given URL with a retry mechanism using requests.

        Session and urllib3's Retry utility for built-in exponential backoff.

        Args:
            url (str): The URL to make the request to.
            method (str): The HTTP method to use (like 'GET' or 'POST'). Case-insensitive.
            data (dict or str): (Optional) Dictionary, bytes, or file-like object to send in the body of the request.
                                Typically used for 'application/x-www-form-urlencoded' or raw data.
            json (dict): (Optional) A JSON serializable dictionary to send in the body of the request.
                        Automatically sets 'Content-Type: application/json'.
            headers (dict): (Optional) Dictionary of HTTP Headers to send with the Request.
            timeout (int or tuple): How many seconds to wait for the server to send data before giving up.
                                    Can be a float, or a (connect timeout, read timeout) tuple.

        Returns:
            requests.Response or None: The response object if successful, None otherwise.
        """
        if url is None:
            logger.error("Invalid URL provided. URL must be a non-empty string.")
            return None

        method = method.upper()  # Ensure method is uppercase for consistency
        try:
            logger.debug(
                f"Attempting to make {method} request to {url} (max retries: {self.__total_retries}, backoff factor: {self.__backoff_factor})..."
            )

            # Make the request based on the method
            response = self.__http_session.request(method, url, data=data, json=json, headers=headers, timeout=timeout)
            response.raise_for_status()  # Raise an HTTPError for bad responses (4xx or 5xx)
            return response
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed after all retries: {e}")
            return None

    def close(self):
        """Close the HTTP session."""
        self.__http_session.close()
        logger.debug("HTTP session closed.")
