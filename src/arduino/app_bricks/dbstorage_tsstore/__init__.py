# SPDX-FileCopyrightText: Copyright (C) 2025 ARDUINO SA <http://www.arduino.cc>
#
# SPDX-License-Identifier: MPL-2.0

from arduino.app_utils import Logger
from typing import Any
import yaml
import time

from influxdb_client import InfluxDBClient, Point, WritePrecision, BucketRetentionRules

from arduino.app_internal.core import get_brick_compose_file, parse_docker_compose_variable
from arduino.app_utils import brick, Logger

logger = Logger("TimeSeriesStore")

base_influx_host = "dbstorage-influx"
base_influx_port = 8086


class TimeSeriesStoreError(Exception):
    """Custom exception raised for TimeSeriesStore database operation errors."""

    pass


def _convert_days_to_seconds(days: int) -> int:
    """Convert days to seconds."""
    return days * 24 * 60 * 60


class _InfluxDBHandler:
    """Base class for handling InfluxDB connections and operations.

    This class initializes the InfluxDB client and provides methods for writing and querying data.
    It automatically loads configuration from Docker Compose infrastructure and manages
    database authentication, bucket settings, and connection parameters.

    Note:
        It is intended to be subclassed for specific database operations. Use TimeSeriesStore for time series operations.
    """

    def __init__(self, host: str = base_influx_host, port: int = base_influx_port, retention_days: int = 7):
        """Initialize the InfluxDB client with the provided host and port.

        Args:
            host (str, optional): The hostname of the InfluxDB server. Defaults to "dbstorage-influx".
            port (int, optional): The port number of the InfluxDB server. Defaults to 8086.
            retention_days (int, optional): Number of days to retain data in the InfluxDB bucket. Defaults to 7.
        """
        self.name = "dbstorage"
        self.host = host
        self.port = port
        infra = self.load_default_infra()
        env_dict = infra["services"]["dbstorage-influx"]["environment"]
        self.url = f"http://{self.host}:{self.port}"
        self.token = parse_docker_compose_variable(env_dict["DOCKER_INFLUXDB_INIT_ADMIN_TOKEN"])[0][1]
        self.org = env_dict["DOCKER_INFLUXDB_INIT_ORG"]
        self.bucket = env_dict["DOCKER_INFLUXDB_INIT_BUCKET"]
        self.client: InfluxDBClient = None
        self.retention_days = retention_days

    def start(self):
        """Establish a connection to the InfluxDB server.

        This method creates the InfluxDB client connection, initializes write and query APIs,
        and configures the data retention policy for the bucket. The connection is established
        with the parameters specified during initialization.

        Raises:
            TimeSeriesStoreError: If there is an error connecting to the InfluxDB server.
        """
        try:
            with InfluxDBClient(url=self.url, token=self.token, org=self.org) as client:
                self.client = client
                self.write_api = client.write_api(write_precision=WritePrecision.MS)
                self.query_api = client.query_api()
                # Update data retention of the bucket
                bucket = self.client.buckets_api().find_bucket_by_name(self.bucket)
                bucket.retention_rules = [BucketRetentionRules(type="expire", every_seconds=_convert_days_to_seconds(self.retention_days))]
                self.client.buckets_api().update_bucket(bucket)
            logger.info(f"Connected to InfluxDB: {self.url}")
        except Exception as e:
            raise TimeSeriesStoreError(f"Error connecting to InfluxDB: {e}") from e

    def stop(self):
        """Close the InfluxDB database connection.

        Properly closes the client connection and releases associated resources.
        Should be called when finished with the time series store to ensure
        proper cleanup.
        """
        self.client.close()

    def load_default_infra(self):
        """Load the default InfluxDB compose file for the brick.

        This method looks for a YAML file named 'module_compose.yaml' in the current module's directory.
        If the file is found, it loads the content and returns it as a dictionary.
        If the file is not found, it logs an error message.

        Returns:
            dict: The content of the compose file as a dictionary.
        """
        pathfile = get_brick_compose_file(self.__class__)
        if pathfile:
            with open(pathfile) as f:
                compose_content = yaml.safe_load(f)
                logger.debug(f"Loading compose file: {compose_content}")
                return compose_content
        else:
            logger.error("Error: Could not find module_compose.yaml")
            return None

    def get_client(self) -> InfluxDBClient:
        """Returns the InfluxDB client instance."""
        return self.client


def _is_valid_time(value: str) -> bool:
    import re
    from datetime import datetime

    try:
        if not isinstance(value, str):
            return False
        # Check for relative period (e.g., -1d, -2h, -30m)
        if re.fullmatch(r"-\d+[smhdw]", value):
            return True
        # Check for RFC3339 timestamp
        try:
            # Accepts e.g. 2024-06-25T12:34:56Z
            datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
            return True
        except ValueError:
            pass
        if value == "now()":
            return True
        return False
    except Exception as e:
        raise e


@brick
class TimeSeriesStore(_InfluxDBHandler):
    """Time series database handler for storing and retrieving data using InfluxDB.

    This class extends the base InfluxDB handler and provides methods for writing samples to the database.
    It allows writing and reading individual measurements with their values and timestamps.
    """

    def __init__(self, host: str = base_influx_host, port: int = base_influx_port, retention_days: int = 7):
        """Initialize the InfluxDB persistence handler.

        Args:
            host (str, optional): The hostname of the InfluxDB server.
                Defaults to "dbstorage-influx".
            port (int, optional): The port number of the InfluxDB server.
                Defaults to 8086.
            retention_days (int, optional): The number of days to retain data in the
                InfluxDB bucket. Defaults to 7.
        """
        super().__init__(host, port, retention_days)

    def write_sample(self, measure: str, value: Any, ts: int = 0, measurement_name: str = "arduino"):
        """Write a time series sample to the InfluxDB database.

        Stores a single data point with the specified measurement field, value, and timestamp.
        If no timestamp is provided, the current time is used automatically.

        Args:
            measure (str): The name of the measurement field (e.g., "temperature", "humidity").
                This acts as the column name for the data point.
            value (Any): The numeric or string value to store. Supports int, float, str, and bool types.
            ts (int, optional): The timestamp in milliseconds since epoch.
                Defaults to 0 (current time).
            measurement_name (str, optional): The measurement container name that groups
                related fields together. Defaults to "arduino".

        Raises:
            TimeSeriesStoreError: If there is an error writing the sample to the InfluxDB database,
                such as connection failures or invalid data types.
        """
        try:
            if ts <= 0:
                ts = int(time.time_ns() / 1000000)
            point = Point(measurement_name).field(measure, value).time(ts, WritePrecision.MS)

            self.write_api.write(bucket=self.bucket, record=point)

        except Exception as e:
            raise TimeSeriesStoreError(f"Error writing sample to InfluxDB: {e}") from e

    def read_last_sample(self, measure: str, measurement_name: str = "arduino", start_from: str = "-1d") -> tuple | None:
        """Read the last sample of a specific measurement from the InfluxDB database.

        Retrieves the latest data point for the specified measurement field within
        the given time range.

        Args:
            measure (str): The name of the measurement field to query (e.g., "temperature").
            measurement_name (str, optional): The measurement container name to search within.
                Defaults to "arduino".
            start_from (str, optional): The time range to search within. Supports relative
                periods like "-1d" (1 day), "-2h" (2 hours), "-30m" (30 minutes) or
                RFC3339 timestamps like "2024-01-01T00:00:00Z". Defaults to "-1d".

        Returns:
            tuple | None: A tuple containing (field_name, timestamp_iso, value) where:
                - field_name (str): The measurement field name
                - timestamp_iso (str): ISO format timestamp string
                - value (Any): The stored value
                Returns None if no data is found in the specified time range.

        Raises:
            TimeSeriesStoreError: If the start_from value is invalid or if there is an error querying the InfluxDB database.
        """
        try:
            if not _is_valid_time(start_from):
                raise TimeSeriesStoreError(f"Invalid start_from value: {start_from}. Must be a valid time period or timestamp.")

            query = f'''
            from(bucket: "{self.bucket}")
            |> range(start: {start_from})
            |> filter(fn: (r) => r["_measurement"] == "{measurement_name}")
            |> filter(fn: (r) => r["_field"] == "{measure}")
            |> last()
            '''

            result = self.query_api.query(org=self.org, query=query)

            if result:
                for table in result:
                    for record in table.records:
                        return measure, record.get_time().isoformat(), record.get_value()
            else:
                return None

        except Exception as e:
            raise TimeSeriesStoreError(f"Error reading last sample from InfluxDB: {e}") from e

    def read_samples(
        self,
        measure: str,
        measurement_name: str = "arduino",
        start_from: str = "-1d",
        end_to: str = None,
        aggr_window: str = None,
        aggr_func: str = None,
        limit: int = 1000,
        order: str = "asc",
    ) -> list:
        """Read all samples of a specific measurement from the InfluxDB database.

        Retrieves multiple data points for the specified measurement field with support
        for time range filtering, data aggregation, and result ordering.

        Args:
            measure (str): The name of the measurement field to query (e.g., "temperature").
            measurement_name (str, optional): The measurement container name to search within.
                Defaults to "arduino".
            start_from (str, optional): The start time for the query range. Supports relative
                periods ("-7d", "-1h") or RFC3339 timestamps. Defaults to "-1d".
            end_to (str, optional): The end time for the query range. Supports same formats
                as start_from or "now()". Defaults to None (current time).
            aggr_window (str, optional): Time window for data aggregation (e.g., "1h" for hourly,
                "30m" for 30-minute intervals). Must be used with aggr_func. Defaults to None.
            aggr_func (str, optional): Aggregation function to apply within each window.
                Supported values: "mean", "max", "min", "sum". Must be used with aggr_window.
                Defaults to None.
            limit (int, optional): Maximum number of samples to return. Must be positive.
                Defaults to 1000.
            order (str, optional): Sort order for results by timestamp. Must be "asc"
                (ascending, oldest first) or "desc" (descending, newest first). Defaults to "asc".

        Returns:
            list: List of tuples, each containing (field_name, timestamp_iso, value) where:
                - field_name (str): The measurement field name
                - timestamp_iso (str): ISO format timestamp string
                - value (Any): The stored or aggregated value
                Empty list if no data found in the specified range.

        Raises:
            TimeSeriesStoreError: If any parameter is invalid, such as:
                - Invalid time format in start_from or end_to
                - Invalid order value (not "asc" or "desc")
                - Invalid limit value (not positive integer)
                - Invalid aggregation function
                - Mismatched aggr_window and aggr_func (one specified without the other)
                - Database query errors
        """
        try:
            if not _is_valid_time(start_from):
                raise TimeSeriesStoreError(f"Invalid start_from value: {start_from}. Must be a valid time period or timestamp.")
            if end_to is not None and not _is_valid_time(end_to):
                raise TimeSeriesStoreError(f"Invalid end_to value: {end_to}. Must be a valid time period or timestamp.")
            if end_to is None:
                end_to = "now()"

            if order.lower() not in ["asc", "desc"]:
                raise TimeSeriesStoreError(f"Invalid order value: {order}. Must be 'asc' or 'desc'.")

            if limit <= 0:
                raise TimeSeriesStoreError(f"Invalid limit value: {limit}. Must be a positive integer.")

            if aggr_func not in [None, "mean", "max", "min", "sum"]:
                raise TimeSeriesStoreError(f"Invalid aggregation function: {aggr_func}. Must be one of 'mean', 'max', 'min' or 'sum'.")

            if aggr_window is None and aggr_func is not None:
                raise TimeSeriesStoreError("If aggr_func is specified, aggr_window must also be specified.")
            if aggr_window is not None and aggr_func is None:
                raise TimeSeriesStoreError("If aggr_window is specified, aggr_func must also be specified.")

            query = f'''
            from(bucket: "{self.bucket}")
            |> range(start: {start_from}, stop: {end_to})
            |> filter(fn: (r) => r["_measurement"] == "{measurement_name}")
            |> filter(fn: (r) => r["_field"] == "{measure}")
            '''

            if aggr_window and aggr_func:
                query += f"|> aggregateWindow(every: {aggr_window}, fn: {aggr_func})"

            query += f'|> sort(columns: ["_time"], desc: {"false" if order.lower() == "asc" else "true"})'
            query += f"|> limit(n: {limit})"

            result = self.query_api.query(org=self.org, query=query)

            samples = []
            if result:
                for table in result:
                    for record in table.records:
                        value = record.get_value()
                        if value is not None:
                            samples.append((measure, record.get_time().isoformat(), value))
            return samples

        except Exception as e:
            raise TimeSeriesStoreError(f"Error reading samples from InfluxDB: {e}") from e
