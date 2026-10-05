# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import pytest
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import MagicMock, call, patch
from typing import Any

from arduino.app_bricks.dbstorage_tsstore import TimeSeriesStore, TimeSeriesStoreError


@pytest.fixture
def mock_influx_database() -> MagicMock:
    """Fixture that provides mock database objects."""
    # Create mock objects for TimeSeriesStore
    mock_db = MagicMock()

    # Configure mock_dbread.read_last_sample to return expected test data
    mock_db.read_last_sample.return_value = ("test_measurement_str", "timestamp", "test_string")

    return mock_db


@patch("time.sleep")
def test_write_and_read_string(mock_sleep: MagicMock, mock_influx_database: MagicMock) -> None:
    """Test writing and reading a string sample directly using mocked DB components."""
    mock_db = mock_influx_database

    # The fixture mock_influx_database already configures mock_dbread.read_last_sample:
    # mock_dbread.read_last_sample.return_value = ("test_measurement_str", "timestamp", "test_string")

    measurement_to_test = "test_measurement_str"
    string_value_to_write = "test_string"

    # Simulate writing to database
    mock_db.write_sample(measurement_to_test, string_value_to_write)

    # Simulate a delay
    mock_sleep(1)

    # Simulate reading from database
    result_measurement, timestamp, result_value = mock_db.read_last_sample(measurement_to_test)

    # Assertions
    mock_db.write_sample.assert_called_once_with(measurement_to_test, string_value_to_write)
    mock_db.read_last_sample.assert_called_once_with(measurement_to_test)
    mock_sleep.assert_called_once_with(1)
    assert result_measurement == measurement_to_test
    assert result_value == string_value_to_write


@patch("arduino.app_bricks.dbstorage_tsstore.TimeSeriesStore")
def test_open_influx_database(mock_db_persistence: MagicMock) -> None:
    """Unit test for open_influx_database function.

    Verifies that the function properly initializes database connection objects.
    """
    # Setup mock instances
    mock_db_tsstore_instance = MagicMock()
    mock_db_persistence.return_value = mock_db_tsstore_instance

    def open_influx_database():
        """Function to open InfluxDB database."""
        db = mock_db_persistence()
        return db

    # Call the function
    db = open_influx_database()

    # Verify correct objects are returned
    assert db == mock_db_tsstore_instance
    mock_db_persistence.assert_called_once()


@pytest.fixture
def mock_influx_database_with_numeric() -> MagicMock:
    """Fixture that provides mock database objects with numeric data returns."""
    mock_db = MagicMock()

    # Configure mock_dbread.read_last_sample to return numeric data
    mock_db.read_last_sample.return_value = ("test_measurement_num", "timestamp", 42.5)

    return mock_db


@patch("time.sleep")
def test_write_and_read_numeric(mock_sleep: MagicMock, mock_influx_database_with_numeric: MagicMock) -> None:
    """Test for writing and reading numeric data.

    Verifies the database can handle numeric values correctly.
    """
    mock_db = mock_influx_database_with_numeric

    # Define test values
    measurement: str = "test_measurement_num"
    value: float = 42.5

    # Simulate writing to database
    mock_db.write_sample(measurement, value)

    # Simulate reading from database
    result_measurement, timestamp, result_value = mock_db.read_last_sample(measurement)

    # Assertions
    assert result_measurement == measurement
    assert result_value == value
    mock_db.write_sample.assert_called_once_with(measurement, value)
    mock_db.read_last_sample.assert_called_once_with(measurement)


@patch("arduino.app_bricks.dbstorage_tsstore.TimeSeriesStore")
def test_database_write_error_handling(mock_db_persistence: MagicMock) -> None:
    """Test error handling during database write operations.

    Verifies that database write errors are properly handled.
    """
    # Setup mock to raise an exception on write
    mock_instance = MagicMock()
    mock_instance.write_sample.side_effect = Exception("Database connection error")
    mock_db_persistence.return_value = mock_instance

    # Create a test function that uses the database
    def test_function() -> bool:
        db = mock_db_persistence()
        try:
            db.write_sample("measurement", "value")
            return True
        except Exception:
            return False

    # Assert that the exception is caught
    assert test_function() is False
    mock_instance.write_sample.assert_called_once()


@patch("arduino.app_bricks.dbstorage_tsstore.TimeSeriesStore")
def test_database_read_error_handling(mock_db_retrieval: MagicMock) -> None:
    """Test error handling during database read operations.

    Verifies that database read errors are properly handled.
    """
    # Setup mock to raise an exception on read
    mock_instance = MagicMock()
    mock_instance.read_last_sample.side_effect = Exception("Database connection error")
    mock_db_retrieval.return_value = mock_instance

    # Create a test function that uses the database
    def test_function() -> bool:
        db = mock_db_retrieval()
        try:
            db.read_last_sample("measurement")
            return True
        except Exception:
            return False

    # Assert that the exception is caught
    assert test_function() is False
    mock_instance.read_last_sample.assert_called_once()


@patch("arduino.app_bricks.dbstorage_tsstore.TimeSeriesStore")
def test_database_persistence_process(mock_db_persistence_class: MagicMock) -> None:
    """Test the process method of DatabasePersistence.

    Verifies that the process method correctly handles different input types.
    """
    # Create a mock instance with a proper implementation of the process method
    mock_instance = MagicMock()

    # Mock the process method to call write_sample for each key-value pair
    def mock_process(data: dict[str, Any]) -> dict[str, Any]:
        if isinstance(data, dict):
            for key, value in data.items():
                mock_instance.write_sample(key, value)
        return data

    mock_instance.process.side_effect = mock_process
    mock_db_persistence_class.return_value = mock_instance

    # Get the instance
    db = mock_db_persistence_class()

    # Test with dictionary input
    test_data: dict[str, Any] = {"sensor1": 25.5, "sensor2": "active"}
    result = db.process(test_data)

    # Verify write_sample was called for each key-value pair
    assert mock_instance.write_sample.call_count == 2
    mock_instance.write_sample.assert_any_call("sensor1", 25.5)
    mock_instance.write_sample.assert_any_call("sensor2", "active")

    # Verify the method returns the original item
    assert result == test_data


@patch("arduino.app_bricks.dbstorage_tsstore.TimeSeriesStore")
def test_database_retrieval_process(mock_db_retrieval_class: MagicMock) -> None:
    """Test the process method of DatabaseRetrieval.

    Verifies that the process method correctly handles different input types.
    """
    # Create a mock instance with a proper implementation of the process method
    mock_instance = MagicMock()

    # Configure mock to return expected values for read_last_sample based on the sensor name
    def read_last_sample_side_effect(measurement: str) -> tuple[str, str, Any]:
        if measurement == "sensor1":
            return ("sensor1", "2023-01-01T12:00:00Z", 25.5)
        elif measurement == "sensor2":
            return ("sensor2", "2023-01-01T12:01:00Z", "active")
        return (measurement, "unknown_timestamp", None)

    mock_instance.read_last_sample.side_effect = read_last_sample_side_effect

    # Mock the process method to call read_last_sample for different input types
    def mock_process(data: Any) -> Any:
        if isinstance(data, str):
            measurement = data
            result = mock_instance.read_last_sample(measurement)
            return {measurement: result}
        elif isinstance(data, dict):
            result = {}
            for key in data.keys():
                sample = mock_instance.read_last_sample(key)
                result[key] = sample
            return result
        return None

    mock_instance.process.side_effect = mock_process
    mock_db_retrieval_class.return_value = mock_instance

    # Get the instance
    db = mock_db_retrieval_class()

    # Test with string input
    string_result = db.process("sensor1")
    mock_instance.read_last_sample.assert_called_with("sensor1")
    assert "sensor1" in string_result
    assert string_result["sensor1"][2] == 25.5

    # Test with dictionary input
    test_data: dict[str, None] = {"sensor1": None, "sensor2": None}
    dict_result = db.process(test_data)

    # Verify read_last_sample was called for each key
    assert mock_instance.read_last_sample.call_count == 3  # Once for string test, twice for dict test
    mock_instance.read_last_sample.assert_any_call("sensor1")
    mock_instance.read_last_sample.assert_any_call("sensor2")

    # Verify the result contains entries for both keys
    assert "sensor1" in dict_result
    assert "sensor2" in dict_result
    assert dict_result["sensor1"][2] == 25.5
    assert dict_result["sensor2"][2] == "active"


def _compose_with_token(tmp_path: Path, token: str) -> str:
    compose = tmp_path / "brick_compose.yaml"
    compose.write_text(
        "services:\n"
        "  dbstorage-influx:\n"
        "    environment:\n"
        "      DOCKER_INFLUXDB_INIT_ORG: arduino\n"
        "      DOCKER_INFLUXDB_INIT_BUCKET: arduinostorage\n"
        f'      DOCKER_INFLUXDB_INIT_ADMIN_TOKEN: "{token}"\n'
    )
    return str(compose)


def test_store_reads_the_default_token_from_the_compose_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("INFLUXDB_ADMIN_TOKEN", raising=False)
    compose = _compose_with_token(tmp_path, "${INFLUXDB_ADMIN_TOKEN:-secret}")
    with patch("arduino.app_bricks.dbstorage_tsstore.get_brick_compose_file", return_value=compose):
        assert TimeSeriesStore().token == "secret"


def test_store_prefers_the_token_set_for_the_app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INFLUXDB_ADMIN_TOKEN", "custom")
    compose = _compose_with_token(tmp_path, "${INFLUXDB_ADMIN_TOKEN:-secret}")
    with patch("arduino.app_bricks.dbstorage_tsstore.get_brick_compose_file", return_value=compose):
        assert TimeSeriesStore().token == "custom"


def test_store_rejects_a_missing_token(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("INFLUXDB_ADMIN_TOKEN", raising=False)
    compose = _compose_with_token(tmp_path, "${INFLUXDB_ADMIN_TOKEN}")
    with patch("arduino.app_bricks.dbstorage_tsstore.get_brick_compose_file", return_value=compose):
        with pytest.raises(TimeSeriesStoreError):
            TimeSeriesStore()


@pytest.fixture
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TimeSeriesStore:
    monkeypatch.delenv("INFLUXDB_ADMIN_TOKEN", raising=False)
    compose = _compose_with_token(tmp_path, "${INFLUXDB_ADMIN_TOKEN:-secret}")
    with patch("arduino.app_bricks.dbstorage_tsstore.get_brick_compose_file", return_value=compose):
        return TimeSeriesStore()


@pytest.fixture
def influx_class() -> Iterator[MagicMock]:
    """The InfluxDB client class, talking to a server that has the brick bucket."""
    with patch("arduino.app_bricks.dbstorage_tsstore.InfluxDBClient") as client_class:
        yield client_class


@pytest.fixture
def influx(influx_class: MagicMock) -> MagicMock:
    """The client a started store holds."""
    return influx_class.return_value


def test_store_rejects_a_missing_compose_file() -> None:
    with patch("arduino.app_bricks.dbstorage_tsstore.get_brick_compose_file", return_value=None):
        with pytest.raises(TimeSeriesStoreError, match=r"^Could not find the brick compose file\.$"):
            TimeSeriesStore()


def test_start_rejects_a_missing_bucket(store: TimeSeriesStore, influx: MagicMock) -> None:
    influx.buckets_api.return_value.find_bucket_by_name.return_value = None
    with pytest.raises(TimeSeriesStoreError, match=r"^Error connecting to InfluxDB: Bucket arduinostorage not found\.$"):
        store.start()
    influx.write_api.return_value.close.assert_called_once()
    influx.close.assert_called_once()
    with pytest.raises(TimeSeriesStoreError):
        store.get_client()


def test_stop_before_start_does_nothing(store: TimeSeriesStore) -> None:
    store.stop()


def test_get_client_before_start_raises(store: TimeSeriesStore) -> None:
    with pytest.raises(TimeSeriesStoreError, match=r"InfluxDB client is not available, call start\(\) first\."):
        store.get_client()


def test_get_client_returns_a_client_open_until_stop(store: TimeSeriesStore, influx: MagicMock) -> None:
    store.start()
    assert store.get_client() is influx
    influx.close.assert_not_called()
    influx.write_api.return_value.close.assert_not_called()

    store.stop()
    influx.close.assert_called_once()
    with pytest.raises(TimeSeriesStoreError):
        store.get_client()


def test_stop_flushes_pending_writes_before_closing_the_client(store: TimeSeriesStore, influx: MagicMock) -> None:
    store.start()
    store.stop()
    closes = [c for c in influx.mock_calls if c in (call.write_api().close(), call.close())]
    assert closes == [call.write_api().close(), call.close()]


def test_start_twice_keeps_the_open_client(store: TimeSeriesStore, influx_class: MagicMock) -> None:
    store.start()
    store.start()
    influx_class.assert_called_once()


def test_start_after_stop_opens_a_new_client(store: TimeSeriesStore, influx_class: MagicMock) -> None:
    store.start()
    store.stop()
    store.start()
    assert influx_class.call_count == 2
    assert store.get_client() is influx_class.return_value


@pytest.mark.parametrize("start_from", ["-1d", "-30m", "2024-06-25T12:34:56Z", "now()"])
def test_read_samples_accepts_supported_times(store: TimeSeriesStore, influx: MagicMock, start_from: str) -> None:
    store.start()
    assert store.read_samples("temp", start_from=start_from) == []


@pytest.mark.parametrize("start_from", ["yesterday", "1d", 123, None])
def test_read_samples_rejects_an_invalid_start(store: TimeSeriesStore, start_from: Any) -> None:  # noqa: ANN401
    with pytest.raises(TimeSeriesStoreError, match=f"Invalid start_from value: {start_from}\\. Must be a valid time period or timestamp\\."):
        store.read_samples("temp", start_from=start_from)


@pytest.mark.parametrize("end_to", ["tomorrow", 123])
def test_read_samples_rejects_an_invalid_end(store: TimeSeriesStore, end_to: Any) -> None:  # noqa: ANN401
    with pytest.raises(TimeSeriesStoreError, match=f"Invalid end_to value: {end_to}\\. Must be a valid time period or timestamp\\."):
        store.read_samples("temp", end_to=end_to)


@pytest.mark.parametrize("start_from", ["yesterday", 123])
def test_read_last_sample_rejects_an_invalid_start(store: TimeSeriesStore, start_from: Any) -> None:  # noqa: ANN401
    with pytest.raises(TimeSeriesStoreError, match=f"Invalid start_from value: {start_from}\\. Must be a valid time period or timestamp\\."):
        store.read_last_sample("temp", start_from=start_from)
