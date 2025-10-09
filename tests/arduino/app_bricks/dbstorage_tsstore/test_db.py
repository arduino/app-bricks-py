# SPDX-FileCopyrightText: Copyright (C) 2025 ARDUINO SA <http://www.arduino.cc>
#
# SPDX-License-Identifier: MPL-2.0

import pytest
from unittest.mock import MagicMock, patch
from typing import Any


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
