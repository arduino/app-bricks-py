# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

import threading
import numpy as np


class SlidingWindowBuffer:
    """A sliding window buffer for managing generic streaming data.
    This buffer allows a single producer to push data and a single consumer to pull that data in a sliding window
    manner. The data type (dtype) and item shape are inferred from the first array pushed to the buffer.

    By providing a slide_amount < window_size, the buffer implements a sliding window where older data is
    repeated but newer data (with slide_amount length) is always available.
    By providing a slide_amount == window_size, the buffer implements a tumbling window where older data
    is never repeated and only new data (with window_size length) is always available.
    """

    def __init__(self, window_size: int, slide_amount: int, capacity: int = None):
        """Initializes the sliding window buffer.

        Args:
            window_size (int): The size of the sliding window.
            slide_amount (int): The amount by which the window slides each time data is pulled.
            capacity (int, optional): The maximum number of items the buffer can hold. If None, defaults to 2 * window_size.
            dtype (np.dtype, optional): The data type of the buffer. Defaults to np.int16.

        Raises:
            ValueError: If window_size, slide_amount or capacity have incorrect values.
            TypeError: If dtype is not a valid NumPy dtype.
        """
        if window_size <= 0 or slide_amount <= 0:
            raise ValueError("window_size and slide_amount must be positive")
        if slide_amount > window_size:
            raise ValueError("slide_amount cannot be greater than window_size")

        self.window_size = int(window_size)
        self.slide_amount = int(slide_amount)
        self.capacity = capacity or 2 * int(window_size)
        self.capacity = int(self.capacity)
        if self.capacity < self.window_size + self.slide_amount:
            raise ValueError("Capacity is too small for the given window_size and slide_amount.")

        self._buffer: np.ndarray = None
        self._dtype: np.dtype = None
        self._condition = threading.Condition()

        self._write_index: int = 0
        self._read_index: int = 0
        self._data_count: int = 0
        self._new_data_count: int = 0

    def push(self, data: np.ndarray) -> bool:
        """Attempts to add a NumPy array of data to the buffer.

        Args:
            data (np.ndarray): The array of data to push. Its dtype must match the buffer's declared dtype.

        Returns:
            bool: True if the data was successfully pushed, False if it would overflow the buffer.

        Raises:
            TypeError: If data_array has wrong type with respect to the buffer's declared dtype.
        """
        if not isinstance(data, np.ndarray):
            raise TypeError(f"Input data must be a np.ndarray, not {type(data)}.")

        num_items = len(data)
        if num_items == 0:
            return True

        with self._condition:
            if self._buffer is None:
                # Lazily initialize buffer with item shape and dtype from data
                self._dtype = data.dtype
                item_shape = data.shape[1:]
                buffer_shape = (self.capacity,) + item_shape
                self._buffer = np.empty(buffer_shape, dtype=self._dtype)
            elif data.dtype != self._dtype:
                raise TypeError(f"Inconsistent item type: buffer has item type {self._dtype}, not {data.dtype}.")
            elif data.shape[1:] != self._buffer.shape[1:]:
                raise ValueError(f"Inconsistent item shape: buffer has item shape {self._buffer.shape[1:]}, not {data.shape[1:]}")
            if self._data_count + num_items > self.capacity:
                # Buffer overflow!
                return False

            # Write data and handle wrap-arounds
            end_index = self._write_index + num_items
            if end_index <= self.capacity:
                # No wrap-around: write directly
                self._buffer[self._write_index : end_index] = data
            else:
                # Wrap-around: write in two parts, the first goes to the right, the second to the left of the buffer
                part1_len = self.capacity - self._write_index
                self._buffer[self._write_index :] = data[:part1_len]

                part2_len = num_items - part1_len
                self._buffer[:part2_len] = data[part1_len:]

            self._write_index = (self._write_index + num_items) % self.capacity
            self._data_count += num_items
            self._new_data_count += num_items

            # Notify the consumer that we have data available
            if self._new_data_count >= self.slide_amount:
                self._condition.notify()

        return True

    def pull(self, timeout: float = None) -> np.ndarray:
        """Retrieves a window of data as a NumPy array.
        Blocks until a window of window_size with at least slide_amount of
        new data is available or the provided timeout expires.

        Args:
            timeout (float, optional): The maximum time to wait for data. If None, waits indefinitely.

        Returns:
            np.ndarray: A NumPy array containing the data in the sliding window.
        """
        with self._condition:
            has_data = self._condition.wait_for(lambda: self.has_data(), timeout=timeout)
            if not has_data:
                # If wait_for timed out, return an empty array
                if self._buffer is not None:
                    empty_shape = (0,) + self._buffer.shape[1:]
                    return np.empty(empty_shape, dtype=self._dtype)
                else:
                    return np.array([], dtype=self._dtype)  # Return a simple 1D empty array.

            start = self._read_index
            end = start + self.window_size

            if end <= self.capacity:
                # No wrap-around: return a direct view of the data. O(1) operation.
                window = self._buffer[start:end]
            else:
                # Wraps around: concatenate two views. Creates a copy.
                end_wrapped = end % self.capacity
                window = np.concatenate((self._buffer[start:], self._buffer[:end_wrapped]))

            self._read_index = (self._read_index + self.slide_amount) % self.capacity
            self._data_count -= self.slide_amount
            self._new_data_count -= self.slide_amount

            return window

    def flush(self) -> None:
        """Clears the buffer to its initial empty state and notifies any waiting threads."""
        with self._condition:
            self._write_index = 0
            self._read_index = 0
            self._data_count = 0
            self._new_data_count = 0
            # Unblock any waiting threads that the state has changed
            self._condition.notify_all()

    def has_data(self) -> bool:
        """Checks if there is a full window of data ready to be pulled without blocking.

        Returns:
            bool: True if a call to pull() would not block waiting for data, False otherwise.
        """
        with self._condition:
            return self._buffer is not None and self._data_count >= self.window_size and self._new_data_count >= self.slide_amount
