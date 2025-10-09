# SPDX-FileCopyrightText: Copyright (C) 2025 ARDUINO SA <http://www.arduino.cc>
#
# SPDX-License-Identifier: MPL-2.0

# EXAMPLE_NAME = Expose functions to the microcontroller
import time
from arduino.app_utils import *


@provide()
def add_numbers(a: int, b: int):
    """Performs a sum operation named 'add_numbers'."""
    print(f"'add_numbers' called with: {a}, {b}")
    return a + b


@provide("math.subtract")  # Uses 'math.subtract' as the RPC method name
def subtract_numbers(a: int, b: int):
    """Performs a subtraction operation named 'math.subtract'."""
    print(f"'math.subtract' called with: {a}, {b}")
    return a - b


@provide()
def print_result(message: str):
    """Prints a text message."""
    print(f"'print_result' called with: {message}")


while True:
    time.sleep(10)
