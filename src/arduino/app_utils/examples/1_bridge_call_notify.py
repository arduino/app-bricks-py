# SPDX-FileCopyrightText: Copyright (C) 2025 ARDUINO SA <http://www.arduino.cc>
#
# SPDX-License-Identifier: MPL-2.0

# EXAMPLE_NAME = Call functions exposed by the microcontroller
# EXAMPLE_REQUIRES = Ensure the MCU has registered the functions below
from arduino.app_utils import *


# The code that will actually run is on the MCU, not on the Linux/Python side,
# so we can omit the function bodies and leave the "..." (ellipsis) placeholder.


@call()
def add_numbers(num1: int, num2: int) -> int:
    """Calls 'add_numbers' on the MCU, waiting indefinitely for a response."""
    ...


@call("math.subtract", timeout=3)
def sub_numbers(num1: int, num2: int) -> int:
    """Calls 'math.subtract' on the MCU, waiting by default at most 3s for a response."""
    ...


@notify()
def print_result(message: str):
    """Calls 'print_result' on the MCU, without waiting for a response."""
    ...


try:
    add_result = add_numbers(1, 2)  # Wait indefinitely for response
    print(f"Result of add_numbers(1, 2): {add_result}")
    print_result(f"The result is: {add_result}")

    add_result = add_numbers(3, 4, timeout=1)  # Wait at most 1s for response
    print(f"Result of add_numbers(3, 4, timeout=1): {add_result}")
    print_result(f"The result is: {add_result}")

    sub_result = sub_numbers(2, 1)  # Wait at most 3s for response (default)
    print(f"Result of sub_numbers(2, 1): {sub_result}")
    print_result(f"The result is: {sub_result}")

    sub_result = sub_numbers(4, 3, timeout=1)
    print(f"Result of sub_numbers(4, 3, timeout=1): {sub_result}")
    print_result(f"The result is: {sub_result}")
except Exception as e:
    print(f"An error occurred: {e}")
