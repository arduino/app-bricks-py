# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

import asyncio
import time


class AsyncRateLimiter:
    """Helper class for async rate limiting."""

    def __init__(self, calls_per_second: int):
        if calls_per_second <= 0:
            raise ValueError("calls_per_second must be greater than 0")
        self._min_interval = 1.0 / calls_per_second
        self._lock = asyncio.Lock()
        self._last_call_time = 0.0

    async def wait(self):
        """Wait if necessary to maintain the desired rate."""
        async with self._lock:
            current_time = time.monotonic()
            elapsed = current_time - self._last_call_time
            wait_time = self._min_interval - elapsed
            if wait_time > 0:
                await asyncio.sleep(wait_time)
            self._last_call_time = time.monotonic()
