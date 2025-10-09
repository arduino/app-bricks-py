# SPDX-FileCopyrightText: Copyright (C) 2025 ARDUINO SA <http://www.arduino.cc>
#
# SPDX-License-Identifier: MPL-2.0

import asyncio
import queue
import threading
from typing import Any, Optional
from .constants import _SHUTDOWN
from .limiter import AsyncRateLimiter
from arduino.app_utils import Logger

logger = Logger("pipeline.adapter")


# These classes are used to adapt the original bricks to the asyncio API. They are responsible for wrapping the original
# bricks and providing a consistent interface for the pipeline.


class AsyncBrickAdapter:
    """Base class for brick adapters, normalizing to an async API."""

    def __init__(self, original_brick: Any, rate_limit: Optional[int] = None):
        self.original_brick = original_brick
        self.rate_limit = rate_limit
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def set_loop(self, loop: asyncio.AbstractEventLoop):
        self._loop = loop

    async def start(self):
        """Normalized async start method."""
        logger.debug(f"Running start method for {type(self.original_brick).__name__}")
        return await self._execute_maybe_sync("start")

    async def stop(self):
        """Normalized async stop method."""
        logger.debug(f"Starting stop method for {type(self.original_brick).__name__}")
        return await self._execute_maybe_sync("stop")

    async def _execute_maybe_sync(self, method_name: str, *args: Any) -> Any:
        """Helper to execute a method, handling sync/async via executor."""
        if not self._loop:
            raise RuntimeError("Loop not set for adapter execution")
        if not hasattr(self.original_brick, method_name):
            # The start and stop methods are optional
            return

        method = getattr(self.original_brick, method_name)
        if not callable(method):
            raise TypeError(f"Method {method_name} is not callable on {type(self.original_brick).__name__}")

        if asyncio.iscoroutinefunction(method):
            return await method(*args)
        else:
            return await self._loop.run_in_executor(None, method, *args)


class AsyncSourceAdapter(AsyncBrickAdapter):
    """Adapter for async sources."""

    def __init__(self, original_brick: Any, rate_limit: Optional[int] = None):
        super().__init__(original_brick, rate_limit)

        self._produce_method = getattr(self.original_brick, "produce", None)
        if not callable(self._produce_method) or not asyncio.iscoroutinefunction(self._produce_method):
            raise TypeError(f"Method 'produce' not found or not async on {type(self.original_brick).__name__}")
        self._limiter = AsyncRateLimiter(rate_limit) if rate_limit else None

    async def produce(self, *args: Any) -> Any:
        """Normalized async produce, calls original async method."""
        if self._limiter:
            await self._limiter.wait()
        return await self._produce_method(*args)


class AsyncBlockingSourceAdapter(AsyncBrickAdapter):
    """Adapter for synchronous sources that might block indefinitely.
    Manages a daemon thread internally to avoid blocking the event loop.
    """

    def __init__(self, original_brick: Any, rate_limit: Optional[int] = None):
        super().__init__(original_brick, rate_limit)

        self._produce_method = getattr(self.original_brick, "produce", None)
        if not callable(self._produce_method) or asyncio.iscoroutinefunction(self._produce_method):
            raise TypeError(f"Method 'produce' not found or async on {type(self.original_brick).__name__}")

        # Dedicated limiter for the data emission by this adapter
        self._limiter = AsyncRateLimiter(rate_limit) if rate_limit else None

        # Internal queue for daemon thread -> async communication
        self._data_queue = queue.Queue(1)
        self._stop_event = threading.Event()
        self._producer_thread: Optional[threading.Thread] = None

    async def start(self):
        """Start the original brick and the internal blocking producer thread."""
        await super().start()

        if self._producer_thread is None or not self._producer_thread.is_alive():
            self._stop_event.clear()
            # Clear queue in case of restart
            while not self._data_queue.empty():
                try:
                    self._data_queue.get_nowait()
                except queue.Empty:
                    break
            self._producer_thread = threading.Thread(
                target=self._producer_loop, name=f"BlockingProducer-{type(self.original_brick).__name__}", daemon=True
            )
            self._producer_thread.start()
            logger.debug(f"Started internal producer thread for {type(self.original_brick).__name__}")

    async def stop(self):
        """Signal the producer thread and the original brick to stop."""
        # Signal producer thread to stop and unblock consumer task
        self.unblock_producer()

        # Briefly wait for thread
        if self._producer_thread and self._producer_thread.is_alive():
            self._producer_thread.join(timeout=1.0)

        # Stop the original brick using base class method (will run in executor)
        await super().stop()

    def unblock_producer(self):
        """Signals the producer thread and injects sentinel to unblock consumer."""
        if not self._stop_event.is_set() and self._producer_thread and self._producer_thread.is_alive():
            logger.debug(f"Adapter for {type(self.original_brick).__name__}: signaling stop event and injecting sentinel.")
            # Allow the producer thread to stop cleanly on its next iteration
            self._stop_event.set()

            # Put sentinel to unblock the _data_queue.get() call in produce()
            try:
                self._data_queue.put_nowait(_SHUTDOWN)
            except queue.Full:
                logger.warning(f"Adapter for {type(self.original_brick).__name__}: could not inject sentinel, queue full.")
            except Exception as e:
                logger.warning(f"Adapter for {type(self.original_brick).__name__}: error injecting sentinel: {e}")
        else:
            logger.debug(f"Adapter for {type(self.original_brick).__name__}: stop already signaled.")

    async def produce(self, *args: Any) -> Any:
        """Normalized async produce, gets data from internal queue populated by the daemon thread
        and applies emission rate limit.
        """
        if not self._loop:
            raise RuntimeError("Loop not set for adapter execution")
        if self._stop_event.is_set() or not self._producer_thread or not self._producer_thread.is_alive():
            logger.debug(f"Producer thread for {type(self.original_brick).__name__} not running in produce().")
            # Might happen if start wasn't called or thread died. Return None to signal end.
            return None

        # Rate limiting is applied at emission time, before getting the actual data to emit
        if self._limiter:
            await self._limiter.wait()

        data = await self._loop.run_in_executor(None, self._data_queue.get)
        if data is _SHUTDOWN:
            logger.debug(f"Adapter {type(self.original_brick).__name__} received sentinel from internal queue.")
            return None
        self._data_queue.task_done()

        return data

    # TODO: we can probably avoid propagating the _SHUTDOWN sentinel and simply return.
    # self._producer_thread.is_alive() in produce() should take care of this situation.
    def _producer_loop(self):
        """Target for the internal daemon thread. Transfers data from the blocking produce method to the async one."""
        try:
            while not self._stop_event.is_set():
                try:
                    data = self._produce_method()
                    if data is None:
                        logger.debug(f"Internal producer thread ({type(self.original_brick).__name__}): produce returned None. Stopping.")
                        self._data_queue.put(_SHUTDOWN)
                        break
                    if not self._stop_event.is_set():
                        self._data_queue.put(data)
                    else:
                        break
                except Exception as e:
                    logger.exception(f"Error in internal producer thread ({type(self.original_brick).__name__}): {e}")
                    self._data_queue.put(_SHUTDOWN)  # Signal error
                    break
        finally:
            logger.debug(f"Internal producer thread finished for {type(self.original_brick).__name__}.")
            try:
                self._data_queue.put_nowait(_SHUTDOWN)  # Ensure sentinel
            except queue.Full:
                pass
            except Exception as e:
                logger.warning(f"Exception putting final sentinel from producer thread: {e}")


class AsyncProcessorAdapter(AsyncBrickAdapter):
    def __init__(self, original_brick: Any, rate_limit: Optional[int] = None):
        super().__init__(original_brick, rate_limit)

        self._process_method = getattr(self.original_brick, "process", None)
        if not callable(self._process_method):
            raise TypeError(f"Method 'process' not found on {type(self.original_brick).__name__}")

        self._is_sync = not asyncio.iscoroutinefunction(self._process_method)
        self._limiter = AsyncRateLimiter(rate_limit) if rate_limit else None

    async def process(self, *args: Any) -> Any:
        if not self._loop and self._is_sync:
            raise RuntimeError("Loop not set for executing sync process")

        if self._limiter:
            await self._limiter.wait()

        if self._is_sync:
            return await self._loop.run_in_executor(None, self._process_method, *args)
        else:
            return await self._process_method(*args)


class AsyncSinkAdapter(AsyncBrickAdapter):
    def __init__(self, original_brick: Any, rate_limit: Optional[int] = None):
        super().__init__(original_brick, rate_limit)

        self._consume_method = getattr(self.original_brick, "consume", None)
        if not callable(self._consume_method):
            raise TypeError(f"Method 'consume' not found on {type(self.original_brick).__name__}")

        self._is_sync = not asyncio.iscoroutinefunction(self._consume_method)
        self._limiter = AsyncRateLimiter(rate_limit) if rate_limit else None

    async def consume(self, *args: Any) -> Any:
        if not self._loop and self._is_sync:
            raise RuntimeError("Loop not set for executing sync consume")

        if self._limiter:
            await self._limiter.wait()

        if self._is_sync:
            return await self._loop.run_in_executor(None, self._consume_method, *args)
        else:
            return await self._consume_method(*args)


def create_adapter(brick: Any, brick_type: str, rate_limit: Optional[int] = None) -> AsyncBrickAdapter:
    """Factory function that creates the appropriate adapter for the provided brick_type."""
    original_brick = brick
    method_name = ""
    SyncAdapterClass = None
    AsyncAdapterClass = None

    # Determine adapter classes and method based on type
    if brick_type == "source":
        # Producers might need a blocking adapter, run_in_executor is NOT fine in that case!
        method_name = "produce"
        SyncAdapterClass = AsyncBlockingSourceAdapter
        AsyncAdapterClass = AsyncSourceAdapter
    elif brick_type == "processor":
        method_name = "process"
        SyncAdapterClass = AsyncProcessorAdapter
        AsyncAdapterClass = AsyncProcessorAdapter  # Use same adapter
    elif brick_type == "sink":
        method_name = "consume"
        SyncAdapterClass = AsyncSinkAdapter
        AsyncAdapterClass = AsyncSinkAdapter  # Use same adapter
    else:
        raise ValueError(f"Unknown brick type: {brick_type}")

    # Handle simple callables by wrapping them first
    is_simple_callable = callable(brick) and not (hasattr(brick, method_name) or hasattr(brick, "start") or hasattr(brick, "stop"))
    if is_simple_callable:

        class FuncHolder:
            pass

        original_brick = FuncHolder()
        setattr(original_brick, method_name, brick)
        logger.debug(f"Wrapping callable {getattr(brick, '__name__', 'unknown')} as a simple {brick_type} object.")

    # Check if the core method exists on the brick
    core_method = getattr(original_brick, method_name, None)
    if not callable(core_method):
        raise TypeError(f"{brick_type.capitalize()} brick must have a callable '{method_name}' method.")

    # Decide which adapter to use based on sync/async nature
    is_sync = not asyncio.iscoroutinefunction(core_method)
    AdapterClass = SyncAdapterClass if is_sync else AsyncAdapterClass

    try:
        return AdapterClass(original_brick, rate_limit)
    except TypeError as e:
        raise TypeError(f"{brick_type.capitalize()} brick error: {e}") from e
