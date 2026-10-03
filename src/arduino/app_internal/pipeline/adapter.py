# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import asyncio
import inspect
import queue
import threading
from collections.abc import Callable
from typing import Any, Literal, overload
from .limiter import AsyncRateLimiter
from arduino.app_utils import Logger

logger = Logger("pipeline.adapter")


# These classes are used to adapt the original bricks to the asyncio API. They are responsible for wrapping the original
# bricks and providing a consistent interface for the pipeline.


class AsyncBrickAdapter:
    """Base class for brick adapters, normalizing to an async API."""

    def __init__(self, original_brick: Any, rate_limit: int | None = None) -> None:  # noqa: ANN401
        self.original_brick = original_brick
        self.rate_limit = rate_limit
        self._loop: asyncio.AbstractEventLoop | None = None

    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def _running_loop(self) -> asyncio.AbstractEventLoop:
        if self._loop is None:
            raise RuntimeError("Loop not set for adapter execution")
        return self._loop

    async def start(self) -> Any:  # noqa: ANN401
        """Normalized async start method."""
        logger.debug(f"Running start method for {type(self.original_brick).__name__}")
        return await self._execute_maybe_sync("start")

    async def stop(self) -> Any:  # noqa: ANN401
        """Normalized async stop method."""
        logger.debug(f"Starting stop method for {type(self.original_brick).__name__}")
        return await self._execute_maybe_sync("stop")

    async def _execute_maybe_sync(self, method_name: str, *args: Any) -> Any:  # noqa: ANN401
        """Helper to execute a method, handling sync/async via executor."""
        loop = self._running_loop()
        if not hasattr(self.original_brick, method_name):
            # The start and stop methods are optional
            return

        method = getattr(self.original_brick, method_name)
        if not callable(method):
            raise TypeError(f"Method {method_name} is not callable on {type(self.original_brick).__name__}")

        if inspect.iscoroutinefunction(method):
            return await method(*args)
        else:
            return await loop.run_in_executor(None, method, *args)


class AsyncSourceAdapter(AsyncBrickAdapter):
    """Adapter for async sources."""

    def __init__(self, original_brick: Any, rate_limit: int | None = None) -> None:  # noqa: ANN401
        super().__init__(original_brick, rate_limit)

        produce_method = getattr(self.original_brick, "produce", None)
        if not callable(produce_method) or not inspect.iscoroutinefunction(produce_method):
            raise TypeError(f"Method 'produce' not found or not async on {type(self.original_brick).__name__}")
        self._produce_method: Callable[..., Any] = produce_method
        self._limiter = AsyncRateLimiter(rate_limit) if rate_limit else None

    async def produce(self, *args: Any) -> Any:  # noqa: ANN401
        """Normalized async produce, calls original async method."""
        if self._limiter:
            await self._limiter.wait()
        return await self._produce_method(*args)


class AsyncBlockingSourceAdapter(AsyncBrickAdapter):
    """Adapter for synchronous sources that might block indefinitely.
    Manages a daemon thread internally to avoid blocking the event loop.
    """

    def __init__(self, original_brick: Any, rate_limit: int | None = None) -> None:  # noqa: ANN401
        super().__init__(original_brick, rate_limit)

        produce_method = getattr(self.original_brick, "produce", None)
        if not callable(produce_method) or inspect.iscoroutinefunction(produce_method):
            raise TypeError(f"Method 'produce' not found or async on {type(self.original_brick).__name__}")
        self._produce_method: Callable[..., Any] = produce_method

        # Dedicated limiter for the data emission by this adapter
        self._limiter = AsyncRateLimiter(rate_limit) if rate_limit else None

        # Internal queue for daemon thread -> async communication
        self._data_queue: queue.Queue[Any] = queue.Queue(1)
        self._stop_event = threading.Event()
        self._producer_thread: threading.Thread | None = None

    async def start(self) -> None:
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

    async def stop(self) -> None:
        """Signal the producer thread and the original brick to stop."""
        # Signal producer thread to stop and unblock consumer task
        self.unblock_producer()

        # Briefly wait for thread
        if self._producer_thread and self._producer_thread.is_alive():
            self._producer_thread.join(timeout=1.0)

        # Stop the original brick using base class method (will run in executor)
        await super().stop()

    def unblock_producer(self) -> None:
        """Signals the producer thread and injects sentinel to unblock consumer."""
        if not self._stop_event.is_set() and self._producer_thread and self._producer_thread.is_alive():
            logger.debug(f"Adapter for {type(self.original_brick).__name__}: signaling stop event and injecting sentinel.")
            # Allow the producer thread to stop cleanly on its next iteration
            self._stop_event.set()

            # Put sentinel to unblock the _data_queue.get() call in produce()
            try:
                self._data_queue.put_nowait(None)
            except queue.Full:
                logger.warning(f"Adapter for {type(self.original_brick).__name__}: could not inject sentinel, queue full.")
            except Exception as e:
                logger.warning(f"Adapter for {type(self.original_brick).__name__}: error injecting sentinel: {e}")
        else:
            logger.debug(f"Adapter for {type(self.original_brick).__name__}: stop already signaled.")

    async def produce(self, *args: Any) -> Any:  # noqa: ANN401
        """Normalized async produce, gets data from internal queue populated by the daemon thread
        and applies emission rate limit.
        """
        loop = self._running_loop()
        if self._stop_event.is_set() or not self._producer_thread or not self._producer_thread.is_alive():
            logger.debug(f"Producer thread for {type(self.original_brick).__name__} not running in produce().")
            # Might happen if start wasn't called or thread died. Return None to signal end.
            return None

        # Rate limiting is applied at emission time, before getting the actual data to emit
        if self._limiter:
            await self._limiter.wait()

        data = await loop.run_in_executor(None, self._data_queue.get)
        if data is None:
            logger.debug(f"Adapter {type(self.original_brick).__name__} received sentinel from internal queue.")
            return None
        self._data_queue.task_done()

        return data

    def _producer_loop(self) -> None:
        """Target for the internal daemon thread. Transfers data from the blocking produce method to the async one."""
        try:
            while not self._stop_event.is_set():
                try:
                    data = self._produce_method()
                    if data is None:
                        logger.debug(f"Internal producer thread ({type(self.original_brick).__name__}): produce returned None. Stopping.")
                        self._data_queue.put(None)
                        break
                    if not self._stop_event.is_set():
                        self._data_queue.put(data)
                    else:
                        break
                except Exception as e:
                    logger.exception(f"Error in internal producer thread ({type(self.original_brick).__name__}): {e}")
                    self._data_queue.put(None)  # Signal error
                    break
        finally:
            logger.debug(f"Internal producer thread finished for {type(self.original_brick).__name__}.")
            try:
                self._data_queue.put_nowait(None)  # Ensure sentinel
            except queue.Full:
                pass
            except Exception as e:
                logger.warning(f"Exception putting final sentinel from producer thread: {e}")


class AsyncProcessorAdapter(AsyncBrickAdapter):
    def __init__(self, original_brick: Any, rate_limit: int | None = None) -> None:  # noqa: ANN401
        super().__init__(original_brick, rate_limit)

        process_method = getattr(self.original_brick, "process", None)
        if not callable(process_method):
            raise TypeError(f"Method 'process' not found on {type(self.original_brick).__name__}")
        self._process_method: Callable[..., Any] = process_method

        self._is_sync = not inspect.iscoroutinefunction(process_method)
        self._limiter = AsyncRateLimiter(rate_limit) if rate_limit else None

    async def process(self, *args: Any) -> Any:  # noqa: ANN401
        loop = self._running_loop() if self._is_sync else None

        if self._limiter:
            await self._limiter.wait()

        if loop is not None:
            return await loop.run_in_executor(None, self._process_method, *args)
        return await self._process_method(*args)


class AsyncSinkAdapter(AsyncBrickAdapter):
    def __init__(self, original_brick: Any, rate_limit: int | None = None) -> None:  # noqa: ANN401
        super().__init__(original_brick, rate_limit)

        consume_method = getattr(self.original_brick, "consume", None)
        if not callable(consume_method):
            raise TypeError(f"Method 'consume' not found on {type(self.original_brick).__name__}")
        self._consume_method: Callable[..., Any] = consume_method

        self._is_sync = not inspect.iscoroutinefunction(consume_method)
        self._limiter = AsyncRateLimiter(rate_limit) if rate_limit else None

    async def consume(self, *args: Any) -> Any:  # noqa: ANN401
        loop = self._running_loop() if self._is_sync else None

        if self._limiter:
            await self._limiter.wait()

        if loop is not None:
            return await loop.run_in_executor(None, self._consume_method, *args)
        return await self._consume_method(*args)


@overload
def create_adapter(brick: Any, brick_type: Literal["source"], rate_limit: int | None = None) -> AsyncSourceAdapter | AsyncBlockingSourceAdapter: ...  # noqa: ANN401


@overload
def create_adapter(brick: Any, brick_type: Literal["processor"], rate_limit: int | None = None) -> AsyncProcessorAdapter: ...  # noqa: ANN401


@overload
def create_adapter(brick: Any, brick_type: Literal["sink"], rate_limit: int | None = None) -> AsyncSinkAdapter: ...  # noqa: ANN401


def create_adapter(brick: Any, brick_type: str, rate_limit: int | None = None) -> AsyncBrickAdapter:  # noqa: ANN401
    """Factory function that creates the appropriate adapter for the provided brick_type."""
    original_brick = brick
    method_name = ""
    SyncAdapterClass: type[AsyncBrickAdapter]
    AsyncAdapterClass: type[AsyncBrickAdapter]

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
    is_sync = not inspect.iscoroutinefunction(core_method)
    AdapterClass = SyncAdapterClass if is_sync else AsyncAdapterClass

    try:
        return AdapterClass(original_brick, rate_limit)
    except TypeError as e:
        raise TypeError(f"{brick_type.capitalize()} brick error: {e}") from e
