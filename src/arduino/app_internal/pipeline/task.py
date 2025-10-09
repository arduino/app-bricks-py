# SPDX-FileCopyrightText: Copyright (C) 2025 ARDUINO SA <http://www.arduino.cc>
#
# SPDX-License-Identifier: MPL-2.0

import asyncio
from typing import Generic, Optional
from concurrent.futures import CancelledError as FutureCancelledError
from .constants import _SHUTDOWN, T_IN, T_OUT
from .adapter import AsyncBrickAdapter, AsyncProcessorAdapter, AsyncSinkAdapter
from arduino.app_utils import Logger

logger = Logger("pipeline.task")


# The following classes handle the lifecycle of bricks inside the asyncio loop. They are responsible for the starting
# and stopping of the bricks, as well as the management of the queues between them.


class PipelineTask:
    """Hierarchy root for classes that adapt bricks to the asyncio's tasks API."""

    def __init__(self, adapter: AsyncBrickAdapter):
        self.adapter = adapter
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._task: Optional[asyncio.Task] = None

    def set_loop(self, loop: asyncio.AbstractEventLoop):
        self._loop = loop
        self.adapter.set_loop(loop)

    async def start(self) -> asyncio.Task:
        if not self._loop:
            raise RuntimeError("Event loop not set")

        log.debug(f"Starting user brick via adapter {type(self.adapter.original_brick).__name__}")
        await self.adapter.start()

        if self._task is None or self._task.done():
            brick_name = type(self.adapter.original_brick).__name__
            task_name = type(self).__name__
            self._task = self._loop.create_task(self._run(), name=f"{task_name}-{brick_name}")
            log.debug(f"Created task for {brick_name}")

        return self._task

    async def stop(self):
        """Stops the task gracefully ."""
        if self._task and not self._task.done():
            try:
                await self._task
            except (asyncio.CancelledError, FutureCancelledError):
                log.warning(f"Task {self._task.get_name()} cancelled during graceful stop.")
            except Exception as e:
                log.exception(f"Task {self._task.get_name()} raised exception during stop wait: {e}")

        log.debug(f"Stopping user brick via adapter {type(self.adapter.original_brick).__name__}")
        await self.adapter.stop()

    async def _run(self):
        raise NotImplementedError


class SourceTask(PipelineTask, Generic[T_OUT]):
    def __init__(self, adapter: AsyncBrickAdapter, queue_size: int = 1):
        super().__init__(adapter)
        self.adapter: AsyncBrickAdapter = adapter
        self.output_queue = asyncio.Queue(queue_size)

    async def _run(self):
        brick_name = type(self.adapter.original_brick).__name__
        log.info(f"Source task run loop started for {brick_name}.")
        if not self._loop:
            raise RuntimeError("Loop not set")

        try:
            while True:
                try:
                    # Handles rate limit, sync/async and blocking variant internally
                    data = await self.adapter.produce()
                    if data is None:
                        log.info(f"Source adapter {brick_name} indicated end of stream.")
                        break
                    await self.output_queue.put(data)
                except AttributeError:
                    log.exception(f"Adapter for {brick_name} missing produce?")
                    break
                except (asyncio.CancelledError, FutureCancelledError):
                    log.info(f"Source task task {brick_name} cancelled.")
                    raise
                except Exception as e:
                    log.exception(f"Error in source task {brick_name}: {e}")
                    break
        finally:
            log.info(f"Source task {brick_name} finished. Signaling downstream.")
            await self.output_queue.put(_SHUTDOWN)


class ProcessorTask(PipelineTask, Generic[T_IN, T_OUT]):
    def __init__(self, adapter: AsyncProcessorAdapter, queue_size: int = 1):
        super().__init__(adapter)
        self.adapter: AsyncProcessorAdapter = adapter
        self.input_queue: Optional[asyncio.Queue[T_IN]] = None
        self.output_queue: asyncio.Queue[T_OUT] = asyncio.Queue(queue_size)

    async def _run(self):
        brick_name = type(self.adapter.original_brick).__name__
        if not self.input_queue:
            log.error(f"Input queue not set for processor {brick_name}")
            return
        log.info(f"Processor task run loop started for {brick_name}.")

        try:
            while True:
                data_in = await self.input_queue.get()
                try:
                    if data_in is _SHUTDOWN:
                        log.debug(f"Processor {brick_name} got sentinel.")
                        break
                    # Handles rate limit and sync/async variations internally
                    data_out = await self.adapter.process(data_in)
                    if data_out is not None:
                        await self.output_queue.put(data_out)
                    else:
                        log.debug(f"Processor {brick_name} filtered data.")
                except AttributeError:
                    log.exception(f"Adapter for {brick_name} missing process?")
                    break
                except (asyncio.CancelledError, FutureCancelledError):
                    log.info(f"Processor task {brick_name} cancelled.")
                    raise
                except Exception as e:
                    log.exception(f"Error processing in {brick_name}: {e}")
                    break
                finally:
                    self.input_queue.task_done()
        finally:
            log.info(f"Processor task {brick_name} finished. Signaling downstream.")
            await self.output_queue.put(_SHUTDOWN)


class SinkTask(PipelineTask, Generic[T_IN]):
    def __init__(self, adapter: AsyncSinkAdapter, queue_size: int = 1):
        super().__init__(adapter)
        self.adapter: AsyncSinkAdapter = adapter
        self.input_queue: Optional[asyncio.Queue[T_IN]] = None

    async def _run(self):
        brick_name = type(self.adapter.original_brick).__name__
        if not self.input_queue:
            log.error(f"Input queue not set for sink {brick_name}")
            return
        log.info(f"Sink task run loop started for {brick_name}.")

        try:
            while True:
                data_in = await self.input_queue.get()
                try:
                    if data_in is _SHUTDOWN:
                        log.debug(f"Sink {brick_name} got sentinel.")
                        break
                    # Handles rate limit, sync/async internally
                    await self.adapter.consume(data_in)
                except AttributeError:
                    log.exception(f"Adapter for {brick_name} missing consume?")
                    break
                except (asyncio.CancelledError, FutureCancelledError):
                    log.info(f"Sink task {brick_name} cancelled.")
                    raise
                except Exception as e:
                    log.exception(f"Error consuming in {brick_name}: {e}")
                    break
                finally:
                    self.input_queue.task_done()
        finally:
            log.info(f"Sink task {brick_name} finished.")
