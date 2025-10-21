# SPDX-FileCopyrightText: Copyright (C) 2025 ARDUINO SA <http://www.arduino.cc>
#
# SPDX-License-Identifier: MPL-2.0

import asyncio
import logging
import threading
from concurrent.futures import Future, CancelledError as FutureCancelledError
from typing import Any, Optional
from .adapter import create_adapter
from .task import PipelineTask, SourceTask, ProcessorTask, SinkTask
from arduino.app_utils import Logger

logger = Logger("pipeline.main")


class Pipeline:
    def __init__(self, debug: bool = False):
        if debug:
            logger.setLevel(logging.DEBUG)
        self._steps: list[PipelineTask] = []
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._loop_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._pipeline_future: Optional[Future] = None  # Represents the overall pipeline task
        self._running = False

    def add_source(self, brick: Any, rate_limit: Optional[int] = None, queue_size: int = 1):
        if self._running:
            raise RuntimeError("Cannot add bricks while pipeline is running.")
        if self._steps:
            raise ValueError("Source must be the first brick added.")

        try:
            adapter = create_adapter(brick, "source", rate_limit)
        except TypeError:
            raise

        self._steps.append(SourceTask(adapter))
        logger.debug(f"Added Source task for: {type(adapter.original_brick).__name__}")

        return self

    def add_processor(self, brick: Any, rate_limit: Optional[int] = None, queue_size: int = 1):
        if self._running:
            raise RuntimeError("Cannot add bricks while pipeline is running.")
        if not self._steps:
            raise ValueError("Cannot add processor before a source.")
        if isinstance(self._steps[-1], SinkTask):
            raise ValueError("Cannot add processor after a sink.")

        try:
            adapter = create_adapter(brick, "processor", rate_limit)
        except TypeError:
            raise

        self._steps.append(ProcessorTask(adapter))
        logger.debug(f"Added Processor task for: {type(adapter.original_brick).__name__}")

        return self

    def add_sink(self, brick: Any, rate_limit: Optional[int] = None, queue_size: int = 1):
        if self._running:
            raise RuntimeError("Cannot add bricks while pipeline is running.")
        if not self._steps:
            raise ValueError("Cannot add sink before a source.")
        if isinstance(self._steps[-1], SinkTask):
            raise ValueError("Cannot add sink after another sink.")

        try:
            adapter = create_adapter(brick, "sink", rate_limit)
        except TypeError:
            raise

        self._steps.append(SinkTask(adapter))
        logger.debug(f"Added Sink task for: {type(adapter.original_brick).__name__}")

        return self

    def start(self):
        """Starts the pipeline in a background thread."""
        if self._running:
            logger.warning("Pipeline is already running.")
            return

        logger.debug("Starting pipeline...")
        if len(self._steps) < 2:
            raise ValueError("Pipeline must have at least a source and a sink.")

        self._stop_event.clear()
        loop_ready_event = threading.Event()

        self._loop_thread = threading.Thread(target=self._run_loop, args=(loop_ready_event,), name="PipelineEventLoop", daemon=True)
        self._loop_thread.start()

        # Wait for the loop to be ready in the background thread
        ready = loop_ready_event.wait(timeout=10.0)
        if not ready:
            # Cleanup if loop failed to start
            if self._loop_thread.is_alive():
                # Try to signal stop if loop object exists, otherwise just hope thread exits
                if self._loop:
                    try:
                        self._loop.call_soon_threadsafe(self._loop.stop)
                    except Exception:
                        pass
                self._loop_thread.join(5)
            self._loop = None
            self._loop_thread = None
            raise RuntimeError("Pipeline event loop failed to start.")

        self._running = True
        logger.debug("Pipeline started successfully.")

    def stop(self):
        """Stops the pipeline gracefully."""
        if not self._running or not self._loop_thread or not self._loop_thread.is_alive() or not self._loop:
            logger.warning("Pipeline is not running or already stopped.")
            return

        logger.debug("Stopping pipeline...")

        # Schedule the async stop logic in the event loop thread
        future = asyncio.run_coroutine_threadsafe(self._async_stop_pipeline(), self._loop)
        try:
            # Wait for the async stop sequence to complete (with timeout)
            future.result(timeout=70.0)  # Should be > shutdown timeout inside _async_stop_pipeline
            logger.debug("Async stop sequence completed.")
        except TimeoutError:
            logger.exception("Timeout waiting for pipeline stop sequence to complete.")
            # If timeout occurs, try to cancel remaining tasks forcefully
            if not future.done():
                future.cancel()
            if self._pipeline_future and not self._pipeline_future.done():
                self._loop.call_soon_threadsafe(self._pipeline_future.cancel)
        except Exception as e:
            logger.exception(f"Error during pipeline stop sequence: {e}")
            # May need forceful cleanup here too

        # Signal the event loop to stop running
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)

        # Wait for the event loop thread to terminate
        self._loop_thread.join(timeout=10.0)
        if self._loop_thread.is_alive():
            logger.warning("Pipeline event loop thread did not terminate cleanly.")

        self._running = False
        self._loop = None
        self._loop_thread = None
        self._pipeline_future = None
        logger.debug("Pipeline stopped.")

    def _run_loop(self, loop_ready_event: threading.Event):
        """Main loop."""
        try:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            logger.debug("Internal event loop started.")
            loop_ready_event.set()
            # Run the main async pipeline logic, this will block until the pipeline is stopped or finished
            self._loop.run_until_complete(self._async_run_pipeline())
        except Exception as e:
            logger.exception(f"Exception in pipeline event loop: {e}")
            raise
        finally:
            try:
                logger.debug("Closing internal event loop...")
                # Ensure all pending tasks are cancelled/finished before stopping loop
                tasks = asyncio.all_tasks(self._loop)
                if tasks:
                    logger.debug(f"Waiting for {len(tasks)} remaining tasks before stopping loop...")
                    for task in tasks:
                        if not task.done():
                            task.cancel()
                    self._loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))

                self._loop.run_until_complete(self._loop.shutdown_asyncgens())
            except Exception as e:
                logger.exception(f"Error during event loop cleanup: {e}")
            finally:
                if self._loop and not self._loop.is_closed():
                    self._loop.close()
                self._loop = None
                logger.debug("Internal event loop stopped.")

    async def _async_run_pipeline(self):
        """The main async logic using Adapters."""
        if not self._loop:
            raise RuntimeError("Pipeline internal loop not available.")
        if len(self._steps) < 2:
            raise ValueError("Pipeline must have at least a source and a sink.")

        # Set the loop for all steps (which passes it to adapters)
        for step in self._steps:
            step.set_loop(self._loop)

        # Link steps
        logger.debug("Linking pipeline step queues...")
        for i in range(len(self._steps) - 1):
            prev_step = self._steps[i]
            next_step = self._steps[i + 1]
            prev_output_queue = getattr(prev_step, "output_queue", None)
            if prev_output_queue and hasattr(next_step, "input_queue"):
                next_step.input_queue = prev_output_queue
                logger.debug(f"Linked output queue of step {i} to input queue of step {i + 1}")
            else:
                err_msg = f"Cannot link step {i} ({type(prev_step)}) to step {i + 1} ({type(next_step)}): incompatible queue attributes."
                logger.error(err_msg)
                raise TypeError(err_msg)

        try:
            logger.debug("Starting steps...")
            # Start steps
            steps = [await step.start() for step in self._steps]
            logger.debug(f"Launched {len(steps)} steps.")

            # Gather and await them
            self._pipeline_future = asyncio.gather(*steps)
            await self._pipeline_future
            logger.debug("Pipeline async run completed normally (all tasks finished).")
        except (asyncio.CancelledError, FutureCancelledError):
            logger.warn("Pipeline async run cancelled.")
        except Exception as e:
            logger.exception(f"Pipeline async run failed: {e}")
        finally:
            logger.debug("Entering final cleanup phase for all steps...")
            # Ensure resources are cleaned up
            for step in self._steps:
                try:
                    await step.stop()
                except Exception as e:
                    logger.exception(f"Error while stopping {type(step.adapter.original_brick).__name__}: {e}")
            logger.debug("Final cleanup phase completed.")

    async def _async_stop_pipeline(self):
        """Coroutine scheduled by stop() to ensure pipeline finishes.
        Unblocks source if needed and ensures the main gather future completes.
        Final cleanup is handled by _async_run_pipeline's finally block.
        """
        logger.debug("Initiating async stop sequence...")
        if not self._steps or not self._pipeline_future:
            logger.warning("Stop called but pipeline seems inactive or not properly started.")
            return

        source_wrapper = self._steps[0]
        unblock_method = getattr(source_wrapper.adapter, "unblock_producer", None)
        if callable(unblock_method):
            logger.debug("Attempting to unblock source producer...")
            try:
                unblock_method()
            except Exception as e:
                logger.exception(f"Error unblocking source adapter producer: {e}")

        if self._pipeline_future and not self._pipeline_future.done():
            logger.debug("Waiting for pipeline tasks to finish after stop initiated...")
            try:
                await asyncio.wait_for(self._pipeline_future, timeout=60.0)
                logger.debug("Pipeline tasks finished after stop initiated.")
            except asyncio.TimeoutError:
                logger.warning("Pipeline tasks did not finish within timeout even after unblocking source. Cancelling remaining.")
                if not self._pipeline_future.done():
                    self._pipeline_future.cancel()
                    try:
                        await self._pipeline_future
                    except (asyncio.CancelledError, FutureCancelledError):
                        pass  # Expected
            except (asyncio.CancelledError, FutureCancelledError):
                logger.warning("Pipeline task group was already cancelled.")
            except Exception as e:
                logger.exception(f"Error waiting for pipeline tasks during stop: {e}")
        else:
            logger.debug("Pipeline task group was already done.")

        logger.debug("Async stop sequence finished.")
