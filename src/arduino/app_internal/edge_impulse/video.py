# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""The base of the bricks running an Edge Impulse model on a live video stream."""

import inspect
import os
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np

from arduino.app_internal.ei_inference import InferenceClient, Result
from arduino.app_peripherals.camera import BaseCamera, Camera
from arduino.app_utils import Logger, brick
from arduino.app_utils.image.adjustments import compress_to_jpeg

from .model import EdgeImpulseModel
from .video_stream import VideoStreamServer

logger = Logger("VideoInference")

type DetectionCallback = Callable[[], None] | Callable[[dict[str, Any]], None] | Callable[[dict[str, Any], bytes | None], None]
"""Callback accepted by `on_detect`: no arguments, the detection details dict, or the dict plus the camera `frame`."""
type AllDetectionsCallback = Callable[[dict[str, Any]], None] | Callable[[dict[str, Any], bytes | None], None]
"""Callback accepted by `on_detect_all`: the detections dict, optionally followed by the camera `frame`."""


class VideoInference(EdgeImpulseModel):
    """A brick running an Edge Impulse model on the frames of a camera, reacting to what it reports.

    The camera frames go to the inference service as they are, the results come back in their coordinates
    and `_process_result`, which every subclass implements, turns each of them into the payload of the
    callbacks: per-label callbacks registered with `on_detect`, debounced and never run twice at once, and a
    catch-all one registered with `on_detect_all`. The video is served with the overlay `_annotate` draws,
    for browsers and embedded iframes: with viewers every camera frame is captured and drawn with the latest
    result, without viewers frames are captured only when the model is free. The connection to the service
    is retried until it is ready and reopened when it is lost.
    """

    ALL_HANDLERS_KEY = "__ALL"
    STREAM_PORT = 4912  # the port the Edge Impulse runner container used to serve the video on

    _DETECTION_LOCK_TO = 0.01  # seconds to wait for a detection lock before discarding the detection signal

    def __init__(
        self,
        camera: BaseCamera | None = None,
        confidence: float = 0.3,
        debounce_sec: float = 0.0,
        camera_preview: bool = False,
        stream_port: int | None = STREAM_PORT,
    ) -> None:
        """Initialize the brick.

        Args:
            camera (BaseCamera): The camera instance to use for capturing video. If None, a default camera will be initialized.
            confidence (float): Score the results must reach, applied by the model. Default is 0.3 (30%).
            debounce_sec (float): Minimum seconds between repeated callbacks for the same label. Default is 0 seconds.
            camera_preview (bool): Receive the current camera frame on callback invocation, as a raw JPEG without
                the overlay. Default is False.
            stream_port (int | None): Port of the MJPEG stream of the video with the overlay, the one external
                viewers embed. Default is 4912, None disables the stream.

        Raises:
            RuntimeError: If no model is configured.
        """
        super().__init__(confidence=confidence)
        self._camera = camera if camera else Camera()
        self._debounce_sec = debounce_sec
        self._last_detected: dict[str, float] = {}
        self._camera_preview = camera_preview

        self._handlers_lock = threading.Lock()
        self._handlers: dict[str, Callable[[dict | None, bytes | None], None]] = {}
        self._detection_locks: dict[str, threading.Lock] = {}  # one per label, a handler never runs twice at once
        self._detection_locks_lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=5, thread_name_prefix=f"{type(self).__name__}Handler")

        self._is_running = threading.Event()
        self._stream = VideoStreamServer(os.getenv("BIND_ADDRESS", "0.0.0.0"), stream_port) if stream_port is not None else None
        logger.info(f"[{type(self).__name__}] Model: {self._model}")

    @property
    def stream_port(self) -> int | None:
        """The port serving the video with the overlay, None when the stream is disabled."""
        return self._stream.port if self._stream else None

    def on_detect(self, object: str, callback: DetectionCallback) -> None:  # noqa: A002
        """Register a callback invoked when a **specific label** is reported.

        Args:
            object (str): The label to react to.
            callback (DetectionCallback): A plain function taking either no parameters, or one parameter receiving
                the details dict of the label, as the brick shapes it. A function that also declares a `frame`
                parameter receives the current camera frame as raw JPEG bytes (or None, see `camera_preview`).

        Raises:
            TypeError: If `callback` is not a function.
        """
        if not inspect.isfunction(callback):
            raise TypeError("Callback must be a callable function.")
        with self._handlers_lock:
            if object in self._handlers:
                logger.warning(f"Handler for object '{object}' already exists. Overwriting.")
            self._handlers[object] = self._bind(callback)

    def on_detect_all(self, callback: AllDetectionsCallback) -> None:
        """Register a callback invoked for **every result with something to report**.

        Args:
            callback (AllDetectionsCallback): A plain function taking one dict argument, everything the result
                reported as the brick shapes it. A function that also declares a `frame` parameter receives the
                current camera frame as raw JPEG bytes (or None, see `camera_preview`).

        Raises:
            TypeError: If `callback` is not a function.
        """
        if not inspect.isfunction(callback):
            raise TypeError("Callback must be a callable function.")
        with self._handlers_lock:
            self._handlers[self.ALL_HANDLERS_KEY] = self._bind(callback)

    @staticmethod
    def _bind(callback: Callable) -> Callable[[dict | None, bytes | None], None]:
        """Adapt a handler to (payload, frame) once, according to the parameters it declares."""
        parameters = inspect.signature(callback).parameters
        if len(parameters) == 0:
            return lambda payload, frame: callback()
        if "frame" in parameters:
            return lambda payload, frame: callback(payload, frame=frame)
        return lambda payload, frame: callback(payload)

    def start(self) -> None:
        """Start the camera and the video stream; the model is opened by the inference loop."""
        self._camera.start()
        if self._stream:
            self._stream.start()
        self._is_running.set()

    def stop(self) -> None:
        """Stop the brick and release its resources, the service releases the model."""
        self._is_running.clear()
        self._close()
        if self._stream:
            self._stream.stop()
        self._camera.stop()
        self._executor.shutdown(wait=False, cancel_futures=True)

    @brick.execute
    def inference_loop(self) -> None:
        """Feed the camera frames to the model and dispatch its results, until stopped.

        Submits a frame each time the model has a free slot and a receiver thread dispatches the results as they
        arrive. Without viewers of the video, frames are captured only when the model is free: a model slower
        than the camera lowers the capture rate to its own and no work is spent on frames that would be dropped.
        With viewers, every camera frame is captured and streamed with the overlay of the latest result, so the
        video keeps the camera rate whatever the model takes. Waits for the service and its model, and
        reconnects when the connection is lost.
        """
        while self._is_running.is_set():
            client = self._connect(self._is_running)
            if client is None:
                continue
            receiver = threading.Thread(target=self._receive_results, args=(client,), daemon=True, name=f"{type(self).__name__}Results")
            receiver.start()
            try:
                while self._is_running.is_set() and not client.closed:
                    if not self._has_viewers and not client.wait_idle(0.5):
                        continue  # the model is still busy, check the stop flag and wait again
                    frame = self._camera.capture()
                    if frame is None:
                        time.sleep(0.01)  # Brief sleep if no image available
                        continue
                    client.submit(frame, keep_frame=self._camera_preview)  # skipped while the slots are taken
                    if self._has_viewers:  # checked again: a viewer may have arrived during the wait
                        self._publish(frame)
                if self._is_running.is_set():
                    logger.warning("Inference service connection lost. Reconnecting...")
            except ConnectionError as e:
                if self._is_running.is_set():
                    logger.warning(f"Inference service connection lost: {e}. Reconnecting...")
            except Exception as e:
                logger.exception(f"Failed to process the frames: {e}")
                self._is_running.wait(self._RETRY_SEC)
            finally:
                self._close()
                receiver.join()

    def _receive_results(self, client: InferenceClient) -> None:
        """Dispatch the results of the frames in flight until the connection closes."""
        try:
            while True:
                result = client.get_result(timeout=0.5)
                if result is not None:
                    self._process_result(result)
        except ConnectionError:
            pass

    def _process_result(self, result: Result) -> None:
        """Turn one result into callbacks and into what the video shows; every brick implements it.

        `result.frame`, present with `camera_preview`, is the frame the preview callbacks receive.
        """
        raise NotImplementedError

    @property
    def _has_viewers(self) -> bool:
        """True while someone watches the video stream, the only time frames are worth rendering."""
        return self._stream is not None and self._stream.has_clients

    def _publish(self, frame: np.ndarray) -> None:
        """Stream the camera frame with the overlay drawn on it."""
        annotated = self._annotate(frame)
        if annotated is not None and self._stream is not None:
            self._stream.publish(annotated)

    def _annotate(self, frame: np.ndarray) -> bytes | None:
        """The frame with the overlay of the latest result, as JPEG bytes for the video stream; plain by default."""
        jpeg = compress_to_jpeg(frame)
        return jpeg.tobytes() if jpeg is not None else None

    def _encode_preview(self, frame: np.ndarray | None) -> bytes | None:
        """The camera frame as JPEG bytes for the handlers, None unless camera_preview is enabled."""
        if frame is None or not self._camera_preview:
            return None
        jpeg = compress_to_jpeg(frame)
        return jpeg.tobytes() if jpeg is not None else None

    def _get_detection_lock(self, detection: str) -> threading.Lock:
        """The lock of a label, created on first use."""
        with self._detection_locks_lock:
            if detection not in self._detection_locks:
                self._detection_locks[detection] = threading.Lock()
            return self._detection_locks[detection]

    def _execute_handler(self, key: str, payload: dict | None = None, frame: bytes | None = None) -> None:
        """Run the handler registered for the key on the executor, unless it is running already or debounced.

        Args:
            key (str): The handler key, either a label or ``ALL_HANDLERS_KEY``.
            payload (dict): The data to pass to the handler.
            frame (bytes): The raw JPEG camera frame, if available.
        """
        with self._handlers_lock:
            handler = self._handlers.get(key)
        if not handler:
            return

        detection_lock = self._get_detection_lock(key)
        if not detection_lock.acquire(timeout=self._DETECTION_LOCK_TO):
            logger.debug(f"Handler for '{key}' is already running, skipping.")
            return

        now = time.time()
        if now - self._last_detected.get(key, 0) >= self._debounce_sec:
            self._last_detected[key] = now
        else:
            detection_lock.release()
            return

        def _run() -> None:
            try:
                logger.debug(f"Detected: {key}, invoking handler.")
                handler(payload, frame)
            finally:
                detection_lock.release()

        try:
            self._executor.submit(_run)
        except RuntimeError:  # the executor was shut down before the task could be submitted
            detection_lock.release()
