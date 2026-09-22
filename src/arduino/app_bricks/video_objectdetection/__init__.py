# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import inspect
import os
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from arduino.app_internal import ei_inference
from arduino.app_internal.core.module import get_brick_config, get_brick_configured_model, load_model_list
from arduino.app_internal.ei_inference import InferenceClient, Result, ServerError, model_name_from_path
from arduino.app_peripherals.camera import BaseCamera, Camera
from arduino.app_utils import Logger, brick
from arduino.app_utils.image.adjustments import compress_to_jpeg

from .video_stream import BoxStabilizer, LabelColors, VideoStreamServer, draw_detections

logger = Logger("VideoObjectDetection")

MODEL_VARIABLE = "EI_V_OBJ_DETECTION_MODEL"
STREAM_PORT = 4912  # The port the Edge Impulse runner container used to serve the video on

type DetectionCallback = Callable[[], None] | Callable[[dict], None] | Callable[[dict, bytes | None], None]
"""Callback accepted by `on_detect`: no arguments, the detection details dict, or the dict plus the camera `frame`."""
type AllDetectionsCallback = Callable[[dict], None] | Callable[[dict, bytes | None], None]
"""Callback accepted by `on_detect_all`: the detections dict, optionally followed by the camera `frame`."""


@brick
class VideoObjectDetection:
    """Module for object detection on a **live video stream** using a specified machine learning model.

    This brick:
      - Streams the camera frames to the Edge Impulse inference service over its Unix socket, one in flight at a time.
      - Receives the bounding boxes, in the coordinates of the camera frame.
      - Filters detections by a configurable confidence threshold.
      - Debounces repeated triggers of the same label.
      - Invokes per-label callbacks and/or a catch-all callback.
      - Streams the video with the bounding boxes on port 4912, for browsers and embedded iframes: every camera
        frame is drawn with the boxes of the latest inference.
    """

    ALL_HANDLERS_KEY = "__ALL"

    _DETECTION_LOCK_TO = 0.01  # Seconds to wait for a detection lock before discarding the detection signal
    _RETRY_SEC = 2.0  # Seconds between attempts to reach the inference service

    def __init__(
        self,
        camera: BaseCamera | None = None,
        confidence: float = 0.3,
        debounce_sec: float = 0.0,
        camera_preview: bool = False,
        model: str | None = None,
        stream_port: int | None = STREAM_PORT,
    ) -> None:
        """Initialize the VideoObjectDetection class.

        Args:
            camera (BaseCamera): The camera instance to use for capturing video. If None, a default camera will be initialized.
            confidence (float): Confidence level for detection. Default is 0.3 (30%).
            debounce_sec (float): Minimum seconds between repeated detections of the same object. Default is 0 seconds.
            camera_preview (bool): Receive current camera frame on callback invocation.
                Frame is a raw jpeg-encoded image without bounding boxes applied on it. Default is False.
            model (str): Name of the model on the inference service, its `.eim` path relative to the models
                directory without extension. Defaults to the model configured for the brick (EI_V_OBJ_DETECTION_MODEL).
            stream_port (int | None): Port of the MJPEG stream of the video with the bounding boxes, the one
                external viewers embed. Default is 4912, None disables the stream.

        Raises:
            RuntimeError: If no model is configured.
        """
        self._camera = camera if camera else Camera()

        self._confidence = confidence
        self._debounce_sec = debounce_sec
        self._last_detected: dict[str, float] = {}
        self._camera_preview = camera_preview
        self._model = model or self._configured_model()

        self._handlers_lock = threading.Lock()
        self._handlers = {}  # Dictionary to hold handlers for different actions

        self._detection_locks = {}  # Per-detection locks for fine-grained concurrency control
        self._detection_locks_lock = threading.Lock()  # Lock to protect _detection_locks dict

        self._executor = ThreadPoolExecutor(max_workers=5, thread_name_prefix="VideoObjectDetectionHandler")

        self._is_running = threading.Event()
        self._client: InferenceClient | None = None
        self._client_lock = threading.Lock()
        self._stream = VideoStreamServer(os.getenv("BIND_ADDRESS", "0.0.0.0"), stream_port) if stream_port is not None else None
        self._colors = LabelColors()
        self._boxes = BoxStabilizer()  # what the video shows: the boxes of the results, steadied across them

        logger.info(f"[{self.__class__.__name__}] Model: {self._model}")

    @classmethod
    def _configured_model(cls) -> str:
        """The model configured for the brick, as the inference service names it.

        The app CLI sets the model variable when the app selects a model; for the default model of the board
        the path comes from the models list, through the model the brick configuration selects.
        """
        model_path = os.getenv(MODEL_VARIABLE) or cls._default_model_path()
        if not model_path:
            raise RuntimeError(f"No model configured: set the {MODEL_VARIABLE} variable or pass model= to the brick.")
        try:
            return model_name_from_path(model_path)
        except ValueError as e:
            raise RuntimeError(str(e)) from e

    @classmethod
    def _default_model_path(cls) -> str | None:
        """The model path the models list configures for this brick and its default model, None when unknown."""
        brick_config = get_brick_config(cls)
        brick_id = brick_config.get("id") if brick_config else None
        model_id = get_brick_configured_model(brick_id, brick_config) if brick_id else None
        models = load_model_list() or {}
        entry = models.get(model_id) if model_id else None
        if entry is None:
            return None
        for brick in entry.bricks:
            if brick.id == brick_id and MODEL_VARIABLE in brick.model_configuration:
                logger.info(f"Model '{model_id}' selected for the brick")
                return brick.model_configuration[MODEL_VARIABLE]
        return None

    @property
    def model(self) -> str:
        """The name of the model on the inference service."""
        return self._model

    @property
    def stream_port(self) -> int | None:
        """The port serving the video with the bounding boxes, None when the stream is disabled."""
        return self._stream.port if self._stream else None

    def on_detect(self, object: str, callback: DetectionCallback) -> None:
        """Register a callback invoked when a **specific label** is detected.

        Args:
            object (str): The label of the object to check for in the classification results.
            callback (DetectionCallback): A plain function taking either no parameters, or one
                parameter receiving the detection details dict
                `{"confidence": float, "bounding_box_xyxy": (x1, y1, x2, y2)}`. A function that also
                declares a `frame` parameter receives the current camera frame as raw JPEG bytes
                (or None when no preview frame is available, see `camera_preview`).

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
        """Register a callback invoked for **every detection event**.

        This is useful to receive a consolidated dictionary of detections for each frame.

        Args:
            callback (AllDetectionsCallback): A plain function taking one dict argument mapping
                each detected label to the list of its detections, with the shape
                `{label: [{"confidence": float, "bounding_box_xyxy": (x1, y1, x2, y2)}, ...], ...}`.
                A function that also declares a `frame` parameter receives the current camera
                frame as raw JPEG bytes (or None when no preview frame is available, see
                `camera_preview`).

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
        """Start the video object detection process."""
        self._camera.start()
        if self._stream:
            self._stream.start()
        self._is_running.set()

    def stop(self) -> None:
        """Stop the video object detection process and release resources, the service releases the model."""
        self._is_running.clear()
        self._close_client()
        if self._stream:
            self._stream.stop()
        self._camera.stop()
        self._executor.shutdown(wait=False, cancel_futures=True)

    @brick.execute
    def detection_loop(self) -> None:
        """Object detection main loop.

        Submits a frame each time the model is free, one in flight at a time, and a receiver thread dispatches the
        results as they arrive. Without viewers of the video, frames are captured only when the model is free: a
        model slower than the camera lowers the capture rate to its own and no work is spent on frames that would
        be dropped. With viewers, every camera frame is captured and streamed with the boxes of the latest result,
        so the video keeps the camera rate whatever the model takes. Waits for the service and its model, and
        reconnects when the connection is lost, until stopped.
        """
        while self._is_running.is_set():
            client = self._connect()
            if client is None:
                continue
            receiver = threading.Thread(target=self._receive_results, args=(client,), daemon=True, name="VideoObjectDetectionResults")
            receiver.start()
            try:
                while self._is_running.is_set() and not client.closed:
                    if not self._has_viewers and not client.wait_idle(0.5):
                        continue  # the model is still busy, check the stop flag and wait again
                    frame = self._camera.capture()
                    if frame is None:
                        time.sleep(0.01)  # Brief sleep if no image available
                        continue
                    client.submit(frame, keep_frame=self._camera_preview)  # skipped while a frame is in flight
                    if self._has_viewers:  # checked again: a viewer may have arrived during the wait
                        self._publish(frame)
                if self._is_running.is_set():
                    logger.warning("Inference service connection lost. Reconnecting...")
            except ConnectionError as e:
                if self._is_running.is_set():
                    logger.warning(f"Inference service connection lost: {e}. Reconnecting...")
            except Exception as e:
                logger.exception(f"Failed to process detection: {e}")
                self._is_running.wait(self._RETRY_SEC)
            finally:
                self._close_client()
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

    @property
    def _has_viewers(self) -> bool:
        """True while someone watches the video stream, the only time frames are worth rendering."""
        return self._stream is not None and self._stream.has_clients

    def _connect(self) -> InferenceClient | None:
        """Open the model on the inference service, retrying until it is ready or the brick is stopped."""
        while self._is_running.is_set():
            try:
                client = InferenceClient(self._model, ei_inference.DEFAULT_SOCKET_PATH)
            except ServerError as e:
                logger.error(f"The inference service refused model '{self._model}' ({e.code}): {e}. Retrying...")
            except OSError:
                logger.debug(f"Waiting for the inference service at {ei_inference.DEFAULT_SOCKET_PATH}. Retrying...")
            else:
                with self._client_lock:
                    self._client = client
                logger.info(f"Connected to the inference service, model '{client.model}' input {client.input_size[0]}x{client.input_size[1]}")
                return client
            self._is_running.wait(self._RETRY_SEC)
        return None

    def _close_client(self) -> None:
        with self._client_lock:
            client, self._client = self._client, None
        if client is not None:
            client.close()

    def _process_result(self, result: Result) -> None:
        """Turn the boxes of one frame into detections, feed the video boxes and invoke the handlers.

        `result.frame`, present with `camera_preview`, is the frame the preview callbacks receive.
        """
        if not result.ok:
            logger.warning(f"Inference failed ({result.error_code}): {result.error}")
            return
        frame = result.frame

        detections = {}
        for box in result.boxes:
            if box.score < self._confidence:
                continue
            xyxy_bbox = (round(box.x), round(box.y), round(box.x + box.w), round(box.y + box.h))
            detections.setdefault(box.label, []).append({"confidence": box.score, "bounding_box_xyxy": xyxy_bbox})
        self._boxes.update(result.boxes, self._confidence, (time.monotonic_ns() - result.ts_ns) / 1e9)
        if not detections:
            return

        preview = self._encode_preview(frame) if frame is not None else None
        for label, label_detections in detections.items():
            for detection_details in label_detections:
                self._execute_handler(key=label, payload=detection_details, frame=preview)
        self._execute_handler(key=self.ALL_HANDLERS_KEY, payload=detections, frame=preview)

    def _publish(self, frame: np.ndarray) -> None:
        """Stream the camera frame with the steadied boxes drawn on it."""
        annotated = self._annotate(frame, self._boxes.visible())
        if annotated is not None and self._stream is not None:
            self._stream.publish(annotated)

    def _annotate(self, frame: np.ndarray, detections: dict) -> bytes | None:
        """The frame with the boxes and labels drawn on a copy, as JPEG bytes for the video stream."""
        jpeg = compress_to_jpeg(draw_detections(frame, detections, self._colors))
        return jpeg.tobytes() if jpeg is not None else None

    def _encode_preview(self, frame: np.ndarray) -> bytes | None:
        """The camera frame as JPEG bytes for the handlers, None unless camera_preview is enabled."""
        if not self._camera_preview:
            return None
        jpeg = compress_to_jpeg(frame)
        return jpeg.tobytes() if jpeg is not None else None

    def _get_detection_lock(self, detection: str) -> threading.Lock:
        """Get or create a lock for a specific detection label.

        Args:
            detection (str): The detection label to get a lock for.

        Returns:
            threading.Lock: The lock for the specified detection.
        """
        with self._detection_locks_lock:
            if detection not in self._detection_locks:
                self._detection_locks[detection] = threading.Lock()
            return self._detection_locks[detection]

    def _execute_handler(self, key: str, payload: dict | None = None, frame: bytes | None = None) -> None:
        """Execute the handler registered for the given key.

        Args:
            key (str): The handler key, either a detection label or ``ALL_HANDLERS_KEY``.
            payload (dict): The data to pass to the handler (detection details or full detections dict).
            frame (bytes): The raw jpeg-encoded camera frame, if available.
        """
        with self._handlers_lock:
            handler = self._handlers.get(key)

        if not handler:
            return

        detection_lock = self._get_detection_lock(key)
        if not detection_lock.acquire(timeout=self._DETECTION_LOCK_TO):
            # Lock is already held by a running handler, discard this detection
            logger.debug(f"Handler for '{key}' is already running, skipping.")
            return

        # Debounce logic: check if enough time has passed since the last detection before invoking the handler
        now = time.time()
        last_time = self._last_detected.get(key, 0)
        if now - last_time >= self._debounce_sec:
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
        except RuntimeError:
            # Executor was shut down before the task could be submitted
            detection_lock.release()

    def override_threshold(self, value: float) -> None:
        """Override the confidence threshold of the detections.

        Args:
            value (float): The new value for the threshold in the range [0.0, 1.0].

        Raises:
            TypeError: If the value is not a number.
        """
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise TypeError("Invalid types for value.")
        logger.info(f"Overriding detection threshold. New confidence: {value}")
        self._confidence = float(value)
