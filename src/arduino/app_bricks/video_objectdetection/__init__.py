# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

import time
import json
import inspect
import threading
import socket
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import base64
from typing import Callable

from websockets.sync.client import connect
from websockets.sync.connection import Connection
from websockets.exceptions import ConnectionClosedOK, ConnectionClosedError

from arduino.app_peripherals.camera import Camera, BaseCamera
from arduino.app_internal.core import load_brick_compose_file, resolve_address
from arduino.app_internal.core import EdgeImpulseRunnerFacade
from arduino.app_utils.image.adjustments import compress_to_jpeg
from arduino.app_utils import brick, Logger

logger = Logger("VideoObjectDetection")


@brick
class VideoObjectDetection:
    """Module for object detection on a **live video stream** using a specified machine learning model.

    This brick:
      - Connects to a model runner over WebSocket.
      - Parses incoming classification messages with bounding boxes.
      - Filters detections by a configurable confidence threshold.
      - Debounces repeated triggers of the same label.
      - Invokes per-label callbacks and/or a catch-all callback.
    """

    ALL_HANDLERS_KEY = "__ALL"

    _DETECTION_LOCK_TO = 0.01  # Seconds to wait for a detection lock before discarding the detection signal

    def __init__(
        self,
        camera: BaseCamera | None = None,
        confidence: float = 0.3,
        debounce_sec: float = 0.0,
        camera_preview: bool = False,
    ):
        """Initialize the VideoObjectDetection class.

        Args:
            camera (BaseCamera): The camera instance to use for capturing video. If None, a default camera will be initialized.
            confidence (float): Confidence level for detection. Default is 0.3 (30%).
            debounce_sec (float): Minimum seconds between repeated detections of the same object. Default is 0 seconds.
            camera_preview (bool): Receive current camera frame on callback invocation.
                Frame is a raw jpeg-encoded image without bounding boxes applied on it. Default is False.

        Raises:
            RuntimeError: If the host address could not be resolved.
        """
        self._camera = camera if camera else Camera()

        self._confidence = confidence
        self._debounce_sec = debounce_sec
        self._last_detected: dict[str, float] = {}
        self._camera_preview = camera_preview
        self._last_camera_frame: str | None = None
        self._camera_preview_lock = threading.Lock()

        self._handlers_lock = threading.Lock()
        self._handlers = {}  # Dictionary to hold handlers for different actions

        self._detection_locks = {}  # Per-detection locks for fine-grained concurrency control
        self._detection_locks_lock = threading.Lock()  # Lock to protect _detection_locks dict

        self._executor = ThreadPoolExecutor(max_workers=5, thread_name_prefix="VideoObjectDetectionHandler")

        self._is_running = threading.Event()

        infra = load_brick_compose_file(self.__class__)
        if infra is None or "services" not in infra:
            raise RuntimeError("Infrastructure configuration could not be loaded.")
        for k, v in infra["services"].items():
            self._host = k
            break  # Only one service is expected

        self._host = resolve_address(self._host)
        if not self._host:
            raise RuntimeError("Host address could not be resolved. Please check your configuration.")

        self._uri = f"ws://{self._host}:4912"
        logger.info(f"[{self.__class__.__name__}] Host: {self._host} - URL: {self._uri}")

    def on_detect(self, object: str, callback: Callable[[], None]):
        """Register a callback invoked when a **specific label** is detected.

        Args:
            object (str): The label of the object to check for in the classification results.
            callback (Callable[[], None]): A function with **no parameters**.

        Raises:
            TypeError: If `callback` is not a function.
            ValueError: If `callback` accepts any parameters.
        """
        if not inspect.isfunction(callback):
            raise TypeError("Callback must be a callable function.")

        with self._handlers_lock:
            if object in self._handlers:
                logger.warning(f"Handler for object '{object}' already exists. Overwriting.")
            self._handlers[object] = callback

    def on_detect_all(self, callback: Callable[[dict], None]):
        """Register a callback invoked for **every detection event**.

        This is useful to receive a consolidated dictionary of detections for each frame.

        Args:
            callback (Callable[[dict], None]): A function that accepts **one dict argument** with
                the shape `{label: confidence, ...}`.

        Raises:
            TypeError: If `callback` is not a function.
            ValueError: If `callback` does not accept exactly one argument.
        """
        if not inspect.isfunction(callback):
            raise TypeError("Callback must be a callable function.")

        with self._handlers_lock:
            self._handlers[self.ALL_HANDLERS_KEY] = callback

    def start(self):
        """Start the video object detection process."""
        self._camera.start()
        self._is_running.set()

    def stop(self):
        """Stop the video object detection process and release resources."""
        self._is_running.clear()
        self._camera.stop()
        self._executor.shutdown(wait=False, cancel_futures=True)

    @brick.execute
    def object_detection_loop(self):
        """Object detection main loop.

        Maintains WebSocket connection to the model runner and processes object detection messages.
        Retries on connection errors until stopped.
        """
        while self._is_running.is_set():
            try:
                with connect(self._uri) as ws:
                    logger.info("WebSocket connection established")
                    while self._is_running.is_set():
                        try:
                            message = ws.recv()
                            if not message:
                                continue
                            if isinstance(message, (bytes, bytearray, memoryview)):
                                message = bytes(message).decode("utf-8")
                            self._process_message(ws, message)
                        except ConnectionClosedOK:
                            raise
                        except (TimeoutError, ConnectionRefusedError, ConnectionClosedError):
                            logger.warning(f"WebSocket connection lost. Retrying...")
                            raise
                        except Exception as e:
                            logger.exception(f"Failed to process detection: {e}")
            except ConnectionClosedOK:
                logger.debug(f"WebSocket disconnected cleanly, exiting loop.")
                return
            except (TimeoutError, ConnectionRefusedError, ConnectionClosedError):
                logger.debug(f"Waiting for model runner. Retrying...")
                time.sleep(2)
                continue
            except Exception as e:
                logger.exception(f"Failed to establish WebSocket connection to {self._host}: {e}")
                time.sleep(2)

    @brick.execute
    def camera_loop(self):
        """Camera main loop.

        Captures images from the camera and forwards them over the TCP connection.
        Retries on connection errors until stopped.
        """
        while self._is_running.is_set():
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as tcp_socket:
                    tcp_socket.connect((self._host, 5050))
                    logger.info(f"TCP connection established to {self._host}:5050")

                    # Send a priming frame to initialize the EI pipeline and its web server
                    res = (self._camera.resolution[1], self._camera.resolution[0], 3)
                    frame = np.zeros(res, dtype=np.uint8)
                    jpeg_frame = compress_to_jpeg(frame)
                    if jpeg_frame is not None:
                        tcp_socket.sendall(jpeg_frame.tobytes())

                    while self._is_running.is_set():
                        try:
                            frame = self._camera.capture()
                            if frame is None:
                                time.sleep(0.01)  # Brief sleep if no image available
                                continue

                            jpeg_frame = compress_to_jpeg(frame)
                            if jpeg_frame is not None:
                                tcp_socket.sendall(jpeg_frame.tobytes())

                        except (BrokenPipeError, ConnectionResetError, OSError) as e:
                            logger.warning(f"TCP connection lost: {e}. Retrying...")
                            break
                        except Exception as e:
                            logger.exception(f"Error sending image: {e}")

            except (ConnectionRefusedError, OSError) as e:
                logger.debug(f"TCP connection failed: {e}. Retrying in 2 seconds...")
                time.sleep(2)
            except Exception as e:
                logger.exception(f"Unexpected error in TCP loop: {e}")
                time.sleep(2)

    def _process_message(self, ws: Connection, message: str):
        jmsg = json.loads(message)
        if jmsg.get("type") == "hello":
            # Parse hello message to extract model info if needed
            logger.debug(
                f"Connected to model runner: {jmsg}. Configure confidence threshold: {self._confidence}, camera preview: {self._camera_preview}"
            )
            try:
                self._model_info = EdgeImpulseRunnerFacade.parse_model_info_message(jmsg)
                if self._model_info and self._model_info.thresholds is not None:
                    self._override_threshold(ws, self._confidence)
                if self._camera_preview:
                    self._toogle_camera_preview(ws, True)

            except Exception as e:
                logger.error(f"Error parsing WS hello message: {e}")
            return

        elif jmsg.get("type") == "handling-message-success":
            # Ignore handling-message-success messages
            return

        elif jmsg.get("type") == "classification":
            result = jmsg.get("result", {})
            if not isinstance(result, dict):
                return

            bounding_boxes = result.get("bounding_boxes", [])
            if bounding_boxes:
                if len(bounding_boxes) == 0:
                    return

                # If camera preview is enabled, decode the last received frame to pass to handlers
                frame = self._decode_preview_frame()

                # Process each bounding box
                detections = {}
                for box in bounding_boxes:
                    detected_object = box.get("label")
                    if detected_object is None:
                        continue

                    confidence = box.get("value", 0.0)
                    if confidence < self._confidence:
                        continue

                    # Extract bounding box coordinates if needed
                    xyxy_bbox = (
                        box.get("x", 0),
                        box.get("y", 0),
                        box.get("x", 0) + box.get("width", 0),
                        box.get("y", 0) + box.get("height", 0),
                    )

                    detection_details = {"confidence": confidence, "bounding_box_xyxy": xyxy_bbox}
                    if detected_object not in detections:
                        detections[detected_object] = []
                    detections[detected_object].append(detection_details)

                    # Check if the class_id matches any registered handlers
                    self._execute_handler(key=detected_object, payload=detection_details, frame=frame)

                if len(detections) > 0:
                    # If there are detections, invoke the all-detection handler
                    self._execute_handler(key=self.ALL_HANDLERS_KEY, payload=detections, frame=frame)

        elif jmsg.get("type") == "camera-preview":
            # Keep last camera preview frame if needed for callbacks
            img_base64 = jmsg.get("image")
            if img_base64 and self._camera_preview and isinstance(img_base64, str) and img_base64 != "":
                with self._camera_preview_lock:
                    # Image data is base64-encoded string (i.e. data:image/jpeg;base64,...)
                    self._last_camera_frame = img_base64
            return

        else:
            # Leave logging for unknown message types for debugging purposes
            logger.warning(f"Unknown message type: {jmsg.get('type')}")

    def _decode_preview_frame(self) -> bytes | None:
        """Decode the last received camera preview frame from base64 to a NumPy array.

        Returns:
            bytes: The decoded image data as bytes, or None if no valid preview frame is available.
                Image is jpeg encoded.

        """

        if self._camera_preview is False:
            return None

        last_frame = None
        with self._camera_preview_lock:
            if self._last_camera_frame is not None:
                last_frame = self._last_camera_frame

        if last_frame is not None and last_frame != "":
            try:
                split_frame = last_frame.split(",")
                if len(split_frame) != 2:
                    logger.debug(f"Unexpected format for camera preview frame: {last_frame[:50]}...")
                    return None
                return base64.b64decode(split_frame[1])
            except Exception as e:
                logger.error(f"Failed to decode camera preview frame: {e}")
                return None

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

    def _execute_handler(self, key: str, payload: dict | None = None, frame: bytes | None = None):
        """Execute the handler registered for the given key.

        Args:
            key (str): The handler key — either a detection label or ``ALL_HANDLERS_KEY``.
            payload (dict): The data to pass to the handler (detection details or full detections dict).
            frame (bytes): The raw jpeg-encoded camera frame, if available.
        """
        with self._handlers_lock:
            handler = self._handlers.get(key)

        if not handler:
            return

        detection_lock = self._get_detection_lock(key)
        if not detection_lock.acquire(timeout=self._DETECTION_LOCK_TO):
            # Lock is already held by a running handler — discard this detection
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

        def _run():
            try:
                logger.debug(f"Detected: {key}, invoking handler.")
                sig_args = inspect.signature(handler).parameters
                if len(sig_args) == 0:
                    handler()
                else:
                    if sig_args.get("frame") is not None:
                        handler(payload, frame=frame)
                    else:
                        handler(payload)
            finally:
                detection_lock.release()

        try:
            self._executor.submit(_run)
        except RuntimeError:
            # Executor was shut down before the task could be submitted
            detection_lock.release()

    def _send_ws_message(self, ws: Connection, message: dict):
        try:
            ws.send(json.dumps(message))
        except Exception as e:
            logger.error(f"Failed to send message over WebSocket: {e}")

    def override_threshold(self, value: float):
        """Override the threshold for object detection model.

        Args:
            value (float): The new value for the threshold in the range [0.0, 1.0].

        Raises:
            TypeError: If the value is not a number.
            RuntimeError: If the model information is not available or does not support threshold override.
        """
        with connect(self._uri) as ws:
            self._override_threshold(ws, value)

    def _override_threshold(self, ws: Connection, value: float):
        """Override the threshold for object detection model.

        Args:
            ws (ClientConnection): The WebSocket connection to send the message through.
            value (float): The new value for the threshold.

        Raises:
            TypeError: If the value is not a number.
            RuntimeError: If the model information is not available or does not support threshold override.
        """
        if not value or not isinstance(value, (int, float)):
            raise TypeError("Invalid types for value.")

        if self._model_info is None or self._model_info.thresholds is None or len(self._model_info.thresholds) == 0:
            raise RuntimeError("Model information is not available or does not support threshold override.")

        # Get first threshold and extract id. Then override it with the new confidence value.
        th = self._model_info.thresholds[0]
        id = th["id"]
        message = {"type": "threshold-override", "id": id, "key": "min_score", "value": value}

        logger.info(f"Overriding detection threshold. New confidence: {value}")
        ws.send(json.dumps(message))
        # Update local confidence value
        self._confidence = value

    def _toogle_camera_preview(self, ws: Connection, enabled: bool):
        """Toggle the camera preview on the model runner's web server.

        Args:
            ws (ClientConnection): The WebSocket connection to send the message through.
            enabled (bool): Whether to enable or disable the camera preview.

        Raises:
            TypeError: If `enabled` is not a boolean.
        """
        if not isinstance(enabled, bool):
            raise TypeError("Enabled must be a boolean value.")

        message = {"type": "toggle-camera-preview", "enabled": enabled}
        logger.info(f"Toggling camera preview to {'enabled' if enabled else 'disabled'}.")
        ws.send(json.dumps(message))
