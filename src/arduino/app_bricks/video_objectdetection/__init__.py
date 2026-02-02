# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

import time
import json
import inspect
import threading
import socket
import numpy as np
from typing import Callable

from websockets.sync.client import connect, ClientConnection
from websockets.exceptions import ConnectionClosedOK, ConnectionClosedError

from arduino.app_peripherals.camera import Camera, BaseCamera, WebSocketCamera
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

    def __init__(self, camera: BaseCamera = None, confidence: float = 0.3, debounce_sec: float = 0.0):
        """Initialize the VideoObjectDetection class.

        Args:
            camera (BaseCamera): The camera instance to use for capturing video. If None, a default camera will be initialized.
            confidence (float): Confidence level for detection. Default is 0.3 (30%).
            debounce_sec (float): Minimum seconds between repeated detections of the same object. Default is 0 seconds.

        Raises:
            RuntimeError: If the host address could not be resolved.
        """
        self._camera = camera if camera else Camera()

        self._confidence = confidence
        self._debounce_sec = debounce_sec
        self._last_detected: dict[str, float] = {}

        self._handlers = {}  # Dictionary to hold handlers for different actions
        self._handlers_lock = threading.Lock()

        self._is_running = threading.Event()

        infra = load_brick_compose_file(self.__class__)
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
        sig_args = inspect.signature(callback).parameters
        if len(sig_args) > 1:
            raise ValueError("Callback must accept 0 or 1 dictionary argument")

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
        sig_args = inspect.signature(callback).parameters
        if len(sig_args) != 1:
            raise ValueError("Callback must accept exactly one argument: the detected object.")

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

                    if isinstance(self._camera, WebSocketCamera):
                        # Send a priming frame to initialize the EI pipeline and its web server
                        frame = np.zeros((480, 640, 3), dtype=np.uint8)
                        jpeg_frame = compress_to_jpeg(frame)
                        tcp_socket.sendall(jpeg_frame.tobytes())

                    while self._is_running.is_set():
                        try:
                            frame = self._camera.capture()
                            if frame is None:
                                time.sleep(0.01)  # Brief sleep if no image available
                                continue

                            jpeg_frame = compress_to_jpeg(frame)
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

    def _process_message(self, ws: ClientConnection, message: str):
        jmsg = json.loads(message)
        if jmsg.get("type") == "hello":
            # Parse hello message to extract model info if needed
            logger.debug(f"Connected to model runner: {jmsg}")
            try:
                self._model_info = EdgeImpulseRunnerFacade.parse_model_info_message(jmsg)
                if self._model_info and self._model_info.thresholds is not None:
                    self._override_threshold(ws, self._confidence)

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
                    detections[detected_object] = detection_details

                    # Check if the class_id matches any registered handlers
                    self._execute_handler(detection=detected_object, detection_details=detection_details)

                if len(detections) > 0:
                    # If there are detections, invoke the all-detection handler
                    self._execute_global_handler(detections=detections)

        else:
            # Leave logging for unknown message types for debugging purposes
            logger.warning(f"Unknown message type: {jmsg.get('type')}")

    def _execute_handler(self, detection: str, detection_details: dict):
        """Execute the handler for the detected object if it exists.

        Args:
            detection (str): The label of the detected object.
            detection_details (dict): Dictionary containing 'confidence' (the detection confidence)
                and 'bounding_box_xyxy' (the detection bounding box coordinates).
        """
        now = time.time()
        with self._handlers_lock:
            handler = self._handlers.get(detection)
            if handler:
                last_time = self._last_detected.get(detection, 0)
                if now - last_time >= self._debounce_sec:
                    self._last_detected[detection] = now
                    logger.debug(f"Detected object: {detection}, invoking handler.")
                    sig_args = inspect.signature(handler).parameters
                    if len(sig_args) == 0:
                        handler()
                    else:
                        handler(detection_details)

    def _execute_global_handler(self, detections: dict = None):
        """Execute the global handler for the detected object if it exists.

        Args:
            detections (dict): The dictionary of detected objects and their details (e.g., confidence, bounding box).
        """
        now = time.time()
        with self._handlers_lock:
            handler = self._handlers.get(self.ALL_HANDLERS_KEY)
            if handler:
                last_time = self._last_detected.get(self.ALL_HANDLERS_KEY, 0)
                if now - last_time >= self._debounce_sec:
                    self._last_detected[self.ALL_HANDLERS_KEY] = now
                    logger.debug("Detected object: __ALL, invoking handler.")
                    sig_args = inspect.signature(handler).parameters
                    if len(sig_args) == 0:
                        handler()
                    else:
                        handler(detections)

    def _send_ws_message(self, ws: ClientConnection, message: dict):
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

    def _override_threshold(self, ws: ClientConnection, value: float):
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
