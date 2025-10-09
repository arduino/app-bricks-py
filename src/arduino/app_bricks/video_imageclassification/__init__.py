# SPDX-FileCopyrightText: Copyright (C) 2025 ARDUINO SA <http://www.arduino.cc>
#
# SPDX-License-Identifier: MPL-2.0

from arduino.app_utils import brick, Logger
from arduino.app_internal.core import load_brick_compose_file, resolve_address
from arduino.app_internal.core import EdgeImpulseRunnerFacade
import threading
import time
from typing import Callable
from websockets.sync.client import connect, ClientConnection
from websockets.exceptions import ConnectionClosedOK, ConnectionClosedError
import json
import inspect

logger = Logger("VideoImageClassification")


@brick
class VideoImageClassification:
    """Module for image classification on a **live video stream** using a specified machine learning model.

    Provides a way to react to detected classes over a video stream invoking registered actions in real-time.
    """

    ALL_HANDLERS_KEY = "__ALL"

    def __init__(self, confidence: float = 0.3, debounce_sec: float = 0.0):
        """Initialize the VideoImageClassification class.

        Args:
            confidence (float): The minimum confidence level for a classification to be considered valid. Default is 0.3.
            debounce_sec (float): The minimum time in seconds between consecutive detections of the same object
                to avoid multiple triggers. Default is 0 seconds.

        Raises:
             RuntimeError: If the host address could not be resolved.
        """
        self._confidence = confidence
        self._debounce_sec = debounce_sec
        self._last_detected = {}

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

    def on_detect_all(self, callback: Callable[[dict], None]):
        """Register a callback invoked for **every classification event**.

        This callback is useful if you want to process all classified labels in a single
        place, or be notified about any classification regardless of its type.

        Args:
            callback (Callable[[dict], None]):
                A function that accepts **exactly one argument**: a dictionary of
                classifications above the confidence threshold, in the form
                ``{"label": confidence, ...}``.

        Raises:
            TypeError: If `callback` is not a function.
            ValueError: If `callback` does not accept exactly one argument.
        """
        if not inspect.isfunction(callback):
            raise TypeError("Callback must be a callable function.")
        sig_args = inspect.signature(callback).parameters
        if len(sig_args) != 1:
            raise ValueError("Callback must accept exactly one argument (type dictionary): the detected object.")

        with self._handlers_lock:
            self._handlers[self.ALL_HANDLERS_KEY] = callback

    def on_detect(self, object: str, callback: Callable[[], None]):
        """Register a callback invoked when a **specific label** is classified.

        The callback is triggered whenever the given label appears in the classification
        results and passes the confidence and debounce filters.

        Args:
            object (str):
                The label to listen for (e.g., ``"dog"``).
            callback (Callable[[], None]):
                A function with **no parameters** that will be executed when the
                label is detected.

        Raises:
            TypeError: If `callback` is not a function.
            ValueError: If `callback` accepts one or more parameters.

        Notes:
            Registering a second callback for the same label overwrites the existing one.
        """
        if not inspect.isfunction(callback):
            raise TypeError("Callback must be a callable function.")
        sig_args = inspect.signature(callback).parameters
        if len(sig_args) > 0:
            raise ValueError("Callback must not accept any arguments.")

        with self._handlers_lock:
            if object in self._handlers:
                logger.warning(f"Handler for label '{object}' already exists. Overwriting.")
            self._handlers[object] = callback

    def start(self):
        """Start the classification stream.

        This only sets the internal running flag. You must call
        `execute` in a loop or a separate thread to actually begin receiving classification results.
        """
        self._is_running.set()

    def stop(self):
        """Stop the classification stream and release resources.

        This clears the running flag. Any active `execute` loop
        will exit gracefully at its next iteration.
        """
        self._is_running.clear()

    def execute(self):
        """Run the main classification loop.

        Behavior:
            - Opens a WebSocket connection to the model runner.
            - Receives classification messages in real time.
            - Filters classifications below the confidence threshold.
            - Applies debounce rules before invoking callbacks.
            - Retries on transient connection errors until stopped.

        Exceptions:
            ConnectionClosedOK:
                Raised to exit when the server closes the connection cleanly.
            ConnectionClosedError, TimeoutError, ConnectionRefusedError:
                Logged and retried with backoff.
        """
        while self._is_running.is_set():
            try:
                with connect(self._uri) as ws:
                    while self._is_running.is_set():
                        try:
                            message = ws.recv()
                            if not message:
                                continue
                            self._process_message(ws, message)
                        except ConnectionClosedOK:
                            raise
                        except (TimeoutError, ConnectionRefusedError, ConnectionClosedError):
                            logger.warning(f"Connection lost. Retrying...")
                            raise
                        except Exception as e:
                            logger.exception(f"Failed to process detection: {e}")
            except ConnectionClosedOK:
                logger.debug(f"Disconnected cleanly, exiting WebSocket read loop.")
                return
            except (TimeoutError, ConnectionRefusedError, ConnectionClosedError):
                logger.debug(f"Waiting for model runner. Retrying...")
                import time

                time.sleep(2)
                continue
            except Exception as e:
                logger.exception(f"Failed to establish WebSocket connection to {self._host}: {e}")

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

            det_classifications = {}
            classifications = result.get("classification", [])
            if classifications:
                for classification in classifications:
                    confidence = classifications[classification]
                    if confidence < self._confidence:
                        continue
                    det_classifications[classification] = confidence
                    self._execute_handler(classification)

                if len(det_classifications) > 0:
                    # If there are classified objects, invoke the all-detection handler
                    self._execute_handler(self.ALL_HANDLERS_KEY, det_classifications)

        else:
            # Leave logging for unknown message types for debugging purposes
            logger.warning(f"Unknown message type: {jmsg.get('type')}")

    def _execute_handler(self, classification: str, classifications: dict = None):
        """Execute the handler for the detected object if it exists.

        Args:
            classification (str): The classified object to check for in the registered handlers.
            classifications (dict, optional): The full dictionary of classifications if invoking the all-detection handler.
        """
        now = time.time()
        with self._handlers_lock:
            handler = self._handlers.get(classification)
            if handler:
                last_time = self._last_detected.get(classification, 0)
                if now - last_time >= self._debounce_sec:
                    self._last_detected[classification] = now
                    logger.debug(f"Classification: {classification}, invoking handler.")
                    if classifications is None:
                        handler()
                    else:
                        handler(classifications)

    def override_threshold(self, value: float):
        """Override the threshold for image classification model.

        Args:
            value (float): The new value for the threshold.

        Raises:
            TypeError: If the value is not a number.
            RuntimeError: If the model information is not available or does not support threshold override.
        """
        with connect(self._uri) as ws:
            self._override_threshold(ws, value)

    def _override_threshold(self, ws: ClientConnection, value: float):
        """Override the threshold for image classification model.

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
